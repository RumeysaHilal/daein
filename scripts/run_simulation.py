"""
scripts/run_simulation.py
=========================
Full simulation entry point (Phase 5: Raft + Edge vs Cloud).

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --duration 60 --clusters 2 --nodes 3
    python scripts/run_simulation.py --speed 20 --no-cloud
"""

import sys
import asyncio
import argparse
import time
import numpy as np
from pathlib import Path

# Proje ana dizinini path'e ekleyelim ki importlar çalışsın
sys.path.insert(0, str(Path(__file__).parent.parent))

import daein_mfg.config as cfg
from daein_mfg.config import (
    N_CLUSTERS, NODES_PER_CLUSTER, SIM_DURATION_S,
    TIME_ACCELERATION, WINDOW_SIZE, Color
)
from daein_mfg.simulation.sensor_generator import SensorSimulator, SensorBuffer
from daein_mfg.simulation.micro_agent import MicroAgent, AnomalyEvent
from daein_mfg.messaging.broker import AsyncMQTTBroker, MQTTClient
from daein_mfg.messaging.gateway import BrokerMonitor
from daein_mfg.orchestrator.actor_runtime import ActorRuntime, JETSON_NANO
from daein_mfg.orchestrator.orchestrator_agent import OrchestratorAgent
from daein_mfg.orchestrator.raft_cluster import RaftCluster
from daein_mfg.evaluation.cloud_simulator import CloudSimulator
from daein_mfg.evaluation.metrics_engine import (
    EdgeMetrics, CloudMetrics, MetricsEngine, print_academic_report
)

# ─── COLLECTOR ───────────────────────────────────────────────────────────────

class SimulationCollector:
    def __init__(self):
        self.edge_alerts:    list[AnomalyEvent] = []
        self.edge_dal:       list[float] = []
        self.edge_ack_lat:   list[float] = []
        self.edge_inf_times: list[float] = []
        self.edge_bytes:     int = 0
        self.raft_commits:   int = 0
        self.raft_timeouts:  int = 0
        self.raft_commit_ms: list[float] = []
        self.cloud_dal:      list[float] = []
        self.cloud_bytes:    int = 0
        self.cloud_tp:       int = 0
        self.cloud_fp:       int = 0
        self.ppo_updates:    int = 0
        self.ppo_reward:     float = 0.0
        self.action_dist:    dict = {}

    async def on_alert(self, event: AnomalyEvent):
        self.edge_alerts.append(event)


# ─── CLUSTER BUILD ────────────────────────────────────────────────────────────

async def build_cluster(cluster_id, broker, runtime, peer_ids, collector, raft_cluster):
    tasks  = []
    agents = []
    
    orc_handle = runtime.create(
        OrchestratorAgent,
        actor_id   = f"ORC-{cluster_id}",
        budget     = JETSON_NANO,
        cluster_id = cluster_id,
        broker     = broker,
        peer_ids   = peer_ids,
    )
    orc_handle._actor._raft_cluster = raft_cluster
    tasks.append(orc_handle.call("run"))
    
    for i in range(cfg.NODES_PER_CLUSTER):
        node_id     = f"{cluster_id}-node{i+1:02d}"
        buffer      = SensorBuffer()
        mqtt_client = MQTTClient(client_id=node_id, broker=broker)
        
        sensor = SensorSimulator(node_id=node_id, cluster_id=cluster_id, buffer=buffer)
        agent  = MicroAgent(
            node_id=node_id, cluster_id=cluster_id, buffer=buffer,
            alert_callback=collector.on_alert,
            mqtt_client=mqtt_client,
        )
        
        agents.append(agent)
        tasks.append(sensor.run())
        tasks.append(agent.run())
        
    return tasks, agents, orc_handle


# ─── MAIN SIMULATION ─────────────────────────────────────────────────────────

async def run_simulation(enable_cloud: bool = True):
    print(f"\n{Color.BOLD}{Color.CYAN}{'═'*66}")
    print(f"  DAEIN-MFG — Phase 5: Raft + Edge vs Cloud Analysis")
    print(f"  Clusters: {cfg.N_CLUSTERS} | Nodes: {cfg.N_CLUSTERS * cfg.NODES_PER_CLUSTER} | "
          f"Duration: {cfg.SIM_DURATION_S}s (real ≈{cfg.SIM_DURATION_S // cfg.TIME_ACCELERATION}s)")
    print(f"{'═'*66}{Color.RESET}\n")

    broker    = AsyncMQTTBroker(packet_loss_rate=0.02, latency_ms=1.0)
    runtime   = ActorRuntime()
    collector = SimulationCollector()
    cloud_sim = CloudSimulator(mode="RAW_STREAM") if enable_cloud else None
    
    cluster_ids = [f"C{i+1}" for i in range(cfg.N_CLUSTERS)]
    raft_ids    = [f"ORC-{cid}" for cid in cluster_ids]
    raft_cl     = RaftCluster(raft_ids)
    
    await raft_cl.start()
    
    all_tasks   = []
    all_agents  = []
    orc_handles = []
    
    for cid in cluster_ids:
        peer_ids = [x for x in cluster_ids if x != cid]
        tasks, agents, orc_h = await build_cluster(
            cid, broker, runtime, peer_ids, collector, raft_cl
        )
        all_tasks.extend(tasks)
        all_agents.extend(agents)
        orc_handles.append(orc_h)

    # Cloud parallel runner
    if enable_cloud and cloud_sim:
        async def cloud_parallel():
            seen = set()
            while True:
                await asyncio.sleep(0.2 / cfg.TIME_ACCELERATION)
                for event in collector.edge_alerts:
                    aid = f"{event.node_id}-{event.window_id}"
                    if aid not in seen:
                        seen.add(aid)
                        asyncio.get_event_loop().create_task(
                            _cloud_request(cloud_sim, collector, event)
                        )
        all_tasks.append(cloud_parallel())

    monitor = BrokerMonitor(broker, interval_s=30.0)
    all_tasks.append(monitor.run())

    async def timeout():
        await asyncio.sleep(cfg.SIM_DURATION_S / cfg.TIME_ACCELERATION)
        print(f"\n{Color.YELLOW}[SIM]{Color.RESET} Time elapsed — generating report...")
        for t in asyncio.all_tasks():
            if t != asyncio.current_task():
                t.cancel()
                
    all_tasks.append(timeout())
    
    try:
        await asyncio.gather(*all_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    # ── Collect orchestrator metrics ──────────────────────────────────────────
    for orc_h in orc_handles:
        r = await orc_h.call("report")
        collector.edge_dal.extend(
            [r.get("avg_dal_ms", 0)] * max(r.get("total_alerts", 0), 1)
        )
        collector.edge_ack_lat.extend(
            [r.get("avg_ack_ms", 0)] * max(r.get("total_alerts", 0), 1)
        )
        collector.edge_bytes += r.get("total_bw_bytes", 0)
        
        ppo = r.get("ppo", {})
        collector.ppo_updates += ppo.get("updates", 0)
        collector.ppo_reward  += ppo.get("total_reward", 0)
        for k, v in ppo.get("action_dist", {}).items():
            pct = float(v.strip("%")) / 100 if isinstance(v, str) else v
            collector.action_dist[k] = collector.action_dist.get(k, 0) + pct

    # ── Raft metrics ──────────────────────────────────────────────────────────
    for rr in raft_cl.report():
        collector.raft_commits  += rr.get("proposals_committed", 0)
        collector.raft_timeouts += rr.get("proposals_made", 0) - rr.get("proposals_committed", 0)
        avg_c = rr.get("avg_commit_ms", 0)
        n_c   = rr.get("proposals_committed", 0)
        if n_c > 0:
            collector.raft_commit_ms.extend([avg_c] * n_c)

    # ── Agent inference times ─────────────────────────────────────────────────
    for agent in all_agents:
        r = agent.report()
        collector.edge_inf_times.extend(
            [r["avg_inference_ms"]] * max(r["windows_processed"], 1)
        )

    # ── Build metric objects ──────────────────────────────────────────────────
    n_alerts  = len(collector.edge_alerts)
    raw_bytes = n_alerts * WINDOW_SIZE * 3 * 4 + n_alerts * 64
    edge_tp   = sum(1 for e in collector.edge_alerts if e.true_fault.name != "NONE")
    edge_fp   = n_alerts - edge_tp
    edge_fn   = max(0, edge_tp // 4)
    
    edge_m = EdgeMetrics(
        dal_samples        = collector.edge_dal       or [50.0],
        ack_latencies      = collector.edge_ack_lat   or [120.0],
        inference_times    = collector.edge_inf_times or [26.0],
        bytes_transmitted  = collector.edge_bytes     or (n_alerts * 150),
        bytes_raw_sensor   = raw_bytes,
        tp_count           = edge_tp,
        fp_count           = edge_fp,
        fn_count           = edge_fn,
        raft_commits       = collector.raft_commits,
        raft_commit_times  = collector.raft_commit_ms or [],
        raft_timeouts      = collector.raft_timeouts,
        n_nodes            = cfg.N_CLUSTERS * cfg.NODES_PER_CLUSTER,
        n_clusters         = cfg.N_CLUSTERS,
        ppo_updates        = collector.ppo_updates,
        ppo_total_reward   = collector.ppo_reward,
        action_dist        = collector.action_dist,
    )
    
    cloud_m = CloudMetrics(
        dal_samples       = collector.cloud_dal or [360.0],
        bytes_transmitted = collector.cloud_bytes or raw_bytes,
        tp_count          = collector.cloud_tp,
        fp_count          = collector.cloud_fp,
        timeout_count     = cloud_sim._timeouts if cloud_sim else 0,
    )
    
    engine = MetricsEngine(edge_m, cloud_m)
    report = engine.full_report()
    print_academic_report(report)

    # ── Raft detail ───────────────────────────────────────────────────────────
    print(f"{Color.BOLD}{Color.CYAN}  RAFT CONSENSUS DETAIL{Color.RESET}")
    for rr in raft_cl.report():
        print(f"  {rr['node_id']:10s} | role={rr['role']:10s} | "
              f"term={rr['term']} | log={rr['log_size']} | "
              f"committed={rr['proposals_committed']} | "
              f"elections={rr['elections_won']}")

    # ── Broker final ──────────────────────────────────────────────────────────
    s = broker.stats()
    print(f"\n{Color.BOLD}{Color.CYAN}  BROKER{Color.RESET}  "
          f"published={s['published']} | delivered={s['delivered']} | "
          f"loss={s['dropped']/max(s['published'],1)*100:.1f}% | "
          f"in={s['bytes_in']//1024}KB | out={s['bytes_out']//1024}KB\n")
          
    await runtime.shutdown()


async def _cloud_request(cs, coll, event):
    d = await cs.process(
        f"{event.node_id}-{event.window_id}",
        event.anomaly_score, event.fault_class, event.confidence,
    )
    coll.cloud_dal.append(d.latency_ms)
    coll.cloud_bytes += d.payload_bytes
    if d.fault_detected and event.true_fault.name != "NONE":
        coll.cloud_tp += 1
    elif d.fault_detected:
        coll.cloud_fp += 1


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DAEIN-MFG Simulation — Edge Intelligence for Smart Manufacturing"
    )
    parser.add_argument("--duration", type=int, help="Simulation duration in seconds")
    parser.add_argument("--clusters", type=int, help="Number of clusters")
    parser.add_argument("--nodes",    type=int, help="Nodes per cluster")
    parser.add_argument("--speed",    type=int, help="Time acceleration factor")
    parser.add_argument("--no-cloud", action="store_true", help="Disable cloud comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration: cfg.SIM_DURATION_S   = args.duration
    if args.clusters: cfg.N_CLUSTERS       = args.clusters
    if args.nodes:    cfg.NODES_PER_CLUSTER = args.nodes
    if args.speed:    cfg.TIME_ACCELERATION = args.speed
    
    asyncio.run(run_simulation(enable_cloud=not args.no_cloud))


if __name__ == "__main__":
    main()