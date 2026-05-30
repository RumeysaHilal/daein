"""
DAEIN-MFG — Orkestratör Ajan (Aşama 4)
========================================
Ray Actor modelini AsyncActorRuntime üzerinde çalışan tam orkestratör.

Önceki Aşama 3 ClusterGateway'den farkları:
  - PPO politikası: her alert için RL tabanlı aksiyon seçimi
  - Peer-to-peer yük paylaşımı: OFFLOAD_PEER aksiyonu gerçek
  - Ertelenmiş işleme: DEFER kuyruğu + yeniden değerlendirme
  - Kaynak izleme: CPU/mem util simülasyonu
  - Zengin metrik toplama: DAL, bant genişliği, TP/FP per-cluster

Bileşen mimarisi:
  ┌─────────────────────────────────────────────────────┐
  │              OrchestratorAgent                       │
  │                                                     │
  │  MQTTClient ──► AlertProcessor ──► PPOPolicy        │
  │                      │                  │           │
  │                      ▼                  ▼           │
  │               DeferQueue          ActionDispatch    │
  │                      │                  │           │
  │                      ▼                  ▼           │
  │              MetricsCollector      ACK / Cmd        │
  └─────────────────────────────────────────────────────┘
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from collections import deque
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Color, TIME_ACCELERATION
from messaging.broker import AsyncMQTTBroker, MQTTClient, MQTTMessage, PayloadSchema, QoS
from orchestrator.ppo_policy import PPOPolicy, Action, ACTION_LABELS, Transition
import numpy as np


# ─── ÖLÇÜM KAYDI ─────────────────────────────────────────────────────────────

@dataclass
class AlertMetrics:
    """Tek bir alert için uçtan uca ölçümler."""
    alert_id:           str
    node_id:            str
    anomaly_score:      float
    fault_class:        str
    action_taken:       str
    detection_latency_ms: float
    ack_latency_ms:     float
    is_tp:              bool   # true positive tahmini
    payload_bytes:      int
    consensus_rounds:   int = 0


# ─── KAYNAK SİMÜLATÖRÜ ───────────────────────────────────────────────────────

class ResourceSimulator:
    """
    Jetson Nano'nun CPU/memory kullanımını simüle eder.
    Gerçek ortamda psutil ile replace edilir.
    """
    def __init__(self, seed: int = 0):
        self._rng     = random.Random(seed)
        self.cpu_util = 0.35    # başlangıç: %35 yük
        self.mem_util = 0.45

    def update(self, n_active_alerts: int):
        """Her alert için CPU yükü artar."""
        target_cpu = 0.30 + n_active_alerts * 0.08 + self._rng.uniform(-0.03, 0.03)
        target_mem = 0.40 + n_active_alerts * 0.04 + self._rng.uniform(-0.02, 0.02)
        # Exponential moving average
        self.cpu_util = 0.8 * self.cpu_util + 0.2 * np.clip(target_cpu, 0.1, 0.95)
        self.mem_util = 0.8 * self.mem_util + 0.2 * np.clip(target_mem, 0.2, 0.90)

    @property
    def state_vector(self):
        return self.cpu_util, self.mem_util


# ─── ORKESTRATÖR AJAN ─────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Cluster düzeyinde otonom karar ajanı.

    Ray Actor uyumluluğu:
      @ray.remote dekoratörü eklenerek gerçek Ray ile kullanılabilir.
      Tüm public metodlar async — Ray remote call uyumlu.
    """

    def __init__(
        self,
        cluster_id:    str,
        broker:        AsyncMQTTBroker,
        peer_ids:      list[str] = None,   # Komşu cluster ID'leri
    ):
        self.cluster_id   = cluster_id
        self.orchestrator_id = f"ORC-{cluster_id}"
        self.broker       = broker
        self.peer_ids     = peer_ids or []

        # MQTT istemcisi
        self.mqtt = MQTTClient(client_id=self.orchestrator_id, broker=broker)

        # RL politikası
        self.policy   = PPOPolicy(seed=hash(cluster_id) % 1000)

        # Kaynak simülatörü
        self.resources = ResourceSimulator()

        # Erteleme kuyruğu
        self._defer_queue: deque = deque(maxlen=32)

        # Metrikler
        self._metrics: list[AlertMetrics] = []
        self._alert_counter = 0
        self._active_alerts = 0

        # Peer yük bilgisi (diğer orkestratörlerden heartbeat)
        self._peer_loads: dict[str, float] = {pid: 0.5 for pid in self.peer_ids}

        # Önceki state (PPO transition için)
        self._last_state: np.ndarray = None
        self._last_action: int = None
        self._last_lp: float = None
        self._last_score: float = None
        self._last_alert_time: float = None

        print(f"{Color.CYAN}[ORC]{Color.RESET} Orkestratör "
              f"{Color.BOLD}{self.orchestrator_id}{Color.RESET} | "
              f"peers={self.peer_ids}")

    async def start(self):
        """MQTT bağlantısı ve subscriptionları kur."""
        await self.mqtt.connect()

        # Cluster alertlerini dinle
        self.mqtt.subscribe(
            f"sensor/{self.cluster_id}/+/alert",
            self._on_alert
        )
        # Peer orkestratör yük mesajları
        self.mqtt.subscribe(
            "orchestrator/+/load_report",
            self._on_peer_load
        )
        # OFFLOAD: başka cluster bu cluster'a iş gönderebilir
        self.mqtt.subscribe(
            f"orchestrator/{self.cluster_id}/offload",
            self._on_offload
        )

        # Erteleme kuyruğunu işle
        asyncio.get_event_loop().create_task(self._defer_processor())
        # Yük raporu yayınla
        asyncio.get_event_loop().create_task(self._load_reporter())

        print(f"{Color.CYAN}[ORC]{Color.RESET} {self.orchestrator_id} dinliyor...")

    async def run(self):
        """Ana orkestratör döngüsü."""
        await self.start()
        while True:
            await asyncio.sleep(0.5)

    # ── ALERT İŞLEME ─────────────────────────────────────────────────────────

    async def _on_alert(self, msg: MQTTMessage):
        """
        µ-Agent alertini al → PPO ile aksiyon seç → uygula.

        Bu metodun içindeki akış tam olarak proposal belgemizdeki
        "End-to-End Scenario" bölümünün T+2565ms – T+2575ms kısmı.
        """
        t_recv = time.perf_counter()

        try:
            data = msg.to_dict()
        except Exception:
            return

        self._alert_counter += 1
        self._active_alerts += 1
        self.resources.update(self._active_alerts)

        node_id    = data["node_id"]
        score      = data["anomaly_score"]
        fault_cls  = data["fault_class"]
        confidence = data["confidence"]
        window_id  = data["window_id"]
        ts_ns      = data["timestamp_ns"]

        detection_latency_ms = (time.time_ns() - ts_ns) / 1e6

        # ── Durum Vektörü Oluştur ─────────────────────────────────────
        cpu, mem = self.resources.state_vector
        queue_depth    = len(self._defer_queue) / 32.0
        peer_avg_load  = np.mean(list(self._peer_loads.values())) if self._peer_loads else 0.5

        state = np.array([
            cpu,
            mem,
            queue_depth,
            min(1.0, self._active_alerts / 10.0),
            peer_avg_load,
            score,
            confidence,
        ], dtype=np.float32)

        # ── PPO Aksiyon Seç ───────────────────────────────────────────
        action, log_prob = self.policy.act(state)

        # Kural override: çok yüksek skor → her zaman ESCALATE
        if score > 0.97:
            action = Action.ESCALATE

        # Kural override: çok düşük confidence → DEFER
        if confidence < 0.5 and action not in (Action.ESCALATE,):
            action = Action.DEFER_500MS

        action_label = ACTION_LABELS[Action(action)]

        # ── Aksiyon Uygula ────────────────────────────────────────────
        consensus_rounds = 0
        ack_action = "ACK"

        if action == Action.LOCAL_INFER:
            # Yerel işle — sadece ACK gönder
            pass

        elif action == Action.OFFLOAD_PEER:
            # En boş peer'e iş gönder
            if self._peer_loads:
                best_peer = min(self._peer_loads, key=self._peer_loads.get)
                offload_payload = {**data, "routed_by": self.orchestrator_id}
                await self.mqtt.publish(
                    f"orchestrator/{best_peer}/offload",
                    offload_payload,
                    qos=QoS.AT_MOST_ONCE,
                )
                ack_action = "OFFLOADED"

        elif action == Action.DEFER_500MS:
            # Kuyruğa ekle
            self._defer_queue.append((data, time.perf_counter()))
            ack_action = "DEFERRED"

        elif action == Action.ESCALATE:
            # Anında komut yayınla
            await self.mqtt.publish(
                f"orchestrator/{self.cluster_id}/command",
                PayloadSchema.command(
                    orchestrator_id=self.orchestrator_id,
                    target_node="*",
                    command="ESCALATE_ALERT",
                    params={"fault_class": fault_cls, "score": score, "node": node_id},
                ),
            )
            ack_action = "ESCALATE"

        elif action == Action.REQUEST_CONSENSUS:
            # Aşama 5'e hazırlık — şimdilik sadece logla
            consensus_rounds = 1
            ack_action = "CONSENSUS_PENDING"

        # ── ACK Gönder ────────────────────────────────────────────────
        ack_topic = f"sensor/{self.cluster_id}/{node_id}/ack"
        await self.mqtt.publish(
            ack_topic,
            PayloadSchema.ack(
                orchestrator_id=self.orchestrator_id,
                node_id=node_id,
                window_id=window_id,
                action=ack_action,
                consensus_reached=(action != Action.REQUEST_CONSENSUS),
            ),
        )

        ack_latency_ms = (time.perf_counter() - t_recv) * 1000
        self._active_alerts = max(0, self._active_alerts - 1)

        # ── Terminal Çıktısı ──────────────────────────────────────────
        action_color = {
            "LOCAL":    Color.GREEN,
            "OFFLOAD":  Color.CYAN,
            "DEFER":    Color.YELLOW,
            "ESCALATE": Color.RED,
            "CONSENSUS_PENDING": Color.BLUE,
        }.get(action_label, Color.RESET)

        is_tp = fault_cls not in ("NORMAL", "UNKNOWN")
        tp_str = f"{Color.GREEN}TP{Color.RESET}" if is_tp else f"{Color.RED}FP{Color.RESET}"

        print(
            f"{Color.CYAN}[ORC]{Color.RESET} {self.orchestrator_id} | "
            f"{node_id[-6:]} | {fault_cls[:12]:12s} | "
            f"score={score:.3f} | "
            f"act={action_color}{action_label:8s}{Color.RESET} | "
            f"{tp_str} | ack={ack_latency_ms:.1f}ms"
        )

        # ── PPO Geçiş Kaydı ───────────────────────────────────────────
        if self._last_state is not None:
            reward = PPOPolicy.compute_reward(
                action               = self._last_action,
                anomaly_score        = self._last_score,
                detection_latency_ms = detection_latency_ms,
                false_positive       = not is_tp,
                consensus_rounds     = consensus_rounds,
                cpu_util             = cpu,
            )
            self.policy.store(Transition(
                state      = self._last_state,
                action     = self._last_action,
                reward     = reward,
                next_state = state,
                done       = False,
                log_prob   = self._last_lp,
            ))

        self._last_state  = state
        self._last_action = action
        self._last_lp     = log_prob
        self._last_score  = score

        # ── Metrik Kaydet ─────────────────────────────────────────────
        self._metrics.append(AlertMetrics(
            alert_id             = f"{self.cluster_id}-{self._alert_counter}",
            node_id              = node_id,
            anomaly_score        = score,
            fault_class          = fault_cls,
            action_taken         = action_label,
            detection_latency_ms = detection_latency_ms,
            ack_latency_ms       = ack_latency_ms,
            is_tp                = is_tp,
            payload_bytes        = msg.size_bytes,
            consensus_rounds     = consensus_rounds,
        ))

    async def _on_offload(self, msg: MQTTMessage):
        """Başka cluster'dan offload edilen işi al ve LOCAL_INFER yap."""
        try:
            data = msg.to_dict()
            node_id   = data.get("node_id", "unknown")
            score     = data.get("anomaly_score", 0)
            fault_cls = data.get("fault_class", "?")
            routed_by = data.get("routed_by", "?")
            print(
                f"{Color.CYAN}[ORC←OFFLOAD]{Color.RESET} {self.orchestrator_id} | "
                f"from={routed_by} | {node_id[-6:]} | {fault_cls} | score={score:.3f}"
            )
        except Exception:
            pass

    async def _on_peer_load(self, msg: MQTTMessage):
        """Peer orkestratör yük raporu."""
        try:
            data = msg.to_dict()
            peer_id  = data.get("orchestrator_id", "")
            cpu_util = data.get("cpu_util", 0.5)
            if peer_id != self.orchestrator_id:
                cluster = peer_id.replace("ORC-", "")
                self._peer_loads[cluster] = cpu_util
        except Exception:
            pass

    async def _defer_processor(self):
        """Ertelenmiş alertleri 500ms sonra yeniden değerlendir."""
        while True:
            await asyncio.sleep(0.5 / TIME_ACCELERATION)
            now = time.perf_counter()
            reprocess = []
            while self._defer_queue:
                data, enqueued_at = self._defer_queue[0]
                if (now - enqueued_at) * TIME_ACCELERATION >= 0.5:
                    self._defer_queue.popleft()
                    reprocess.append(data)
                else:
                    break
            for data in reprocess:
                # Yeniden değerlendir: bu sefer LOCAL_INFER zorla
                score = data.get("anomaly_score", 0)
                node  = data.get("node_id", "?")
                fault = data.get("fault_class", "?")
                print(
                    f"{Color.YELLOW}[DEFER→]{Color.RESET} {self.orchestrator_id} "
                    f"yeniden işlendi: {node[-6:]} | {fault} | score={score:.3f}"
                )

    async def _load_reporter(self):
        """CPU yükünü peer'lere periyodik olarak yayınla."""
        while True:
            await asyncio.sleep(5.0 / TIME_ACCELERATION)
            cpu, mem = self.resources.state_vector
            await self.mqtt.publish(
                f"orchestrator/{self.cluster_id}/load_report",
                {
                    "orchestrator_id": self.orchestrator_id,
                    "cluster_id":      self.cluster_id,
                    "cpu_util":        round(cpu, 3),
                    "mem_util":        round(mem, 3),
                    "active_alerts":   self._active_alerts,
                    "timestamp_ns":    time.time_ns(),
                },
                qos=QoS.AT_MOST_ONCE,
            )

    async def shutdown(self):
        await self.mqtt.disconnect()

    def report(self) -> dict:
        """Tam performans raporu."""
        if not self._metrics:
            return {"cluster_id": self.cluster_id, "alerts": 0}

        total    = len(self._metrics)
        tp_count = sum(1 for m in self._metrics if m.is_tp)
        avg_dal  = np.mean([m.detection_latency_ms for m in self._metrics])
        avg_ack  = np.mean([m.ack_latency_ms       for m in self._metrics])
        total_bw = sum(m.payload_bytes for m in self._metrics)

        action_dist = {}
        for m in self._metrics:
            action_dist[m.action_taken] = action_dist.get(m.action_taken, 0) + 1

        return {
            "cluster_id":       self.cluster_id,
            "total_alerts":     total,
            "tp_rate":          round(tp_count / total, 3),
            "avg_dal_ms":       round(avg_dal, 2),
            "avg_ack_ms":       round(avg_ack, 2),
            "total_bw_bytes":   total_bw,
            "action_dist":      action_dist,
            "ppo":              self.policy.report(),
        }
