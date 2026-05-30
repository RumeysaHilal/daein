"""
DAEIN-MFG — Raft Cluster Koordinatörü (Aşama 5)
=================================================
Birden fazla RaftNode'u bağlar ve aralarındaki mesaj iletimini yönetir.

OrchestratorAgent.REQUEST_CONSENSUS aksiyonu bu sınıfı çağırır.
Raft peer kuyrukları burada cross-connect edilir.
"""

import asyncio
import time
from orchestrator.raft import RaftNode, VoteBid
from config import Color, TIME_ACCELERATION


class RaftCluster:
    """
    N adet RaftNode'u birbirine bağlayan yönetici.
    Her node diğerlerinin inbox'ına doğrudan yazabilir.
    """

    def __init__(self, node_ids: list[str]):
        self.node_ids = node_ids
        self.nodes: dict[str, RaftNode] = {}

        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self.nodes[nid] = RaftNode(nid, peers)

        # Cross-connect: her node'un peer_queue'su karşı node'un inbox'ı
        for nid, node in self.nodes.items():
            for pid in node.peer_ids:
                peer_node = self.nodes[pid]
                # Bu node'un pid'e gönderdiği mesajlar → peer'in inbox'ına
                node._peer_queues[pid] = peer_node._inbox

        print(f"{Color.CYAN}[RAFT-CLUSTER]{Color.RESET} "
              f"{len(node_ids)} node bağlandı: {node_ids}")

    async def start(self):
        """Tüm Raft node'larını başlat."""
        async def _run_all():
            await asyncio.gather(
                *[node.run() for node in self.nodes.values()],
                return_exceptions=True
            )
        asyncio.get_event_loop().create_task(_run_all())
        await asyncio.sleep(0.4 / TIME_ACCELERATION)

    async def propose_consensus(
        self,
        proposer_id: str,
        anomaly_id: str,
        action: str,
        confidence: float,
        anomaly_score: float,
    ) -> dict:
        """
        Belirli bir node üzerinden konsensüs teklifi başlat.
        Returns: {'committed': bool, 'commit_ms': float, 'leader': str}
        """
        node = self.nodes.get(proposer_id)
        if not node:
            return {"committed": False, "commit_ms": 0, "leader": None}

        bid = VoteBid(
            voter_id         = proposer_id,
            anomaly_id       = anomaly_id,
            local_score      = anomaly_score,
            corroboration    = min(1.0, anomaly_score * 1.1),
            confidence       = confidence,
            suggested_action = action,
        )

        t0    = time.perf_counter()
        entry = await node.propose(anomaly_id, action, confidence, bid)
        ms    = (time.perf_counter() - t0) * 1000

        return {
            "committed": entry is not None and entry.committed,
            "commit_ms": round(ms, 2),
            "leader":    node.leader_id or proposer_id,
            "term":      node.current_term,
        }

    def report(self) -> list[dict]:
        return [n.report() for n in self.nodes.values()]
