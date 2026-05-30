"""
DAEIN-MFG — Raft Konsensüs Protokolü (Aşama 4)
================================================
Çok-orkestratör koordinasyonu için basitleştirilmiş Raft.

Tam Raft'tan farklar (simülasyon için):
  - Log replikasyonu: sadece anomali kararları için
  - Lider seçimi: rastgele timeout + asyncio event
  - Byzantine tolerans: median-of-bids ile basit FT
  - Gerçek Raft: etcd / rqlite kullanımı önerilir

Protokol akışı:
  1. Lider seçimi (election timeout 150–300ms)
  2. Lider teklif yayınlar (propose)
  3. Takipçiler oy kullanır (vote)
  4. Quorum sağlanınca log'a işlenir (commit)
  5. Commit sonrası aksiyon tetiklenir

Akademik not:
  Bu implementasyon CAP teoreminin CP tarafını tercih eder:
  Network partition durumunda consistency > availability.
  Endüstriyel güvenlik sistemleri için doğru tercih.
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import sys, os
# path setup removed — use pip install -e .
from daein_mfg.config import Color, TIME_ACCELERATION


# ─── TİPLER ───────────────────────────────────────────────────────────────────

class RaftRole(Enum):
    FOLLOWER  = auto()
    CANDIDATE = auto()
    LEADER    = auto()


@dataclass
class LogEntry:
    """Raft log kaydı — bir anomali kararını temsil eder."""
    epoch:       int
    anomaly_id:  str
    action:      str       # Karar: ESCALATE, ACK, DEFER, vb.
    confidence:  float     # [0,1]
    proposer_id: str
    committed:   bool = False
    committed_at: float = 0.0


@dataclass
class VoteBid:
    """
    Auction-based vote: her orkestratör kendi skorunu teklif eder.
    Median filtresi ile Byzantine node'lar elenir.
    """
    voter_id:         str
    anomaly_id:       str
    local_score:      float   # Lokal IsolationForest skoru
    corroboration:    float   # Komşu sensör koroborasyonu [0,1]
    confidence:       float
    suggested_action: str


# ─── RAFT NODE ────────────────────────────────────────────────────────────────

class RaftNode:
    """
    Tek bir orkestratörün Raft durumunu yönetir.

    Temel garantiler:
      - Quorum = ⌊N/2⌋ + 1 oydan oluşur
      - Commit sadece quorum sonrası gerçekleşir
      - Lider çöküşünde yeni seçim başlar (timeout: 150–300ms sim)
      - Bid filtreleme: medyan ± 2σ dışındaki teklifler reddedilir
    """

    ELECTION_TIMEOUT_MIN = 0.15 / TIME_ACCELERATION
    ELECTION_TIMEOUT_MAX = 0.30 / TIME_ACCELERATION
    HEARTBEAT_INTERVAL   = 0.05 / TIME_ACCELERATION

    def __init__(self, node_id: str, peer_ids: list[str]):
        self.node_id  = node_id
        self.peer_ids = peer_ids          # Diğer orkestratör kimlikleri
        self.n_nodes  = len(peer_ids) + 1
        self.quorum   = self.n_nodes // 2 + 1

        # Raft durumu
        self.role         = RaftRole.FOLLOWER
        self.current_term = 0
        self.voted_for:   Optional[str] = None
        self.leader_id:   Optional[str] = None
        self.log:         list[LogEntry] = []

        # Oylar ve teklifler
        self._votes_received:  dict[str, bool] = {}
        self._bids_received:   dict[str, VoteBid] = {}
        self._pending_commits: dict[str, asyncio.Event] = {}

        # Zamanlama
        self._last_heartbeat = time.time()
        self._election_timeout = random.uniform(
            self.ELECTION_TIMEOUT_MIN, self.ELECTION_TIMEOUT_MAX
        )

        # Peer iletişim kanalları (orkestratörler arası)
        # Her peer için bir asyncio.Queue — gerçek sistemde gRPC/MQTT
        self._peer_queues: dict[str, asyncio.Queue] = {
            pid: asyncio.Queue(maxsize=50) for pid in peer_ids
        }
        self._inbox: asyncio.Queue = asyncio.Queue(maxsize=100)

        # İstatistikler
        self.stats = {
            "elections_started":   0,
            "elections_won":       0,
            "proposals_made":      0,
            "proposals_committed": 0,
            "bids_rejected":       0,
            "avg_commit_ms":       0.0,
        }

        print(f"{Color.CYAN}[RAFT]{Color.RESET} Node {Color.BOLD}{node_id}{Color.RESET} "
              f"başlatıldı | quorum={self.quorum}/{self.n_nodes}")

    async def run(self):
        """Raft ana döngüsü."""
        await asyncio.gather(
            self._election_timer(),
            self._message_processor(),
        )

    async def propose(
        self,
        anomaly_id:    str,
        action:        str,
        confidence:    float,
        local_bid:     VoteBid,
    ) -> Optional[LogEntry]:
        """
        Anomali kararı için konsensüs teklif et.
        Lider değilsek önce lider seçimi tetikle.
        Returns: Commit edilmiş LogEntry veya None (timeout)
        """
        # Lider değilsek seçimi zorla
        if self.role != RaftRole.LEADER:
            await self._start_election()

        # Hâlâ lider değilsek teklifi en iyi bid'e bırak
        if self.role != RaftRole.LEADER:
            return None

        t0 = time.perf_counter()
        self.stats["proposals_made"] += 1

        entry = LogEntry(
            epoch       = self.current_term,
            anomaly_id  = anomaly_id,
            action      = action,
            confidence  = confidence,
            proposer_id = self.node_id,
        )

        # Kendi teklifimizi kaydet
        self._bids_received = {self.node_id: local_bid}

        # Peerlara yayınla
        msg = {
            "type":      "PROPOSE",
            "term":      self.current_term,
            "proposer":  self.node_id,
            "anomaly_id": anomaly_id,
            "action":    action,
            "confidence": confidence,
            "bid":       {
                "voter_id":         local_bid.voter_id,
                "anomaly_id":       local_bid.anomaly_id,
                "local_score":      local_bid.local_score,
                "corroboration":    local_bid.corroboration,
                "confidence":       local_bid.confidence,
                "suggested_action": local_bid.suggested_action,
            }
        }
        await self._broadcast(msg)

        # Quorum bekleme (simülasyon: timeout 300ms)
        commit_event = asyncio.Event()
        self._pending_commits[anomaly_id] = commit_event

        try:
            await asyncio.wait_for(commit_event.wait(), timeout=0.3 / TIME_ACCELERATION)
            entry.committed    = True
            entry.committed_at = time.time()
            self.log.append(entry)
            self.stats["proposals_committed"] += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Hareketli ortalama
            n = self.stats["proposals_committed"]
            self.stats["avg_commit_ms"] = (
                self.stats["avg_commit_ms"] * (n - 1) + elapsed_ms
            ) / n

            print(
                f"{Color.CYAN}[RAFT ✓]{Color.RESET} {self.node_id} | "
                f"anomaly={anomaly_id[:12]} | action={action} | "
                f"commit={elapsed_ms:.1f}ms | "
                f"bids={len(self._bids_received)}/{self.n_nodes}"
            )
            return entry

        except asyncio.TimeoutError:
            self._pending_commits.pop(anomaly_id, None)
            print(f"{Color.YELLOW}[RAFT ✗]{Color.RESET} {self.node_id} | "
                  f"timeout: {anomaly_id[:12]}")
            return None

    async def receive_bid(self, bid: VoteBid):
        """Bir peer'dan teklif aldık — inbox'a ekle."""
        await self._inbox.put({"type": "BID", "bid": bid})

    async def _message_processor(self):
        """Gelen mesajları işle."""
        while True:
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=0.1)
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "PROPOSE":
            await self._handle_propose(msg)
        elif msg_type == "BID":
            await self._handle_bid(msg["bid"])
        elif msg_type == "VOTE":
            await self._handle_vote(msg)
        elif msg_type == "HEARTBEAT":
            self._last_heartbeat = time.time()
            self.leader_id       = msg.get("leader_id")
            self.role            = RaftRole.FOLLOWER

    async def _handle_propose(self, msg: dict):
        """Liderden gelen teklifi değerlendir, bid gönder."""
        term = msg.get("term", 0)
        if term < self.current_term:
            return  # Eski term, reddet

        self._last_heartbeat = time.time()
        self.role = RaftRole.FOLLOWER

        # Basit koroboras: lokal skor 0.5 varsayılan (gerçekte sensör sorgusu)
        bid = VoteBid(
            voter_id         = self.node_id,
            anomaly_id       = msg["anomaly_id"],
            local_score      = random.uniform(0.4, 0.9),  # Simüle lokal gözlem
            corroboration    = random.uniform(0.3, 0.8),
            confidence       = msg["confidence"],
            suggested_action = msg["action"],
        )

        # Lidere bid gönder
        vote_msg = {"type": "BID", "bid": bid, "from": self.node_id}
        leader = msg["proposer"]
        if leader in self._peer_queues:
            try:
                self._peer_queues[leader].put_nowait(vote_msg)
            except asyncio.QueueFull:
                pass

    async def _handle_bid(self, bid: VoteBid):
        """Peer'dan gelen bid'i işle, quorum kontrolü yap."""
        if not isinstance(bid, VoteBid):
            # Dict'ten VoteBid oluştur
            if isinstance(bid, dict):
                bid = VoteBid(**bid)
            else:
                return

        self._bids_received[bid.voter_id] = bid

        # Quorum kontrolü
        if len(self._bids_received) >= self.quorum:
            anomaly_id = bid.anomaly_id
            if self._validate_bids(anomaly_id):
                if anomaly_id in self._pending_commits:
                    self._pending_commits[anomaly_id].set()

    def _validate_bids(self, anomaly_id: str) -> bool:
        """
        Byzantine filtresi: medyan ± 2σ dışındaki teklifler reddedilir.
        Tek bir sahte node'un kararı etkilemesini önler.
        """
        bids  = list(self._bids_received.values())
        scores = [b.local_score for b in bids]

        if len(scores) < 2:
            return True  # Tek node, filtre uygulama

        median = float(sorted(scores)[len(scores) // 2])
        std    = float(
            (sum((s - median) ** 2 for s in scores) / len(scores)) ** 0.5
        )

        valid = [s for s in scores if abs(s - median) <= 2 * std + 0.01]
        rejected = len(scores) - len(valid)

        if rejected > 0:
            self.stats["bids_rejected"] += rejected

        # Geçerli bid'lerin ortalaması
        avg_valid = sum(valid) / len(valid)
        return avg_valid > 0.3  # Minimum güven eşiği

    async def _start_election(self):
        """Lider seçimi başlat."""
        self.role          = RaftRole.CANDIDATE
        self.current_term += 1
        self.voted_for     = self.node_id
        self._votes_received = {self.node_id: True}
        self.stats["elections_started"] += 1

        vote_req = {
            "type":      "VOTE_REQUEST",
            "term":      self.current_term,
            "candidate": self.node_id,
        }
        await self._broadcast(vote_req)

        # Oylar için kısa bekleme
        await asyncio.sleep(self.ELECTION_TIMEOUT_MIN)

        yes_votes = sum(1 for v in self._votes_received.values() if v)
        if yes_votes >= self.quorum:
            self.role      = RaftRole.LEADER
            self.leader_id = self.node_id
            self.stats["elections_won"] += 1
            print(f"{Color.GREEN}[RAFT]{Color.RESET} {self.node_id} "
                  f"lider seçildi | term={self.current_term}")

    async def _handle_vote(self, msg: dict):
        """Oy isteği veya oyu işle."""
        if msg.get("type") == "VOTE_REQUEST":
            term = msg.get("term", 0)
            if term > self.current_term:
                self.current_term = term
                self.voted_for    = msg["candidate"]
                resp = {"type": "VOTE", "term": term, "granted": True, "from": self.node_id}
                cand = msg["candidate"]
                if cand in self._peer_queues:
                    try:
                        self._peer_queues[cand].put_nowait(resp)
                    except asyncio.QueueFull:
                        pass

        elif msg.get("type") == "VOTE":
            if msg.get("granted") and self.role == RaftRole.CANDIDATE:
                self._votes_received[msg["from"]] = True

    async def _election_timer(self):
        """Election timeout kontrolü."""
        while True:
            await asyncio.sleep(self.ELECTION_TIMEOUT_MAX)
            if self.role == RaftRole.FOLLOWER:
                since_hb = time.time() - self._last_heartbeat
                if since_hb > self._election_timeout:
                    await self._start_election()

    async def _broadcast(self, msg: dict):
        """Tüm peerlara mesaj gönder."""
        for pid, q in self._peer_queues.items():
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def connect_peer_inbox(self, peer_id: str) -> asyncio.Queue:
        """Peer'ın bu node'a mesaj göndereceği kuyruğu döndür."""
        return self._inbox

    def report(self) -> dict:
        return {
            "node_id":    self.node_id,
            "role":       self.role.name,
            "term":       self.current_term,
            "log_size":   len(self.log),
            **self.stats,
        }
