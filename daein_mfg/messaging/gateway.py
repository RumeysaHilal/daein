"""
DAEIN-MFG — Cluster Gateway Orkestratörü (Aşama 3)
====================================================
MQTT tabanlı cluster yöneticisi.

Bu Aşama 3 sürümü:
  - Tüm cluster'ın alert akışını dinler
  - Her alert için ACK gönderir (CONSENSUS_WAIT'i açar)
  - Cluster düzeyinde istatistik toplar
  - Yüksek şiddetli alertlerde komut yayınlar

Aşama 4'te eklenecekler:
  - Ray Actor tabanlı dağıtık yapı
  - PPO tabanlı RL politikası
  - Raft konsensüs protokolü
  - Phi-3-mini SLM entegrasyonu
"""

import asyncio
import time
from dataclasses import dataclass, field
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Color, TIME_ACCELERATION
from messaging.broker import AsyncMQTTBroker, MQTTClient, MQTTMessage, PayloadSchema, QoS


# ─── ALERT KAYDEDICI ──────────────────────────────────────────────────────────

@dataclass
class ClusterAlertRecord:
    """Bir cluster'da gözlenen tüm alertlerin kayıt defteri."""
    cluster_id:   str
    alerts:       list = field(default_factory=list)
    node_health:  dict = field(default_factory=dict)   # node_id → son heartbeat zamanı
    fault_counts: dict = field(default_factory=lambda: defaultdict(int))

    def add_alert(self, data: dict):
        self.alerts.append({**data, "received_at": time.time()})
        self.fault_counts[data.get("fault_class", "UNKNOWN")] += 1

    def summary(self) -> dict:
        total   = len(self.alerts)
        classes = dict(self.fault_counts)
        tp_rate = sum(
            1 for a in self.alerts
            if a.get("fault_class") not in ("NORMAL", "UNKNOWN")
        ) / max(total, 1)
        return {
            "cluster_id":    self.cluster_id,
            "total_alerts":  total,
            "fault_classes": classes,
            "tp_estimate":   round(tp_rate, 3),
        }


# ─── CLUSTER GATEWAY ──────────────────────────────────────────────────────────

class ClusterGateway:
    """
    Tek bir cluster'ı yöneten gateway ajan.

    Sorumlulukları:
      1. sensor/{cluster_id}/+/alert → dinle, ACK gönder
      2. system/heartbeat            → node sağlığı izle
      3. Şiddetli alert → tüm cluster'a REDUCE_SAMPLING komutu
      4. İstatistik toplama ve raporlama
    """

    # Bu eşiğin üstündeki skor "şiddetli" sayılır
    SEVERE_THRESHOLD = 0.95

    def __init__(self, cluster_id: str, broker: AsyncMQTTBroker):
        self.cluster_id    = cluster_id
        self.broker        = broker
        self.gateway_id    = f"GW-{cluster_id}"
        self.record        = ClusterAlertRecord(cluster_id)

        # MQTT istemcisi
        self.mqtt = MQTTClient(client_id=self.gateway_id, broker=broker)

        # Şiddetli alert sayacı (ard arda gelirse yük azalt)
        self._severe_streak = 0

        print(f"{Color.CYAN}[GW]{Color.RESET} Cluster Gateway {Color.BOLD}{self.gateway_id}{Color.RESET} hazır")

    async def run(self):
        """Gateway'i başlat ve dinlemeye geç."""
        await self.mqtt.connect()

        # Tüm cluster node'larının alertlerini dinle
        alert_pattern = f"sensor/{self.cluster_id}/+/alert"
        self.mqtt.subscribe(alert_pattern, self._on_alert)

        # Heartbeat'leri dinle
        self.mqtt.subscribe("system/heartbeat", self._on_heartbeat)

        print(f"{Color.CYAN}[GW]{Color.RESET} {self.gateway_id} "
              f"dinliyor: {alert_pattern}")

        # Sonsuz döngü — asyncio.gather içinde iptal edilene kadar çalışır
        while True:
            await asyncio.sleep(1.0)

    async def _on_alert(self, msg: MQTTMessage):
        """
        µ-Agent'tan gelen anomali alertini işle.

        Akış:
          1. Payload parse et
          2. Kaydet
          3. ACK gönder (µ-Agent'ın CONSENSUS_WAIT'ini aç)
          4. Şiddet kontrolü → gerekirse cluster komut yayınla
        """
        try:
            data       = msg.to_dict()
            node_id    = data["node_id"]
            score      = data["anomaly_score"]
            fault_cls  = data["fault_class"]
            window_id  = data["window_id"]
            confidence = data["confidence"]

            # Kaydet
            self.record.add_alert(data)

            # ── ACK gönder ────────────────────────────────────────────────
            # µ-Agent bu ACK'i alınca CONSENSUS_WAIT → IDLE geçer
            action = "ACK"
            if score > self.SEVERE_THRESHOLD:
                action = "ESCALATE"
                self._severe_streak += 1
            else:
                self._severe_streak = max(0, self._severe_streak - 1)

            ack_topic = f"sensor/{self.cluster_id}/{node_id}/ack"
            ack_payload = PayloadSchema.ack(
                orchestrator_id  = self.gateway_id,
                node_id          = node_id,
                window_id        = window_id,
                action           = action,
                consensus_reached= True,
            )
            await self.mqtt.publish(ack_topic, ack_payload)

            # ── Terminal çıktısı ──────────────────────────────────────────
            score_color = (
                Color.RED    if score > self.SEVERE_THRESHOLD else
                Color.YELLOW if score > 0.7 else
                Color.GREEN
            )
            print(
                f"{Color.CYAN}[GW ←]{Color.RESET} {self.gateway_id} | "
                f"{node_id} | {fault_cls} | "
                f"score={score_color}{score:.3f}{Color.RESET} | "
                f"conf={confidence:.2f} | ack={action}"
            )

            # ── Şiddetli alert: cluster komut yayınla ─────────────────────
            if self._severe_streak >= 3:
                await self._broadcast_command(
                    command="REDUCE_SAMPLING",
                    params={"reason": "SEVERE_FAULT_STREAK", "count": self._severe_streak},
                )
                self._severe_streak = 0

        except Exception as e:
            print(f"{Color.YELLOW}[GW WARN]{Color.RESET} Alert işleme hatası: {e}")

    async def _on_heartbeat(self, msg: MQTTMessage):
        """Node sağlık sinyalini kaydet."""
        try:
            data = msg.to_dict()
            if data.get("cluster_id") == self.cluster_id:
                self.record.node_health[data["node_id"]] = time.time()
        except Exception:
            pass

    async def _broadcast_command(self, command: str, params: dict):
        """Tüm cluster node'larına komut yayınla."""
        cmd_topic = f"orchestrator/{self.cluster_id}/command"
        payload   = PayloadSchema.command(
            orchestrator_id = self.gateway_id,
            target_node     = "*",
            command         = command,
            params          = params,
        )
        await self.mqtt.publish(cmd_topic, payload)
        print(
            f"{Color.YELLOW}[GW →]{Color.RESET} {self.gateway_id} "
            f"broadcast: {command} | {params}"
        )

    def report(self) -> dict:
        return self.record.summary()


# ─── BROKER MONİTÖRÜ ─────────────────────────────────────────────────────────

class BrokerMonitor:
    """
    Broker istatistiklerini periyodik olarak loglar.
    Akademik metrik: bandwidth, mesaj sayısı, paket kaybı.
    """

    def __init__(self, broker: AsyncMQTTBroker, interval_s: float = 10.0):
        self.broker     = broker
        self.interval   = interval_s / TIME_ACCELERATION

    async def run(self):
        while True:
            await asyncio.sleep(self.interval)
            s = self.broker.stats()
            print(
                f"\n{Color.BOLD}[BROKER İSTATİSTİK]{Color.RESET} "
                f"yayın={s['published']} | "
                f"iletilen={s['delivered']} | "
                f"düşen={s['dropped']} | "
                f"gelen={s['bytes_in']//1024}KB | "
                f"giden={s['bytes_out']//1024}KB | "
                f"abonelik={s['active_subscriptions']}\n"
            )
