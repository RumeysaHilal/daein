"""
DAEIN-MFG — Async MQTT Broker & Client (Aşama 3)
==================================================
Gerçek MQTT protokolünü simüle eden in-process broker.

Neden in-process?
  - Dış bağımlılık yok (Mosquitto, EMQX kurulumu gereksiz)
  - Test ortamında deterministik davranış
  - Gerçek MQTT'ye geçiş: MQTTClient sınıfını paho-mqtt ile swap et,
    broker.py'yi tamamen at → µ-Agent kodu değişmez

Desteklenen özellikler:
  - Topic tabanlı publish / subscribe
  - Wildcard: sensor/# veya sensor/C1/+/alert
  - QoS 1 simülasyonu (ack mekanizması)
  - Mesaj gecikmesi ve paket kaybı enjeksiyonu (test için)
  - Retained messages (son mesajı yeni abone alır)
  - Per-topic mesaj istatistikleri

Konu hiyerarşisi (bu projede):
  sensor/{cluster_id}/{node_id}/alert      ← µ-Agent yayınlar
  sensor/{cluster_id}/{node_id}/ack        ← Orkestratör onaylar
  orchestrator/{cluster_id}/command        ← Orkestratör komut gönderir
  system/heartbeat                         ← Tüm node'lar periyodik ping
"""

import asyncio
import time
import fnmatch
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import IntEnum
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Color


# ─── TEMEL TİPLER ─────────────────────────────────────────────────────────────

class QoS(IntEnum):
    AT_MOST_ONCE  = 0   # Fire-and-forget
    AT_LEAST_ONCE = 1   # Ack gerekli (QoS 1 simülasyonu)


@dataclass
class MQTTMessage:
    """
    Tek bir MQTT mesajı.
    Gerçek MQTT paket yapısını taklit eder: topic + payload + metadata.
    """
    topic:      str
    payload:    bytes           # Seri hale getirilmiş veri (Protobuf yerine JSON sim)
    qos:        QoS = QoS.AT_LEAST_ONCE
    retain:     bool = False
    msg_id:     int  = 0        # QoS 1 için mesaj kimliği
    timestamp:  float = field(default_factory=time.time)
    publisher:  str = "unknown"

    @property
    def size_bytes(self) -> int:
        """Gerçek MQTT paket boyutu tahmini (header + topic + payload)."""
        return 2 + len(self.topic.encode()) + len(self.payload)

    def to_dict(self) -> dict:
        return json.loads(self.payload.decode())

    @staticmethod
    def from_dict(topic: str, data: dict, **kwargs) -> "MQTTMessage":
        return MQTTMessage(
            topic=topic,
            payload=json.dumps(data).encode(),
            **kwargs
        )


# ─── IN-PROCESS BROKER ────────────────────────────────────────────────────────

class AsyncMQTTBroker:
    """
    Tam asyncio tabanlı in-process MQTT broker.

    Özellikler:
      - Topic ağacı: dict tabanlı, O(1) kesin eşleşme
      - Wildcard çözümleme: fnmatch ile (+, #)
      - Retained mesajlar: her topic için son mesaj saklanır
      - QoS 1: ack kuyruğu ile mesaj kaybı önleme
      - Ağ simülasyonu: gecikme ve paket kaybı enjeksiyonu
      - İstatistikler: topic başına mesaj sayısı, byte sayısı
    """

    def __init__(
        self,
        packet_loss_rate: float = 0.0,    # 0.0 = kayıp yok, 0.1 = %10 kayıp
        latency_ms: float = 0.0,          # Simüle gecikme (ms)
        max_queue_per_sub: int = 100,      # Abone başına kuyruk boyutu
    ):
        self._packet_loss   = packet_loss_rate
        self._latency_ms    = latency_ms
        self._max_q         = max_queue_per_sub

        # {topic: [asyncio.Queue, ...]}  — abone kuyrukları
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

        # {topic: MQTTMessage}  — retained mesajlar
        self._retained: dict[str, MQTTMessage] = {}

        # {msg_id: asyncio.Event}  — QoS 1 ack bekleyenleri
        self._pending_acks: dict[int, asyncio.Event] = {}

        # İstatistikler
        self._stats = {
            "published":    0,
            "delivered":    0,
            "dropped":      0,
            "bytes_in":     0,
            "bytes_out":    0,
        }

        self._msg_counter = 0
        self._rng = __import__("random").Random(42)

        print(f"{Color.CYAN}[BROKER]{Color.RESET} Async MQTT Broker başlatıldı "
              f"| loss={packet_loss_rate:.0%} | latency={latency_ms:.0f}ms")

    async def publish(self, msg: MQTTMessage) -> bool:
        """
        Mesaj yayınla. QoS 1 ise ack gelene kadar bekle.
        Returns: True → başarılı, False → paket kaybı
        """
        self._msg_counter += 1
        msg.msg_id = self._msg_counter
        self._stats["published"] += 1
        self._stats["bytes_in"] += msg.size_bytes

        # Paket kaybı simülasyonu
        if self._packet_loss > 0 and self._rng.random() < self._packet_loss:
            self._stats["dropped"] += 1
            return False

        # Gecikme simülasyonu
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        # Retained mesaj sakla
        if msg.retain:
            self._retained[msg.topic] = msg

        # Eşleşen tüm abonel kuyruklara ilet
        delivered = 0
        for pattern, queues in self._subscribers.items():
            if self._topic_matches(msg.topic, pattern):
                for q in queues:
                    try:
                        q.put_nowait(msg)
                        delivered += 1
                        self._stats["bytes_out"] += msg.size_bytes
                    except asyncio.QueueFull:
                        self._stats["dropped"] += 1

        self._stats["delivered"] += delivered

        # QoS 1: ack event oluştur
        if msg.qos == QoS.AT_LEAST_ONCE:
            ack_event = asyncio.Event()
            self._pending_acks[msg.msg_id] = ack_event
            # QoS 1 ack timeout = 500ms (simülasyon hızlandırması ile)
            try:
                await asyncio.wait_for(ack_event.wait(), timeout=0.5)
                return True
            except asyncio.TimeoutError:
                # Gerçek MQTT'de yeniden iletilir; simülasyonda logluyoruz
                del self._pending_acks[msg.msg_id]
                return delivered > 0  # En az biri aldıysa başarılı say

        return True

    async def ack(self, msg_id: int):
        """QoS 1 ack gönder (genellikle broker tarafından çağrılır)."""
        if msg_id in self._pending_acks:
            self._pending_acks[msg_id].set()
            del self._pending_acks[msg_id]

    def subscribe(self, topic_pattern: str) -> asyncio.Queue:
        """
        Bir topic pattern'e abone ol.
        Returns: Mesajların geleceği asyncio.Queue
        """
        q = asyncio.Queue(maxsize=self._max_q)
        if topic_pattern not in self._subscribers:
            self._subscribers[topic_pattern] = []
        self._subscribers[topic_pattern].append(q)

        # Retained mesajları hemen ilet
        for topic, retained_msg in self._retained.items():
            if self._topic_matches(topic, topic_pattern):
                try:
                    q.put_nowait(retained_msg)
                except asyncio.QueueFull:
                    pass

        return q

    def unsubscribe(self, topic_pattern: str, queue: asyncio.Queue):
        if topic_pattern in self._subscribers:
            try:
                self._subscribers[topic_pattern].remove(queue)
            except ValueError:
                pass

    def stats(self) -> dict:
        return {**self._stats, "active_subscriptions": sum(len(v) for v in self._subscribers.values())}

    @staticmethod
    def _topic_matches(topic: str, pattern: str) -> bool:
        """
        MQTT wildcard eşleştirme:
          + → tek seviye (sensor/C1/+/alert)
          # → çok seviye (sensor/#)
        """
        if pattern == topic:
            return True
        # MQTT'yi fnmatch pattern'ine çevir
        fnpat = pattern.replace("+", "*").replace("#", "**")
        # # sonrasında başka segment olmamalı
        if "**" in fnpat:
            prefix = fnpat.split("**")[0].rstrip("/")
            return topic.startswith(prefix + "/") or topic == prefix.rstrip("/")
        return fnmatch.fnmatch(topic, fnpat)


# ─── MQTT CLIENT ──────────────────────────────────────────────────────────────

class MQTTClient:
    """
    µ-Agent ve Orkestratör tarafından kullanılan MQTT istemcisi.

    Gerçek MQTT'ye geçiş:
      Bu sınıfın __init__, publish, subscribe metodlarını
      paho.mqtt.client.Client wrapper'ı ile değiştir.
      Dışarıdan görünen API aynı kalır.
    """

    def __init__(self, client_id: str, broker: AsyncMQTTBroker):
        self.client_id = client_id
        self._broker   = broker
        self._subscriptions: dict[str, tuple[asyncio.Queue, Callable]] = {}
        self._listener_tasks: list[asyncio.Task] = []
        self._connected = False

        # İstatistikler
        self.stats = {"published": 0, "received": 0, "bytes_sent": 0}

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False
        for task in self._listener_tasks:
            task.cancel()

    async def publish(
        self,
        topic: str,
        data: dict,
        qos: QoS = QoS.AT_LEAST_ONCE,
        retain: bool = False,
    ) -> bool:
        """
        Dict veriyi JSON serialize edip yayınla.
        Payload boyutu kontrol edilir — edge bütçesi: 512B.
        """
        if not self._connected:
            raise RuntimeError(f"[{self.client_id}] Broker'a bağlı değil")

        msg = MQTTMessage.from_dict(
            topic=topic,
            data=data,
            qos=qos,
            retain=retain,
            publisher=self.client_id,
        )

        # Payload bütçe kontrolü
        from config import MAX_MQTT_PAYLOAD_B
        if msg.size_bytes > MAX_MQTT_PAYLOAD_B:
            print(f"{Color.YELLOW}[WARN]{Color.RESET} {self.client_id}: "
                  f"Payload {msg.size_bytes}B > {MAX_MQTT_PAYLOAD_B}B bütçesi")

        success = await self._broker.publish(msg)
        if success:
            self.stats["published"] += 1
            self.stats["bytes_sent"] += msg.size_bytes
        return success

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[MQTTMessage], Awaitable[None]],
    ):
        """
        Topic pattern'e abone ol, gelen mesajlar callback'e iletilir.
        Callback asyncio coroutine olmalı.
        """
        q = self._broker.subscribe(topic_pattern)
        self._subscriptions[topic_pattern] = (q, callback)

        # Her subscription için ayrı listener görevi
        task = asyncio.get_event_loop().create_task(
            self._listen(topic_pattern, q, callback),
            name=f"mqtt-{self.client_id}-{topic_pattern}"
        )
        self._listener_tasks.append(task)

    async def _listen(
        self,
        pattern: str,
        queue: asyncio.Queue,
        callback: Callable,
    ):
        """Kuyruğu dinle, gelen mesajları callback'e ilet."""
        while self._connected:
            try:
                msg: MQTTMessage = await asyncio.wait_for(queue.get(), timeout=0.5)
                self.stats["received"] += 1

                # QoS 1: broker'a ack gönder
                if msg.qos == QoS.AT_LEAST_ONCE:
                    asyncio.get_event_loop().create_task(
                        self._broker.ack(msg.msg_id)
                    )

                # Callback'i çağır
                try:
                    await callback(msg)
                except Exception as e:
                    print(f"{Color.YELLOW}[WARN]{Color.RESET} {self.client_id} "
                          f"callback hatası ({pattern}): {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# ─── PAYLOAD ŞEMALARI ─────────────────────────────────────────────────────────

class PayloadSchema:
    """
    Tüm MQTT mesaj şemaları burada tanımlanır.
    µ-Agent ve Orkestratör bu metodları kullanarak mesaj oluşturur.
    """

    @staticmethod
    def alert(
        node_id: str,
        cluster_id: str,
        window_id: int,
        anomaly_score: float,
        fault_class: str,
        confidence: float,
        thermal: float,
        timestamp_ns: int,
    ) -> dict:
        """µ-Agent → Orkestratör anomali uyarısı."""
        return {
            "type":          "ALERT",
            "node_id":       node_id,
            "cluster_id":    cluster_id,
            "window_id":     window_id,
            "anomaly_score": round(anomaly_score, 4),
            "fault_class":   fault_class,
            "confidence":    round(confidence, 4),
            "thermal":       round(thermal, 2),
            "timestamp_ns":  timestamp_ns,
        }

    @staticmethod
    def ack(
        orchestrator_id: str,
        node_id: str,
        window_id: int,
        action: str,            # 'ACK', 'ESCALATE', 'DEFER'
        consensus_reached: bool,
    ) -> dict:
        """Orkestratör → µ-Agent onay/aksiyon."""
        return {
            "type":              "ACK",
            "orchestrator_id":   orchestrator_id,
            "node_id":           node_id,
            "window_id":         window_id,
            "action":            action,
            "consensus_reached": consensus_reached,
            "timestamp_ns":      time.time_ns(),
        }

    @staticmethod
    def heartbeat(node_id: str, cluster_id: str, state: str) -> dict:
        """Periyodik sağlık sinyali."""
        return {
            "type":       "HEARTBEAT",
            "node_id":    node_id,
            "cluster_id": cluster_id,
            "state":      state,
            "timestamp_ns": time.time_ns(),
        }

    @staticmethod
    def command(
        orchestrator_id: str,
        target_node: str,
        command: str,           # 'REDUCE_SAMPLING', 'INCREASE_SAMPLING', 'SHUTDOWN'
        params: dict = None,
    ) -> dict:
        """Orkestratör → µ-Agent komut."""
        return {
            "type":           "COMMAND",
            "orchestrator_id": orchestrator_id,
            "target_node":    target_node,
            "command":        command,
            "params":         params or {},
            "timestamp_ns":   time.time_ns(),
        }
