"""
tests/test_broker.py — Async MQTT Broker Testleri

Test kapsamı:
  - Publish / subscribe temel akışı
  - Wildcard eşleştirme (+, #)
  - QoS 1 ack mekanizması
  - Paket kaybı simülasyonu
  - Retained mesajlar
  - Broker istatistikleri
"""

import pytest
import asyncio
from daein_mfg.messaging.broker import (
    AsyncMQTTBroker, MQTTClient, MQTTMessage, PayloadSchema, QoS
)


# ─── BROKER TEMEL ─────────────────────────────────────────────────────────────

class TestAsyncMQTTBroker:

    @pytest.mark.asyncio
    async def test_exact_topic_delivery(self):
        broker = AsyncMQTTBroker()
        q      = broker.subscribe("sensor/C1/node01/alert")

        msg = MQTTMessage.from_dict(
            topic="sensor/C1/node01/alert",
            data={"type": "ALERT", "score": 0.87},
            publisher="test",
        )
        await broker.publish(msg)

        received = await asyncio.wait_for(q.get(), timeout=1.0)
        assert received.to_dict()["score"] == 0.87

    @pytest.mark.asyncio
    async def test_wildcard_plus(self):
        """sensor/C1/+/alert → C1 altındaki tüm node'lar."""
        broker = AsyncMQTTBroker()
        q      = broker.subscribe("sensor/C1/+/alert")

        for node in ("node01", "node02", "node03"):
            msg = MQTTMessage.from_dict(
                f"sensor/C1/{node}/alert",
                {"node": node},
                publisher="test",
            )
            await broker.publish(msg)

        received = []
        for _ in range(3):
            m = await asyncio.wait_for(q.get(), timeout=1.0)
            received.append(m.to_dict()["node"])

        assert set(received) == {"node01", "node02", "node03"}

    @pytest.mark.asyncio
    async def test_wildcard_hash(self):
        """sensor/# → tüm sensor topic'leri."""
        broker = AsyncMQTTBroker()
        q      = broker.subscribe("sensor/#")

        topics = [
            "sensor/C1/node01/alert",
            "sensor/C2/node05/alert",
            "sensor/C3/node12/heartbeat",
        ]
        for t in topics:
            await broker.publish(MQTTMessage.from_dict(t, {"x": 1}, publisher="t"))

        count = 0
        for _ in range(3):
            await asyncio.wait_for(q.get(), timeout=1.0)
            count += 1
        assert count == 3

    @pytest.mark.asyncio
    async def test_no_cross_topic_delivery(self):
        """C1 subscriber'ı C2 mesajını almamalı."""
        broker = AsyncMQTTBroker()
        q1     = broker.subscribe("sensor/C1/+/alert")
        q2     = broker.subscribe("sensor/C2/+/alert")

        await broker.publish(MQTTMessage.from_dict(
            "sensor/C2/node01/alert", {"cluster": "C2"}, publisher="t"
        ))

        # q2 mesajı almalı
        m = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert m.to_dict()["cluster"] == "C2"

        # q1 boş kalmalı
        assert q1.empty()

    @pytest.mark.asyncio
    async def test_retained_message_delivered_on_subscribe(self):
        """Retained mesaj yeni aboneye anında iletilmeli."""
        broker = AsyncMQTTBroker()
        msg    = MQTTMessage.from_dict(
            "sensor/C1/node01/status",
            {"status": "ONLINE"},
            retain=True,
            publisher="t",
        )
        await broker.publish(msg)

        # Sonradan abone ol
        q = broker.subscribe("sensor/C1/node01/status")
        received = await asyncio.wait_for(q.get(), timeout=1.0)
        assert received.to_dict()["status"] == "ONLINE"

    @pytest.mark.asyncio
    async def test_packet_loss_simulation(self):
        """
        %100 paket kaybı broker'da hiçbir mesaj iletmemeli.
        Not: Gerçek ortamda %2–5 kayıp beklenir.
        """
        broker  = AsyncMQTTBroker(packet_loss_rate=1.0)
        q       = broker.subscribe("test/topic")
        msg     = MQTTMessage.from_dict("test/topic", {"x": 1}, publisher="t")
        success = await broker.publish(msg)

        assert not success or q.empty()

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        broker = AsyncMQTTBroker()
        _      = broker.subscribe("test/#")

        for i in range(5):
            await broker.publish(MQTTMessage.from_dict(
                f"test/topic{i}", {"i": i}, publisher="t"
            ))

        s = broker.stats()
        assert s["published"]  == 5
        assert s["delivered"]  == 5
        assert s["bytes_in"]   >  0
        assert s["bytes_out"]  >  0

    def test_payload_size_bytes(self):
        msg = MQTTMessage.from_dict(
            "sensor/C1/node01/alert",
            {"type": "ALERT", "score": 0.92},
            publisher="node01",
        )
        assert msg.size_bytes > 0
        # topic(24) + payload(~40) + header(2) ≈ 66 byte
        assert msg.size_bytes < 512


# ─── MQTT CLIENT ──────────────────────────────────────────────────────────────

class TestMQTTClient:

    @pytest.mark.asyncio
    async def test_client_publish_receive(self):
        broker     = AsyncMQTTBroker()
        publisher  = MQTTClient("pub-01", broker)
        subscriber = MQTTClient("sub-01", broker)

        received = []

        async def handler(msg: MQTTMessage):
            received.append(msg.to_dict())

        await publisher.connect()
        await subscriber.connect()
        subscriber.subscribe("alerts/#", handler)

        await asyncio.sleep(0.05)
        await publisher.publish("alerts/C1/node01", {"score": 0.75})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0]["score"] == 0.75

    @pytest.mark.asyncio
    async def test_client_stats_updated(self):
        broker = AsyncMQTTBroker()
        client = MQTTClient("stats-client", broker)
        await client.connect()

        await client.publish("test/topic", {"x": 1})
        await client.publish("test/topic", {"x": 2})

        assert client.stats["published"]   == 2
        assert client.stats["bytes_sent"]  >  0

    @pytest.mark.asyncio
    async def test_publish_without_connect_raises(self):
        broker = AsyncMQTTBroker()
        client = MQTTClient("no-connect", broker)
        with pytest.raises(RuntimeError, match="bağlı değil"):
            await client.publish("test/t", {"x": 1})


# ─── PAYLOAD ŞEMALARI ─────────────────────────────────────────────────────────

class TestPayloadSchema:

    def test_alert_payload_keys(self):
        p = PayloadSchema.alert(
            node_id="C1-node01", cluster_id="C1",
            window_id=5, anomaly_score=0.87, fault_class="BEARING_EARLY",
            confidence=0.79, thermal=42.0, timestamp_ns=123456789,
        )
        for key in ("type", "node_id", "anomaly_score", "fault_class", "confidence"):
            assert key in p

    def test_ack_payload_keys(self):
        p = PayloadSchema.ack("ORC-C1", "C1-node01", 5, "ACK", True)
        assert p["type"]   == "ACK"
        assert p["action"] == "ACK"

    def test_heartbeat_payload_keys(self):
        p = PayloadSchema.heartbeat("node01", "C1", "SAMPLING")
        assert p["type"]  == "HEARTBEAT"
        assert p["state"] == "SAMPLING"
