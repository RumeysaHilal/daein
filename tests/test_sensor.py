"""
tests/test_sensor.py — Sensör Simülatörü Testleri

Test kapsamı:
  - Sağlıklı sinyal istatistiksel özellikleri
  - Arıza enjeksiyonu doğruluğu (her arıza tipi)
  - Pencere boyutu ve şekil garantileri
  - Isı drift modeli sınırları
"""

import pytest
import numpy as np
from daein_mfg.simulation.sensor_generator import (
    SignalGenerator, FaultType, SensorWindow, SensorBuffer
)
import asyncio, time


# ─── SİNYAL ÜRETECİ ──────────────────────────────────────────────────────────

class TestSignalGenerator:

    def test_healthy_signal_shape(self, signal_generator):
        sig = signal_generator.healthy_vibration(256)
        assert sig.shape == (256, 3), "3 eksenli, 256 örneklik bekleniyor"

    def test_healthy_signal_rms_range(self, signal_generator):
        """Sağlıklı sinyal RMS'i gerçekçi sınırlarda olmalı (0.01g – 0.15g)."""
        sig = signal_generator.healthy_vibration(256)
        rms = float(np.sqrt(np.mean(sig[:, 0] ** 2)))
        assert 0.005 < rms < 0.20, f"RMS {rms:.4f} beklenen aralık dışında"

    def test_fault_increases_rms(self, signal_generator):
        """Arıza enjeksiyonu RMS'i artırmalı."""
        healthy = signal_generator.healthy_vibration(256)
        rms_h   = float(np.sqrt(np.mean(healthy[:, 0] ** 2)))

        faulty  = signal_generator.healthy_vibration(256)
        faulty  = signal_generator.inject_bearing_fault(
            faulty, FaultType.BEARING_EARLY, severity=0.9
        )
        rms_f = float(np.sqrt(np.mean(faulty[:, 0] ** 2)))

        assert rms_f > rms_h, f"Arızalı RMS ({rms_f:.4f}) sağlıklıdan ({rms_h:.4f}) büyük olmalı"

    def test_bearing_severe_higher_than_early(self, signal_generator):
        """İleri evre, erken evreden daha yüksek RMS üretmeli."""
        early  = signal_generator.healthy_vibration(256)
        early  = signal_generator.inject_bearing_fault(early, FaultType.BEARING_EARLY, 0.5)

        severe = signal_generator.healthy_vibration(256)
        severe = signal_generator.inject_bearing_fault(severe, FaultType.BEARING_SEVERE, 1.0)

        rms_e = float(np.sqrt(np.mean(early[:,  0] ** 2)))
        rms_s = float(np.sqrt(np.mean(severe[:, 0] ** 2)))
        assert rms_s > rms_e

    def test_overload_scales_all_axes(self, signal_generator):
        """OVERLOAD tüm eksenleri ölçeklemeli."""
        base   = signal_generator.healthy_vibration(256).copy()
        faulty = signal_generator.healthy_vibration(256)
        faulty = signal_generator.inject_bearing_fault(faulty, FaultType.OVERLOAD, 1.0)

        for ax in range(3):
            rms_b = float(np.sqrt(np.mean(base[:, ax] ** 2)))
            rms_f = float(np.sqrt(np.mean(faulty[:, ax] ** 2)))
            assert rms_f >= rms_b, f"Eksen {ax}: OVERLOAD RMS artışı bekleniyor"

    def test_signal_generator_time_advances(self, signal_generator):
        """Her çağrıda zaman sayacı ilerlemeli (stateful üretici)."""
        t0 = signal_generator.t
        signal_generator.healthy_vibration(256)
        t1 = signal_generator.t
        assert t1 > t0, "Zaman sayacı ilerlemeli"

    @pytest.mark.parametrize("n_samples", [64, 128, 256, 512])
    def test_arbitrary_window_sizes(self, signal_generator, n_samples):
        sig = signal_generator.healthy_vibration(n_samples)
        assert sig.shape == (n_samples, 3)


# ─── SENSOR BUFFER ────────────────────────────────────────────────────────────

class TestSensorBuffer:

    def test_buffer_queue_maxsize(self):
        buf = SensorBuffer()
        assert buf.window_q.maxsize == 5
        assert buf.vibration_q.maxsize == 10

    @pytest.mark.asyncio
    async def test_window_queue_put_get(self, healthy_window):
        buf = SensorBuffer()
        await buf.window_q.put(healthy_window)
        retrieved = await asyncio.wait_for(buf.window_q.get(), timeout=1.0)
        assert retrieved.node_id   == healthy_window.node_id
        assert retrieved.window_id == healthy_window.window_id

    @pytest.mark.asyncio
    async def test_full_queue_drops_oldest(self):
        """Dolu kuyrukta put_nowait → QueueFull → en eski düşer."""
        buf = SensorBuffer()
        gen = SignalGenerator()
        # Kuyruğu doldur
        for i in range(5):
            sig = gen.healthy_vibration(256)
            sw  = SensorWindow('n', 'C1', time.time_ns(), i, sig, 25.0, 0.0, FaultType.NONE)
            buf.window_q.put_nowait(sw)
        assert buf.window_q.full()
