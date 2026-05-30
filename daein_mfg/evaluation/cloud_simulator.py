"""
DAEIN-MFG — Cloud Mimarisi Simülatörü (Aşama 5)
=================================================
Edge sistemiyle karşılaştırma için cloud tabanlı karar zincirini simüle eder.

Cloud modelinde her sensör penceresi:
  1. Ham veriyi (veya feature vektörünü) cloud'a gönderir
  2. Cloud'da inference çalışır
  3. Karar edge'e geri döner
  4. Aktüasyon gerçekleşir

Literatür referansları:
  - Azure IoT Hub E2E latency: 200–800ms (ortalama ~350ms)
  - AWS IoT Core → SageMaker: 150–500ms
  - Ham IMU stream 100Hz × 3-eksen × 4B ≈ 1.2KB/s/node
"""

import asyncio
import time
import random
import numpy as np
from dataclasses import dataclass
from config import Color, WINDOW_SIZE, TIME_ACCELERATION


@dataclass
class CloudLatencyModel:
    """
    Gerçekçi cloud round-trip gecikme dağılımı.
    Log-normal dağılım: internet gecikmeleri için standart model.
    """
    network_rtt_mean:   float = 80.0    # ms
    network_rtt_std:    float = 25.0
    api_queue_mean:     float = 50.0
    api_queue_std:      float = 30.0
    inference_mean:     float = 150.0
    inference_std:      float = 40.0
    packet_loss_rate:   float = 0.05
    retry_penalty_ms:   float = 200.0

    def sample(self) -> float:
        net  = max(10.0, random.gauss(self.network_rtt_mean, self.network_rtt_std))
        q    = max(5.0,  random.gauss(self.api_queue_mean,   self.api_queue_std))
        inf  = max(50.0, random.gauss(self.inference_mean,   self.inference_std))
        ret  = max(10.0, random.gauss(self.network_rtt_mean, self.network_rtt_std))
        total = net + q + inf + ret
        if random.random() < self.packet_loss_rate:
            total += self.retry_penalty_ms
        return total


@dataclass
class CloudDecision:
    anomaly_id:     str
    latency_ms:     float
    payload_bytes:  int
    fault_detected: bool
    fault_class:    str
    confidence:     float
    timed_out:      bool = False


class CloudSimulator:
    """
    Merkezi cloud mimarisini simüle eder.
    RAW_STREAM modunda tüm ham sensör verisi gönderilir.
    """
    CLOUD_TIMEOUT_MS = 2000.0

    def __init__(self, mode: str = "RAW_STREAM"):
        self.mode      = mode
        self._model    = CloudLatencyModel()
        self._decisions: list[CloudDecision] = []
        self._bytes_up  = 0
        self._timeouts  = 0

        # Payload: 256 örnek × 3 eksen × 4B + 64B metadata
        self.payload_bytes = (WINDOW_SIZE * 3 * 4 + 64
                              if mode == "RAW_STREAM" else 14 * 4 + 32)

        print(f"{Color.YELLOW}[CLOUD]{Color.RESET} Simülatör hazır | "
              f"mod={mode} | payload={self.payload_bytes}B")

    async def process(self, anomaly_id, anomaly_score, fault_class, confidence) -> CloudDecision:
        t0         = time.perf_counter()
        latency_ms = self._model.sample()
        timed_out  = latency_ms > self.CLOUD_TIMEOUT_MS

        await asyncio.sleep(latency_ms / 1000 / TIME_ACCELERATION)
        actual_ms = (time.perf_counter() - t0) * 1000 * TIME_ACCELERATION

        self._bytes_up += self.payload_bytes
        if timed_out:
            self._timeouts += 1

        d = CloudDecision(
            anomaly_id    = anomaly_id,
            latency_ms    = actual_ms,
            payload_bytes = self.payload_bytes,
            fault_detected= anomaly_score > 0.55 and not timed_out,
            fault_class   = fault_class if not timed_out else "TIMEOUT",
            confidence    = confidence if not timed_out else 0.0,
            timed_out     = timed_out,
        )
        self._decisions.append(d)
        return d

    def report(self) -> dict:
        n = max(len(self._decisions), 1)
        lats = [d.latency_ms for d in self._decisions] or [0]
        return {
            "n":              n,
            "bytes_up_kb":    round(self._bytes_up / 1024, 1),
            "timeouts":       self._timeouts,
            "timeout_rate":   round(self._timeouts / n, 3),
            "avg_latency_ms": round(float(np.mean(lats)), 1),
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 1),
        }
