"""
DAEIN-MFG — Sensör Simülatörü
==============================
Gerçek IoT sensörlerini taklit eden async veri üreticisi.

Üretilen sinyaller:
  - Titreşim (6-axis IMU @ 100Hz) → bearing fault enjeksiyonu ile
  - Isı (IR @ 10Hz) → Gaussian drift modeli
  - Akustik (@ 8kHz → düşürülmüş 8kHz) → kurtosis tabanlı

Her sensör kendi asyncio coroutine'inde çalışır.
SensorBuffer üzerinden µ-Agent'a veri besler.
"""

import asyncio
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    SAMPLE_RATE_HZ, WINDOW_SIZE, THERMAL_RATE_HZ,
    FAULT_INJECT_INTERVAL_S, FAULT_AMPLITUDE, FAULT_FREQUENCY_HZ,
    TIME_ACCELERATION, Color
)


# ─── VERİ YAPILARI ────────────────────────────────────────────────────────────

class FaultType(Enum):
    NONE            = auto()
    BEARING_EARLY   = auto()   # Erken evre rulman arızası (BPFO)
    BEARING_SEVERE  = auto()   # İleri evre rulman arızası
    OVERLOAD        = auto()   # Aşırı yük (RMS artışı)
    MISALIGNMENT    = auto()   # Mil hizasızlığı (harmonik distorsiyon)


@dataclass
class SensorWindow:
    """
    Bir zaman penceresinin tüm ham verisi.
    µ-Agent bu nesneyi alır, feature extraction yapar, inference çalıştırır.
    """
    node_id:        str
    cluster_id:     str
    timestamp:      float               # Unix timestamp (ns hassasiyet)
    window_id:      int                 # Monotonic sayaç
    vibration:      np.ndarray          # Shape: (WINDOW_SIZE, 3) — X, Y, Z ekseni
    thermal:        float               # Anlık sıcaklık (°C)
    acoustic_rms:   float               # Akustik RMS (0–1 normalized)
    injected_fault: FaultType = FaultType.NONE  # Sadece sim içinde bilinen gerçek etiket


@dataclass
class SensorBuffer:
    """
    Sensör → µ-Agent arası asyncio kuyrukları.
    maxsize=10: eski veriler düşer, backpressure oluşmaz (edge gerçekçiliği).
    """
    vibration_q: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=10))
    window_q:    asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=5))


# ─── SINYAL ÜRETECİ ──────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Fizik tabanlı sentetik sinyal üreteci.
    
    Sağlıklı rulman sinyali: Brownian gürültü + harmonikler
    Arızalı sinyal: Karakteristik frekansa darbe eklenir
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE_HZ):
        self.fs = sample_rate
        self.t  = 0.0          # Global zaman sayacı
        self.rng = np.random.default_rng(seed=42)

    def healthy_vibration(self, n_samples: int) -> np.ndarray:
        """
        Sağlıklı makine titreşimi.
        Model: x(t) = A_rot·sin(2π·f_rot·t) + σ·W(t)
        W(t) = Wiener process (Brownian gürültü)
        """
        t = np.linspace(self.t, self.t + n_samples / self.fs, n_samples)

        # Döner makine temel frekansı (1500 RPM → 25 Hz)
        f_rot   = 25.0
        x_axis  = 0.05 * np.sin(2 * np.pi * f_rot * t)
        x_axis += 0.02 * np.sin(2 * np.pi * 2 * f_rot * t)  # 2. harmonik
        x_axis += 0.005 * self.rng.standard_normal(n_samples)  # gürültü

        y_axis = 0.04 * np.sin(2 * np.pi * f_rot * t + np.pi / 4)
        y_axis += 0.004 * self.rng.standard_normal(n_samples)

        z_axis = 0.02 * self.rng.standard_normal(n_samples)  # axial, düşük

        self.t += n_samples / self.fs
        return np.column_stack([x_axis, y_axis, z_axis])  # (N, 3)

    def inject_bearing_fault(
        self,
        signal: np.ndarray,
        fault_type: FaultType,
        severity: float = 1.0
    ) -> np.ndarray:
        """
        Rulman arızası sinyali ekler.
        
        Fiziksel model:
        x_fault(t) = x_healthy(t) + A·Σ δ(t - n·T_fault) * h(t)
        
        Burada:
          T_fault = 1/f_BPFO  (BPFO: Ball Pass Frequency Outer race)
          h(t) = e^(-α·t)     (exponential decay impulse response)
          A = genlik (severity ile ölçeklenir)
        """
        n = len(signal)
        t = np.linspace(0, n / self.fs, n)
        fault_signal = np.zeros(n)

        if fault_type in (FaultType.BEARING_EARLY, FaultType.BEARING_SEVERE):
            T_fault = 1.0 / FAULT_FREQUENCY_HZ
            amp     = FAULT_AMPLITUDE * severity
            alpha   = 500.0  # decay katsayısı

            # Her darbe anı
            pulse_times = np.arange(0, n / self.fs, T_fault)
            for pt in pulse_times:
                # Darbe başlangıcından itibaren zaman
                tau = t - pt
                # Sadece pozitif tau (causal)
                mask = tau > 0
                fault_signal[mask] += amp * np.exp(-alpha * tau[mask])

            # Erken evre → sadece X eksenine, ileri evre → tüm eksenlere
            if fault_type == FaultType.BEARING_EARLY:
                signal[:, 0] += fault_signal
            else:
                signal[:, 0] += fault_signal
                signal[:, 1] += fault_signal * 0.5
                signal[:, 2] += fault_signal * 0.3

        elif fault_type == FaultType.OVERLOAD:
            # Genel RMS artışı — tüm eksenlerde homojen yük artışı
            scale = 1.0 + (severity * 0.8)
            signal *= scale

        elif fault_type == FaultType.MISALIGNMENT:
            # 2x ve 3x harmonikler baskın olur
            f_rot = 25.0
            t_arr = np.linspace(self.t - n / self.fs, self.t, n)
            harmonic = 0.15 * severity * np.sin(2 * np.pi * 2 * f_rot * t_arr)
            harmonic += 0.08 * severity * np.sin(2 * np.pi * 3 * f_rot * t_arr)
            signal[:, 0] += harmonic

        return signal


# ─── ANA SENSÖR SİMÜLATÖRÜ ───────────────────────────────────────────────────

class SensorSimulator:
    """
    Tek bir IoT düğümünü simüle eder.
    
    Async coroutine'ler:
      - _vibration_loop: 100Hz'de titreşim verisi üretir
      - _thermal_loop:   10Hz'de sıcaklık verisi üretir
      - _window_assembler: buffer'ı pencereye dönüştürüp µ-Agent'a gönderir
      - _fault_injector: periyodik arıza sinyali enjekte eder
    """

    def __init__(self, node_id: str, cluster_id: str, buffer: SensorBuffer):
        self.node_id    = node_id
        self.cluster_id = cluster_id
        self.buffer     = buffer
        self.generator  = SignalGenerator()

        # Durum değişkenleri
        self._current_fault = FaultType.NONE
        self._fault_severity = 0.0
        self._window_counter = 0
        self._vib_accumulator: list = []
        self._current_thermal = 25.0 + np.random.uniform(-2, 2)  # başlangıç sıcaklığı

        # Simülasyon hızlandırma: gerçek bekleme sürelerini böl
        self._tick = (1.0 / SAMPLE_RATE_HZ) / TIME_ACCELERATION

        print(f"{Color.CYAN}[SENSOR]{Color.RESET} Node {Color.BOLD}{node_id}{Color.RESET} başlatıldı → cluster: {cluster_id}")

    async def run(self):
        """Tüm coroutine'leri paralel başlatır."""
        await asyncio.gather(
            self._vibration_loop(),
            self._thermal_loop(),
            self._window_assembler(),
            self._fault_injector(),
        )

    async def _vibration_loop(self):
        """
        100Hz titreşim verisi üretir.
        Her tick'te 1 örnek üretilir, accumulator'a eklenir.
        """
        while True:
            # 1 örnek üret
            sample = self.generator.healthy_vibration(1)

            # Arıza varsa sinyal üzerine ekle
            if self._current_fault != FaultType.NONE:
                sample = self.generator.inject_bearing_fault(
                    sample, self._current_fault, self._fault_severity
                )

            self._vib_accumulator.append(sample[0])  # (3,) array ekle

            # Kuyruk doluysa en eski örneği at (edge gerçekçiliği)
            try:
                self.buffer.vibration_q.put_nowait(sample[0])
            except asyncio.QueueFull:
                try:
                    self.buffer.vibration_q.get_nowait()
                    self.buffer.vibration_q.put_nowait(sample[0])
                except asyncio.QueueEmpty:
                    pass

            await asyncio.sleep(self._tick)

    async def _thermal_loop(self):
        """
        10Hz ısı verisi.
        Gaussian random walk + arıza durumunda ısı artışı.
        """
        while True:
            # Random walk: küçük adımlar
            drift = np.random.normal(0, 0.05)
            self._current_thermal += drift

            # Arıza varsa ısı artar
            if self._current_fault in (FaultType.BEARING_EARLY, FaultType.BEARING_SEVERE):
                self._current_thermal += 0.02 * self._fault_severity

            # Gerçekçi sınırlar
            self._current_thermal = np.clip(self._current_thermal, 18.0, 95.0)

            await asyncio.sleep((1.0 / THERMAL_RATE_HZ) / TIME_ACCELERATION)

    async def _window_assembler(self):
        """
        256 örnek birikince SensorWindow oluşturur ve µ-Agent kuyruğuna atar.
        Bu pencere FFT + inference için kullanılır.
        """
        while True:
            # 256 örnek biriktikçe pencere oluştur
            # Hızlandırılmış simülasyonda accumulator'ı toplu doldur
            if len(self._vib_accumulator) < WINDOW_SIZE:
                needed = WINDOW_SIZE - len(self._vib_accumulator)
                bulk = self.generator.healthy_vibration(needed)
                if self._current_fault != FaultType.NONE:
                    bulk = self.generator.inject_bearing_fault(bulk, self._current_fault, self._fault_severity)
                for s in bulk:
                    self._vib_accumulator.append(s)

            if len(self._vib_accumulator) >= WINDOW_SIZE:
                window_data = np.array(self._vib_accumulator[:WINDOW_SIZE])  # (256, 3)
                self._vib_accumulator = self._vib_accumulator[int(WINDOW_SIZE * 0.5):]  # %50 örtüşme

                sw = SensorWindow(
                    node_id        = self.node_id,
                    cluster_id     = self.cluster_id,
                    timestamp      = time.time_ns(),
                    window_id      = self._window_counter,
                    vibration      = window_data,
                    thermal        = self._current_thermal,
                    acoustic_rms   = float(np.std(window_data[:, 0])),  # basit akustik proxy
                    injected_fault = self._current_fault,
                )
                self._window_counter += 1

                try:
                    self.buffer.window_q.put_nowait(sw)
                except asyncio.QueueFull:
                    # En eski pencereyi at
                    try:
                        self.buffer.window_q.get_nowait()
                        self.buffer.window_q.put_nowait(sw)
                    except asyncio.QueueEmpty:
                        pass

            await asyncio.sleep(self._tick * 2)  # her 2 tick'te kontrol et

    async def _fault_injector(self):
        """
        Periyodik arıza enjeksiyonu.
        Gerçek bir sistemde bu, bilinen etiketli test senaryolarını temsil eder.
        """
        await asyncio.sleep(FAULT_INJECT_INTERVAL_S / TIME_ACCELERATION)

        fault_sequence = [
            (FaultType.BEARING_EARLY,  0.5,  "Erken rulman arızası"),
            (FaultType.BEARING_EARLY,  0.9,  "Erken rulman arızası (şiddetli)"),
            (FaultType.BEARING_SEVERE, 1.0,  "İleri evre rulman arızası"),
            (FaultType.OVERLOAD,       0.7,  "Aşırı yük"),
            (FaultType.MISALIGNMENT,   0.6,  "Mil hizasızlığı"),
            (FaultType.NONE,           0.0,  "Normal operasyon"),
        ]

        for fault_type, severity, label in fault_sequence:
            self._current_fault   = fault_type
            self._fault_severity  = severity

            status = f"{Color.RED}⚠ ARIZA{Color.RESET}" if fault_type != FaultType.NONE else f"{Color.GREEN}✓ NORMAL{Color.RESET}"
            print(
                f"{Color.YELLOW}[INJECT]{Color.RESET} {self.node_id} → "
                f"{status} | {label} | severity={severity:.1f}"
            )

            await asyncio.sleep(FAULT_INJECT_INTERVAL_S / TIME_ACCELERATION)
