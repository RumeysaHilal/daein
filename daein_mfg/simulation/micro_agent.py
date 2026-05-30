"""
DAEIN-MFG — Mikro-Ajan (µ-Agent)
==================================
Her IoT düğümündeki otonom karar birimi.

Görevleri:
  1. SensorWindow al → FFT feature extraction
  2. Anomali skoru hesapla (IsolationForest proxy)
  3. Eşik aşıldıysa AnomalyEvent üret
  4. FSM durumunu güncelle
  5. (Opsiyonel) MQTT'ye yayınla

FSM Durumları:
  IDLE → SAMPLING → INFERENCE → ALERT_STAGE → CONSENSUS_WAIT → IDLE
"""

import asyncio
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    SAMPLE_RATE_HZ, WINDOW_SIZE, ANOMALY_THRESHOLD,
    MAX_INFERENCE_MS, MAX_MQTT_PAYLOAD_B, Color
)
from simulation.sensor_generator import SensorBuffer, SensorWindow, FaultType

# Aşama 2: ML tabanlı scorer (kural tabanlı AnomalyScorer'ın yerini alır)
try:
    from models.ml_scorer import MLAnomalyScorer as AnomalyScorerML
    _ML_SCORER_AVAILABLE = True
except Exception:
    _ML_SCORER_AVAILABLE = False

# Aşama 3: MQTT mesajlaşma
try:
    from messaging.broker import MQTTClient, PayloadSchema, QoS
    _MQTT_AVAILABLE = True
except Exception:
    _MQTT_AVAILABLE = False


# ─── VERİ YAPILARI ────────────────────────────────────────────────────────────

class AgentState(Enum):
    IDLE           = auto()   # Bekleme, düşük güç
    SAMPLING       = auto()   # Pencere toplama
    INFERENCE      = auto()   # Feature extraction + anomali skoru
    ALERT_STAGE    = auto()   # Eşik aşıldı, uyarı hazırlanıyor
    CONSENSUS_WAIT = auto()   # Orkestratörden onay bekleniyor
    ACTUATING      = auto()   # Aksiyon alınıyor


@dataclass
class FeatureVector:
    """
    256-örneklik pencereden çıkarılan 18-boyutlu özellik vektörü.
    
    Frekans özellikleri (FFT tabanlı):
      - dominant_freq: En yüksek güçteki frekans
      - spectral_entropy: Frekans dağılımının düzensizliği
      - bpfo_power: Rulman karakteristik frekansındaki güç
      - top5_freq_power: En güçlü 5 frekansın toplam gücü
    
    Zaman özellikleri (istatistiksel):
      - rms: Karesel ortalama (genel titreşim şiddeti)
      - kurtosis: 4. moment (darbe tipi arızaları yakalar)
      - crest_factor: Tepe/RMS oranı
      - skewness: Asimetri
    """
    # Frekans domain
    dominant_freq:      float
    spectral_entropy:   float
    bpfo_power:         float
    top5_freq_power:    float
    freq_centroid:      float

    # Zaman domain — X ekseni
    rms_x:          float
    kurtosis_x:     float
    crest_factor_x: float
    skewness_x:     float

    # Zaman domain — Y ekseni
    rms_y:          float
    kurtosis_y:     float

    # Çok-modal
    thermal:        float
    acoustic_rms:   float

    # Çapraz korelasyon
    xy_correlation: float

    # Ham vektöre dönüştür (model input için)
    def to_array(self) -> np.ndarray:
        return np.array([
            self.dominant_freq, self.spectral_entropy, self.bpfo_power,
            self.top5_freq_power, self.freq_centroid,
            self.rms_x, self.kurtosis_x, self.crest_factor_x, self.skewness_x,
            self.rms_y, self.kurtosis_y,
            self.thermal, self.acoustic_rms, self.xy_correlation
        ], dtype=np.float32)


@dataclass
class AnomalyEvent:
    """µ-Agent'ın ürettiği uyarı mesajı. MQTT payload'ı buradan serialize edilir."""
    node_id:        str
    cluster_id:     str
    timestamp_ns:   int
    window_id:      int
    anomaly_score:  float
    fault_class:    str         # 'BEARING_EARLY', 'OVERLOAD', vb.
    confidence:     float
    features:       np.ndarray  # 14-dim feature vektörü
    true_fault:     FaultType   # Sadece simülasyon için (ground truth)

    def serialize_size(self) -> int:
        """Tahmini MQTT payload boyutu (byte)."""
        # node_id(20) + cluster_id(5) + timestamp(8) + window_id(4)
        # + scores(8) + fault_class(20) + features(14*4=56) ≈ 121 byte
        return 20 + 5 + 8 + 4 + 8 + 20 + len(self.features) * 4

    def summary(self) -> str:
        return (
            f"node={self.node_id} | score={self.anomaly_score:.3f} | "
            f"class={self.fault_class} | conf={self.confidence:.2f} | "
            f"payload≈{self.serialize_size()}B"
        )


# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

class FeatureExtractor:
    """
    256-örneklik titreşim penceresinden özellik çıkarır.
    Hann penceresi + FFT kullanır.
    
    Hedef: RPi4 simülasyonunda ≤ 3ms
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE_HZ, window_size: int = WINDOW_SIZE):
        self.fs = sample_rate
        self.n  = window_size
        # Hann penceresi: spektral sızıntıyı azaltır
        self.hann = np.hanning(window_size)
        # Frekans ekseni (sadece pozitif frekanslar)
        self.freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
        # BPFO frekans indeksi (87.3 Hz) — rfft boyutu N//2+1
        rfft_size = window_size // 2 + 1
        freq_resolution = sample_rate / window_size   # Hz per bin
        self.bpfo_idx = min(int(87.3 / freq_resolution), rfft_size - 1)

    def extract(self, window: SensorWindow) -> FeatureVector:
        """Ana özellik çıkarım fonksiyonu."""
        t0 = time.perf_counter()

        vib = window.vibration  # (256, 3)
        x, y, z = vib[:, 0], vib[:, 1], vib[:, 2]

        # ── FFT (X ekseni üzerinde) ────────────────────────────────────────
        x_windowed = x * self.hann
        fft_mag    = np.abs(np.fft.rfft(x_windowed))
        fft_power  = fft_mag ** 2

        # Dominant frekans
        dom_idx      = np.argmax(fft_power[1:]) + 1  # DC'yi atla
        dominant_freq = float(self.freqs[dom_idx])

        # Spektral entropi: H = -Σ p·log(p)
        p_norm = fft_power / (fft_power.sum() + 1e-10)
        spectral_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-10)))

        # BPFO frekansındaki güç (rulman arızası göstergesi)
        bpfo_power = float(fft_power[self.bpfo_idx])

        # En güçlü 5 frekansın toplam gücü
        top5_power = float(np.sort(fft_power)[-5:].sum())

        # Frekans ağırlık merkezi (spectral centroid)
        freq_centroid = float(
            np.sum(self.freqs * fft_power) / (fft_power.sum() + 1e-10)
        )

        # ── Zaman Domain — X ──────────────────────────────────────────────
        rms_x         = float(np.sqrt(np.mean(x ** 2)))
        kurtosis_x    = float(self._kurtosis(x))
        crest_factor_x = float(np.max(np.abs(x)) / (rms_x + 1e-10))
        skewness_x    = float(self._skewness(x))

        # ── Zaman Domain — Y ──────────────────────────────────────────────
        rms_y      = float(np.sqrt(np.mean(y ** 2)))
        kurtosis_y = float(self._kurtosis(y))

        # ── Çapraz Korelasyon ─────────────────────────────────────────────
        xy_corr = float(np.corrcoef(x, y)[0, 1])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > MAX_INFERENCE_MS:
            print(f"{Color.YELLOW}[WARN]{Color.RESET} Feature extraction {elapsed_ms:.1f}ms > budget {MAX_INFERENCE_MS}ms")

        return FeatureVector(
            dominant_freq    = dominant_freq,
            spectral_entropy = spectral_entropy,
            bpfo_power       = bpfo_power,
            top5_freq_power  = top5_power,
            freq_centroid    = freq_centroid,
            rms_x            = rms_x,
            kurtosis_x       = kurtosis_x,
            crest_factor_x   = crest_factor_x,
            skewness_x       = skewness_x,
            rms_y            = rms_y,
            kurtosis_y       = kurtosis_y,
            thermal          = window.thermal,
            acoustic_rms     = window.acoustic_rms,
            xy_correlation   = xy_corr,
        )

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Fisher kurtosis (normal dağılım = 0)."""
        mu  = np.mean(x)
        std = np.std(x) + 1e-10
        return float(np.mean(((x - mu) / std) ** 4) - 3)

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        mu  = np.mean(x)
        std = np.std(x) + 1e-10
        return float(np.mean(((x - mu) / std) ** 3))


# ─── ANOMALİ SKORLAYICI ───────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Kural tabanlı anomali skoru hesaplar.
    
    Gerçek projede: TFLite INT8 model veya IsolationForest kullanılır.
    Burada: Belirlenmiş fiziksel eşikler üzerinden ağırlıklı skor.
    
    Bu yaklaşım:
    - TFLite bağımlılığı olmadan test edilebilir (Aşama 1)
    - Yorumlanabilir (her kural ne katkı sağladı görülür)
    - Aşama 2'de gerçek model ile değiştirilecek
    """

    # Eşik değerleri (titreşim analizi literatüründen)
    THRESHOLDS = {
        "kurtosis":      (0.5,  0.30),  # simüle edilmiş sinyal için ayarlı
        "crest_factor":  (3.0,  0.20),
        "bpfo_power":    (1e-5, 0.25),  # FFT power birimi (simülasyon)
        "rms_x":         (0.08, 0.15),  # simüle sinyal genliği ~0.05–0.15g
        "spectral_entropy_low": (1.5, 0.10),
    }

    def score(self, fv: FeatureVector) -> tuple[float, str, float]:
        """
        Returns: (anomaly_score, fault_class, confidence)
        anomaly_score ∈ [0, 1]
        """
        score     = 0.0
        triggered = []

        # Kurtosis kontrolü
        thresh, weight = self.THRESHOLDS["kurtosis"]
        if fv.kurtosis_x > thresh:
            contribution = min(1.0, (fv.kurtosis_x - thresh) / thresh) * weight
            score += contribution
            triggered.append(f"kurtosis={fv.kurtosis_x:.2f}")

        # Crest factor
        thresh, weight = self.THRESHOLDS["crest_factor"]
        if fv.crest_factor_x > thresh:
            contribution = min(1.0, (fv.crest_factor_x - thresh) / thresh) * weight
            score += contribution
            triggered.append(f"crest={fv.crest_factor_x:.2f}")

        # BPFO gücü (rulman arızası karakteristik frekansı)
        thresh, weight = self.THRESHOLDS["bpfo_power"]
        if fv.bpfo_power > thresh:
            contribution = min(1.0, fv.bpfo_power / (thresh * 10)) * weight
            score += contribution
            triggered.append(f"bpfo_power={fv.bpfo_power:.4f}")

        # RMS artışı
        thresh, weight = self.THRESHOLDS["rms_x"]
        if fv.rms_x > thresh:
            contribution = min(1.0, (fv.rms_x - thresh) / thresh) * weight
            score += contribution
            triggered.append(f"rms={fv.rms_x:.3f}")

        # Spektral entropi düşüklüğü
        thresh, weight = self.THRESHOLDS["spectral_entropy_low"]
        if fv.spectral_entropy < thresh:
            contribution = (thresh - fv.spectral_entropy) / thresh * weight
            score += contribution
            triggered.append(f"low_entropy={fv.spectral_entropy:.3f}")

        # Isı etkisi
        if fv.thermal > 60.0:
            score += min(0.1, (fv.thermal - 60.0) / 100.0)
            triggered.append(f"thermal={fv.thermal:.1f}°C")

        score = min(1.0, score)

        # Arıza sınıflandırması
        fault_class, confidence = self._classify(fv, triggered)

        return score, fault_class, confidence

    def _classify(self, fv: FeatureVector, triggered: list) -> tuple[str, float]:
        """Basit kural tabanlı sınıflandırma."""
        # BPFO → rulman
        if fv.bpfo_power > 1e-5 and fv.kurtosis_x > 0.5:
            severity = "EARLY" if fv.kurtosis_x < 3.0 else "SEVERE"
            conf     = min(0.95, 0.5 + fv.kurtosis_x / 10)
            return f"BEARING_{severity}", conf

        # Yüksek RMS + düşük kurtosis → aşırı yük
        if fv.rms_x > 0.08 and fv.kurtosis_x < 0.3:
            return "OVERLOAD", 0.70

        # Harmonik baskınlığı → hizasızlık
        if fv.spectral_entropy < 1.5 and fv.dominant_freq > 40:
            return "MISALIGNMENT", 0.65

        return "UNKNOWN", 0.40


# ─── µ-AGENT FSM ──────────────────────────────────────────────────────────────

class MicroAgent:
    """
    Aşama 3 µ-Ajan: FSM + ML inference + MQTT mesajlaşma.

    Yeni özellikler (Aşama 3):
      - MQTT publish: anomali uyarısını broker'a yayınlar
      - MQTT subscribe: orkestratörden ACK/COMMAND alır
      - Heartbeat: periyodik sağlık sinyali yayınlar
      - CONSENSUS_WAIT: gerçek ACK bekler (simüle timeout ile)
      - alert_callback: geriye dönük uyumluluk için korundu
    """

    def __init__(
        self,
        node_id:        str,
        cluster_id:     str,
        buffer:         SensorBuffer,
        alert_callback: Optional[Callable] = None,
        use_ml:         bool = True,
        mqtt_client:    Optional["MQTTClient"] = None,  # Aşama 3
    ):
        self.node_id        = node_id
        self.cluster_id     = cluster_id
        self.buffer         = buffer
        self.alert_callback = alert_callback
        self.mqtt           = mqtt_client          # Aşama 3: MQTT istemcisi

        self.state     = AgentState.IDLE
        self.extractor = FeatureExtractor()

        # Bekleyen ACK'ler için event map: window_id → asyncio.Event
        self._ack_events: dict[int, asyncio.Event] = {}

        # Scorer seçimi
        if use_ml and _ML_SCORER_AVAILABLE:
            try:
                self.scorer   = AnomalyScorerML()
                self._ml_mode = True
            except FileNotFoundError:
                print(f"{Color.YELLOW}[WARN]{Color.RESET} Model dosyası yok → kural tabanlı scorer")
                self.scorer   = AnomalyScorer()
                self._ml_mode = False
        else:
            self.scorer   = AnomalyScorer()
            self._ml_mode = False

        scorer_label = f"{Color.GREEN}ML (IF+RF){Color.RESET}" if self._ml_mode else f"{Color.YELLOW}Kural tabanlı{Color.RESET}"
        mqtt_label   = f"{Color.GREEN}MQTT ✓{Color.RESET}" if self.mqtt else f"{Color.YELLOW}MQTT yok{Color.RESET}"

        # İstatistikler
        self.stats = {
            "windows_processed":  0,
            "alerts_fired":       0,
            "false_positives":    0,
            "acks_received":      0,
            "ack_timeouts":       0,
            "heartbeats_sent":    0,
            "avg_inference_ms":   0.0,
            "total_inference_ms": 0.0,
            "bytes_published":    0,
        }

        print(f"{Color.BLUE}[AGENT]{Color.RESET} µ-Agent {Color.BOLD}{node_id}{Color.RESET} "
              f"| scorer={scorer_label} | {mqtt_label} | eşik={ANOMALY_THRESHOLD}")

    async def run(self):
        """Ana ajan döngüsü — MQTT ile birlikte çalışır."""
        # MQTT bağlantısı ve subscription'lar
        if self.mqtt:
            await self.mqtt.connect()
            # Orkestratörden ACK mesajlarını dinle
            ack_topic = f"sensor/{self.cluster_id}/{self.node_id}/ack"
            self.mqtt.subscribe(ack_topic, self._on_ack)
            # Orkestratörden komutları dinle
            cmd_topic = f"orchestrator/{self.cluster_id}/command"
            self.mqtt.subscribe(cmd_topic, self._on_command)

        self._transition(AgentState.SAMPLING)

        # Heartbeat ve ana döngü paralel çalışır
        coroutines = [self._main_loop()]
        if self.mqtt:
            coroutines.append(self._heartbeat_loop())

        await asyncio.gather(*coroutines)

    async def _main_loop(self):
        """Pencere işleme döngüsü."""
        while True:
            try:
                window: SensorWindow = await asyncio.wait_for(
                    self.buffer.window_q.get(),
                    timeout=1.0
                )
                await self._process_window(window)
            except asyncio.TimeoutError:
                if self.state == AgentState.SAMPLING:
                    self._transition(AgentState.IDLE)
                await asyncio.sleep(0.01)

    async def _heartbeat_loop(self):
        """
        Periyodik heartbeat yayını.
        Orkestratör bu sinyali kullanarak node sağlığını izler.
        5 heartbeat gelmeyen node = DOWN olarak işaretlenir.
        """
        from config import TIME_ACCELERATION
        interval = 10.0 / TIME_ACCELERATION   # 10s gerçek → hızlandırılmış

        while True:
            await asyncio.sleep(interval)
            if self.mqtt:
                payload = PayloadSchema.heartbeat(
                    node_id=self.node_id,
                    cluster_id=self.cluster_id,
                    state=self.state.name,
                )
                await self.mqtt.publish("system/heartbeat", payload, qos=QoS.AT_MOST_ONCE)
                self.stats["heartbeats_sent"] += 1

    async def _on_ack(self, msg):
        """Orkestratörden ACK geldiğinde."""
        try:
            data = msg.to_dict()
            window_id = data.get("window_id")
            action    = data.get("action", "ACK")

            self.stats["acks_received"] += 1

            # Bekleyen event'i tetikle
            if window_id in self._ack_events:
                self._ack_events[window_id].set()

            action_color = Color.GREEN if action == "ACK" else Color.YELLOW
            print(
                f"{Color.CYAN}[ACK ←]{Color.RESET} {self.node_id} "
                f"win={window_id} | action={action_color}{action}{Color.RESET}"
            )
        except Exception as e:
            print(f"{Color.YELLOW}[WARN]{Color.RESET} ACK parse hatası: {e}")

    async def _on_command(self, msg):
        """Orkestratörden komut geldiğinde."""
        try:
            data    = msg.to_dict()
            target  = data.get("target_node", "*")
            command = data.get("command", "")

            # Sadece bana veya broadcast mesajları
            if target not in ("*", self.node_id):
                return

            print(
                f"{Color.YELLOW}[CMD ←]{Color.RESET} {self.node_id} "
                f"← {command} | params={data.get('params', {})}"
            )

            if command == "REDUCE_SAMPLING":
                # Örnekleme hızını düşür (güç tasarrufu)
                print(f"  {self.node_id}: örnekleme azaltıldı")
            elif command == "SHUTDOWN":
                print(f"  {self.node_id}: kapatılıyor...")
        except Exception as e:
            print(f"{Color.YELLOW}[WARN]{Color.RESET} CMD parse hatası: {e}")

    async def _process_window(self, window: SensorWindow):
        """Tek bir pencereyi işle: feature extraction → scoring → karar."""
        self._transition(AgentState.INFERENCE)
        t0 = time.perf_counter()

        # ── Feature Extraction ────────────────────────────────────────────
        features = self.extractor.extract(window)

        # ── Anomali Skoru ─────────────────────────────────────────────────
        score, fault_class, confidence = self.scorer.score(features)

        # ── İstatistik Güncelle ───────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["windows_processed"] += 1
        self.stats["total_inference_ms"] += elapsed_ms
        self.stats["avg_inference_ms"] = (
            self.stats["total_inference_ms"] / self.stats["windows_processed"]
        )

        # Bütçe aşımı uyarısı
        if elapsed_ms > MAX_INFERENCE_MS:
            print(f"{Color.YELLOW}[WARN]{Color.RESET} {self.node_id} inference {elapsed_ms:.1f}ms > budget")

        # ── Eşik Kontrolü & FSM Geçişi ────────────────────────────────────
        if score > ANOMALY_THRESHOLD:
            self._transition(AgentState.ALERT_STAGE)
            self.stats["alerts_fired"] += 1

            # Ground truth ile karşılaştır (simülasyon avantajı)
            is_true_positive = window.injected_fault != FaultType.NONE
            if not is_true_positive:
                self.stats["false_positives"] += 1

            # AnomalyEvent oluştur
            event = AnomalyEvent(
                node_id       = self.node_id,
                cluster_id    = self.cluster_id,
                timestamp_ns  = window.timestamp,
                window_id     = window.window_id,
                anomaly_score = score,
                fault_class   = fault_class,
                confidence    = confidence,
                features      = features.to_array(),
                true_fault    = window.injected_fault,
            )

            # Payload boyutu kontrol
            payload_size = event.serialize_size()
            assert payload_size <= MAX_MQTT_PAYLOAD_B, \
                f"Payload bütçesi aşıldı: {payload_size}B > {MAX_MQTT_PAYLOAD_B}B"

            # Çıktı
            tp_str = f"{Color.GREEN}TP{Color.RESET}" if is_true_positive else f"{Color.RED}FP{Color.RESET}"
            print(
                f"{Color.RED}[ALERT →]{Color.RESET} {self.node_id} | "
                f"{event.summary()} | {tp_str} | {elapsed_ms:.1f}ms"
            )

            # ── MQTT Publish (Aşama 3) ────────────────────────────────────
            if self.mqtt:
                alert_topic = f"sensor/{self.cluster_id}/{self.node_id}/alert"
                payload = PayloadSchema.alert(
                    node_id      = self.node_id,
                    cluster_id   = self.cluster_id,
                    window_id    = event.window_id,
                    anomaly_score= event.anomaly_score,
                    fault_class  = event.fault_class,
                    confidence   = event.confidence,
                    thermal      = window.thermal,
                    timestamp_ns = event.timestamp_ns,
                )
                published = await self.mqtt.publish(alert_topic, payload)
                self.stats["bytes_published"] += len(str(payload).encode())

                if published:
                    # ACK bekle (CONSENSUS_WAIT simülasyonu)
                    self._transition(AgentState.CONSENSUS_WAIT)
                    ack_event = asyncio.Event()
                    self._ack_events[event.window_id] = ack_event
                    try:
                        await asyncio.wait_for(ack_event.wait(), timeout=0.3)
                    except asyncio.TimeoutError:
                        self.stats["ack_timeouts"] += 1
                    finally:
                        self._ack_events.pop(event.window_id, None)
            else:
                self._transition(AgentState.CONSENSUS_WAIT)
                await asyncio.sleep(0.05)

            # Geriye dönük uyumluluk: eski callback
            if self.alert_callback:
                await self.alert_callback(event)

            self._transition(AgentState.IDLE)

        else:
            self._transition(AgentState.SAMPLING)
            # Normal pencere — sadece debug modunda logla
            # print(f"[OK] {self.node_id} score={score:.3f} < {ANOMALY_THRESHOLD}")

    def _transition(self, new_state: AgentState):
        """FSM durum geçişi."""
        # Geçerli geçişler
        valid_transitions = {
            AgentState.IDLE:           [AgentState.SAMPLING],
            AgentState.SAMPLING:       [AgentState.INFERENCE, AgentState.IDLE],
            AgentState.INFERENCE:      [AgentState.ALERT_STAGE, AgentState.SAMPLING],
            AgentState.ALERT_STAGE:    [AgentState.CONSENSUS_WAIT],
            AgentState.CONSENSUS_WAIT: [AgentState.ACTUATING, AgentState.IDLE],
            AgentState.ACTUATING:      [AgentState.IDLE],
        }

        if new_state not in valid_transitions.get(self.state, []):
            # Geçersiz geçiş — zorla geçiş yapmak yerine uyar
            pass  # Aşama 4'te strict mode eklenecek

        self.state = new_state

    def report(self) -> dict:
        """Ajan istatistik raporu."""
        total = self.stats["windows_processed"]
        alerts = self.stats["alerts_fired"]
        fps = self.stats["false_positives"]
        return {
            "node_id":           self.node_id,
            "windows_processed": total,
            "alerts_fired":      alerts,
            "false_positives":   fps,
            "false_positive_rate": fps / max(alerts, 1),
            "avg_inference_ms":  round(self.stats["avg_inference_ms"], 2),
        }
