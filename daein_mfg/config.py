"""
DAEIN-MFG — Merkezi Yapılandırma
Tüm sistem parametreleri burada tanımlanır.
Gerçek donanıma geçince sadece bu dosyayı değiştirmen yeterli.
"""

# ─── SENSÖR AYARLARI ─────────────────────────────────────────────────────────

SAMPLE_RATE_HZ      = 100       # Titreşim sensörü örnekleme frekansı
WINDOW_SIZE         = 256       # FFT pencere boyutu (örnek sayısı)
WINDOW_OVERLAP      = 0.5       # %50 örtüşme (sliding window)
THERMAL_RATE_HZ     = 10        # Isı sensörü frekansı
AUDIO_RATE_HZ       = 8000      # Akustik sensör (düşürülmüş, edge için)

# ─── AJAN AYARLARI ────────────────────────────────────────────────────────────

N_CLUSTERS          = 3         # Simülasyondaki cluster sayısı
NODES_PER_CLUSTER   = 4         # Her cluster'daki µ-Agent sayısı

# Anomali eşiği: bu değerin üstüne çıkınca ajan ALERT moduna geçer
ANOMALY_THRESHOLD   = 0.55

# Donanım bütçeleri (simülasyonda zorlanır)
MAX_INFERENCE_MS    = 8         # RPi4 için max inference süresi
MAX_MQTT_PAYLOAD_B  = 512       # MQTT mesaj boyut limiti (byte)

# ─── ARIZA ENJEKSİYON AYARLARI ───────────────────────────────────────────────

FAULT_INJECT_INTERVAL_S = 30    # Kaç saniyede bir arıza enjekte edilsin
FAULT_AMPLITUDE         = 0.35  # Arıza sinyalinin genliği (g cinsinden)
FAULT_FREQUENCY_HZ      = 87.3  # BPFO — Bearing Pass Frequency Outer race

# ─── SİMÜLASYON AYARLARI ─────────────────────────────────────────────────────

SIM_DURATION_S      = 120       # Toplam simülasyon süresi (saniye)
TIME_ACCELERATION   = 10        # 10x hızlandırma (120s → 12s gerçek süre)
LOG_LEVEL           = "INFO"    # DEBUG | INFO | WARNING

# ─── MQTT AYARLARI ────────────────────────────────────────────────────────────

MQTT_BROKER_HOST    = "localhost"
MQTT_BROKER_PORT    = 1883
MQTT_TOPIC_TEMPLATE = "sensor/{cluster_id}/{node_id}/alert"

# ─── RENK KODLARI (terminal çıktısı için) ────────────────────────────────────
class Color:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
