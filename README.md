# DAEIN-MFG

**Decentralized Agentic Edge Intelligence Network for Proactive Fault Prediction and Autonomous Self-Healing in Smart Manufacturing**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Özet

DAEIN-MFG, endüstriyel IoT ortamlarında **bulut bağımlılığı olmadan** gerçek zamanlı makine arızası tespiti ve otonom müdahale gerçekleştiren bir çok-ajanlı kenar zekası simülasyonudur.

Üç paradigmanın kesişiminde konumlanır:

| Paradigma | Bu Projede Karşılığı |
|---|---|
| Agentic Edge Intelligence | FSM tabanlı µ-Agent + PPO Orkestratör |
| Agentic IoT | MQTT üzerinden otonom sensör-aktüatör koordinasyonu |
| Distributed Data Engineering | Async stream pipeline + Raft konsensüs |

### Temel Akademik Katkılar

1. **Edge Advantage Function (EAF):** Gecikme, bant genişliği ve veri uyumluluğunu tek bir skorda birleştiren ölçüm çerçevesi.
2. **Hibrit Raft + Auction Konsensüs:** Arıza şiddet oylaması için sealed-bid açık artırma mekanizması ile genişletilmiş Raft.
3. **İki Aşamalı ML Skorlayıcı:** IsolationForest (sürekli anomali skoru) + RandomForest (tetiklemede sınıflandırma) mimarisi.
4. **Resource-Aware PPO:** Donanım bütçesi kısıtlarını durum uzayına dahil eden RL politikası.

---

## Sistem Mimarisi

```
TIER 2: GATEWAY ORKESTRATÖRÜ (Jetson Nano sim.)
  ┌──────────────────────────────────────────────┐
  │  MQTTClient ──► PPO Politikası               │
  │  RaftNode   ──► Auction Bid Mekanizması      │
  │  ActorHandle ── Ray-uyumlu async proxy       │
  └──────────────────────────────────────────────┘
                     ▲ MQTT
TIER 1: µ-AGENT (Raspberry Pi 4 sim.)
  ┌──────────────────────────────────────────────┐
  │  FSM: IDLE→SAMPLING→INFERENCE→ALERT→WAIT    │
  │  IsolationForest + RandomForest (INT8 proxy) │
  │  FFT feature extraction (256-sample window)  │
  └──────────────────────────────────────────────┘
                     ▲ asyncio
SENSÖR SİMÜLATÖRÜ
  Titreşim (100Hz) + Isı (10Hz) + Rulman arıza enjeksiyonu
```

---

## Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/[username]/daein-mfg.git
cd daein-mfg

# 2. Sanal ortam oluştur (Python 3.10+)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Geliştirme modunda kur (import daein_mfg çalışması için)
pip install -e .
```

---

## Hızlı Başlangıç

### Adım 1 — Modelleri Eğit

```bash
python scripts/train_models.py
```

Çıktı:
```
Veri seti üretiliyor... 2000 örnek, 14 özellik
IsolationForest eğitiliyor...
  Normal  → ortalama anomali skoru: 0.312
  Anormal → ortalama anomali skoru: 0.718
RandomForest eğitiliyor...
  5-fold CV F1 (weighted): 0.9821 ± 0.0043
Kaydedildi: isolation_forest.joblib (648 KB)
             fault_classifier.joblib  (40 KB)
```

### Adım 2 — Simülasyonu Çalıştır

```bash
python scripts/run_simulation.py
```

Çıktı:
```
DAEIN-MFG — Aşama 5: Raft + Edge vs Cloud Analizi
Clusters: 3 | Nodes: 12 | Süre: 120s (gerçek ≈12s)
...
1. DETECTION-TO-ACTUATION LATENCY (DAL)
   Edge  (DAEIN-MFG) : ort=442ms  P95=450ms
   Cloud (Baseline)  : ort=920ms  P95=1899ms
   Azalma: +52%  |  Cohen's d = 1.559 (Büyük etki)

7. EDGE ADVANTAGE FUNCTION (EAF)
   EAF = +1.0439  →  Edge üstün ✓
```

### Adım 3 — Testleri Çalıştır

```bash
pytest tests/ -v --cov=daein_mfg
```

---

## Proje Yapısı

```
daein-mfg/
│
├── README.md                        # Bu dosya
├── requirements.txt                 # Bağımlılıklar
├── setup.py                         # Paket kurulum
├── .gitignore
├── LICENSE
│
├── daein_mfg/                       # Ana Python paketi
│   ├── config.py                    # Merkezi parametreler
│   │
│   ├── simulation/
│   │   ├── sensor_generator.py      # Fizik tabanlı sensör simülatörü
│   │   └── micro_agent.py           # µ-Agent FSM + ML inference + MQTT
│   │
│   ├── messaging/
│   │   ├── broker.py                # In-process async MQTT broker
│   │   └── gateway.py               # Cluster gateway (Aşama 3)
│   │
│   ├── orchestrator/
│   │   ├── actor_runtime.py         # Ray Actor proxy (async)
│   │   ├── ml_scorer.py             # IsolationForest + RF scorer
│   │   ├── ppo_policy.py            # PPO kaynak tahsis politikası
│   │   ├── orchestrator_agent.py    # Ana orkestratör ajanı
│   │   ├── raft.py                  # Raft + Auction-bid konsensüs
│   │   └── raft_cluster.py          # Çok-node Raft koordinatörü
│   │
│   └── evaluation/
│       ├── cloud_simulator.py       # Cloud gecikme/bant modeli
│       └── metrics_engine.py        # EAF + akademik metrik raporu
│
├── scripts/
│   ├── train_models.py              # Model eğitimi (bağımsız çalışır)
│   └── run_simulation.py            # Tam simülasyon (Aşama 5)
│
├── tests/
│   ├── conftest.py                  # Paylaşılan fixture'lar
│   ├── test_sensor.py               # Sensör simülatörü testleri
│   ├── test_micro_agent.py          # µ-Agent FSM + inference testleri
│   ├── test_broker.py               # MQTT broker testleri
│   ├── test_raft.py                 # Raft konsensüs testleri
│   └── test_metrics.py              # EAF ve metrik hesaplama testleri
│
├── notebooks/
│   └── results_analysis.ipynb      # Sonuç görselleştirme
│
└── docs/
    └── architecture.md             # Detaylı sistem mimarisi
```

---

## Konfigürasyon

Tüm parametreler `daein_mfg/config.py` içinde merkezi olarak yönetilir:

```python
# Anahtar parametreler
SAMPLE_RATE_HZ        = 100      # Titreşim sensörü örnekleme hızı
WINDOW_SIZE           = 256      # FFT pencere boyutu
ANOMALY_THRESHOLD     = 0.55     # µ-Agent alarm eşiği
N_CLUSTERS            = 3        # Cluster sayısı
NODES_PER_CLUSTER     = 4        # Cluster başına node
SIM_DURATION_S        = 120      # Simülasyon süresi
TIME_ACCELERATION     = 10       # Hız katsayısı
FAULT_INJECT_INTERVAL_S = 30     # Arıza enjeksiyon periyodu
```

---

## Arıza Türleri

| Tür | Fiziksel Model | Sinyal Göstergesi |
|---|---|---|
| `BEARING_EARLY` | BPFO darbesi (87.3 Hz) | Kurtosis > 0.5, BPFO gücü artışı |
| `BEARING_SEVERE` | Tüm eksenlerde darbe | Yüksek RMS, kurtosis > 3.0 |
| `OVERLOAD` | Genel RMS artışı | Düşük kurtosis, yüksek RMS |
| `MISALIGNMENT` | 2x, 3x harmonik baskınlığı | Düşük spektral entropi |

---

## Sonuçlar Özeti

Aşama 5 simülasyonu sonuçları (3 cluster, 12 node, 120s):

| Metrik | Edge (DAEIN-MFG) | Cloud Baseline | Gelişme |
|---|---|---|---|
| Ort. DAL (ms) | 442 | 920 | **−52%** |
| P99 DAL (ms) | 450 | 1961 | **−77%** |
| Bant Genişliği (KB) | 48.8 | 633.9 | **−92% (13×)** |
| F1 Skoru | 0.780 | 0.856 | Cloud yüksek* |
| EAF | +1.044 | — | **Edge üstün** |
| Cohen's d | 1.559 | — | Büyük etki |

*Cloud F1 yüksek çünkü timeout'larda bile mevcut sensör verisi kullanılıyor; gerçek ortamda retry gecikmesi bu avantajı ortadan kaldırır.

---

## Gerçek Donanıma Deployment

### Raspberry Pi 4 (µ-Agent)
```bash
pip install tflite-runtime
# micro_agent.py içindeki MLAnomalyScorer'ı TFLite ile değiştir
# config.py → TIME_ACCELERATION = 1 (gerçek zaman)
```

### NVIDIA Jetson Nano (Orkestratör)
```bash
pip install ray ollama
# actor_runtime.py → ActorRuntime.create() → @ray.remote
# orchestrator_agent.py → Phi-3-mini SLM entegrasyonu
```

### Gerçek MQTT Broker
```bash
pip install paho-mqtt
# messaging/broker.py → MQTTClient → paho.mqtt.client.Client wrapper
# Broker: Mosquitto veya EMQX (lokal kurulum)
```

---

## Geliştirme Yol Haritası

- [x] Aşama 1: Sensör simülatörü + µ-Agent FSM
- [x] Aşama 2: IsolationForest + RandomForest ML
- [x] Aşama 3: Async MQTT broker + ClusterGateway
- [x] Aşama 4: ActorRuntime + PPO RL + Peer Offloading
- [x] Aşama 5: Raft konsensüs + Edge vs Cloud analizi
- [ ] Aşama 6: TFLite INT8 export + gerçek RPi4 deployment
- [ ] Aşama 7: Phi-3-mini SLM entegrasyonu (Jetson Nano)
- [ ] Aşama 8: CWRU Bearing Dataset ile gerçek veri doğrulama

---

## Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.
