# DAEIN-MFG — Sistem Mimarisi

> Bu belge makale / tez için "System Design" bölümünün taslağıdır.

---

## 1. Genel Bakış

DAEIN-MFG üç katmanlı bir hiyerarşi üzerine inşa edilmiştir:

```
KATMAN 3: BULUT (opsiyonel — sadece offline model eğitimi)
KATMAN 2: GATEWAY ORKESTRATÖRÜ  (Jetson Nano / eşdeğer)
KATMAN 1: µ-AGENT               (Raspberry Pi 4 / ESP32-S3)
```

Her katman kendi bağımsız karar döngüsüne sahiptir. Üst katmanlar yalnızca
alt katmanların üretemediği kararlar için devreye girer. Bu prensip
**subsidyarity** olarak adlandırılır ve sistemin toplam gecikme bütçesini
minimize eder.

---

## 2. µ-Agent (Tier 1)

### 2.1 Donanım Profili (Simüle)

| Parametre | Raspberry Pi 4 | ESP32-S3 |
|---|---|---|
| CPU | 4× Cortex-A72 @ 1.8GHz | 2× LX7 @ 240MHz |
| RAM | 2 GB (2GB limit) | 8 MB PSRAM |
| Ağ | 802.11n, 20Mbps | BLE/WiFi, 1Mbps |
| Güç | ~5W | ~0.3W |
| Model boyutu | ≤ 2MB (TFLite INT8) | ≤ 256KB (TFLite Micro) |

### 2.2 Sonlu Durum Makinesi

```
IDLE ──► SAMPLING ──► INFERENCE ──► ALERT_STAGE ──► CONSENSUS_WAIT ──► IDLE
                         │
                         └──► SAMPLING  (eşik altı)
```

**Durum Geçiş Koşulları:**
- `IDLE → SAMPLING`: ilk sensör verisi geldiğinde
- `SAMPLING → INFERENCE`: 256 örneklik pencere tamamlandığında
- `INFERENCE → ALERT_STAGE`: anomaly_score > θ (0.55)
- `INFERENCE → SAMPLING`: anomaly_score ≤ θ
- `ALERT_STAGE → CONSENSUS_WAIT`: MQTT publish başarılı
- `CONSENSUS_WAIT → IDLE`: ACK alındı veya timeout (300ms)

### 2.3 Feature Extraction Pipeline

```
Ham titreşim (256 × 3)
    │
    ├─ Hann penceresi
    │
    ├─ FFT (rfft, 129 bin)
    │    ├─ Dominant frekans
    │    ├─ Spektral entropi
    │    ├─ BPFO gücü (87.3 Hz)
    │    ├─ Top-5 güç toplamı
    │    └─ Frekans ağırlık merkezi
    │
    └─ Zaman domain
         ├─ RMS (X, Y)
         ├─ Kurtosis (X, Y)
         ├─ Crest factor (X)
         ├─ Skewness (X)
         └─ XY cross-correlation

Çıktı: 14-boyutlu float32 vektör
```

### 2.4 İki Aşamalı ML Skorlayıcı

```
Pencere → IsolationForest → anomaly_score ∈ [0,1]
                │
                │ score > 0.40 ?
                ▼
          RandomForest → fault_class + confidence
```

**IsolationForest (unsupervised):**
- Sadece normal örneklerle eğitilir
- Her pencerede çalışır (düşük maliyet)
- Teorik temel: anormal noktalar daha kısa yollarla izole edilir

**RandomForest (supervised):**
- 5 sınıf: NORMAL, BEARING_EARLY, BEARING_SEVERE, OVERLOAD, MISALIGNMENT
- Sadece IF eşiği aşıldığında tetiklenir
- 5-fold CV F1 ≈ 0.98

---

## 3. Orkestratör (Tier 2)

### 3.1 PPO Politika Ağı

```
Durum S (7 boyut):
  [cpu_util, mem_util, queue_depth, n_active_alerts,
   peer_avg_load, anomaly_score, confidence]

Policy Network: 7 → 32 → 16 → 5 (ReLU + Softmax)
Value Network:  7 → 32 → 16 → 1  (ReLU)

Aksiyon A (5 ayrık):
  0: LOCAL_INFER       → yerel işle
  1: OFFLOAD_PEER      → en boş komşuya gönder
  2: DEFER_500MS       → ertele, yeniden değerlendir
  3: ESCALATE          → anında aktüatör tetikle
  4: REQUEST_CONSENSUS → Raft oyu iste
```

**Ödül Fonksiyonu:**
```
R = -0.3·(τ/500) + 0.4·TP_bonus - 0.5·FP_penalty
    -0.2·max(0, CPU-0.7) - 0.1·(consensus_rounds/5)
    + 0.5·[score>0.9 AND TP]
    - 0.3·[ESCALATE AND score<0.6]
```

### 3.2 Raft + Auction Konsensüs

**Neden hibrit?**
Standart Raft log replikasyonu deterministik karar alır. Ancak endüstriyel
arıza tespitinde "en güçlü kanıta sahip node kazanmalı" prensibi
gereklidir. Sealed-bid auction bu amaca hizmet eder.

**Protokol Akışı:**
```
1. REQUEST_CONSENSUS → Raft liderine teklif gönder
2. Lider tüm peer'lerden VoteBid toplar
3. bid = f(local_score, corroboration, confidence)
4. Quorum (⌊N/2⌋+1) bid alındıktan sonra log'a commit
5. Commit → tüm node'lara AppendEntries
6. µ-Agent → ACK (CONSENSUS_COMMITTED)
```

**Hata Toleransı:**
- Node çöküşü: Raft election timeout = 150–300ms
- Paket kaybı: Sequence number ile gap tespiti
- Byzantine bid: Median filtresi (outlier rejection)

---

## 4. Mesajlaşma Katmanı

### 4.1 MQTT Konu Hiyerarşisi

```
sensor/{cluster_id}/{node_id}/alert      → µ-Agent yayınlar
sensor/{cluster_id}/{node_id}/ack        → Orkestratör yanıt verir
orchestrator/{cluster_id}/command        → Cluster komutları
orchestrator/{cluster_id}/load_report    → Yük paylaşımı
orchestrator/{cluster_id}/offload        → İş yönlendirme
system/heartbeat                         → Tüm node'lar
```

### 4.2 Payload Boyut Bütçeleri

| Mesaj Tipi | Boyut | Açıklama |
|---|---|---|
| µ-Agent alert | ~147B | JSON serileştirilmiş anomali raporu |
| Orkestratör ACK | ~120B | Aksiyon + konsensüs durumu |
| Heartbeat | ~80B | Durum + timestamp |
| Cluster komut | ~150B | Komut + parametreler |

---

## 5. Değerlendirme Çerçevesi

### 5.1 Edge Advantage Function

```
EAF(τ, β, ρ) = α₁·(1/τ_edge - 1/τ_cloud)
             + α₂·(β_saved / β_total)
             + α₃·ρ_compliance

α₁ = 0.40 (gecikme ağırlığı)
α₂ = 0.35 (bant genişliği ağırlığı)
α₃ = 0.25 (veri uyumluluğu ağırlığı)
```

EAF > 0 → Edge sistemin cloud'a kıyasla üstünlüğü

### 5.2 Ölçüm Metrikleri

**Sistem Metrikleri:**
- DAL: Detection-to-Actuation Latency (ms)
- BER: Bandwidth Efficiency Ratio
- Raft commit süresi (ms)
- PPO ödül yakınsaması

**Domain Metrikleri:**
- F1, Precision, Recall (her arıza tipi için)
- False Positive Rate
- Cohen's d (etki büyüklüğü)

---

## 6. Gerçek Donanıma Geçiş Yol Haritası

| Bileşen | Simülasyon | Gerçek Deployment |
|---|---|---|
| Sensör | `SignalGenerator` | RPi4 UART/I2C |
| Inference | scikit-learn | TFLite INT8 |
| MQTT | In-process broker | Mosquitto / EMQX |
| Actor Runtime | `ActorRuntime` | `@ray.remote` |
| SLM | (stub) | Phi-3-mini via Ollama |
| Zaman | 10× hızlandırılmış | Gerçek zamanlı |

Geçiş için değişmesi gereken dosyalar:
1. `daein_mfg/messaging/broker.py` → `paho-mqtt` wrapper
2. `daein_mfg/orchestrator/ml_scorer.py` → `tflite-runtime`
3. `daein_mfg/orchestrator/actor_runtime.py` → `@ray.remote`
4. `daein_mfg/config.py` → `TIME_ACCELERATION = 1`

Tüm diğer kod değişmez.
