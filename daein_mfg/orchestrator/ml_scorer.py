"""
DAEIN-MFG — ML Tabanlı Anomali Skorlayıcı (Aşama 2)
=====================================================
Kural tabanlı AnomalyScorer'ın yerini alır.
micro_agent.py içinde drop-in replacement olarak kullanılır.

Mimari:
  IsolationForest  → anomaly_score ∈ [0,1]   (her pencerede çalışır)
  RandomForest     → fault_class + confidence (sadece eşik aşıldığında)

Bu ikili yapı edge sistemlerinde tercih edilir:
  - IF hafif, unsupervised → sürekli çalışabilir
  - RF daha ağır, supervised → sadece gerektiğinde tetiklenir
"""

import numpy as np
import joblib
from pathlib import Path
import time


FAULT_LABEL_MAP = {
    0: "NORMAL",
    1: "BEARING_EARLY",
    2: "BEARING_SEVERE",
    3: "OVERLOAD",
    4: "MISALIGNMENT",
}


class MLAnomalyScorer:
    """
    IsolationForest + RandomForest tabanlı anomali skorlayıcı.
    AnomalyScorer (kural tabanlı) ile aynı arayüzü paylaşır →
    micro_agent.py'de sadece sınıf adı değişir, başka hiçbir şey değişmez.
    """

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            # Bu dosyanın bulunduğu klasör = models/
            model_dir = Path(__file__).parent

        iso_path = model_dir / "isolation_forest.joblib"
        clf_path = model_dir / "fault_classifier.joblib"

        if not iso_path.exists() or not clf_path.exists():
            raise FileNotFoundError(
                f"Model dosyaları bulunamadı: {model_dir}\n"
                "Önce 'python models/train.py' komutunu çalıştır."
            )

        self._iso = joblib.load(iso_path)   # IsolationForest pipeline
        self._clf = joblib.load(clf_path)   # RandomForest pipeline

        # Kalibrasyon parametreleri (eğitimde gömüldü)
        self._s_min = self._iso._score_min
        self._s_max = self._iso._score_max

        print(f"  [MLScorer] IsolationForest + RandomForest yüklendi ✓")
        print(f"             Modeller: {iso_path.name}, {clf_path.name}")

    def score(self, fv) -> tuple[float, str, float]:
        """
        AnomalyScorer.score() ile aynı imza:
          Returns: (anomaly_score, fault_class, confidence)

        fv: FeatureVector nesnesi (micro_agent.py'den)
        """
        x = fv.to_array().reshape(1, -1)   # (1, 14)

        # ── Adım 1: IsolationForest → anomali skoru ───────────────────────
        t0 = time.perf_counter()
        raw = self._iso.decision_function(x)[0]
        anomaly_score = 1.0 - (raw - self._s_min) / (self._s_max - self._s_min + 1e-10)
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
        iso_ms = (time.perf_counter() - t0) * 1000

        # ── Adım 2: RandomForest → sınıf (sadece eşik aşıldıysa) ─────────
        # Skor düşükse classifier çalıştırmayı atla (edge tasarrufu)
        CLASSIFIER_THRESHOLD = 0.40   # IF skoru bu değerin üstündeyse RF devreye girer

        if anomaly_score >= CLASSIFIER_THRESHOLD:
            t1 = time.perf_counter()
            class_idx   = int(self._clf.predict(x)[0])
            class_probs = self._clf.predict_proba(x)[0]
            confidence  = float(class_probs[class_idx])
            fault_class = FAULT_LABEL_MAP.get(class_idx, "UNKNOWN")
            clf_ms = (time.perf_counter() - t1) * 1000
        else:
            fault_class = "NORMAL"
            confidence  = 1.0 - anomaly_score
            clf_ms      = 0.0

        total_ms = iso_ms + clf_ms

        # Bütçe aşımı log'u (RPi4 → 8ms hedef)
        if total_ms > 8.0:
            print(f"  [WARN] MLScorer {total_ms:.1f}ms > 8ms bütçe "
                  f"(IF={iso_ms:.1f}ms, RF={clf_ms:.1f}ms)")

        return anomaly_score, fault_class, confidence

    def benchmark(self, n_runs: int = 500) -> dict:
        """
        Inference hız testi. Simülasyon raporunda kullanılır.
        n_runs pencere üzerinde ortalama/std/p99 hesaplar.
        """
        dummy = np.random.default_rng(0).random((1, 14)).astype(np.float32)

        # Dummy FeatureVector proxy
        class FVProxy:
            def to_array(self): return dummy[0]

        fv = FVProxy()
        latencies = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.score(fv)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        return {
            "n_runs":   n_runs,
            "mean_ms":  round(float(latencies.mean()), 3),
            "std_ms":   round(float(latencies.std()),  3),
            "p99_ms":   round(float(np.percentile(latencies, 99)), 3),
            "max_ms":   round(float(latencies.max()),  3),
            "budget_ok": bool(np.percentile(latencies, 99) < 8.0),
        }
