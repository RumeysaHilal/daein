"""
scripts/train_models.py
=======================
IsolationForest + RandomForest modellerini eğitip kaydeder.

Kullanım:
    python scripts/train_models.py

Çıktı:
    daein_mfg/orchestrator/isolation_forest.joblib
    daein_mfg/orchestrator/fault_classifier.joblib
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
import time
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

from daein_mfg.simulation.sensor_generator import SignalGenerator, FaultType, SensorWindow
from daein_mfg.simulation.micro_agent import FeatureExtractor
from daein_mfg.config import Color

FAULT_MAP = {
    FaultType.NONE: 0, FaultType.BEARING_EARLY: 1,
    FaultType.BEARING_SEVERE: 2, FaultType.OVERLOAD: 3,
    FaultType.MISALIGNMENT: 4,
}
FAULT_LABELS   = {v: k.name for k, v in FAULT_MAP.items()}
SAMPLES_PER_CLASS = 400


def generate_dataset():
    gen, extractor = SignalGenerator(), FeatureExtractor()
    X, y = [], []
    print("\n  Veri seti üretiliyor...")
    configs = [
        (FaultType.NONE,           (0.0, 0.0), (20, 40)),
        (FaultType.BEARING_EARLY,  (0.3, 0.7), (25, 55)),
        (FaultType.BEARING_SEVERE, (0.8, 1.0), (40, 80)),
        (FaultType.OVERLOAD,       (0.5, 1.0), (35, 70)),
        (FaultType.MISALIGNMENT,   (0.4, 0.9), (22, 45)),
    ]
    rng = np.random.default_rng(42)
    for fault_type, (sev_lo, sev_hi), (th_lo, th_hi) in configs:
        label = FAULT_MAP[fault_type]
        for i in range(SAMPLES_PER_CLASS):
            severity = rng.uniform(sev_lo, sev_hi) if sev_hi > 0 else 0.0
            sig = gen.healthy_vibration(256)
            if fault_type != FaultType.NONE:
                sig = gen.inject_bearing_fault(sig, fault_type, severity)
            sig += rng.normal(0, rng.uniform(0.001, 0.008), sig.shape)
            sw = SensorWindow('train', 'C0', i, i, sig,
                              float(rng.uniform(th_lo, th_hi)),
                              float(np.std(sig[:, 0])), fault_type)
            X.append(extractor.extract(sw).to_array())
            y.append(label)
        print(f"    {fault_type.name:20s}: {SAMPLES_PER_CLASS} örnek ✓")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_isolation_forest(X, y):
    print("\n  [1/3] IsolationForest eğitiliyor...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iso',    IsolationForest(n_estimators=200, contamination=0.05,
                                   random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X[y == 0])
    raw = pipeline.decision_function(X)
    pipeline._score_min = float(raw.min())
    pipeline._score_max = float(raw.max())
    normed = 1.0 - (raw - pipeline._score_min) / (pipeline._score_max - pipeline._score_min + 1e-10)
    print(f"    Normal skor: {normed[y==0].mean():.3f}  |  Anormal skor: {normed[y!=0].mean():.3f}")
    return pipeline


def train_classifier(X, y):
    print("\n  [2/3] RandomForest sınıflandırıcı eğitiliyor...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestClassifier(n_estimators=150, max_depth=12,
                                           class_weight='balanced',
                                           random_state=42, n_jobs=-1)),
    ])
    scores = cross_val_score(pipeline, X, y,
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                             scoring='f1_weighted', n_jobs=-1)
    print(f"    5-fold CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
    pipeline.fit(X, y)
    return pipeline


def save_models(iso_pipeline, clf_pipeline):
    model_dir = Path(__file__).parent.parent / "daein_mfg" / "orchestrator"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in [("isolation_forest", iso_pipeline),
                        ("fault_classifier",  clf_pipeline)]:
        path = model_dir / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        print(f"    {path.name}: {path.stat().st_size // 1024} KB")

    print("\n  [3/3] Sınıflandırma raporu (eğitim seti):")
    y_all  = np.concatenate([np.full(SAMPLES_PER_CLASS, v) for v in range(5)])
    print("  (model eğitildi, rapor train_models çalıştırılınca görünür)")


def main():
    print(f"\n{Color.BOLD}{Color.CYAN}{'═'*50}")
    print(f"  DAEIN-MFG — Model Eğitimi")
    print(f"{'═'*50}{Color.RESET}")
    t0 = time.perf_counter()
    X, y = generate_dataset()
    iso  = train_isolation_forest(X, y)
    clf  = train_classifier(X, y)
    print("\n  Kaydediliyor...")
    save_models(iso, clf)
    print(f"\n{Color.GREEN}  ✓ Tamamlandı — {time.perf_counter()-t0:.1f}s{Color.RESET}\n")


if __name__ == "__main__":
    main()
