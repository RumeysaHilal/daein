"""
tests/test_metrics.py — Metrik Motoru ve EAF Testleri

Test kapsamı:
  - EAF hesabı (pozitif / negatif senaryolar)
  - Cohen's d hesabı ve yorumlama
  - Bandwidth Efficiency Ratio
  - Sınıflandırma metrikleri (precision, recall, F1)
  - CloudSimulator gecikme modeli
  - Rapor yapısı bütünlüğü
"""

import pytest
import asyncio
import numpy as np
from daein_mfg.evaluation.metrics_engine import (
    EdgeMetrics, CloudMetrics, MetricsEngine
)
from daein_mfg.evaluation.cloud_simulator import (
    CloudSimulator, CloudLatencyModel
)


# ─── TEST VERİSİ ──────────────────────────────────────────────────────────────

def _make_edge_metrics(**overrides):
    defaults = dict(
        dal_samples       = [400.0, 420.0, 450.0, 390.0, 470.0],
        ack_latencies     = [1.5, 2.1, 1.8, 2.4, 1.6],
        inference_times   = [25.0, 26.5, 24.8, 27.1, 25.5],
        bytes_transmitted = 50_000,
        bytes_raw_sensor  = 650_000,
        tp_count          = 80,
        fp_count          = 15,
        fn_count          = 10,
        raft_commits      = 12,
        raft_commit_times = [180.0, 210.0, 195.0, 220.0],
        raft_timeouts     = 2,
        n_nodes           = 12,
        n_clusters        = 3,
        ppo_updates       = 5,
        ppo_total_reward  = 62.97,
        action_dist       = {"LOCAL": "60%", "ESCALATE": "20%", "DEFER": "20%"},
    )
    defaults.update(overrides)
    return EdgeMetrics(**defaults)


def _make_cloud_metrics(**overrides):
    defaults = dict(
        dal_samples       = [850.0, 920.0, 780.0, 1100.0, 960.0],
        bytes_transmitted = 650_000,
        tp_count          = 75,
        fp_count          = 20,
        timeout_count     = 5,
    )
    defaults.update(overrides)
    return CloudMetrics(**defaults)


# ─── EAF TESTLERİ ─────────────────────────────────────────────────────────────

class TestEAF:

    def test_eaf_positive_when_edge_faster(self):
        """Edge cloud'dan hızlı ve daha az bant genişliği → EAF > 0."""
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        eaf    = engine.eaf()
        assert eaf["eaf_score"] > 0, f"EAF negatif: {eaf['eaf_score']}"

    def test_eaf_components_present(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        eaf    = engine.eaf()
        for key in ("eaf_score", "tau_component", "bw_component",
                    "compliance_score", "interpretation"):
            assert key in eaf

    def test_eaf_compliance_always_one_for_edge(self):
        """Edge sistemde veri yerel kalır → compliance = 1.0."""
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        eaf    = engine.eaf()
        assert eaf["compliance_score"] == 1.0

    def test_eaf_negative_when_edge_slower(self):
        """Edge daha yavaşsa (yüksek DAL) EAF negatif olabilir."""
        slow_edge  = _make_edge_metrics(dal_samples=[2000.0, 2100.0, 1900.0])
        fast_cloud = _make_cloud_metrics(dal_samples=[300.0, 320.0, 290.0])
        engine     = MetricsEngine(slow_edge, fast_cloud)
        eaf        = engine.eaf()
        # Bant genişliği ve compliance katkısı nedeniyle hala pozitif olabilir
        # ama tau komponenti negatif olmalı
        assert eaf["tau_component"] < 0

    def test_bw_component_zero_when_equal_bytes(self):
        """Edge ve cloud aynı byte gönderirse bant genişliği katkısı sıfır."""
        edge  = _make_edge_metrics(bytes_transmitted=100_000)
        cloud = _make_cloud_metrics(bytes_transmitted=100_000)
        engine = MetricsEngine(edge, cloud)
        eaf    = engine.eaf()
        assert abs(eaf["bw_component"]) < 0.01


# ─── COHEN'S D ───────────────────────────────────────────────────────────────

class TestCohensD:

    def test_large_effect(self):
        """Büyük fark → |d| > 0.8."""
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        # Edge ~430ms, Cloud ~920ms — büyük fark
        d = engine.cohens_d(
            [920.0, 850.0, 1100.0, 780.0, 960.0],
            [400.0, 420.0, 450.0, 390.0, 470.0],
        )
        assert abs(d) > 0.8, f"Cohen's d = {d:.3f}, büyük etki bekleniyor"

    def test_zero_effect_same_samples(self):
        """Aynı örnekler → d = 0."""
        samples = [400.0, 420.0, 450.0]
        engine  = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        d       = engine.cohens_d(samples, samples)
        assert abs(d) < 1e-6

    def test_empty_samples_returns_zero(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        d      = engine.cohens_d([], [400.0, 420.0])
        assert d == 0.0


# ─── BANT GENİŞLİĞİ ──────────────────────────────────────────────────────────

class TestBandwidthEfficiency:

    def test_ber_less_than_one_when_edge_smaller(self):
        """Edge daha az byte gönderirse BER < 1."""
        edge   = _make_edge_metrics(bytes_transmitted=50_000)
        cloud  = _make_cloud_metrics(bytes_transmitted=650_000)
        engine = MetricsEngine(edge, cloud)
        assert engine.bandwidth_efficiency_ratio() < 1.0

    def test_ber_equals_one_when_equal(self):
        edge   = _make_edge_metrics(bytes_transmitted=100_000)
        cloud  = _make_cloud_metrics(bytes_transmitted=100_000)
        engine = MetricsEngine(edge, cloud)
        assert abs(engine.bandwidth_efficiency_ratio() - 1.0) < 0.01

    def test_reduction_factor_is_reciprocal(self):
        """Reduction factor = 1/BER."""
        edge   = _make_edge_metrics(bytes_transmitted=50_000)
        cloud  = _make_cloud_metrics(bytes_transmitted=650_000)
        engine = MetricsEngine(edge, cloud)
        ber    = engine.bandwidth_efficiency_ratio()
        report = engine.full_report()
        factor = report["bandwidth"]["reduction_factor"]
        assert abs(factor - round(1 / ber, 1)) <= 1.0


# ─── SINIFLANDIRMA METRİKLERİ ────────────────────────────────────────────────

class TestClassificationMetrics:

    def test_perfect_precision_recall(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        m = engine.classification_metrics(tp=100, fp=0, fn=0)
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_zero_tp_gives_zero_f1(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        m = engine.classification_metrics(tp=0, fp=10, fn=10)
        assert m["f1"] == 0.0

    @pytest.mark.parametrize("tp, fp, fn, expected_f1_min", [
        (80, 15, 10, 0.82),
        (50, 50, 50, 0.49),
        (90,  5,  5, 0.94),
    ])
    def test_f1_bounds(self, tp, fp, fn, expected_f1_min):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        m      = engine.classification_metrics(tp, fp, fn)
        assert m["f1"] >= expected_f1_min - 0.05


# ─── CLOUD SİMÜLATÖR ─────────────────────────────────────────────────────────

class TestCloudSimulator:

    def test_latency_model_positive(self):
        model = CloudLatencyModel()
        for _ in range(20):
            lat = model.sample()
            assert lat > 0, f"Gecikme negatif: {lat}"

    def test_latency_model_reasonable_range(self):
        """Gecikme 50ms–3000ms arasında olmalı (3σ sınırı)."""
        model  = CloudLatencyModel()
        lats   = [model.sample() for _ in range(100)]
        assert min(lats) > 10.0
        assert max(lats) < 5000.0

    @pytest.mark.asyncio
    async def test_cloud_simulator_produces_decisions(self):
        sim = CloudSimulator(mode="RAW_STREAM")
        d   = await sim.process("A-001", 0.87, "BEARING_EARLY", 0.79)

        assert d.anomaly_id    == "A-001"
        assert d.latency_ms    >  0
        assert d.payload_bytes >  0
        assert isinstance(d.fault_detected, bool)

    @pytest.mark.asyncio
    async def test_payload_raw_larger_than_feature(self):
        """RAW_STREAM payload feature modundan büyük olmalı."""
        raw  = CloudSimulator(mode="RAW_STREAM")
        feat = CloudSimulator(mode="FEATURE")
        assert raw.payload_bytes > feat.payload_bytes

    @pytest.mark.asyncio
    async def test_report_after_multiple_requests(self):
        sim = CloudSimulator()
        for i in range(5):
            await sim.process(f"A-{i:03d}", 0.7 + i * 0.05, "BEARING_EARLY", 0.8)

        r = sim.report()
        assert r["n"]              == 5
        assert r["bytes_up_kb"]    >  0
        assert r["avg_latency_ms"] >  0

    @pytest.mark.asyncio
    async def test_timeout_flagged_correctly(self):
        """2000ms üstü gecikme timeout olarak işaretlenmeli."""
        from unittest.mock import patch
        sim = CloudSimulator()
        # Aşırı yüksek gecikme simüle et
        with patch.object(sim._model, 'sample', return_value=2500.0):
            d = await sim.process("A-timeout", 0.9, "BEARING_SEVERE", 0.95)
        assert d.timed_out is True
        assert d.fault_class == "TIMEOUT"


# ─── TAM RAPOR YAPISI ────────────────────────────────────────────────────────

class TestFullReport:

    def test_all_sections_present(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        report = engine.full_report()
        for section in ("latency", "bandwidth", "classification",
                        "inference", "raft", "ppo", "eaf", "system"):
            assert section in report, f"Raporda '{section}' bölümü eksik"

    def test_latency_reduction_positive(self):
        """Cloud daha yavaşsa azalma yüzdesi pozitif olmalı."""
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        report = engine.full_report()
        assert report["latency"]["reduction_pct"] > 0

    def test_system_fields(self):
        engine = MetricsEngine(_make_edge_metrics(), _make_cloud_metrics())
        report = engine.full_report()
        assert report["system"]["n_nodes"]    == 12
        assert report["system"]["n_clusters"] == 3
