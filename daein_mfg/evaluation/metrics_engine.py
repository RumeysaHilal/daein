"""
DAEIN-MFG — Akademik Metrik Motoru (Aşama 5)
=============================================
Edge sistemi ile cloud mimarisini karşılaştıran istatistiksel analiz.

Temel katkı: Edge Advantage Function (EAF)

  EAF(τ, β, ρ) = α₁·(1/τ_edge - 1/τ_cloud)
               + α₂·(β_saved / β_total)
               + α₃·ρ_compliance

  τ = ortalama Detection-to-Actuation Latency (ms)
  β = iletilen toplam veri (KB)
  ρ = veri yerel kalma oranı (0→1)

Ek metrikler:
  - Cohen's d: etki büyüklüğü (latency farkının pratik önemi)
  - Bandwidth Efficiency Ratio (BER): edge / cloud bant genişliği
  - Fault Detection Rate (FDR) karşılaştırması
  - Raft konsensüs yakınsama istatistikleri
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import Color


# ─── VERİ YAPILARI ────────────────────────────────────────────────────────────

@dataclass
class EdgeMetrics:
    """Edge sisteminden toplanan ölçümler."""
    dal_samples:        list[float]   # Detection-to-Actuation Latency (ms)
    ack_latencies:      list[float]   # ACK round-trip süreleri
    inference_times:    list[float]   # µ-Agent inference süreleri (ms)
    bytes_transmitted:  int           # Toplam MQTT payload (byte)
    bytes_raw_sensor:   int           # Ham sensör veri boyutu (byte)
    tp_count:           int
    fp_count:           int
    fn_count:           int           # Kaçırılan arızalar (simüle)
    raft_commits:       int
    raft_commit_times:  list[float]   # Konsensüs süresi (ms)
    raft_timeouts:      int
    n_nodes:            int
    n_clusters:         int
    ppo_updates:        int
    ppo_total_reward:   float
    action_dist:        dict


@dataclass
class CloudMetrics:
    """Cloud simülatöründen toplanan ölçümler."""
    dal_samples:       list[float]
    bytes_transmitted: int
    tp_count:          int
    fp_count:          int
    timeout_count:     int


# ─── METRİK HESAPLAMA ─────────────────────────────────────────────────────────

class MetricsEngine:

    # EAF ağırlıkları (domain expertinden)
    ALPHA_LATENCY    = 0.40
    ALPHA_BANDWIDTH  = 0.35
    ALPHA_COMPLIANCE = 0.25

    def __init__(self, edge: EdgeMetrics, cloud: CloudMetrics):
        self.edge  = edge
        self.cloud = cloud

    # ── Temel İstatistikler ───────────────────────────────────────────────────

    def _stats(self, samples: list) -> dict:
        if not samples:
            return {"mean": 0, "std": 0, "p50": 0, "p95": 0, "p99": 0}
        a = np.array(samples)
        return {
            "mean": round(float(a.mean()), 2),
            "std":  round(float(a.std()),  2),
            "p50":  round(float(np.percentile(a, 50)), 2),
            "p95":  round(float(np.percentile(a, 95)), 2),
            "p99":  round(float(np.percentile(a, 99)), 2),
        }

    # ── Cohen's d (Etki Büyüklüğü) ───────────────────────────────────────────

    def cohens_d(self, a: list, b: list) -> float:
        """
        Cohen's d = (μ_A - μ_B) / pooled_std
        |d| < 0.2 küçük, 0.2–0.8 orta, > 0.8 büyük etki
        """
        if not a or not b:
            return 0.0
        na, nb  = len(a), len(b)
        ma, mb  = np.mean(a), np.mean(b)
        va, vb  = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled  = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        return float((ma - mb) / (pooled + 1e-10))

    # ── Edge Advantage Function ───────────────────────────────────────────────

    def eaf(self) -> dict:
        """
        EAF(τ, β, ρ) hesapla.
        Pozitif EAF → edge sistemin cloud'dan iyi olduğunu gösterir.
        """
        e_lat = np.mean(self.edge.dal_samples)   if self.edge.dal_samples   else 1.0
        c_lat = np.mean(self.cloud.dal_samples)  if self.cloud.dal_samples  else 1.0

        # Latency komponenti: 1/τ_edge - 1/τ_cloud (küçük τ = iyi)
        tau_component = (1.0 / max(e_lat, 1) - 1.0 / max(c_lat, 1)) * 1000

        # Bandwidth komponenti: ne kadar veri tasarrufu sağlandı
        total_bw = max(self.cloud.bytes_transmitted, 1)
        saved    = max(0, self.cloud.bytes_transmitted - self.edge.bytes_transmitted)
        bw_component = saved / total_bw

        # Compliance: veri edge'de kaldığı oran
        # Edge sistemde tüm işlem yerel → ρ=1.0
        # Cloud sistemde hiç yerel kalmaz → ρ=0.0
        compliance_component = 1.0   # Edge her zaman tam compliance

        eaf_score = (
            self.ALPHA_LATENCY    * tau_component +
            self.ALPHA_BANDWIDTH  * bw_component +
            self.ALPHA_COMPLIANCE * compliance_component
        )

        return {
            "eaf_score":          round(float(eaf_score), 4),
            "tau_component":      round(float(tau_component), 4),
            "bw_component":       round(float(bw_component), 4),
            "compliance_score":   1.0,
            "interpretation":     "Edge üstün ✓" if eaf_score > 0 else "Cloud üstün",
        }

    # ── Sınıflandırma Metrikleri ──────────────────────────────────────────────

    def classification_metrics(self, tp, fp, fn) -> dict:
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-10)
        return {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
        }

    # ── Bant Genişliği Verimliliği ────────────────────────────────────────────

    def bandwidth_efficiency_ratio(self) -> float:
        """BER = edge_bytes / cloud_bytes. Küçük BER = edge daha verimli."""
        return round(
            self.edge.bytes_transmitted / max(self.cloud.bytes_transmitted, 1), 4
        )

    # ── Tam Rapor ─────────────────────────────────────────────────────────────

    def full_report(self) -> dict:
        e_lat_stats = self._stats(self.edge.dal_samples)
        c_lat_stats = self._stats(self.cloud.dal_samples)
        e_inf_stats = self._stats(self.edge.inference_times)
        r_com_stats = self._stats(self.edge.raft_commit_times)
        eaf_result  = self.eaf()
        d_latency   = self.cohens_d(self.cloud.dal_samples, self.edge.dal_samples)
        ber         = self.bandwidth_efficiency_ratio()

        edge_cls  = self.classification_metrics(
            self.edge.tp_count, self.edge.fp_count, self.edge.fn_count
        )
        cloud_cls = self.classification_metrics(
            self.cloud.tp_count, self.cloud.fp_count,
            max(0, self.edge.tp_count - self.cloud.tp_count)
        )

        return {
            "latency": {
                "edge_dal":  e_lat_stats,
                "cloud_dal": c_lat_stats,
                "reduction_pct": round(
                    (c_lat_stats["mean"] - e_lat_stats["mean"]) /
                    max(c_lat_stats["mean"], 1) * 100, 1
                ),
                "cohens_d": round(d_latency, 3),
                "cohens_d_interp": (
                    "Büyük etki (|d|>0.8)" if abs(d_latency) > 0.8
                    else "Orta etki" if abs(d_latency) > 0.2
                    else "Küçük etki"
                ),
            },
            "bandwidth": {
                "edge_kb":   round(self.edge.bytes_transmitted / 1024, 1),
                "cloud_kb":  round(self.cloud.bytes_transmitted / 1024, 1),
                "ber":       ber,
                "reduction_factor": round(1 / max(ber, 0.001), 1),
            },
            "classification": {
                "edge":  edge_cls,
                "cloud": cloud_cls,
            },
            "inference": {
                "edge_inf_ms": e_inf_stats,
            },
            "raft": {
                "commits":      self.edge.raft_commits,
                "timeouts":     self.edge.raft_timeouts,
                "commit_ms":    r_com_stats,
                "success_rate": round(
                    self.edge.raft_commits /
                    max(self.edge.raft_commits + self.edge.raft_timeouts, 1), 3
                ),
            },
            "ppo": {
                "updates":      self.edge.ppo_updates,
                "total_reward": self.edge.ppo_total_reward,
                "action_dist":  self.edge.action_dist,
            },
            "eaf": eaf_result,
            "system": {
                "n_nodes":    self.edge.n_nodes,
                "n_clusters": self.edge.n_clusters,
            },
        }


# ─── TERMİNAL RAPOR YAZICI ───────────────────────────────────────────────────

def print_academic_report(report: dict):
    """Formatlanmış akademik raporu terminale yazdır."""
    B = Color.BOLD
    R = Color.RESET
    C = Color.CYAN
    G = Color.GREEN
    Y = Color.YELLOW

    def bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
        n = int(min(value / max(max_val, 1e-9), 1.0) * width)
        return f"[{'█'*n}{'░'*(width-n)}]"

    print(f"\n{B}{C}{'═'*65}")
    print(f"  DAEIN-MFG — AKADEMİK PERFORMANS RAPORU (Aşama 5)")
    print(f"  Sistem: {report['system']['n_clusters']} cluster × "
          f"{report['system']['n_nodes']//report['system']['n_clusters']} node")
    print(f"{'═'*65}{R}")

    # ── Gecikme ────────────────────────────────────────────────────────
    lat = report["latency"]
    e   = lat["edge_dal"]
    cl  = lat["cloud_dal"]
    print(f"\n{B}  1. DETECTION-TO-ACTUATION LATENCY (DAL){R}")
    print(f"  {'':20s}  {'Ortalama':>10}  {'P95':>8}  {'P99':>8}")
    print(f"  {'Edge (DAEIN-MFG)':20s}  {e['mean']:>8.0f}ms  {e['p95']:>6.0f}ms  {e['p99']:>6.0f}ms")
    print(f"  {'Cloud Baseline':20s}  {cl['mean']:>8.0f}ms  {cl['p95']:>6.0f}ms  {cl['p99']:>6.0f}ms")
    print(f"  Azalma  : {G}{lat['reduction_pct']:+.1f}%{R}  |  "
          f"Cohen's d = {lat['cohens_d']:.3f} ({lat['cohens_d_interp']})")
    max_lat = max(e["mean"], cl["mean"])
    print(f"  Edge    : {G}{bar(cl['mean']-e['mean'], max_lat)}{R} "
          f"{lat['reduction_pct']:.0f}% daha hızlı")

    # ── Bant Genişliği ─────────────────────────────────────────────────
    bw = report["bandwidth"]
    print(f"\n{B}  2. BANT GENİŞLİĞİ VERİMLİLİĞİ{R}")
    print(f"  Edge iletimi   : {bw['edge_kb']:>8.1f} KB")
    print(f"  Cloud iletimi  : {bw['cloud_kb']:>8.1f} KB")
    print(f"  BER            : {bw['ber']:.4f}  →  Edge {G}{bw['reduction_factor']:.0f}×{R} daha az bant genişliği")

    # ── Sınıflandırma ─────────────────────────────────────────────────
    cls = report["classification"]
    ec  = cls["edge"]
    cc  = cls["cloud"]
    print(f"\n{B}  3. SINIFLANDIRMA METRİKLERİ{R}")
    print(f"  {'':20s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'Edge (DAEIN-MFG)':20s}  {ec['precision']:>10.3f}  {ec['recall']:>8.3f}  {ec['f1']:>8.3f}")
    print(f"  {'Cloud Baseline':20s}  {cc['precision']:>10.3f}  {cc['recall']:>8.3f}  {cc['f1']:>8.3f}")

    # ── Inference ─────────────────────────────────────────────────────
    inf = report["inference"]["edge_inf_ms"]
    print(f"\n{B}  4. EDGE INFERENCE SÜRESİ (µ-Agent){R}")
    print(f"  Ort={inf['mean']:.1f}ms  P95={inf['p95']:.1f}ms  P99={inf['p99']:.1f}ms  "
          f"(hedef <8ms RPi4 sim, ~25ms Python IF)")

    # ── Raft ──────────────────────────────────────────────────────────
    raft = report["raft"]
    print(f"\n{B}  5. RAFT KONSENSÜS METRİKLERİ{R}")
    if raft["commits"] > 0:
        rc = raft["commit_ms"]
        print(f"  Commit sayısı  : {raft['commits']}  |  "
              f"Başarı oranı: {G}{raft['success_rate']:.0%}{R}  |  "
              f"Timeout: {raft['timeouts']}")
        print(f"  Commit süresi  : ort={rc['mean']:.1f}ms  "
              f"p95={rc['p95']:.1f}ms  p99={rc['p99']:.1f}ms")
    else:
        print(f"  Konsensüs talebi: {raft['commits']} "
              f"(REQUEST_CONSENSUS aksiyonu seçilmedi)")

    # ── PPO ───────────────────────────────────────────────────────────
    ppo = report["ppo"]
    print(f"\n{B}  6. PPO POLİTİKA PERFORMANSI{R}")
    print(f"  Güncelleme sayısı : {ppo['updates']}  |  "
          f"Toplam ödül: {G}{ppo['total_reward']:+.2f}{R}")
    print(f"  Aksiyon dağılımı  : ", end="")
    for act, pct in ppo["action_dist"].items():
        print(f"{act}={pct} ", end="")
    print()

    # ── EAF ───────────────────────────────────────────────────────────
    eaf = report["eaf"]
    score_color = G if eaf["eaf_score"] > 0 else Y
    print(f"\n{B}  7. EDGE ADVANTAGE FUNCTION (EAF){R}")
    print(f"  EAF = α₁·τ_component + α₂·β_component + α₃·ρ_compliance")
    print(f"      = 0.40 × {eaf['tau_component']:+.4f}")
    print(f"      + 0.35 × {eaf['bw_component']:+.4f}")
    print(f"      + 0.25 × {eaf['compliance_score']:+.4f}")
    print(f"      = {score_color}{B}{eaf['eaf_score']:+.4f}{R}  →  {eaf['interpretation']}")

    print(f"\n{B}{C}{'═'*65}{R}\n")
