"""
DAEIN-MFG — PPO Kaynak Tahsis Politikası (numpy tabanlı)
=========================================================
Torch olmadan sıfırdan yazılmış hafif PPO implementasyonu.

Karar problemi:
  Orkestratör her anomali alertinde şunu sormalı:
  "Bu anomaliyi nasıl işlemeliyim? Yerel mi, komşuya mı, ertele mi?"

State space (7 boyut):
  [cpu_util, mem_util, queue_depth, n_active_alerts,
   peer_avg_load, anomaly_score, anomaly_confidence]

Action space (5 ayrık aksiyon):
  0: LOCAL_INFER     → bu orkestratörde işle
  1: OFFLOAD_PEER    → en boş komşuya gönder
  2: DEFER_500MS     → 500ms bekle, yeniden değerlendir
  3: ESCALATE        → hemen aktüatör tetikle
  4: REQUEST_CONSENSUS → diğer cluster'lardan oy iste (Aşama 5)

PPO Mimarisi:
  Policy network : Linear(7→32) → ReLU → Linear(32→16) → ReLU → Linear(16→5)
  Value network  : Linear(7→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from enum import IntEnum
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Color

# ─── AKSİYON UZAYI ───────────────────────────────────────────────────────────

class Action(IntEnum):
    LOCAL_INFER         = 0
    OFFLOAD_PEER        = 1
    DEFER_500MS         = 2
    ESCALATE            = 3
    REQUEST_CONSENSUS   = 4

ACTION_LABELS = {
    Action.LOCAL_INFER:       "LOCAL",
    Action.OFFLOAD_PEER:      "OFFLOAD",
    Action.DEFER_500MS:       "DEFER",
    Action.ESCALATE:          "ESCALATE",
    Action.REQUEST_CONSENSUS: "CONSENSUS",
}

N_STATES  = 7
N_ACTIONS = 5


# ─── AĞIRLIK BAŞLATMA ─────────────────────────────────────────────────────────

def _he_init(fan_in, fan_out, rng):
    return rng.normal(0, np.sqrt(2.0/fan_in), (fan_in, fan_out)).astype(np.float32)

def _zeros(n):
    return np.zeros(n, dtype=np.float32)

def _relu(x):
    return np.maximum(0, x)

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ─── GEÇİŞ TAMPONU ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool
    log_prob:   float


# ─── PPO POLİTİKA ─────────────────────────────────────────────────────────────

class PPOPolicy:
    """
    Kaynak tahsis kararları için PPO politikası.
    Her 32 geçişten sonra online güncelleme yapar.
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)

        self.weights = {
            # Policy net
            "W1": _he_init(N_STATES,  32, rng), "b1": _zeros(32),
            "W2": _he_init(32,        16, rng), "b2": _zeros(16),
            "W3": _he_init(16, N_ACTIONS, rng), "b3": _zeros(N_ACTIONS),
            # Value net
            "vW1": _he_init(N_STATES, 32, rng), "vb1": _zeros(32),
            "vW2": _he_init(32,       16, rng), "vb2": _zeros(16),
            "vW3": _he_init(16,        1, rng), "vb3": _zeros(1),
        }

        # Kural tabanlı önyükleme
        self.weights["b3"][Action.ESCALATE]   = 1.5
        self.weights["b3"][Action.LOCAL_INFER] = 0.5

        self.gamma      = 0.99
        self.lam        = 0.95
        self.eps_clip   = 0.2
        self.lr         = 3e-4
        self.update_freq = 32

        self._buffer: list[Transition] = []
        self.stats = {
            "updates":       0,
            "total_reward":  0.0,
            "action_counts": [0] * N_ACTIONS,
            "avg_entropy":   0.0,
        }

        print(f"  [PPO] Politika: 7→32→16→{N_ACTIONS} | "
              f"γ={self.gamma} λ={self.lam} ε={self.eps_clip}")

    def _policy_probs(self, state):
        h1 = _relu(state @ self.weights["W1"] + self.weights["b1"])
        h2 = _relu(h1    @ self.weights["W2"] + self.weights["b2"])
        return _softmax(h2 @ self.weights["W3"] + self.weights["b3"])

    def _value(self, state):
        h1 = _relu(state @ self.weights["vW1"] + self.weights["vb1"])
        h2 = _relu(h1    @ self.weights["vW2"] + self.weights["vb2"])
        return float((h2 @ self.weights["vW3"] + self.weights["vb3"])[0])

    def act(self, state: np.ndarray, deterministic: bool = False):
        """Durum → aksiyon + log olasılık."""
        probs  = self._policy_probs(state)
        action = int(np.argmax(probs)) if deterministic else int(np.random.choice(N_ACTIONS, p=probs))
        lp     = float(np.log(probs[action] + 1e-8))

        self.stats["action_counts"][action] += 1
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        self.stats["avg_entropy"] = 0.99 * self.stats["avg_entropy"] + 0.01 * entropy

        return action, lp

    def store(self, t: Transition):
        """Geçiş ekle; buffer dolunca güncelle."""
        self._buffer.append(t)
        self.stats["total_reward"] += t.reward
        if len(self._buffer) >= self.update_freq:
            self._ppo_update()
            self._buffer.clear()

    def _ppo_update(self):
        """
        PPO-Clip tek adım güncelleme.
        Numerik gradyan — torch yokken analitik türev yerine.
        Sadece son katman ağırlıkları (W3, b3) güncellenir (hız için).
        """
        buf       = self._buffer
        states    = np.array([t.state    for t in buf], dtype=np.float32)
        actions   = np.array([t.action   for t in buf], dtype=np.int32)
        rewards   = np.array([t.reward   for t in buf], dtype=np.float32)
        old_lps   = np.array([t.log_prob for t in buf], dtype=np.float32)
        dones     = np.array([t.done     for t in buf], dtype=np.float32)

        # GAE
        values = np.array([self._value(s) for s in states], dtype=np.float32)
        adv    = np.zeros(len(buf), dtype=np.float32)
        gae    = 0.0
        for i in reversed(range(len(buf))):
            nv   = values[i+1] if i+1 < len(buf) else 0.0
            d    = rewards[i] + self.gamma * nv * (1-dones[i]) - values[i]
            gae  = d + self.gamma * self.lam * (1-dones[i]) * gae
            adv[i] = gae

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy gradient — numerik
        eps_g = 1e-3
        for key in ("W3", "b3"):
            grad = np.zeros_like(self.weights[key])
            for idx in np.ndindex(self.weights[key].shape):
                orig = self.weights[key][idx]
                losses = []
                for sign in (+eps_g, -eps_g):
                    self.weights[key][idx] = orig + sign
                    total = 0.0
                    for i, (s, a) in enumerate(zip(states, actions)):
                        p   = self._policy_probs(s)
                        nlp = np.log(p[a] + 1e-8)
                        r   = np.exp(nlp - old_lps[i])
                        rc  = np.clip(r, 1-self.eps_clip, 1+self.eps_clip)
                        total -= min(r * adv[i], rc * adv[i])
                    losses.append(total)
                self.weights[key][idx] = orig
                grad[idx] = (losses[0] - losses[1]) / (2 * eps_g)
            self.weights[key] -= self.lr * np.clip(grad, -0.5, 0.5)

        self.stats["updates"] += 1

    @staticmethod
    def compute_reward(
        action: int,
        anomaly_score: float,
        detection_latency_ms: float,
        false_positive: bool,
        consensus_rounds: int,
        cpu_util: float,
    ) -> float:
        """
        Ödül fonksiyonu:
          R = -α·latency - β·FP_penalty + γ·TP_bonus - δ·cpu - ε·consensus
        """
        latency_n = min(1.0, detection_latency_ms / 500.0)
        tp_bonus  = (1.0 - float(false_positive)) * anomaly_score
        fp_pen    = float(false_positive) * 0.8
        cpu_pen   = max(0.0, cpu_util - 0.7)
        cons_pen  = min(1.0, consensus_rounds / 5.0)

        r = -0.3*latency_n + 0.4*tp_bonus - 0.5*fp_pen - 0.2*cpu_pen - 0.1*cons_pen

        if anomaly_score > 0.9 and not false_positive:
            r += 0.5
        if action == Action.ESCALATE and anomaly_score < 0.6:
            r -= 0.3

        return float(np.clip(r, -2.0, 2.0))

    def report(self) -> dict:
        total = max(sum(self.stats["action_counts"]), 1)
        return {
            "updates":      self.stats["updates"],
            "total_reward": round(self.stats["total_reward"], 2),
            "avg_entropy":  round(self.stats["avg_entropy"], 4),
            "action_dist":  {
                ACTION_LABELS[Action(i)]: f"{c/total:.1%}"
                for i, c in enumerate(self.stats["action_counts"])
            },
        }
