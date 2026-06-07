# DAEIN-MFG

**Decentralized Agentic Edge Intelligence Network for Proactive Fault Prediction and Autonomous Self-Healing in Smart Manufacturing**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Summary

DAEIN-MFG is a multi-agent edge intelligence simulation that performs real-time machine fault detection and autonomous response in industrial IoT environments **without cloud dependency**.

It sits at the intersection of three paradigms:

| Paradigm | Role in This Project |
|---|---|
| Agentic Edge Intelligence | FSM-based µ-Agent + PPO Orchestrator |
| Agentic IoT | Autonomous sensor-actuator coordination over MQTT |
| Distributed Data Engineering | Async stream pipeline + Raft consensus |

### Key Academic Contributions

1. **Edge Advantage Function (EAF):** A measurement framework that combines latency, bandwidth, and data compliance into a single score.
2. **Hybrid Raft + Auction Consensus:** Raft extended with a sealed-bid auction mechanism for fault severity voting.
3. **Two-Stage ML Scorer:** IsolationForest (continuous anomaly score) + RandomForest (classification on trigger) architecture.
4. **Resource-Aware PPO:** An RL policy that incorporates hardware budget constraints into the state space.

---

## System Architecture

```
TIER 2: GATEWAY ORCHESTRATOR (Jetson Nano sim.)
  ┌──────────────────────────────────────────────┐
  │  MQTTClient ──► PPO Policy                   │
  │  RaftNode   ──► Auction Bid Mechanism        │
  │  ActorHandle ── Ray-compatible async proxy   │
  └──────────────────────────────────────────────┘
                     ▲ MQTT
TIER 1: µ-AGENT (Raspberry Pi 4 sim.)
  ┌──────────────────────────────────────────────┐
  │  FSM: IDLE→SAMPLING→INFERENCE→ALERT→WAIT    │
  │  IsolationForest + RandomForest (INT8 proxy) │
  │  FFT feature extraction (256-sample window)  │
  └──────────────────────────────────────────────┘
                     ▲ asyncio
SENSOR SIMULATOR
  Vibration (100Hz) + Temperature (10Hz) + Bearing fault injection
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/RumeysaHilal/daein-mfg.git
cd daein-mfg

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install in development mode (for `import daein_mfg` to work)
pip install -e .
```

---

## Quick Start

### Step 1 — Train Models

```bash
python scripts/train_models.py
```

Output:
```
Generating dataset... 2000 samples, 14 features
Training IsolationForest...
  Normal  → mean anomaly score: 0.312
  Abnormal → mean anomaly score: 0.718
Training RandomForest...
  5-fold CV F1 (weighted): 0.9821 ± 0.0043
Saved: isolation_forest.joblib (648 KB)
       fault_classifier.joblib  (40 KB)
```

### Step 2 — Run Simulation

```bash
python scripts/run_simulation.py
```

Output:
```
DAEIN-MFG — Phase 5: Raft + Edge vs Cloud Analysis
Clusters: 3 | Nodes: 12 | Duration: 120s (real ≈12s)
...
1. DETECTION-TO-ACTUATION LATENCY (DAL)
   Edge  (DAEIN-MFG) : avg=442ms  P95=450ms
   Cloud (Baseline)  : avg=920ms  P95=1899ms
   Reduction: +52%  |  Cohen's d = 1.559 (Large effect)

7. EDGE ADVANTAGE FUNCTION (EAF)
   EAF = +1.0439  →  Edge superior ✓
```

### Step 3 — Run Tests

```bash
pytest tests/ -v --cov=daein_mfg
```

---

## Project Structure

```
daein-mfg/
│
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── setup.py                         # Package setup
├── .gitignore
├── LICENSE
│
├── daein_mfg/                       # Main Python package
│   ├── config.py                    # Central parameters
│   │
│   ├── simulation/
│   │   ├── sensor_generator.py      # Physics-based sensor simulator
│   │   └── micro_agent.py           # µ-Agent FSM + ML inference + MQTT
│   │
│   ├── messaging/
│   │   ├── broker.py                # In-process async MQTT broker
│   │   └── gateway.py               # Cluster gateway (Phase 3)
│   │
│   ├── orchestrator/
│   │   ├── actor_runtime.py         # Ray Actor proxy (async)
│   │   ├── ml_scorer.py             # IsolationForest + RF scorer
│   │   ├── ppo_policy.py            # PPO resource allocation policy
│   │   ├── orchestrator_agent.py    # Main orchestrator agent
│   │   ├── raft.py                  # Raft + Auction-bid consensus
│   │   └── raft_cluster.py          # Multi-node Raft coordinator
│   │
│   └── evaluation/
│       ├── cloud_simulator.py       # Cloud latency/bandwidth model
│       └── metrics_engine.py        # EAF + academic metrics report
│
├── scripts/
│   ├── train_models.py              # Model training (standalone)
│   └── run_simulation.py            # Full simulation (Phase 5)
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_sensor.py               # Sensor simulator tests
│   ├── test_micro_agent.py          # µ-Agent FSM + inference tests
│   ├── test_broker.py               # MQTT broker tests
│   ├── test_raft.py                 # Raft consensus tests
│   └── test_metrics.py              # EAF and metric calculation tests
│
├── notebooks/
│   └── results_analysis.ipynb      # Results visualization
│
└── docs/
    └── architecture.md             # Detailed system architecture
```

---

## Configuration

All parameters are centrally managed in `daein_mfg/config.py`:

```python
# Key parameters
SAMPLE_RATE_HZ        = 100      # Vibration sensor sampling rate
WINDOW_SIZE           = 256      # FFT window size
ANOMALY_THRESHOLD     = 0.55     # µ-Agent alert threshold
N_CLUSTERS            = 3        # Number of clusters
NODES_PER_CLUSTER     = 4        # Nodes per cluster
SIM_DURATION_S        = 120      # Simulation duration
TIME_ACCELERATION     = 10       # Speed multiplier
FAULT_INJECT_INTERVAL_S = 30     # Fault injection period
```

---

## Fault Types

| Type | Physical Model | Signal Indicator |
|---|---|---|
| `BEARING_EARLY` | BPFO impulse (87.3 Hz) | Kurtosis > 0.5, BPFO power increase |
| `BEARING_SEVERE` | Impulse on all axes | High RMS, kurtosis > 3.0 |
| `OVERLOAD` | General RMS increase | Low kurtosis, high RMS |
| `MISALIGNMENT` | 2x, 3x harmonic dominance | Low spectral entropy |

---

## Results Summary

Phase 5 simulation results (3 clusters, 12 nodes, 120s):

| Metric | Edge (DAEIN-MFG) | Cloud Baseline | Improvement |
|---|---|---|---|
| Avg. DAL (ms) | 442 | 920 | **−52%** |
| P99 DAL (ms) | 450 | 1961 | **−77%** |
| Bandwidth (KB) | 48.8 | 633.9 | **−92% (13×)** |
| F1 Score | 0.780 | 0.856 | Cloud higher* |
| EAF | +1.044 | — | **Edge superior** |
| Cohen's d | 1.559 | — | Large effect |

*Cloud F1 is higher because available sensor data is used even on timeouts; in real-world conditions, retry latency would eliminate this advantage.

---

## Deployment on Real Hardware

### Raspberry Pi 4 (µ-Agent)
```bash
pip install tflite-runtime
# Replace MLAnomalyScorer in micro_agent.py with TFLite
# config.py → TIME_ACCELERATION = 1 (real time)
```

### NVIDIA Jetson Nano (Orchestrator)
```bash
pip install ray ollama
# actor_runtime.py → ActorRuntime.create() → @ray.remote
# orchestrator_agent.py → Phi-3-mini SLM integration
```

### Real MQTT Broker
```bash
pip install paho-mqtt
# messaging/broker.py → MQTTClient → paho.mqtt.client.Client wrapper
# Broker: Mosquitto or EMQX (local installation)
```

---

## Development Roadmap

- [x] Phase 1: Sensor simulator + µ-Agent FSM
- [x] Phase 2: IsolationForest + RandomForest ML
- [x] Phase 3: Async MQTT broker + ClusterGateway
- [x] Phase 4: ActorRuntime + PPO RL + Peer Offloading
- [x] Phase 5: Raft consensus + Edge vs Cloud analysis
- [ ] Phase 6: TFLite INT8 export + real RPi4 deployment
- [ ] Phase 7: Phi-3-mini SLM integration (Jetson Nano)
- [ ] Phase 8: Real data validation with CWRU Bearing Dataset

---

## License

MIT License — See [LICENSE](LICENSE) for details.
