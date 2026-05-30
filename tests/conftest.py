"""
DAEIN-MFG — Paylaşılan Test Fixture'ları
Tüm test dosyaları bu fixture'ları kullanır.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import asyncio
import numpy as np
from daein_mfg.simulation.sensor_generator import (
    SignalGenerator, SensorBuffer, SensorWindow, FaultType
)
from daein_mfg.simulation.micro_agent import FeatureExtractor, AnomalyScorer
import time


@pytest.fixture
def signal_generator():
    return SignalGenerator()


@pytest.fixture
def sensor_buffer():
    return SensorBuffer()


@pytest.fixture
def healthy_window(signal_generator):
    """Normal (arızasız) SensorWindow."""
    sig = signal_generator.healthy_vibration(256)
    return SensorWindow(
        node_id='test-node', cluster_id='C1',
        timestamp=time.time_ns(), window_id=0,
        vibration=sig, thermal=25.0, acoustic_rms=0.03,
        injected_fault=FaultType.NONE
    )


@pytest.fixture
def bearing_early_window(signal_generator):
    """Erken evre rulman arızası penceresi."""
    sig = signal_generator.healthy_vibration(256)
    sig = signal_generator.inject_bearing_fault(sig, FaultType.BEARING_EARLY, severity=0.9)
    return SensorWindow(
        node_id='test-node', cluster_id='C1',
        timestamp=time.time_ns(), window_id=1,
        vibration=sig, thermal=45.0, acoustic_rms=0.15,
        injected_fault=FaultType.BEARING_EARLY
    )


@pytest.fixture
def bearing_severe_window(signal_generator):
    """İleri evre rulman arızası penceresi."""
    sig = signal_generator.healthy_vibration(256)
    sig = signal_generator.inject_bearing_fault(sig, FaultType.BEARING_SEVERE, severity=1.0)
    return SensorWindow(
        node_id='test-node', cluster_id='C1',
        timestamp=time.time_ns(), window_id=2,
        vibration=sig, thermal=65.0, acoustic_rms=0.22,
        injected_fault=FaultType.BEARING_SEVERE
    )


@pytest.fixture
def feature_extractor():
    return FeatureExtractor()


@pytest.fixture
def anomaly_scorer():
    return AnomalyScorer()
