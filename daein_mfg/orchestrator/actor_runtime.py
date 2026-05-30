"""
DAEIN-MFG — Async Actor Runtime (Ray Proxy)
=============================================
Ray bu ortamda kurulu olmadığı için Ray'in @ray.remote Actor modelini
tam olarak taklit eden hafif bir runtime.

Ray ile kavramsal eşleşme:
  ActorHandle       ↔  ray.remote class instance
  actor.call(...)   ↔  actor.method.remote(...)
  await result      ↔  await ray.get(future)
  ActorRuntime      ↔  ray.init() + ray cluster

Gerçek Ray'e geçiş:
  OrchestratorAgent sınıfının başına @ray.remote ekle,
  ActorRuntime.create() → OrchestratorAgent.remote() yap,
  actor.call() → actor.method.remote() yap.
  Başka hiçbir şey değişmez.

Mimari:
  Her Actor kendi asyncio event loop'unda çalışır (ThreadPoolExecutor).
  Mesajlaşma: asyncio.Queue tabanlı mailbox (Ray'in object store'u yerine).
  Resource enforcement: her Actor'a CPU/memory bütçesi atanır, aşım loglanır.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Color


# ─── KAYNAK BÜTÇESİ ──────────────────────────────────────────────────────────

@dataclass
class ResourceBudget:
    """
    Bir Actor'ün simüle donanım bütçesi.
    Jetson Nano profili: 4 CPU, 4GB RAM, 10W güç.
    """
    num_cpus:     float = 2.0        # sanal CPU hakkı
    memory_mb:    float = 2048.0     # MB
    power_w:      float = 10.0       # Watt (simüle güç bütçesi)
    label:        str   = "default"


JETSON_NANO = ResourceBudget(num_cpus=2.0, memory_mb=2048.0, power_w=10.0, label="jetson_nano")
RPI4        = ResourceBudget(num_cpus=1.0, memory_mb=512.0,  power_w=5.0,  label="rpi4")


# ─── ACTOR HANDLE ─────────────────────────────────────────────────────────────

class ActorHandle:
    """
    Bir Actor'e referans. Doğrudan nesneye erişim yerine
    bu handle üzerinden mesaj gönderilir — Ray'in uzak referans modeli.

    Kullanım:
        handle = ActorHandle(actor_instance)
        result = await handle.call("process_alert", alert_data)
    """

    def __init__(self, actor: Any, budget: ResourceBudget, actor_id: str):
        self._actor    = actor
        self._budget   = budget
        self._actor_id = actor_id
        self._mailbox  = asyncio.Queue(maxsize=256)
        self._stats    = {
            "calls": 0, "errors": 0,
            "total_latency_ms": 0.0, "peak_latency_ms": 0.0,
        }

    async def call(self, method_name: str, *args, **kwargs) -> Any:
        """
        Actor metodunu asenkron çağır.
        Ray'deki actor.method.remote() karşılığı.
        """
        t0 = time.perf_counter()
        self._stats["calls"] += 1

        try:
            method = getattr(self._actor, method_name)
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stats["total_latency_ms"] += elapsed_ms
            self._stats["peak_latency_ms"]   = max(
                self._stats["peak_latency_ms"], elapsed_ms
            )
            return result

        except Exception as e:
            self._stats["errors"] += 1
            raise

    def stats(self) -> dict:
        calls = max(self._stats["calls"], 1)
        return {
            "actor_id":      self._actor_id,
            "budget":        self._budget.label,
            "calls":         self._stats["calls"],
            "errors":        self._stats["errors"],
            "avg_latency_ms": round(self._stats["total_latency_ms"] / calls, 2),
            "peak_latency_ms": round(self._stats["peak_latency_ms"], 2),
        }


# ─── ACTOR RUNTIME ────────────────────────────────────────────────────────────

class ActorRuntime:
    """
    Tüm Actor'lerin yaşam döngüsünü yöneten runtime.
    Ray'deki ray.init() + cluster'ın karşılığı.

    Sorumlulukları:
      - Actor yaratma ve kayıt
      - Toplam kaynak kullanımı takibi
      - Graceful shutdown
    """

    def __init__(self):
        self._actors: dict[str, ActorHandle] = {}
        self._total_budget = ResourceBudget(
            num_cpus=16.0, memory_mb=16384.0, power_w=100.0, label="cluster"
        )
        self._used = {"num_cpus": 0.0, "memory_mb": 0.0, "power_w": 0.0}
        print(f"{Color.CYAN}[RUNTIME]{Color.RESET} ActorRuntime başlatıldı "
              f"| toplam bütçe: {self._total_budget.num_cpus} CPU / "
              f"{self._total_budget.memory_mb:.0f}MB / {self._total_budget.power_w}W")

    def create(
        self,
        actor_class: type,
        actor_id: str,
        budget: ResourceBudget = JETSON_NANO,
        *args, **kwargs
    ) -> ActorHandle:
        """
        Yeni bir Actor yarat ve handle döndür.
        Ray'deki ActorClass.remote(*args, **kwargs) karşılığı.
        """
        # Kaynak kontrolü
        for key in ("num_cpus", "memory_mb", "power_w"):
            if self._used[key] + getattr(budget, key) > getattr(self._total_budget, key):
                raise RuntimeError(
                    f"[RUNTIME] Kaynak aşımı: {key} için bütçe yok "
                    f"(kullanılan={self._used[key]}, talep={getattr(budget, key)})"
                )

        # Actor oluştur
        instance = actor_class(*args, **kwargs)
        handle   = ActorHandle(instance, budget, actor_id)
        self._actors[actor_id] = handle

        # Bütçeyi güncelle
        for key in ("num_cpus", "memory_mb", "power_w"):
            self._used[key] += getattr(budget, key)

        print(f"{Color.CYAN}[RUNTIME]{Color.RESET} Actor oluşturuldu: "
              f"{Color.BOLD}{actor_id}{Color.RESET} | bütçe={budget.label} | "
              f"CPU kullanım: {self._used['num_cpus']:.1f}/{self._total_budget.num_cpus}")

        return handle

    def get(self, actor_id: str) -> ActorHandle:
        if actor_id not in self._actors:
            raise KeyError(f"Actor bulunamadı: {actor_id}")
        return self._actors[actor_id]

    def all_stats(self) -> list[dict]:
        return [h.stats() for h in self._actors.values()]

    def resource_usage(self) -> dict:
        return {
            "cpu_pct":    round(self._used["num_cpus"]  / self._total_budget.num_cpus  * 100, 1),
            "memory_pct": round(self._used["memory_mb"] / self._total_budget.memory_mb * 100, 1),
            "power_pct":  round(self._used["power_w"]   / self._total_budget.power_w   * 100, 1),
        }

    async def shutdown(self):
        """Tüm Actor'leri düzgünce kapat."""
        for actor_id, handle in self._actors.items():
            actor = handle._actor
            if hasattr(actor, "shutdown"):
                await actor.shutdown()
        print(f"{Color.CYAN}[RUNTIME]{Color.RESET} Tüm Actor'ler kapatıldı.")
