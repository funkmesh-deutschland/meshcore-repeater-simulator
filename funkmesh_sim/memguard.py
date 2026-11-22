"""Schutz vor OOM: prüft RAM/VRAM, bevor große Puffer angelegt werden."""
from __future__ import annotations

import psutil

from . import config


class MemoryBudgetError(RuntimeError):
    pass


def ram_budget_bytes() -> int:
    avail = psutil.virtual_memory().available
    return int(min(avail * config.RAM_FRACTION, config.RAM_MAX_GB * 2**30))


def require_ram(nbytes: int, what: str) -> None:
    budget = ram_budget_bytes()
    if nbytes > budget:
        raise MemoryBudgetError(
            f"{what}: benötigt {nbytes / 2**20:.0f} MiB RAM, Budget {budget / 2**20:.0f} MiB "
            f"(verfügbar {psutil.virtual_memory().available / 2**20:.0f} MiB). "
            "Radius/Auflösung verringern oder FUNKMESH_RAM_MAX_GB anpassen."
        )


def vram_free_bytes() -> int:
    from numba import cuda

    free, _total = cuda.current_context().get_memory_info()
    return int(free)


def require_vram(nbytes: int, what: str) -> None:
    free = vram_free_bytes()
    budget = int(free * config.VRAM_FRACTION)
    if nbytes > budget:
        raise MemoryBudgetError(
            f"{what}: benötigt {nbytes / 2**20:.0f} MiB VRAM, Budget {budget / 2**20:.0f} MiB "
            f"(frei {free / 2**20:.0f} MiB)."
        )


def require_disk(path, nbytes: float, what: str, reserve_gb: float = 5.0) -> None:
    """Freien Plattenplatz prüfen (Memory-Maps wachsen sonst bis die Platte voll ist)."""
    import shutil

    free = shutil.disk_usage(path).free
    if nbytes + reserve_gb * 2**30 > free:
        raise MemoryBudgetError(
            f"{what}: benötigt {nbytes / 2**30:.1f} GiB Plattenplatz, frei {free / 2**30:.1f} GiB "
            f"(+{reserve_gb:g} GiB Reserve). Kleineres Gebiet oder gröberes Raster wählen."
        )
