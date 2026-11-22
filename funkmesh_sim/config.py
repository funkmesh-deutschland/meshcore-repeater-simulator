"""Pfade und Speicher-Budgets.

Alle großen Arrays liegen als memory-mapped .npy-Dateien auf der Platte; RAM und VRAM
werden vor jeder großen Allokation geprüft (siehe :mod:`funkmesh_sim.memguard`).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REGIONS_DIR = PACKAGE_DIR / "regions"
STATIC_DIR = PACKAGE_DIR / "static"
MAPPER_DIR = PACKAGE_DIR / "mapper"


def data_dir() -> Path:
    """Arbeitsverzeichnis für Geodaten, Caches und Projekte (``FUNKMESH_DATA``)."""
    p = Path(os.environ.get("FUNKMESH_DATA", PROJECT_ROOT / "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def region_dir(region: str) -> Path:
    p = data_dir() / "regions" / region
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir(region: str) -> Path:
    p = data_dir() / "raw" / region
    p.mkdir(parents=True, exist_ok=True)
    return p


def projects_dir() -> Path:
    p = data_dir() / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Höchstanteil des *verfügbaren* RAM, den eine Berechnung belegen darf.
RAM_FRACTION = float(os.environ.get("FUNKMESH_RAM_FRACTION", "0.35"))
# Absolutes RAM-Limit in GiB (zusätzlich zum Anteil).
RAM_MAX_GB = float(os.environ.get("FUNKMESH_RAM_MAX_GB", "6"))
# Höchstanteil des *freien* VRAM.
VRAM_FRACTION = float(os.environ.get("FUNKMESH_VRAM_FRACTION", "0.6"))
# GPU abschalten (z. B. zum Testen der CPU-Variante).
DISABLE_GPU = os.environ.get("FUNKMESH_NO_GPU", "") not in ("", "0", "false")
