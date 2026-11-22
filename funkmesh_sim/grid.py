"""Gebietsraster: Gelände, Gebäude- und Vegetationshöhen in ETRS89/UTM32 (EPSG:25832).

Dateien in ``data/regions/<name>/``:

* ``meta.json``   – Ausdehnung, Auflösung, Quellen
* ``terrain.npy`` – float32, Geländehöhe [m ü. NHN] (DGM)
* ``bld.npy``     – uint16, Gebäudehöhe über Gelände [dm] (LoD2-Dachflächen), 0 = kein Gebäude
* ``veg.npy``     – uint16, Vegetationshöhe über Gelände [dm] (nDOM ohne Gebäude), 0 = frei

Zeile 0 ist der Nordrand (wie bei GeoTIFF). Zellmittelpunkt (r, c):
``x = x0 + (c + 0.5) * res``, ``y = y1 - (r + 0.5) * res``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from . import config

CRS_EPSG = 25832


@dataclass
class GridMeta:
    name: str
    res: float
    x0: float  # linker Rand [m]
    y1: float  # oberer Rand [m]
    nx: int
    ny: int
    sources: dict
    complete: bool = False
    info: dict = field(default_factory=dict)   # Titel, Grenzen (lon/lat), Aufbauoptionen

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.res

    @property
    def y0(self) -> float:
        return self.y1 - self.ny * self.res

    def to_rc(self, x, y):
        c = (np.asarray(x) - self.x0) / self.res - 0.5
        r = (self.y1 - np.asarray(y)) / self.res - 0.5
        return r, c


class RegionGrid:
    def __init__(self, path: Path):
        self.path = Path(path)
        m = json.loads((self.path / "meta.json").read_text())
        names = set(GridMeta.__dataclass_fields__)
        self.meta = GridMeta(**{k: v for k, v in m.items() if k in names})
        self._arrays: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls, region: str) -> "RegionGrid":
        return cls(config.region_dir(region))

    @staticmethod
    def exists(region: str) -> bool:
        return (config.region_dir(region) / "meta.json").exists()

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.path / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    @property
    def terrain(self):
        return self.array("terrain")

    @property
    def bld(self):
        return self.array("bld")

    @property
    def veg(self):
        return self.array("veg")

    def sample_point(self, x: float, y: float) -> dict:
        r, c = self.meta.to_rc(x, y)
        ri, ci = int(round(float(r))), int(round(float(c)))
        if not (0 <= ri < self.meta.ny and 0 <= ci < self.meta.nx):
            return {"inside": False}
        return {
            "inside": True,
            "terrain": float(self.terrain[ri, ci]),
            "building": float(self.bld[ri, ci]) / 10.0,
            "vegetation": float(self.veg[ri, ci]) / 10.0,
        }


def write_meta(path: Path, meta: GridMeta) -> None:
    (Path(path) / "meta.json").write_text(json.dumps(asdict(meta), indent=1))
