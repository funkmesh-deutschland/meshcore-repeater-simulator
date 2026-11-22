"""Projekt: Repeater-Liste und Einstellungen (JSON in ``data/projects/<name>.json``)."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields

from . import config
from .radio.engine import Environment, Repeater


@dataclass
class Radio:
    """MeshCore-Funkparameter (Voreinstellung: EU/UK Narrow, wie im Coesfelder Netz)."""
    freq_mhz: float = 869.618
    bw_khz: float = 62.5
    sf: int = 8
    cr: int = 8
    preamble: int = 16
    nf_db: float = 6.0


@dataclass
class Node:
    """Endgerät (Companion), am Körper getragen."""
    tx_dbm: float = 22.0
    gain_dbi: float = 0.0
    body_loss_db: float = 4.0    # Körpernähe 868 MHz, typ. 3..8 dB
    height_m: float = 1.5


@dataclass
class Statistics:
    sigma_outdoor_db: float = 5.5   # Ortsvariabilität (ITU-R P.1812 / P.1546)
    sigma_indoor_db: float = 8.0    # zusätzlich Gebäudeeintritt (ITU-R P.2109)
    target_prob: float = 0.9        # Ziel-Ortswahrscheinlichkeit für "versorgt"
    calib_offset_db: float = 0.0    # aus Messungen kalibrierter Zuschlag auf den Pfadverlust


@dataclass
class Compute:
    out_res: float = 10.0
    radius_km: float = 15.0


@dataclass
class Project:
    name: str = "standard"
    region: str = "coesfeld"
    repeaters: list = field(default_factory=list)
    radio: Radio = field(default_factory=Radio)
    node: Node = field(default_factory=Node)
    env: Environment = field(default_factory=Environment)
    stats: Statistics = field(default_factory=Statistics)
    compute: Compute = field(default_factory=Compute)

    def environment(self) -> Environment:
        e = Environment(**asdict(self.env))
        e.freq_mhz = self.radio.freq_mhz
        e.rx_height_m = self.node.height_m
        return e

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        def build(dc, data):
            names = {f.name for f in fields(dc)}
            return dc(**{k: v for k, v in (data or {}).items() if k in names})

        p = cls(name=d.get("name", "standard"), region=d.get("region", "coesfeld"))
        p.repeaters = [build(Repeater, r) for r in d.get("repeaters", [])]
        p.radio = build(Radio, d.get("radio"))
        p.node = build(Node, d.get("node"))
        p.env = build(Environment, d.get("env"))
        p.stats = build(Statistics, d.get("stats"))
        p.compute = build(Compute, d.get("compute"))
        return p


def _coerce(cur, v):
    if isinstance(cur, bool):
        return v in (True, 1, "1", "true", "True", "on")
    if cur is None or v is None:
        return v
    return type(cur)(v)


class ProjectStore:
    def __init__(self, name: str = "standard", region: str = "coesfeld"):
        self.path = config.projects_dir() / f"{name}.json"
        self.lock = threading.RLock()
        if self.path.exists():
            self.project = Project.from_dict(json.loads(self.path.read_text()))
        else:
            self.project = Project(name=name, region=region)
            self.save()

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.project.to_dict(), indent=1, ensure_ascii=False))
            tmp.replace(self.path)

    def add_repeater(self, data: dict) -> Repeater:
        with self.lock:
            names = {f.name for f in fields(Repeater)}
            d = {k: v for k, v in data.items() if k in names}
            d.setdefault("id", uuid.uuid4().hex[:10])
            d.setdefault("name", f"Repeater {len(self.project.repeaters) + 1}")
            r = Repeater(**d)
            self.project.repeaters.append(r)
            self.save()
            return r

    def update_repeater(self, rid: str, data: dict) -> Repeater:
        with self.lock:
            for r in self.project.repeaters:
                if r.id == rid:
                    for k, v in data.items():
                        if k != "id" and hasattr(r, k):
                            setattr(r, k, _coerce(getattr(r, k), v))
                    self.save()
                    return r
        raise KeyError(rid)

    def delete_repeater(self, rid: str) -> None:
        with self.lock:
            self.project.repeaters = [r for r in self.project.repeaters if r.id != rid]
            self.save()

    def update_settings(self, section: str, data: dict) -> None:
        with self.lock:
            obj = getattr(self.project, section)
            for k, v in data.items():
                if hasattr(obj, k):
                    cur = getattr(obj, k)
                    setattr(obj, k, _coerce(cur, v))
            self.save()
