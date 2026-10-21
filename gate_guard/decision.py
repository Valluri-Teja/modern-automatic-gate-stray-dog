"""Gate decision state machine: NORMAL -> WARNING -> CLOSED (latched) -> NORMAL."""
from dataclasses import dataclass, field
from enum import Enum


class GateState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CLOSED = "CLOSED"


@dataclass
class Decision:
    state: GateState
    previous: GateState
    new_warnings: list = field(default_factory=list)
    new_entries: list = field(default_factory=list)

    @property
    def changed(self):
        return self.state != self.previous


class DecisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = GateState.NORMAL
        self._last_threat = float("-inf")

    def evaluate(self, tracks, now, frame_w):
        cfg = self.cfg
        warn_px = cfg.warn_margin * frame_w
        entering = warning = threat = False
        new_w, new_e = [], []

        for t in tracks:
            if cfg.ignore_accompanied and t.accompanied:
                t.status = "ignored"
                continue
            if t.in_zone:
                threat = True
                if t.dwell(now) >= cfg.dwell_seconds:
                    t.status, entering = "entry", True
                    if not t.entry_logged:
                        t.entry_logged = True
                        new_e.append(t)
                else:
                    t.status, warning = "in_zone", True
            elif t.dist > -warn_px and t.approaching:
                t.status, warning, threat = "approaching", True, True
            else:
                t.status = "tracking"
            if t.status in ("in_zone", "approaching", "entry") and not t.warned:
                t.warned = True
                if t.status != "entry":
                    new_w.append(t)

        if threat:
            self._last_threat = now
        prev = self.state
        if entering:
            new = GateState.CLOSED
        elif prev == GateState.CLOSED and now - self._last_threat < cfg.clear_seconds:
            new = GateState.CLOSED          # latched until the area stays clear
        elif warning:
            new = GateState.WARNING
        else:
            new = GateState.NORMAL
        self.state = new
        return Decision(new, prev, new_w, new_e)
