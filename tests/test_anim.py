"""Gate animation tests (no camera / YOLO needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gate_guard import overlay
from gate_guard.decision import GateState
from gate_guard.overlay import GateAnimator


def run(anim, state, seconds, fps=30, t0=0.0):
    t = t0
    for _ in range(int(seconds * fps)):
        anim.update(state, t)
        t += 1.0 / fps
    return t


def test_gate_closes_fully_then_reopens():
    a = GateAnimator(speed=1.0)
    t = run(a, GateState.CLOSED, 1.5)
    assert a.pos == 1.0 and a.label == "GATE CLOSED"
    run(a, GateState.NORMAL, 1.5, t0=t)
    assert a.pos == 0.0 and a.label == "GATE OPEN"


def test_warning_stops_half_way():
    a = GateAnimator(speed=1.0)
    run(a, GateState.WARNING, 2.0)
    assert abs(a.pos - 0.45) < 1e-9
    assert a.label == "GATE HALF-CLOSED"


def test_motion_is_gradual_and_labelled():
    a = GateAnimator(speed=0.5)
    a.update(GateState.NORMAL, 0.0)
    a.update(GateState.CLOSED, 0.1)
    a.update(GateState.CLOSED, 0.2)
    assert 0.0 < a.pos < 0.2 and a.label == "CLOSING..."


def test_widget_draws_in_every_position():
    for pos in (0.0, 0.3, 0.45, 0.8, 1.0):
        frame = np.zeros((480, 640, 3), np.uint8)
        a = GateAnimator(); a.pos = pos
        overlay.draw_gate(frame, a, 40)
        assert frame.sum() > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
