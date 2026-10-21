"""Logic tests - no camera, no YOLO needed.   python tests/test_logic.py   (or pytest)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate_guard.config import Config
from gate_guard.decision import DecisionEngine, GateState
from gate_guard.tracking import Detection, TrackBook
from gate_guard.zone import GateZone

W, H, FPS = 1000, 600, 30
ZONE = GateZone([(0.6, 0.3), (0.9, 0.3), (0.9, 0.95), (0.6, 0.95)])   # x 600-900, y 180-570


def dog(x, y, tid=1):
    return Detection(tid, (x - 40, y - 80, x + 40, y), 0.9)


def run(cfg, path, persons=None, tail=0):
    """path = list of (x, y) foot positions (None = dog not visible), one per frame."""
    book, engine = TrackBook(cfg), DecisionEngine(cfg)
    states, decisions = [], []
    for i, pos in enumerate(path):
        now = i / FPS
        dets = [dog(*pos)] if pos else []
        tracks = book.update(dets, persons or [], ZONE, (W, H), now)
        d = engine.evaluate(tracks, now, W)
        states.append(d.state)
        decisions.append(d)
    return states, decisions


def walk(x0, x1, y, seconds):
    n = int(seconds * FPS)
    return [(x0 + (x1 - x0) * i / (n - 1), y) for i in range(n)]


def test_zone_geometry():
    assert ZONE.contains((700, 400), W, H)
    assert not ZONE.contains((300, 400), W, H)
    assert ZONE.signed_distance((700, 400), W, H) > 0
    assert ZONE.signed_distance((300, 400), W, H) < 0


def test_walk_in_closes_gate_then_reopens():
    cfg = Config()
    path = walk(100, 750, 400, 4) + [(750, 400)] * 60          # approach, stand in zone
    path += [None] * int(6 * FPS)                               # dog leaves
    states, decs = run(cfg, path)
    assert GateState.WARNING in states
    assert GateState.CLOSED in states
    assert states.index(GateState.WARNING) < states.index(GateState.CLOSED)
    assert states[-1] == GateState.NORMAL                       # reopened after clear_seconds
    assert sum(len(d.new_entries) for d in decs) == 1           # entry logged once
    assert sum(1 for d in decs if d.changed and d.state == GateState.CLOSED) == 1


def test_dog_far_away_never_triggers():
    states, _ = run(Config(), [(150, 400)] * 120)
    assert set(states) == {GateState.NORMAL}


def test_brief_touch_below_dwell_does_not_close():
    cfg = Config(dwell_seconds=1.5)
    path = [(300, 400)] * 30 + [(650, 400)] * int(0.5 * FPS) + [(300, 400)] * 60
    states, _ = run(cfg, path)
    assert GateState.CLOSED not in states


def test_zone_edge_flicker_still_counts_as_dwell():
    cfg = Config(dwell_seconds=1.5, zone_grace=0.5)
    path = []
    for i in range(90):                                         # 3 s flickering around the edge
        path.append((650, 400) if i % 6 else (590, 400))        # outside for 1 frame in 6
    states, _ = run(cfg, path)
    assert GateState.CLOSED in states


def test_accompanied_dog_ignored_only_when_enabled():
    person = [(690, 120, 760, 400)]                             # person right next to the dog
    path = [(700, 400)] * 120
    s_off, _ = run(Config(ignore_accompanied=False), path, person)
    s_on, _ = run(Config(ignore_accompanied=True), path, person)
    assert GateState.CLOSED in s_off
    assert GateState.CLOSED not in s_on


def test_dog_moving_away_is_not_approaching():
    path = walk(580, 300, 400, 2)                               # near the gate but leaving
    states, _ = run(Config(), path)
    assert GateState.CLOSED not in states


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
