"""Tests for ID merging, flicker filtering and the evaluation maths."""
import csv
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import read_close_times, score_clip, summarise
from gate_guard.config import Config
from gate_guard.decision import DecisionEngine
from gate_guard.metrics import average_precision, iou_matrix, match_event_times, match_image
from gate_guard.tracking import Detection, TrackBook
from gate_guard.zone import GateZone

W, H, FPS = 1000, 600, 30
ZONE = GateZone([(0.6, 0.3), (0.9, 0.3), (0.9, 0.95), (0.6, 0.95)])


def play(cfg, seq):
    """seq items: None or (raw_id, x, y)  -> returns (book, decisions)"""
    book, engine, decs = TrackBook(cfg), DecisionEngine(cfg), []
    for i, item in enumerate(seq):
        now = i / FPS
        dets = [] if item is None else [Detection(item[0], (item[1] - 40, item[2] - 80, item[1] + 40, item[2]), 0.9)]
        tracks = book.update(dets, [], ZONE, (W, H), now)
        decs.append(engine.evaluate(tracks, now, W))
    return book, decs


def count_entries(decs):
    return sum(len(d.new_entries) for d in decs)


# ---------------- ID merging ----------------
def test_id_switch_is_merged_into_one_dog():
    seq = [(1, 700, 400)] * 60 + [None] * 12 + [(2, 705, 400)] * 60
    book, decs = play(Config(), seq)
    assert len(book.seen_ids) == 1
    assert count_entries(decs) == 1
    assert len(book.merger.map) == 2          # two raw tracker ids ...
    assert set(book.merger.map.values()) == {1}   # ... one dog


def test_far_away_new_dog_is_not_merged():
    seq = [(1, 700, 400)] * 60 + [None] * 12 + [(2, 200, 400)] * 60
    book, _ = play(Config(), seq)
    assert len(book.seen_ids) == 2


def test_merge_can_be_disabled():
    seq = [(1, 700, 400)] * 60 + [None] * 12 + [(2, 705, 400)] * 60
    book, decs = play(Config(no_merge=True), seq)
    assert len(book.seen_ids) == 2
    assert count_entries(decs) == 2


def test_long_gap_means_a_new_dog():
    seq = [(1, 700, 400)] * 60 + [None] * 120 + [(2, 705, 400)] * 60     # 4 s gap > merge_window
    book, _ = play(Config(), seq)
    assert len(book.seen_ids) == 2


def test_one_frame_flicker_is_not_a_dog():
    seq = [None] * 10 + [(1, 700, 400)] * 3 + [None] * 30
    book, decs = play(Config(), seq)
    assert len(book.seen_ids) == 0
    assert count_entries(decs) == 0


# ---------------- metrics ----------------
def test_iou_and_matching():
    a = [[0, 0, 10, 10]]
    assert abs(iou_matrix(a, [[0, 0, 10, 10]])[0, 0] - 1.0) < 1e-9
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0
    scores, flags, missed = match_image([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8],
                                        [[1, 1, 10, 10]], 0.5, 0.0)
    assert flags == [True, False] and missed == 0
    _, flags2, missed2 = match_image([[0, 0, 10, 10]], [0.3], [[1, 1, 10, 10]], 0.5, 0.5)
    assert flags2 == [] and missed2 == 1      # prediction below score threshold -> missed


def test_average_precision():
    assert abs(average_precision([0.9, 0.8], [True, True], 2) - 1.0) < 1e-9
    assert average_precision([], [], 3) == 0.0
    assert 0 < average_precision([0.9, 0.8, 0.7], [True, False, True], 2) < 1


def test_event_matching_and_delay():
    matches, missed, false_preds = match_event_times([4.5, 30.0], [3.0, 20.0], 1.0, 6.0)
    assert matches == [(3.0, 4.5)]
    assert missed == [20.0]
    assert false_preds == [30.0]


def test_score_clip_and_summary_from_csv():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "events.csv"
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "video_time_s", "event", "track_id"])
            w.writerow(["x", "4.50", "GATE_CLOSE", ""])
            w.writerow(["x", "6.00", "UNAUTHORIZED_ENTRY", "1"])
            w.writerow(["x", "40.00", "GATE_CLOSE", ""])
        times = read_close_times(f)
    assert times == [4.5, 40.0]
    r = score_clip([3.0, 20.0], times)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 1, 1)
    s = summarise([r])
    assert s["real_entries"] == 2 and s["caught"] == 1 and s["false_closes"] == 1
    assert abs(s["delay_mean"] - 1.5) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
