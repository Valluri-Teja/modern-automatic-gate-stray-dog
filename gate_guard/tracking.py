"""Per-dog track state: distance to gate zone, dwell time, approach direction."""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    track_id: int
    box: tuple            # x1, y1, x2, y2 (pixels)
    conf: float
    label: str = "dog"

    @property
    def foot(self):
        """Bottom-centre of the box: where the dog touches the ground.
        Better than the box centre for 'is it standing in the gate area'."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, float(y2))


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    box: tuple
    conf: float
    max_conf: float
    dist: float = -1e9                     # signed px distance to zone (>=0 inside)
    history: deque = field(default_factory=lambda: deque(maxlen=240))
    in_zone_since: Optional[float] = None
    last_in_zone: Optional[float] = None
    accompanied: bool = False
    approaching: bool = False
    status: str = "tracking"               # tracking|approaching|in_zone|entry|ignored
    warned: bool = False
    entry_logged: bool = False

    @property
    def in_zone(self):
        return self.in_zone_since is not None

    def dwell(self, now):
        return (now - self.in_zone_since) if self.in_zone_since is not None else 0.0

    def visible(self, now):
        return abs(self.last_seen - now) < 1e-9


def _overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def is_accompanied(dog_box, person_boxes, margin):
    x1, y1, x2, y2 = dog_box
    m = margin * max(x2 - x1, y2 - y1)
    grown = (x1 - m, y1 - m, x2 + m, y2 + m)
    return any(_overlaps(grown, p) for p in person_boxes)


class TrackBook:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.seen_ids = set()

    def update(self, dets, persons, zone, size, now):
        w, h = size
        cfg = self.cfg
        for d in dets:
            if d.track_id < 0:
                continue
            t = self.tracks.get(d.track_id)
            if t is None:
                t = TrackState(d.track_id, now, now, d.box, d.conf, d.conf)
                self.tracks[d.track_id] = t
                self.seen_ids.add(d.track_id)
            t.last_seen, t.box, t.conf = now, d.box, d.conf
            t.max_conf = max(t.max_conf, d.conf)
            t.dist = zone.signed_distance(d.foot, w, h)
            t.history.append((now, t.dist))
            while t.history and now - t.history[0][0] > cfg.approach_window * 2:
                t.history.popleft()
            if t.dist >= 0:
                if t.in_zone_since is None:
                    t.in_zone_since = now
                t.last_in_zone = now
            t.accompanied = is_accompanied(d.box, persons, cfg.accompany_margin) if persons else False
            t.approaching = self._approaching(t, now, w)

        for t in self.tracks.values():          # zone flicker tolerance
            if (t.in_zone_since is not None and t.last_in_zone is not None
                    and now - t.last_in_zone > cfg.zone_grace):
                t.in_zone_since = None

        for tid in [k for k, t in self.tracks.items() if now - t.last_seen > cfg.track_timeout]:
            del self.tracks[tid]
        return list(self.tracks.values())

    def _approaching(self, t, now, width):
        win = self.cfg.approach_window
        past = [(ts, d) for ts, d in t.history if now - ts <= win]
        if len(past) < 3 or now - past[0][0] < win * 0.5:
            return False
        # signed distance grows as the dog gets closer to / deeper into the zone
        return (t.dist - past[0][1]) > self.cfg.approach_min_move * width
