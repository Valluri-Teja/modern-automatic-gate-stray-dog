"""Everything drawn on the video: zone, boxes, IDs, status banner, FPS, counters."""
import cv2
import numpy as np

from .decision import GateState

STATUS_COLOUR = {            # BGR
    "tracking": (0, 200, 0), "approaching": (0, 165, 255),
    "in_zone": (0, 110, 255), "entry": (0, 0, 255), "ignored": (170, 170, 170)}
STATE_COLOUR = {GateState.NORMAL: (0, 160, 0), GateState.WARNING: (0, 150, 255),
                GateState.CLOSED: (0, 0, 220)}
STATE_TEXT = {GateState.NORMAL: "GATE OPEN - NORMAL",
              GateState.WARNING: "WARNING - DOG NEAR GATE",
              GateState.CLOSED: "GATE CLOSED - UNAUTHORIZED ENTRY"}


class GateAnimator:
    """Smoothly moves a virtual gate: open (0.0) -> half closed on WARNING -> fully closed on CLOSED."""
    TARGET = {GateState.NORMAL: 0.0, GateState.WARNING: 0.45, GateState.CLOSED: 1.0}

    def __init__(self, speed=0.9):
        self.pos = 0.0                  # 0 = open, 1 = closed
        self.speed = speed              # fraction of the full travel per second
        self.target = 0.0
        self._last = None

    def update(self, state, now):
        self.target = self.TARGET[state]
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), 0.25)
        self._last = now
        step = self.speed * dt
        if self.pos < self.target:
            self.pos = min(self.target, self.pos + step)
        elif self.pos > self.target:
            self.pos = max(self.target, self.pos - step)
        return self.pos

    @property
    def label(self):
        if self.pos >= 0.999:
            return "GATE CLOSED"
        if abs(self.pos - self.target) > 0.005:
            return "CLOSING..." if self.target > self.pos else "OPENING..."
        return "GATE OPEN" if self.pos <= 0.001 else "GATE HALF-CLOSED"


def _gate_colour(pos):
    """green (open) -> orange (half) -> red (closed), as BGR"""
    stops = [(0.0, (60, 180, 60)), (0.5, (0, 165, 255)), (1.0, (0, 0, 220))]
    pos = min(max(pos, 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if pos <= p1:
            k = (pos - p0) / (p1 - p0)
            return tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
    return stops[-1][1]


def draw_gate(frame, anim, y_top):
    """Little animated gate in the top-right corner: two leaves slide together as the gate closes."""
    h, w = frame.shape[:2]
    pw = max(200, int(w * 0.20))
    ph = int(pw * 0.80)
    x0, y0 = w - pw - 12, y_top + 10
    sc = pw / 260.0
    pos = anim.pos
    col = _gate_colour(pos)

    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + pw, y0 + ph), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (220, 220, 220), 1)

    post = max(6, int(10 * sc))
    gx1, gx2 = x0 + post + 6, x0 + pw - post - 6          # opening between the posts
    gy1, gy2 = y0 + int(14 * sc), y0 + ph - int(44 * sc)
    cv2.rectangle(frame, (x0 + 6, gy1 - 6), (x0 + 6 + post, gy2 + 6), (170, 170, 170), -1)   # posts
    cv2.rectangle(frame, (x0 + pw - 6 - post, gy1 - 6), (x0 + pw - 6, gy2 + 6), (170, 170, 170), -1)
    cv2.line(frame, (gx1, gy2 + 6), (gx2, gy2 + 6), (120, 120, 120), 2)                       # ground

    half = (gx2 - gx1) / 2.0
    cover = int(half * pos)
    if cover > 1:
        bars = max(2, int(cover / (12 * sc)))
        for x_a, x_b in ((gx1, gx1 + cover), (gx2 - cover, gx2)):          # left leaf, right leaf
            cv2.rectangle(frame, (x_a, gy1), (x_b, gy2), col, -1)
            cv2.rectangle(frame, (x_a, gy1), (x_b, gy2), (255, 255, 255), 1)
            for i in range(1, bars):
                bx = int(x_a + (x_b - x_a) * i / bars)
                cv2.line(frame, (bx, gy1), (bx, gy2), (255, 255, 255), 1)
        if pos >= 0.999:                                                     # lock when closed
            cx, cy, r = (gx1 + gx2) // 2, (gy1 + gy2) // 2, max(6, int(9 * sc))
            cv2.rectangle(frame, (cx - r, cy - r // 2), (cx + r, cy + r), (255, 255, 255), -1)
            cv2.ellipse(frame, (cx, cy - r // 2), (int(r * 0.7), int(r * 0.9)), 0, 180, 360, (255, 255, 255), 2)

    label = anim.label
    cv2.putText(frame, label, (x0 + 10, y0 + ph - int(14 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(sc, 0.8), (255, 255, 255), max(1, int(2 * sc)))
    cv2.circle(frame, (x0 + pw - 16, y0 + ph - int(20 * sc)), max(5, int(7 * sc)), col, -1)


def draw(frame, zone, tracks, persons, state, fps, stats, now, gate_anim=None):
    h, w = frame.shape[:2]
    sc = max(0.5, w / 1280.0 * 0.7)
    th = max(1, int(round(sc * 2)))
    poly = zone.polygon_px(w, h).astype(np.int32)
    col = STATE_COLOUR[state]

    if getattr(zone, "visible", True):
        shade = frame.copy()
        cv2.fillPoly(shade, [poly], col)
        cv2.addWeighted(shade, 0.22, frame, 0.78, 0, frame)
        cv2.polylines(frame, [poly], True, col, th + 1)

    for p in persons:
        cv2.rectangle(frame, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (200, 120, 0), 1)

    for t in tracks:
        if not t.visible(now):
            continue
        c = STATUS_COLOUR.get(t.status, (0, 200, 0))
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, th + 1)
        label = f"Dog #{t.track_id} {t.conf:.2f}"
        if t.in_zone:
            label += f" | {t.dwell(now):.1f}s in zone"
        elif t.approaching:
            label += " | approaching"
        if t.accompanied:
            label += " | with person"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, sc * 0.8, c, th)
        cv2.circle(frame, (int((x1 + x2) / 2), y2), 5, c, -1)

    bar = int(44 * sc / 0.7)
    cv2.rectangle(frame, (0, 0), (w, bar), col, -1)
    cv2.putText(frame, STATE_TEXT[state], (10, int(bar * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, sc * 1.1, (255, 255, 255), th + 1)
    if gate_anim is not None:
        draw_gate(frame, gate_anim, bar)
    info = (f"FPS {fps:4.1f}  dogs seen {stats['dogs_seen']}  "
            f"warnings {stats['warnings']}  entries {stats['entries']}")
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.8, (255, 255, 255), th)
    return frame
