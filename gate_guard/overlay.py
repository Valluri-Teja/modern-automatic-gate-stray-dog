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


def draw(frame, zone, tracks, persons, state, fps, stats, now):
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
    info = (f"FPS {fps:4.1f}  dogs seen {stats['dogs_seen']}  "
            f"warnings {stats['warnings']}  entries {stats['entries']}")
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.8, (255, 255, 255), th)
    return frame
