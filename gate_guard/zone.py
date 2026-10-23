"""Restricted gate zone: a polygon stored in normalised (0-1) coordinates,
so the same zone works at any resolution."""
import json
from pathlib import Path

import cv2
import numpy as np


# Automatic zones (normalised coordinates), no clicking needed.
PRESETS = {
    "full": [(-0.05, -0.05), (1.05, -0.05), (1.05, 1.05), (-0.05, 1.05)],   # whole frame
    "center": [(0.25, 0.40), (0.75, 0.40), (0.85, 1.05), (0.15, 1.05)],     # lower-centre area
}


class GateZone:
    def __init__(self, points_norm):
        if len(points_norm) < 3:
            raise ValueError("A zone needs at least 3 points")
        self.points = [(float(x), float(y)) for x, y in points_norm]
        self.visible = True          # False = do not shade the zone (whole-frame mode)
        self._cache = {}

    # ---- geometry ----
    def polygon_px(self, w, h):
        key = (w, h)
        if key not in self._cache:
            self._cache[key] = np.array(
                [[x * w, y * h] for x, y in self.points], dtype=np.float32).reshape(-1, 1, 2)
        return self._cache[key]

    def signed_distance(self, pt, w, h):
        """Pixels from pt to the zone edge: >= 0 inside, < 0 outside."""
        return float(cv2.pointPolygonTest(
            self.polygon_px(w, h), (float(pt[0]), float(pt[1])), True))

    def contains(self, pt, w, h):
        return self.signed_distance(pt, w, h) >= 0

    # ---- persistence ----
    def save(self, path):
        Path(path).write_text(json.dumps({"points": self.points}, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["points"])


def select_zone_interactive(frame, window="Draw gate zone"):
    """Click corners of the restricted area.
    Left-click add | right-click undo | c clear | Enter confirm | Esc cancel.
    Returns normalised points or None."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1280.0 / w)
    base = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    result = None
    while True:
        canvas = base.copy()
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], len(pts) >= 3, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(canvas, p, 5, (0, 0, 255), -1)
        cv2.putText(canvas, "Click gate-zone corners | Enter=save | right-click=undo | c=clear | Esc=cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) >= 3:
            ch, cw = base.shape[:2]
            result = [(x / cw, y / ch) for x, y in pts]
            break
        if key == ord("c"):
            pts.clear()
        if key == 27:
            break
    cv2.destroyWindow(window)
    cv2.waitKey(1)
    return result
