"""Event log (CSV), snapshots and short video clips (pre-roll + post-roll)."""
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2


class ClipRecorder:
    """Keeps the last few seconds in a small ring buffer; on trigger writes them
    to an mp4 and keeps recording for `post` more seconds."""

    def __init__(self, out_dir, pre=3.0, post=3.0, width=640, fps=15):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pre, self.post, self.width = pre, post, width
        self.buf = deque(maxlen=max(1, int(pre * fps)))
        self.active = []          # [writer, frames_left]

    def _small(self, frame):
        h, w = frame.shape[:2]
        s = self.width / float(w)
        return cv2.resize(frame, (self.width, int(h * s))) if s < 1 else frame.copy()

    def push(self, frame):
        small = self._small(frame)
        for item in list(self.active):
            item[0].write(small)
            item[1] -= 1
            if item[1] <= 0:
                item[0].release()
                self.active.remove(item)
        self.buf.append(small)

    def trigger(self, name, fps):
        fps = max(5.0, min(30.0, fps))
        if not self.buf:
            return None
        path = self.dir / f"{name}.mp4"
        h, w = self.buf[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in self.buf:
            writer.write(f)
        self.active.append([writer, int(self.post * fps)])
        return path

    def close(self):
        for writer, _ in self.active:
            writer.release()
        self.active.clear()


class EventLogger:
    HEADER = ["time", "video_time_s", "event", "track_id", "confidence",
              "distance_px", "dwell_s", "gate_state", "snapshot"]

    def __init__(self, run_dir, t0=0.0):
        self.t0 = t0                      # clock value at the first frame
        self.run_dir = Path(run_dir)
        self.snap_dir = self.run_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.run_dir / "events.csv", "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def log(self, event, now, track=None, gate_state="", frame=None, snapshot=False):
        video_t = now - self.t0
        snap = ""
        tid = track.track_id if track else ""
        if snapshot and frame is not None:
            stamp = datetime.now().strftime("%H%M%S")
            snap = str(self.snap_dir / f"{event}_{tid}_{stamp}.jpg")
            cv2.imwrite(snap, frame)
        self._w.writerow([
            datetime.now().isoformat(timespec="seconds"), f"{video_t:.2f}", event, tid,
            f"{track.conf:.2f}" if track else "",
            f"{track.dist:.0f}" if track else "",
            f"{track.dwell(now):.1f}" if track else "",
            gate_state, snap])
        self._f.flush()
        print(f"[event] t={video_t:6.1f}s  {event:<20} id={tid}  gate={gate_state}")
        return snap

    def close(self):
        self._f.close()

