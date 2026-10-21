"""Webcam / video file / RTSP input.
Live sources are read on a background thread (always the newest frame, no lag build-up)
and reconnect automatically if the stream drops."""
import threading
import time

import cv2


def is_live(source) -> bool:
    s = str(source).lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://"))


class VideoSource:
    def __init__(self, source):
        self.live = is_live(source)
        self.source = int(source) if str(source).isdigit() else source
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._idx = 0
        self._frame, self._seq, self._last_seq = None, 0, 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self.live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                fails = 0
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                continue
            fails += 1
            if fails >= 30:
                print("[video] stream lost, reconnecting...")
                self._cap.release()
                time.sleep(2.0)
                self._cap = cv2.VideoCapture(self.source)
                fails = 0
            else:
                time.sleep(0.03)

    def read(self, timeout=3.0):
        """Returns (ok, frame, timestamp_seconds).
        Files: timestamp = frame_index / fps. Live: monotonic clock."""
        if not self.live:
            ok, frame = self._cap.read()
            if not ok:
                return False, None, 0.0
            ts = self._idx / self.fps
            self._idx += 1
            return True, frame, ts
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    return True, self._frame, time.monotonic()
            time.sleep(0.005)
        return False, None, time.monotonic()

    def release(self):
        self._stop.set()
        time.sleep(0.05)
        self._cap.release()
