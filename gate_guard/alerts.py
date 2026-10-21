"""Spoken alert. Non-blocking (the old version froze the video while speaking)
and rate-limited by a cooldown."""
import shutil
import subprocess
import sys
import time


class Announcer:
    def __init__(self, enabled=True, cooldown=10.0):
        self.enabled, self.cooldown = enabled, cooldown
        self._last, self._proc = float("-inf"), None
        self._cmd = None
        if sys.platform == "darwin" and shutil.which("say"):
            self._cmd = "say"
        elif shutil.which("espeak"):
            self._cmd = "espeak"

    def say(self, text):
        if not self.enabled:
            return False
        now = time.monotonic()
        if now - self._last < self.cooldown:
            return False
        self._last = now
        if self._cmd and (self._proc is None or self._proc.poll() is not None):
            self._proc = subprocess.Popen([self._cmd, text],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)   # terminal bell fallback
        return True
