"""All tunable settings in one place. Every field is also a command-line flag."""
import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ---- input / model ----
    source: str = ""                  # empty = pick a video file in a window | 0 = webcam | path | rtsp:// URL
    model: str = "yolo26n.pt"         # or your fine-tuned best.pt
    conf: float = 0.40                # detection confidence threshold
    iou: float = 0.50                 # NMS IoU threshold
    imgsz: int = 640
    device: str = "auto"              # auto | mps | cpu | 0 (CUDA)
    tracker: str = "bytetrack.yaml"   # or botsort.yaml

    # ---- gate zone ----
    zone_file: str = "zone.json"
    select_zone: bool = False         # force re-drawing the zone
    zone_mode: str = "full"           # full (default, no clicking) | center | auto-gate | manual

    # ---- decision logic (distances are fractions of frame width) ----
    dwell_seconds: float = 1.5        # time inside zone before ENTRY event
    warn_margin: float = 0.12         # "near the gate" band outside the zone
    approach_window: float = 1.0      # seconds used to judge movement direction
    approach_min_move: float = 0.015  # min movement toward zone inside that window
    zone_grace: float = 0.5           # tolerate detection flicker at zone edge
    clear_seconds: float = 3.0        # gate stays closed this long after dog leaves
    track_timeout: float = 2.0        # forget a lost track after this long
    min_hits: int = 5                 # frames a dog must be tracked before it counts (filters flicker)
    merge_window: float = 2.5         # a "new" dog appearing within this many seconds ...
    merge_distance: float = 1.0       # ... and this close (x dog size) to a lost dog = same dog
    no_merge: bool = False            # disable ID merging (to compare before/after)
    ignore_accompanied: bool = False  # ignore dogs that have a person next to them
    accompany_margin: float = 0.5     # how close a person must be (x dog size)

    # ---- outputs ----
    out_dir: str = "runs"
    save_video: bool = False          # write annotated output video
    save_clips: bool = False          # short clip around every entry event
    clip_pre: float = 3.0
    clip_post: float = 3.0
    no_display: bool = False          # headless mode
    no_sound: bool = False
    no_gate_anim: bool = False        # hide the animated gate widget
    announce_cooldown: float = 10.0

    # ---- gate hardware ----
    gate: str = "sim"                 # sim | serial
    serial_port: str = ""             # e.g. /dev/cu.usbmodem1101
    serial_baud: int = 9600


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(
        description="Stray dog gate guard: YOLO26 detection + tracking + gate-zone decision")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action="store_true", default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    cfg = Config(**vars(ap.parse_args(argv)))
    if cfg.select_zone:
        cfg.zone_mode = "manual"      # --select-zone alone means: let me draw the zone
    return cfg
