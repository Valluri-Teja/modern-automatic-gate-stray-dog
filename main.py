#!/usr/bin/env python3
"""Stray Dog Gate Guard
Camera -> YOLO26 detection -> tracking -> gate-zone analysis -> gate decision.

  python main.py                                   # opens a window: pick a video, dogs detected at once
  python main.py --source 0                        # webcam
  python main.py --source videos/gate.mp4 --save-video
  python main.py --source videos/gate.mp4 --zone-mode manual --select-zone   # draw your own gate zone
  python main.py --model runs_train/dog_yolo26n/weights/best.pt
Keys: q/Esc quit | p pause | z redraw zone | s snapshot
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from gate_guard import overlay
from gate_guard.overlay import GateAnimator
from gate_guard.alerts import Announcer
from gate_guard.config import parse_args
from gate_guard.decision import DecisionEngine, GateState
from gate_guard.events import ClipRecorder, EventLogger
from gate_guard.gate_controller import make_gate
from gate_guard.tracking import TrackBook
from gate_guard.video_source import VideoSource
from gate_guard.zone import PRESETS, GateZone, select_zone_interactive


def pick_video():
    """Open a file window so you can choose a video (no command-line path needed)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"), ("All files", "*.*")])
        root.destroy()
        return path
    except Exception as e:
        print(f"[source] could not open the file window: {e}")
        return ""


def setup_zone(cfg, frame, run_dir):
    mode = cfg.zone_mode
    if mode in PRESETS:
        print(f"[zone] automatic '{mode}' zone (no clicking)")
        zone = GateZone(PRESETS[mode])
        zone.visible = mode != "full"
        return zone
    if mode == "auto-gate":
        pts = None
        try:
            from gate_guard.gate_finder import find_gate_zone
            pts = find_gate_zone(frame)
        except Exception as e:
            print(f"[zone] automatic gate detection failed: {e}")
        if pts is None:
            print("[zone] no gate found, using the 'center' zone instead")
            return GateZone(PRESETS["center"])
        zone = GateZone(pts)
        preview = frame.copy()
        h, w = preview.shape[:2]
        cv2.polylines(preview, [zone.polygon_px(w, h).astype("int32")], True, (0, 255, 255), 3)
        cv2.imwrite(str(run_dir / "auto_zone.jpg"), preview)
        print(f"[zone] gate found automatically, preview saved to {run_dir / 'auto_zone.jpg'}")
        return zone
    if mode != "manual":
        sys.exit(f"Unknown --zone-mode '{mode}'. Use manual, full, center or auto-gate.")
    if not cfg.select_zone and Path(cfg.zone_file).exists():
        return GateZone.load(cfg.zone_file)
    if cfg.no_display:
        sys.exit(f"No zone file '{cfg.zone_file}'. Run once without --no-display to draw it.")
    pts = select_zone_interactive(frame)
    if pts is None:
        sys.exit("Zone selection cancelled.")
    zone = GateZone(pts)
    zone.save(cfg.zone_file)
    print(f"[zone] saved to {cfg.zone_file}")
    return zone


def main():
    cfg = parse_args()
    if not cfg.source:
        cfg.source = pick_video()
        if not cfg.source:
            sys.exit("No video selected. Use --source 0 for the webcam or --source <video path>.")
    run_dir = Path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    src = VideoSource(cfg.source)
    ok, first, t_first = src.read(timeout=10)
    if not ok:
        sys.exit("Could not read a frame from the source.")
    if src.live:
        # Webcams need ~1 s for auto-exposure; the first frames are often black.
        for _ in range(45):
            ok2, f2, t2 = src.read(timeout=1)
            if ok2:
                first, t_first = f2, t2
        if first.mean() < 3:
            print("[camera] the image is still black - close other apps using the camera, "
                  "check the privacy shutter/key, or try --source 1")
    zone = setup_zone(cfg, first, run_dir)

    from gate_guard.detector import DogDetector      # imported late: slow import
    detector = DogDetector(cfg)
    book, engine = TrackBook(cfg), DecisionEngine(cfg)
    gate_anim = None if cfg.no_gate_anim else GateAnimator()
    gate, speaker = make_gate(cfg), Announcer(not cfg.no_sound, cfg.announce_cooldown)
    log = EventLogger(run_dir, t0=t_first)
    clips = ClipRecorder(run_dir / "clips", cfg.clip_pre, cfg.clip_post) if cfg.save_clips else None
    writer = None
    stats = {"dogs_seen": 0, "warnings": 0, "entries": 0, "gate_closures": 0}
    fps, frames, t_start = 0.0, 0, time.time()
    pending, paused = (ok, first, t_first), False
    print(f"[run] output folder: {run_dir}")

    try:
        while True:
            if pending:
                ok, frame, now = pending
                pending = None
            else:
                ok, frame, now = src.read()
            if not ok:
                if not src.live:
                    break                          # end of video file
                if not cfg.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue                           # live: wait / reconnect

            t_loop = time.time()
            h, w = frame.shape[:2]
            dogs, persons = detector.track(frame)
            active = book.update(dogs, persons, zone, (w, h), now)
            decision = engine.evaluate(active, now, w)
            stats["dogs_seen"] = len(book.seen_ids)
            stats["raw_track_ids"] = len(book.merger.map)      # before merging (shows ID switches)

            if gate_anim is not None:
                gate_anim.update(decision.state, now)
            overlay.draw(frame, zone, active, persons, decision.state, fps, stats, now, gate_anim)
            st = decision.state.value
            for t in decision.new_warnings:
                stats["warnings"] += 1
                log.log("DOG_IN_ZONE" if t.in_zone else "DOG_APPROACHING", now, t, st)
            for t in decision.new_entries:
                stats["entries"] += 1
                log.log("UNAUTHORIZED_ENTRY", now, t, st, frame, snapshot=True)
                if clips:
                    clips.trigger(f"entry_{t.track_id}_{datetime.now():%H%M%S}", fps or 15)
            if decision.changed:
                if decision.state == GateState.CLOSED:
                    stats["gate_closures"] += 1
                    gate.close()
                    speaker.say("Attention. Dog detected at the gate. Gate closed.")
                    log.log("GATE_CLOSE", now, None, st)
                elif decision.previous == GateState.CLOSED:
                    gate.open()
                    log.log("GATE_OPEN", now, None, st)

            if clips:
                clips.push(frame)
            if cfg.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(run_dir / "output.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             src.fps, (w, h))
                writer.write(frame)

            frames += 1
            inst = 1.0 / max(1e-6, time.time() - t_loop)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            if not cfg.no_display:
                cv2.imshow("Stray Dog Gate Guard", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    cv2.imwrite(str(run_dir / f"manual_{frames}.jpg"), frame)
                if key == ord("z"):
                    cfg.select_zone = True
                    new_pts = select_zone_interactive(frame)
                    if new_pts:
                        zone = GateZone(new_pts)
                        zone.save(cfg.zone_file)
                if key == ord("p"):
                    while cv2.waitKey(50) & 0xFF != ord("p"):
                        pass
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        elapsed = max(1e-6, time.time() - t_start)
        stats.update(frames=frames, avg_fps=round(frames / elapsed, 1), source=str(cfg.source),
                     model=cfg.model)
        (run_dir / "summary.json").write_text(json.dumps(stats, indent=2))
        print("[summary]", stats)
        if clips:
            clips.close()
        if writer:
            writer.release()
        log.close()
        gate.shutdown()
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
