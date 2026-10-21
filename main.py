#!/usr/bin/env python3
"""Stray Dog Gate Guard
Camera -> YOLO26 detection -> tracking -> gate-zone analysis -> gate decision.

  python main.py --source 0 --select-zone          # webcam, draw the zone first
  python main.py --source videos/gate.mp4 --save-video
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
from gate_guard.alerts import Announcer
from gate_guard.config import parse_args
from gate_guard.decision import DecisionEngine, GateState
from gate_guard.events import ClipRecorder, EventLogger
from gate_guard.gate_controller import make_gate
from gate_guard.tracking import TrackBook
from gate_guard.video_source import VideoSource
from gate_guard.zone import GateZone, select_zone_interactive


def setup_zone(cfg, frame):
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
    run_dir = Path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    src = VideoSource(cfg.source)
    ok, first, t_first = src.read(timeout=10)
    if not ok:
        sys.exit("Could not read a frame from the source.")
    zone = setup_zone(cfg, first)

    from gate_guard.detector import DogDetector      # imported late: slow import
    detector = DogDetector(cfg)
    book, engine = TrackBook(cfg), DecisionEngine(cfg)
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

            overlay.draw(frame, zone, active, persons, decision.state, fps, stats, now)
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
