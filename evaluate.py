#!/usr/bin/env python3
"""Measure how good the system is.

1) Dog DETECTION quality on labelled images (precision / recall / F1 / AP50):
     python evaluate.py detect --data final_dataset --split val

2) Gate-EVENT quality on your test videos (entries caught, false closes, delay, ID count):
     python evaluate.py events --clips eval_clips.csv
   Compare with ID merging switched off:
     python evaluate.py events --clips eval_clips.csv --tag nomerge --args "--no-merge"
"""
import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gate_guard.metrics import average_precision, match_event_times, match_image

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = ROOT / "eval_results"


# --------------------------------------------------------------------------- detection
def read_gt(label_file, w, h, gt_class):
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) == gt_class:
                xc, yc, bw, bh = (float(v) for v in parts[1:5])
                boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                              (xc + bw / 2) * w, (yc + bh / 2) * h])
    return boxes


def cmd_detect(a):
    import cv2
    from ultralytics import YOLO
    from gate_guard.detector import pick_device

    root = Path(a.data)
    img_dir, lbl_dir = root / "images" / a.split, root / "labels" / a.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT) if img_dir.exists() else []
    if not images:
        sys.exit(f"No images found in {img_dir}")
    model = YOLO(a.model)
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    dog_ids = [k for k, v in names.items() if v == "dog"]
    if not dog_ids:
        sys.exit(f"Model has no 'dog' class: {names}")
    device = pick_device(a.device)
    print(f"[detect] {len(images)} images | model {a.model} | device {device}")

    data = []
    for i, p in enumerate(images, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_gt(lbl_dir / (p.stem + ".txt"), w, h, a.gt_class)
        res = model.predict(img, conf=a.min_conf, iou=0.5, imgsz=a.imgsz, device=device,
                            classes=dog_ids, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            boxes, scores = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        else:
            boxes, scores = np.zeros((0, 4)), np.zeros(0)
        data.append((boxes, scores, gt))
        if i % 25 == 0:
            print(f"  {i}/{len(images)}")

    n_gt = sum(len(g) for _, _, g in data)
    all_scores, all_flags = [], []
    for boxes, scores, gt in data:
        s, f, _ = match_image(boxes, scores, gt, a.iou, 0.0)
        all_scores += s
        all_flags += f
    ap = average_precision(all_scores, all_flags, n_gt)

    rows = []
    for c in (0.25, 0.40, 0.50, 0.60, 0.70):
        tp = fp = fn = 0
        for boxes, scores, gt in data:
            s, f, missed = match_image(boxes, scores, gt, a.iou, c)
            tp += sum(f)
            fp += len(f) - sum(f)
            fn += missed
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((c, tp, fp, fn, prec, rec, f1))

    OUT.mkdir(exist_ok=True)
    with open(OUT / "detection.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["conf", "TP", "FP", "FN", "precision", "recall", "F1"])
        for r in rows:
            w_.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.3f}"])
    print(f"\nDog detection on {len(data)} images, {n_gt} labelled dogs (IoU >= {a.iou})")
    print(f"AP50 = {ap:.3f}")
    print(f"{'conf':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'precision':>10} {'recall':>8} {'F1':>6}")
    for c, tp, fp, fn, p_, r_, f_ in rows:
        print(f"{c:5.2f} {tp:5d} {fp:5d} {fn:5d} {p_:10.3f} {r_:8.3f} {f_:6.3f}")
    print(f"\nSaved {OUT / 'detection.csv'}")
    print("Note: if these images come from COCO, the pretrained model has probably seen them "
          "during training, so the numbers are optimistic. Say so in your report.")


# --------------------------------------------------------------------------- events
def read_close_times(events_csv):
    times = []
    with open(events_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["event"] == "GATE_CLOSE":
                times.append(float(row["video_time_s"]))
    return times


def score_clip(gt_times, close_times, before=1.0, after=6.0):
    matches, missed, false_preds = match_event_times(close_times, gt_times, before, after)
    return {"gt": len(gt_times), "pred": len(close_times), "tp": len(matches),
            "fn": len(missed), "fp": len(false_preds), "delays": [p - g for g, p in matches]}


def summarise(results):
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    delays = [d for r in results for d in r["delays"]]
    out = {"clips": len(results), "real_entries": tp + fn, "caught": tp, "missed": fn,
           "false_closes": fp,
           "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1)}
    if delays:
        out.update(delay_mean=statistics.mean(delays), delay_median=statistics.median(delays),
                   delay_max=max(delays))
    return out


def parse_times(text):
    return [float(x) for x in (text or "").replace(",", ";").split(";") if x.strip()]


def run_clip(video, out_dir, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cmd = [sys.executable, str(ROOT / "main.py"), "--source", str(video), "--no-display",
           "--no-sound", "--out-dir", str(out_dir)] + extra
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    runs = [d for d in out_dir.iterdir() if d.is_dir() and d.stat().st_mtime >= start - 1]
    if proc.returncode != 0 or not runs:
        print(proc.stdout[-800:], proc.stderr[-800:])
        raise RuntimeError(f"main.py failed on {video}")
    return max(runs, key=lambda d: d.stat().st_mtime)


def cmd_events(a):
    clips_file = Path(a.clips)
    rows = list(csv.DictReader(open(clips_file, newline="")))
    extra = shlex.split(a.args or "")
    results, table = [], []
    for row in rows:
        video = Path(row["video"])
        if not video.exists() and (clips_file.parent / video).exists():
            video = clips_file.parent / video
        print(f"[events] {video.name} ...", flush=True)
        run_dir = run_clip(video, ROOT / "eval_runs" / a.tag / video.stem, extra)
        close_times = read_close_times(run_dir / "events.csv")
        summary = json.loads((run_dir / "summary.json").read_text())
        gt = parse_times(row.get("entry_times"))
        r = score_clip(gt, close_times, a.before, a.after)
        r.update(video=video.name, dogs_seen=summary.get("dogs_seen"),
                 raw_ids=summary.get("raw_track_ids"),
                 dogs_true=int(row["dogs"]) if (row.get("dogs") or "").strip() else None)
        results.append(r)
        table.append(r)

    s = summarise(results)
    id_err = [abs(r["dogs_seen"] - r["dogs_true"]) for r in results
              if r["dogs_true"] is not None and r["dogs_seen"] is not None]
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"events_{a.tag}_per_clip.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["video", "true_entries", "gate_closes", "caught", "missed", "false_closes",
                     "delays_s", "dogs_true", "dogs_seen_after_merge", "raw_tracker_ids"])
        for r in table:
            w_.writerow([r["video"], r["gt"], r["pred"], r["tp"], r["fn"], r["fp"],
                         ";".join(f"{d:.1f}" for d in r["delays"]), r["dogs_true"], r["dogs_seen"], r["raw_ids"]])
    lines = [f"# Gate-event evaluation ({a.tag})", "",
             f"- Clips: {s['clips']}",
             f"- Real entries: {s['real_entries']}  |  caught: {s['caught']}  |  missed: {s['missed']}",
             f"- Entry recall: {s['recall']:.2f}  |  gate-close precision: {s['precision']:.2f}",
             f"- False gate closes: {s['false_closes']}"]
    if "delay_mean" in s:
        lines.append(f"- Delay entry -> gate close: mean {s['delay_mean']:.1f}s, "
                     f"median {s['delay_median']:.1f}s, max {s['delay_max']:.1f}s")
    if id_err:
        lines.append(f"- Dog-count error (mean |seen - true|): {statistics.mean(id_err):.2f}")
        lines.append(f"- Tracker IDs before merging: {sum(r['raw_ids'] or 0 for r in results)}  |  "
                     f"dogs after merging: {sum(r['dogs_seen'] or 0 for r in results)}  |  "
                     f"true dogs: {sum(r['dogs_true'] or 0 for r in results if r['dogs_true'] is not None)}")
    (OUT / f"events_{a.tag}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="dog detection precision/recall on labelled images")
    d.add_argument("--data", default="final_dataset")
    d.add_argument("--split", default="val")
    d.add_argument("--model", default="yolo26n.pt")
    d.add_argument("--gt-class", type=int, default=0, help="class id of 'dog' in your label files")
    d.add_argument("--iou", type=float, default=0.5)
    d.add_argument("--min-conf", type=float, default=0.05)
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--device", default="auto")
    e = sub.add_parser("events", help="entries caught / false closes / delay on test videos")
    e.add_argument("--clips", default="eval_clips.csv")
    e.add_argument("--tag", default="merge", help="name for this run (e.g. merge / nomerge)")
    e.add_argument("--args", default="", help='extra main.py flags, e.g. "--no-merge --conf 0.5"')
    e.add_argument("--before", type=float, default=1.0, help="a close may come this early (s)")
    e.add_argument("--after", type=float, default=6.0, help="... or this late after the true entry (s)")
    a = ap.parse_args()
    {"detect": cmd_detect, "events": cmd_events}[a.cmd](a)


if __name__ == "__main__":
    main()
