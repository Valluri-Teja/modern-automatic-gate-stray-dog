#!/usr/bin/env python3
"""Fine-tune YOLO26n on your dog dataset (the 'next step' from Review 1).

  python train.py --data final_dataset --epochs 50
Expected layout:  final_dataset/images/{train,val}  +  final_dataset/labels/{train,val}
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # M1: allow CPU fallback for rare ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="final_dataset", help="dataset root folder")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    from ultralytics import YOLO
    from gate_guard.detector import pick_device

    root = Path(a.data).resolve()
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: dog\n")
        print(f"[train] wrote {yaml_path}")

    model = YOLO(a.model)
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=pick_device(a.device), project="runs_train", name="dog_yolo26n")
    m = model.val()
    print(f"[train] mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
    print("[train] weights: runs_train/dog_yolo26n/weights/best.pt")


if __name__ == "__main__":
    main()
