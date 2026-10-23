"""EXPERIMENTAL: find the gate automatically with an open-vocabulary detector (YOLO-World).
The zone = the gate box + a strip of ground in front of it.
The first run downloads the model (and may install CLIP, which needs git)."""


def find_gate_zone(frame, prompts=("gate", "door", "entrance"),
                   model_name="yolov8s-worldv2.pt", conf=0.05):
    from ultralytics import YOLOWorld
    model = YOLOWorld(model_name)
    model.set_classes(list(prompts))
    res = model.predict(frame, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    best = int(res.boxes.conf.argmax())
    x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[best].tolist())
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    zx1, zx2 = max(0.0, x1 - 0.15 * bw), min(float(w), x2 + 0.15 * bw)
    zy1, zy2 = y1, min(float(h), y2 + 0.35 * bh)
    return [(zx1 / w, zy1 / h), (zx2 / w, zy1 / h), (zx2 / w, zy2 / h), (zx1 / w, zy2 / h)]
