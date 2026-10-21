"""YOLO26 detection + built-in tracking (Ultralytics).
Class ids are looked up by NAME ('dog', 'person'), so it works with both the
COCO-pretrained model (dog=16) and your fine-tuned single-class model (dog=0)."""
from .tracking import Detection


def pick_device(pref):
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class DogDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.model = YOLO(cfg.model)
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.dog_ids = [k for k, v in names.items() if v == "dog"]
        self.person_ids = [k for k, v in names.items() if v == "person"]
        if not self.dog_ids:
            raise RuntimeError(f"Model {cfg.model} has no 'dog' class: {names}")
        self.names = names
        self.classes = self.dog_ids + self.person_ids
        print(f"[detector] {cfg.model} on {self.device} | dog ids={self.dog_ids} "
              f"person ids={self.person_ids or 'none (accompanied-dog check disabled)'}")

    def track(self, frame):
        """Returns (dogs: list[Detection], person_boxes: list[tuple])."""
        c = self.cfg
        res = self.model.track(frame, persist=True, tracker=c.tracker, conf=c.conf,
                               iou=c.iou, imgsz=c.imgsz, device=self.device,
                               classes=self.classes, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return [], []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else [-1] * len(cls)
        dogs, persons = [], []
        for b, k, p, i in zip(xyxy, cls, conf, ids):
            box = tuple(float(v) for v in b)
            if int(k) in self.dog_ids:
                dogs.append(Detection(int(i), box, float(p)))
            else:
                persons.append(box)
        return dogs, persons
