"""Evaluation maths (pure numpy, no camera / YOLO needed)."""
import numpy as np


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5, score_thr=0.0):
    """Greedy matching, highest score first.
    Returns (scores, tp_flags, n_missed_gt) for the predictions with score >= score_thr."""
    scores = np.asarray(pred_scores, dtype=float)
    keep = [int(i) for i in np.argsort(-scores) if scores[i] >= score_thr]
    boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gts = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ious = iou_matrix(boxes[keep], gts)
    used, flags = set(), []
    for row in range(len(keep)):
        best_j, best_iou = -1, iou_thr
        for j in range(len(gts)):
            if j not in used and ious[row, j] >= best_iou:
                best_j, best_iou = j, ious[row, j]
        if best_j >= 0:
            used.add(best_j)
        flags.append(best_j >= 0)
    return [float(scores[i]) for i in keep], flags, len(gts) - len(used)


def average_precision(scores, tp_flags, n_gt):
    """AP at a fixed IoU (all-point interpolation)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(tp_flags, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_event_times(pred_times, gt_times, before=1.0, after=6.0):
    """Match predicted gate-close times to ground-truth entry times.
    A prediction p matches a truth g when  -before <= p - g <= after  (seconds).
    Returns (matches [(g, p)], missed_gt [g], false_preds [p])."""
    preds = sorted(float(p) for p in pred_times)
    used, matches, missed = set(), [], []
    for g in sorted(float(x) for x in gt_times):
        hit = None
        for k, p in enumerate(preds):
            if k not in used and -before <= p - g <= after:
                hit = k
                break
        if hit is None:
            missed.append(g)
        else:
            used.add(hit)
            matches.append((g, preds[hit]))
    false_preds = [p for k, p in enumerate(preds) if k not in used]
    return matches, missed, false_preds
