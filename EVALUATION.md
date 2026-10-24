# Evaluation guide

Two things are measured: (1) how well dogs are **detected**, (2) how well the **gate decision** works on videos.

## 1. Dog detection (images with labels)
    python evaluate.py detect --data final_dataset --split val
Prints AP50 and precision / recall / F1 at several confidence thresholds, and saves `eval_results/detection.csv`.
Note: the 777 images come from COCO, which the pretrained YOLO26n has probably already seen in training,
so these numbers are optimistic. State this in your report, and prefer your own gate footage when you have it.

## 2. Gate events (your test videos)
1. Collect 5-10 clips (include some with **no dog** - they test false alarms).
2. Copy `eval_clips.example.csv` to `eval_clips.csv` and fill one row per clip:
   - `video`: path to the clip
   - `entry_times`: seconds when a dog really enters (the moment it should trigger the gate); separate several with `;`; leave empty for a clip with no dog
   - `dogs`: how many different dogs really appear (optional, used to check the dog count)
3. Run:
       python evaluate.py events --clips eval_clips.csv
   It runs every clip headless and prints: entries caught / missed, false gate closes, delay from entry to gate close, and the dog count vs the truth.
   A gate close counts as correct if it happens between 1 s before and 6 s after the true entry (change with `--before` / `--after`).
4. Show the effect of ID merging (before / after):
       python evaluate.py events --clips eval_clips.csv --tag nomerge --args "--no-merge"
   Compare `eval_results/events_merge_summary.md` with `events_nomerge_summary.md`.

With the default whole-frame zone, "entry" means the dog is clearly visible; the gate closes after `--dwell-seconds` (1.5 s),
so a delay of about 2 s is expected. Use `--zone-mode manual` for a real gate zone.
Results are saved in `eval_results/`.

## How event counting works now
- A dog only counts after it has been tracked for `--min-hits` frames (default 5), which removes one-frame flicker.
- If the tracker gives a dog a new ID after a turn or short occlusion, the new ID is merged into the lost dog when it appears
  within `--merge-window` seconds (2.5) and `--merge-distance` x dog size (1.0) of where that dog was last seen.
  Use `--no-merge` to switch this off.
- `summary.json` now shows `raw_track_ids` (before merging) next to `dogs_seen` (after merging).
- `entries` counts distinct dogs that entered; `gate_closures` counts gate-close incidents.
- Limit: two different dogs that swap places within the merge window can be merged wrongly.
