# Stray Dog Gate Guard (v2)

Improved version of [rydercalmdown/dog_detector](https://github.com/rydercalmdown/dog_detector)
(YOLOv3 + COCO alert script) rebuilt for the project
"Modern Automatic Gate System to Restrict Stray Dog Entries".

    Camera -> YOLO26n detection -> tracking (ByteTrack) -> gate-zone analysis -> gate decision

## What changed from the original
| Original | This version |
|---|---|
| YOLOv3 via OpenCV DNN, manual weights/cfg files | YOLO26n via Ultralytics (auto-downloads weights, uses Apple M1 GPU `mps`) |
| Detection only, no memory between frames | Tracking with persistent dog IDs |
| Alert for any dog anywhere in the frame | Restricted gate **zone** (polygon you draw), dwell time, approach direction |
| `os.system('say ...')` froze the video | Non-blocking speech + cooldown |
| No logs | CSV event log, snapshots, optional video clips, summary.json |
| Hard-coded settings | Every setting is a command-line flag (`config.py`) |
| Webcam / file only, crashes if stream drops | Webcam, file, RTSP/CCTV with auto-reconnect |
| - | Simulated gate + optional serial output (Arduino/ESP32) |
| - | Tests, fine-tuning script (`train.py`) |

## Setup (macOS)
    python3 -m venv env && source env/bin/activate
    pip install -U -r requirements.txt

## Run
    python main.py --source 0 --select-zone            # webcam; draw the zone first
    python main.py --source videos/gate.mp4 --save-video
    python main.py --source rtsp://user:pass@ip/stream --save-clips
    python main.py --model runs_train/dog_yolo26n/weights/best.pt   # your fine-tuned model

Zone drawing: left-click corners, right-click undo, `c` clear, Enter save, Esc cancel.
The zone is saved to `zone.json` and reused next time.
Keys while running: `q`/Esc quit, `p` pause, `z` redraw zone, `s` snapshot.

Outputs go to `runs/<timestamp>/`: `events.csv`, `snapshots/`, `clips/`, `output.mp4`, `summary.json`.

## Decision logic
- **NORMAL** (gate open): no dog threat.
- **WARNING**: a dog is moving toward the gate and is within `--warn-margin` of the zone, or is inside the zone for less than `--dwell-seconds`.
- **CLOSED**: a dog stays inside the zone for `--dwell-seconds` (default 1.5 s) -> `UNAUTHORIZED_ENTRY` event, snapshot, spoken alert, gate close.
  The gate reopens after the area has been clear for `--clear-seconds`.
- "Inside the zone" uses the bottom-centre of the dog's box (its feet).
- `--ignore-accompanied` ignores dogs with a person next to them (a simple stray heuristic).
  It is off by default, which matches your slide: any dog entering the zone is a potential unauthorized entry.
  It needs a model that detects `person` (the COCO model does; a dog-only fine-tuned model does not).

## Real gate hardware (future scope)
`python main.py --gate serial --serial-port /dev/cu.usbmodemXXXX` (needs `pip install pyserial`)
sends `CLOSE\n` / `OPEN\n` over USB serial. Write the Arduino/ESP32 side to drive the motor or relay.

## Fine-tuning
    python train.py --data final_dataset --epochs 50
Note: the COCO-pretrained YOLO26n already detects dogs and was trained on COCO, so fine-tuning on a COCO
subset may not improve it. It pays off with campus-specific images (your gate, camera angle, lighting, night).
Compare mAP of the pretrained model and the fine-tuned one on your own test clips.

## Tests
    python tests/test_logic.py        # zone, tracking, state machine - no camera or GPU needed

## Credits
Original idea and base: https://github.com/rydercalmdown/dog_detector (check its license before redistributing).
