# ACVAS — Ambient Context Volume Adaptation System

## Overview

ACVAS is a real-time system that **listens to ambient audio** via the laptop microphone, **classifies the environment** (silent, office, home, crowded) using Google's YAMNet deep learning model, and **automatically adjusts system volume** to match the context. A phone-accessible web dashboard provides live monitoring and manual volume override via WebSocket.

### Architecture Summary

![Architecture Diagram](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas_final_architecture.png)

The system runs on a **laptop** (Python 3.10+ / asyncio) and serves a **phone browser** dashboard (vanilla JS). The data flow is:

```
Mic → Audio Capture → YAMNet Inference → Context Classifier → Hysteresis Filter → Volume Actuator
                                                    ↓                    ↓
                                              WebSocket Server → Phone Dashboard
                                                    ↓
                                              Event Logger (CSV)
```

---

## Proposed Changes

All files are **new** — this is a greenfield implementation. The project root is `c:\Users\shiva\Personal\Projects\ACVAS\acvas\`.

---

### Configuration & Dependencies

#### [NEW] [config.yaml](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/config.yaml)
Central configuration file. All tunable parameters live here — no magic numbers in code.

| Parameter | Value | Purpose |
|---|---|---|
| `sample_rate` | 16000 | PyAudio sampling rate (YAMNet expects 16kHz) |
| `chunk_duration_sec` | 1 | Audio chunk length in seconds |
| `ws_port` | 8765 | WebSocket server port |
| `http_port` | 8766 | HTTP server port for dashboard |
| `hysteresis_frames` | 5 | Sliding window size for hysteresis filter |
| `hysteresis_majority` | 3 | Minimum votes to confirm environment change |
| `min_hold_seconds` | 3 | Cooldown between environment transitions |
| `confidence_threshold` | 0.15 | Minimum YAMNet confidence to classify |
| `volume_ramp_step` | 0.05 | Volume change increment (5%) |
| `ramp_interval_ms` | 50 | Delay between volume ramp steps |

**Environments block** maps labels → target volume + YAMNet class indices:
- `silent` → volume 0.0, indices [0, 9] (Silence, Speech)
- `office` → volume 0.30, indices [1, 2, 3, 132] (Speech, keyboard, etc.)
- `home` → volume 0.75, indices [137, 138, 139, 140] (domestic sounds)
- `crowded` → volume 1.0, indices [5, 6, 40, 396] (crowd, chatter, music)

#### [NEW] [requirements.txt](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/requirements.txt)
Dependencies: `pyaudio`, `numpy`, `tensorflow`, `tensorflow-hub`, `websockets`, `pyyaml`, `pycaw` (Windows), `comtypes` (Windows).

---

### M1 — Audio Capture

#### [NEW] [capture.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/capture.py)

**Responsibility**: Continuously capture 1-second audio chunks from the laptop microphone.

**Implementation**:
- Single async function: `async def run_capture(audio_queue: asyncio.Queue, config: dict)`
- Opens a PyAudio stream: `rate=16000`, `channels=1`, `format=paInt16`, `frames_per_buffer=16000`
- Infinite loop reads one chunk per iteration
- Converts raw bytes → `numpy.float32`, normalizes by dividing by `32768.0` to get [-1.0, 1.0] range
- Puts waveform into `audio_queue` — if queue is full (`maxsize=5`), drops the chunk silently (no blocking)

> [!NOTE]
> PyAudio's `stream.read()` is a blocking call. Since it runs in an async function, we'll use `asyncio.get_event_loop().run_in_executor(None, stream.read, chunk_size)` to avoid blocking the event loop.

---

### M2 — YAMNet Inference

#### [NEW] [inference.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/inference.py)

**Responsibility**: Load the YAMNet model and run inference on audio chunks to identify dominant sound class.

**Implementation**:
- Module-level global: `model = None`
- `def load_model()`: Calls `hub.load("https://tfhub.dev/google/yamnet/1")`, assigns to `model`. This is a **blocking** call that downloads ~20MB on first run, so it runs once at startup before the asyncio loop.
- `def run_yamnet(waveform: np.ndarray) -> tuple[int, float]`:
  - Calls `model(waveform)` → returns `(scores, embeddings, spectrogram)`
  - `mean_scores = np.mean(scores.numpy(), axis=0)` → shape `(521,)` — averages across time frames
  - Returns `(int(np.argmax(mean_scores)), float(np.max(mean_scores)))` — the top class index and its confidence

> [!IMPORTANT]
> YAMNet inference is CPU-bound (~50-200ms per chunk depending on hardware). It will be dispatched via `run_in_executor` from `main.py` to avoid blocking the event loop.

---

### M3 — Context Classifier

#### [NEW] [classifier.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/classifier.py)

**Responsibility**: Map YAMNet class indices to environment labels (silent/office/home/crowded).

**Implementation**:
- Hardcoded `YAMNET_ENV_MAP` dict mirroring the `environments` block in `config.yaml` for quick index → env lookup
- `def classify(top_idx: int, score: float, config: dict) -> tuple[str, float]`:
  - If `score < config["confidence_threshold"]` → returns `("unknown", score)`
  - Iterates over `config["environments"]`, checks if `top_idx` is in any env's `yamnet_indices`
  - Returns `(env_label, score)` on first match, otherwise `("unknown", score)`

---

### M4 — Hysteresis Filter

#### [NEW] [hysteresis.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/hysteresis.py)

**Responsibility**: Prevent rapid environment switching — require sustained detection before committing to a transition.

**Implementation**:
- `class HysteresisFilter`:
  - `__init__(self, config)`: Creates `collections.deque(maxlen=config["hysteresis_frames"])`, stores config, initializes `self.last_change_time = 0` and `self.last_confirmed = None`
  - `def update(self, env_label: str) -> str | None`:
    - Appends `env_label` to deque
    - Counts most common label using `collections.Counter`
    - **Three conditions must ALL be true** to confirm a transition:
      1. Count ≥ `config["hysteresis_majority"]` (e.g., 3 out of 5)
      2. `time.time() - self.last_change_time >= config["min_hold_seconds"]` (cooldown elapsed)
      3. Label differs from `self.last_confirmed` (actually a *new* environment)
    - If all true: updates timestamps, returns the new label
    - Otherwise: returns `None` (no transition)

---

### M5 — Volume Actuator

#### [NEW] [actuator.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/actuator.py)

**Responsibility**: Smoothly ramp system volume to a target level.

**Implementation**:
- `def get_current_volume() -> float`: Returns current system volume as `0.0–1.0`
- `def set_volume(target: float)`: Ramps from current to target in 5% steps with 50ms sleep between steps
- **OS detection** via `sys.platform == "win32"`:
  - **Windows**: Uses `pycaw` — `AudioUtilities.GetSpeakers()`, casts to `IAudioEndpointVolume`, calls `SetMasterVolumeLevelScalar(level, None)`
  - **Linux**: Uses `subprocess.run(["amixer", "set", "Master", f"{int(target*100)}%"])`

> [!NOTE]
> The ramping loop uses `time.sleep()` (blocking) which is acceptable since `set_volume` is always dispatched via `run_in_executor` from the main pipeline.

---

### M6 — WebSocket Server & HTTP Server

#### [NEW] [server.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/server.py)

**Responsibility**: Serve the phone dashboard over HTTP and maintain real-time WebSocket communication.

**Implementation**:
- `CLIENTS = set()` — global set of connected WebSocket objects
- `async def broadcast(message: dict)`: JSON-encodes and sends to all clients, silently removes disconnected ones
- `async def ws_handler(websocket)`: 
  - Adds client to `CLIENTS` on connect, removes on disconnect
  - Listens for incoming messages — if `msg["type"] == "override"`, calls `actuator.set_volume(msg["volume"])` via `run_in_executor`
- `async def serve_http()`: Uses `http.server.SimpleHTTPRequestHandler` serving the `static/` directory on `config["http_port"]` (8766)
- `async def start_server(config)`: Starts `websockets.serve(ws_handler, "0.0.0.0", config["ws_port"])`

---

### M7 — Event Logger

#### [NEW] [logger.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/logger.py)

**Responsibility**: Append environment transition events to a CSV log file.

**Implementation**:
- `def log_event(env: str, confidence: float, volume: float)`:
  - Appends one row to `acvas_log.csv`
  - Columns: `timestamp` (ISO format), `env`, `confidence` (2 decimal places), `volume` (2 decimal places)
  - Creates file with header row on first call if it doesn't exist

---

### Orchestrator

#### [NEW] [main.py](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/main.py)

**Responsibility**: Wire all modules together, manage the async event loop.

**Implementation**:
1. Load `config.yaml` with `pyyaml`
2. Call `inference.load_model()` — **blocking**, runs before the event loop starts
3. Create three async queues:
   - `audio_queue = asyncio.Queue(maxsize=5)` — raw waveforms
   - `event_queue = asyncio.Queue(maxsize=5)` — (top_idx, score) tuples
   - `broadcast_queue = asyncio.Queue(maxsize=10)` — broadcast payloads
4. Launch **5 concurrent coroutines** with `asyncio.gather()`:

| Coroutine | Reads From | Writes To | Description |
|---|---|---|---|
| `capture.run_capture()` | Microphone | `audio_queue` | Captures 1s audio chunks |
| `inference_loop()` | `audio_queue` | `event_queue` | Runs YAMNet via `run_in_executor` |
| `pipeline_loop()` | `event_queue` | `broadcast_queue` | Classifier → Hysteresis → Actuator → Logger |
| `broadcast_loop()` | `broadcast_queue` | WebSocket clients | Sends JSON updates to phone |
| `server.start_server()` | WebSocket clients | — | Accepts connections + overrides |

---

### Phone Dashboard

#### [NEW] [index.html](file:///c:/Users/shiva/Personal/Projects/ACVAS/acvas/static/index.html)

**Responsibility**: Real-time monitoring dashboard accessible from any phone on the same WiFi network.

**Implementation** — single self-contained HTML file, no frameworks, no npm:
- **WebSocket client**: Connects to `ws://<SERVER_IP>:8765` where `SERVER_IP = window.location.hostname`
- **Auto-reconnect**: On WebSocket `onclose`, waits 3 seconds then reconnects
- **UI Elements**:
  - **Environment display**: Large label with colour badge + confidence percentage
  - **Volume bar**: Animated 0–100% bar with CSS transition (500ms)
  - **Event log**: Last 20 events showing time · env · confidence · volume
  - **Manual override**: Range input (0–100) + button, sends `{type: "override", volume: slider.value/100}` via WebSocket
- **Colour mapping**: silent=green, office=blue, home=amber, crowded=red, unknown=gray

---

## Open Questions

> [!IMPORTANT]
> **Python environment**: Should I create a virtual environment (`venv`) inside the project and install dependencies, or do you have a preferred environment setup? Note that `pyaudio` may need a system-level package (`portaudio`) on some platforms.

> [!IMPORTANT]
> **TensorFlow variant**: The standard `tensorflow` package is ~500MB+. Would you prefer `tensorflow-cpu` instead (smaller, sufficient since YAMNet doesn't need GPU)? Or do you have TensorFlow already installed?

> [!NOTE]
> **Config typo**: The `neccessary_details.txt` file shows `yamlsample_rate: 16000` on line 33 (missing newline after `yaml`). I'll treat this as `sample_rate: 16000` being the first key in `config.yaml`. Please confirm this is correct.

---

## Execution Order

I'll implement files in dependency order — leaves first, orchestrator last:

| Phase | Files | Rationale |
|---|---|---|
| **1** | `config.yaml`, `requirements.txt` | Foundation — no dependencies |
| **2** | `logger.py` | Standalone utility, no internal deps |
| **3** | `actuator.py` | Standalone, OS-level interaction |
| **4** | `inference.py` | Depends only on TF Hub (external) |
| **5** | `classifier.py` | Depends on config structure only |
| **6** | `hysteresis.py` | Depends on config structure only |
| **7** | `capture.py` | Depends on asyncio + config |
| **8** | `server.py` | Depends on `actuator.py` |
| **9** | `static/index.html` | Depends on server message format |
| **10** | `main.py` | Orchestrator — depends on everything |

---

## Verification Plan

### Manual Verification
1. **Syntax check**: Run `python -m py_compile <file>` on each `.py` file to confirm no syntax errors
2. **Import check**: Run `python -c "import capture, inference, classifier, hysteresis, actuator, server, logger"` from the `acvas/` directory
3. **End-to-end**: Run `python main.py` and verify:
   - YAMNet model loads without errors
   - Audio capture starts (microphone access granted)
   - Dashboard accessible at `http://localhost:8766` from a browser
   - WebSocket connection established from dashboard
   - Environment labels and volume changes appear on the dashboard
