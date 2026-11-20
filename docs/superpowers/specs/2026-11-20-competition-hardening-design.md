# Competition Hardening Design

## Goal

Upgrade the competition project from a feature-rich detection system into a
demonstrable, evidence-backed, privacy-aware care system. The work covers four
required tracks:

1. Algorithm evaluation evidence center.
2. One-click competition demo reliability kit.
3. Multimodal risk fusion and active emergency closure.
4. Privacy, security, and trustworthiness.

The competition story becomes: measurable detection, reliable offline-capable
demonstration, active response after an alarm, and responsible handling of
elderly-care data.

## Current Baseline

The repository already contains:

- A temporal, per-person fall state machine in `deidao/fall_engine.py`.
- Regression coverage for fast falls, intentional lying, sitting, frame-rate
  differences, missing pose, and person tracking.
- A real Huawei Band 9 path through Gadgetbridge SQLite and MQTT, with an
  explicit simulator fallback for demos.
- A two-stage emotion pipeline and a reproducible emotion-model benchmark.
- Feishu push, health dashboard, AI chat, local/cloud LLM switching, and TTS.
- A 14-page competition deck targeting the 2026 Henan vocational competition.

The main gaps are quantitative reporting, failure-proof live demonstration, post-alarm
response closure, and privacy/security hardening.

## Scope

### In Scope

#### 1. Evaluation Evidence Center

Add an offline evaluation tool that runs labeled videos through the same fall
decision logic used by the application and produces judge-ready reports.

Dataset layout:

```text
test_videos/evaluation/
  fall/
  sit_down/
  bend_down/
  lie_down/
  walking/
```

A JSON manifest may supplement filenames when exact scene labels, action start
times, or expected alarm windows differ from the default.

The evaluator reports:

- Per-scene true-positive, false-positive, and false-negative counts.
- Recall, precision, false-alarm rate, and missed-alarm rate.
- Average alarm latency from expected action start.
- Alarm count and triggered feature reasons per video.
- Processing FPS and optional CPU/GPU device information.
- Markdown and JSON outputs under `output/evaluation/fall/`.

The emotion benchmark remains in `qingxu`. A small report aggregator combines
fall and emotion results into one Markdown evidence file for the deck.

#### 2. Demo Reliability Kit

Add a preflight and demo launcher. It checks:

- Required Python packages and versions.
- YOLO, MediaPipe, ViT/HSEmotion, and Vosk artifacts.
- Camera readability.
- MySQL availability.
- MQTT/Broker reachability.
- Audio output, TTS, and alarm sound.
- Kimi/Ollama configuration.
- Feishu webhook configuration.
- Backup video presence.

Profiles:

- `standard`: full system with all configured services.
- `offline`: local video, local models, no cloud LLM or external webhook.
- `no_band`: simulator-backed health dashboard and normal vision detection.

The tool reports blocking issues, degraded capabilities, and the selected demo
path. It never labels simulated health values as real device data.

#### 3. Risk Engine and Emergency Closure

Add a Qt-free `RiskEngine` that converts independent signals into one care
decision.

Inputs:

- Fall decision and confidence from the temporal fall engine.
- Heart rate, SpO2, blood-pressure estimate, temperature, and battery from the
  health adapter.
- Current visual emotion and recent emotion stability.
- Time context such as night hours.
- User response state.

Outputs:

- Level: `LOW`, `WATCH`, `HIGH`, or `EMERGENCY`.
- Reasons: machine-readable feature names plus Chinese display text.
- Recommended action.
- Response deadline for active escalation.

The engine must never suppress an already confirmed fall merely to reduce
false alarms. Its role is tiering and escalation; a confirmed fall remains at
least `HIGH`.

Emergency flow:

1. A `HIGH` or `EMERGENCY` event triggers local voice inquiry.
2. The UI exposes an immediate “我没事” response.
3. If Vosk is available, a small affirmative-response phrase list is accepted.
4. If no response arrives before the deadline, the event escalates.
5. Escalation sends the existing Feishu notification with clear response
   instructions.
6. Family or admin marks the event as confirmed, false alarm, waiting for
   help, or processed.
7. Every state transition is persisted.

#### 4. Privacy and Security

Authentication:

- Replace plaintext/default database passwords with a salted PBKDF2 hash.
- Migrate existing local accounts only when their current password verifies.
- Keep a temporary migration-safe login path clearly marked in code and remove
  it after local databases are upgraded.

Audit and storage:

- Add an audit log for login, failed login, settings changes, exports, alarm
  processing, and privacy-mode changes.
- Add an `emergency_events` table for the complete escalation timeline.
- Add configurable retention for videos and exported health data.
- Add a one-click cleanup action for expired artifacts.

Video privacy:

- Add privacy modes: `off`, `face_blur`, `body_blur`, and `skeleton`.
- `face_blur` is the default mode for newly saved alarm recordings.
- Alarm videos honor the mode before writing to `result/`.
- The UI states the active privacy mode.
- Tests verify that face regions cannot be reconstructed from the default
  blurred output and that transform failures prevent an unsafe save when a
  privacy mode other than `off` is selected.

Secrets:

- API keys, webhooks, and database credentials are read from environment
  variables or the existing local `.env` mechanism.
- No new secret is written to a tracked configuration file.

### Out of Scope

- WebRTC automatic outbound calls.
- Millimeter-wave radar integration.
- LSTM/GRU fall classification.
- Cloud RDS migration.
- Multi-camera concurrency.
- Medical diagnosis or certified blood-pressure claims.
- Daily/weekly caregiver reports; this may be added later only if time remains.

## Architecture

### New Components

```text
tools/evaluate_fall_videos.py
tools/aggregate_evaluation.py
tools/demo_preflight.py
deidao/risk_engine.py
deidao/emergency_controller.py
deidao/privacy.py
```

`risk_engine.py`, `emergency_controller.py`, and privacy transforms stay free
of Qt imports so their state transitions can be tested without a window.

### Existing Components Extended

- `deidao/detector.py`: emits structured fall events and privacy-safe video
  frames.
- `deidao/dialogs.py`: renders risk level, response controls, privacy mode,
  and preflight status.
- `deidao/database.py`: owns schema migration and audit/emergency persistence.
- `deidao/device_adapter.py`: continues to expose real and simulated data
  without mixing their provenance.
- `tools/gadgetbridge_mqtt_bridge.py`: remains the real-device bridge.

### Data Flow

```text
Camera/video
  -> temporal fall engine
  -> RiskEngine
       <- health adapter
       <- emotion pipeline
       <- time/response context
  -> EmergencyController
       -> local voice inquiry
       -> response deadline / family action
       -> Feishu escalation
       -> emergency_events and audit log
  -> privacy-safe alarm recording
```

Evaluation runs outside Qt:

```text
labeled videos -> detector/fall engine -> event matcher
               -> metrics -> JSON/Markdown evidence report
```

## Error Handling

- Evaluation reports unreadable videos and invalid labels instead of aborting
  the whole run.
- Preflight separates blocking failures from optional degraded capabilities.
- Offline mode never invokes cloud LLM or external webhooks.
- Speech recognition failure leaves the on-screen response button usable.
- Feishu failure is logged but does not stop local recording or database
  persistence.
- MQTT and simulator fallback preserve existing provenance wording.
- Privacy transform failure prevents saving an unsafe recording when a privacy
  mode other than `off` is selected.
- Database migration failure leaves the existing schema readable and reports a
  clear startup error.

## Testing

Add focused tests for:

1. Fall metric calculation from synthetic event timelines.
2. Report generation and malformed media handling.
3. Preflight dependency/model/service decisions.
4. Risk levels and machine-readable reasons.
5. Confirmed fall cannot be downgraded below `HIGH`.
6. Inquiry response cancels escalation.
7. Missing response escalates exactly once.
8. Family processing states and persistence transitions.
9. Password hashing, migration, and failed login audit.
10. Face/body blur and skeleton privacy transforms.
11. Retention cleanup only deletes intended generated artifacts.

Existing tests in `test/` must continue to pass. The final verification command
is the project's full pytest suite plus the newly added evaluator smoke test.

## Competition Deliverables

- One-command fall evaluation report.
- Combined algorithm evidence page.
- Preflight and demo profiles.
- Reliable offline/no-band demo path.
- Active emergency closure demo.
- Privacy-aware recording and authentication.
- Updated README and deck text with unified model names and non-medical
  disclaimers.

## Acceptance Criteria

- The current full test suite passes after the work.
- Fall evaluation produces JSON and Markdown reports with explicit sample
  counts and metrics.
- Offline and no-band profiles complete the core demo without cloud services.
- A fall triggers inquiry, waits for response, escalates, and records the
  family action.
- User passwords are stored as salted hashes in new or migrated databases.
- Audit rows capture security-relevant actions.
- Alarm videos honor the selected privacy mode.
- PPT model names match the implemented cloud/local providers.
- Every headline metric in the deck can be traced to a generated report.

## Implementation Order

1. Baseline audit and evidence-center tests.
2. Fall evaluator and report aggregation.
3. Demo preflight and profiles.
4. Risk engine and unit tests.
5. Emergency UI/controller and persistence.
6. Password, audit, and schema migration.
7. Video privacy transforms and retention cleanup.
8. README/deck consistency and final full verification.
