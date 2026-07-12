# Proposal: \<Capability name\>

- **Capability id:** (`alignment` | `diarization` | `enhancement` | `separation`)
- **Author / date:**
- **Decision:** Draft | Approved | Rejected
- **Candidate engines:**

## 1. User workflow

What concrete user job requires this? What does success look like in the UI?

## 2. Benchmarks

- Datasets / fixtures
- Accuracy metrics
- Latency / throughput on target CPU and (if any) GPU
- Comparison vs current Faster-Whisper-only path

## 3. Dependencies and licensing

- Python packages and versions
- Model licenses and redistribution terms
- Auth / HF token / click-through requirements (esp. PyAnnote)

## 4. Hardware measurements

| Machine | RAM peak | VRAM peak | Disk for models | Notes |
|---------|----------|-----------|-----------------|-------|
|         |          |           |                 |       |

## 5. Packaging

- How users opt in (`pip install -r requirements-extras.txt` subgroup?)
- What stays out of the base install
- Failure mode when the extra is missing

## 6. Local model acquisition

- Download URLs / cache dirs
- Offline behavior
- Disk-space checks before download

## 7. Resource interaction with ASR

- Can this run in-process with Faster-Whisper loaded?
- Unload order / worker-process need?

## 8. Recommendation

Ship / defer / reject — with rationale.
