# Phase 4 — Optional Advanced Pipelines

Branch: `phase-4-advanced-pipelines`.

### Done (foundation + enhancement spike)

- **P4-000…003** Extras registry, proposals, Setup panel, extras requirements scaffold
- **P4-030** Enhancement proposal **Approved** (Rust `deep-filter` path)
- **P4-060 spike** `vocalforge/enhancement.py` — 16→48→enhance→16 via soxr + CLI  
  Metrics: ~0.6–0.7 s / 3 s clip, ~57 MB child RSS, 0 VRAM  
  Report: `tests/fixtures/enhancement/out/spike_report.json`

### Next

1. **P4-020 sign-off** — read `docs/proposals/diarization.md` and approve/reject conditions
2. If approved → bounded spike: HF token Setup UX + PyAnnote diarize + merge onto Faster-Whisper
3. Other tracks still gated: P4-010 / 040
4. P4-050 resource rules as more extras land

**Enhancement:** home **Enhance** On/Off (`enhance_audio`). Requires `.deps/deep-filter` + `soxr`.

| Track | Topic | Status |
|-------|--------|--------|
| P4-010 | Forced word alignment | Stub |
| P4-020 | Speaker diarization | Proposal ready — await sign-off |
| P4-030 | Speech enhancement | Shipped (Ready) |
| P4-040 | Source separation | Stub |

See `docs/proposals/enhancement.md` and `plan/01-Upgrade/task_breakdown_phase_4.md`.
