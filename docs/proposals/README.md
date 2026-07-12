# Optional extras — evaluation proposals

Advanced pipelines are **not** part of the base VocalForge install.

Before implementing an engine:

1. Copy `_TEMPLATE.md` to a capability file (see stubs below).
2. Complete every section of the template.
3. Get explicit approval to implement (task **P4-060**).
4. Wire the capability through `vocalforge/extras.py` and optional deps only.

## Capability stubs

| File | Capability | Status |
|------|------------|--------|
| `alignment.md` | Forced word alignment | Stub — fill before implement |
| `diarization.md` | Speaker diarization | Approved — P4-020 shipped |
| `enhancement.md` | Speech enhancement / denoising | Approved — P4-060 shipped |
| `separation.md` | Music / vocal source separation | Stub — fill before implement |

Base runtime: `requirements.txt`. Optional stacks: `requirements-extras.txt`.
