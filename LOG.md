# Development log

## 2026-09-20 03:05 EDT — Manual pair foundation and guide compaction

- Added `manual` job mode, interactive `pairing` state, server-owned source/assembly timeline fields, and revision-guarded manual pair persistence.
- Kept interactive manual/annotation jobs admitted so a second upload cannot race the active review.
- Updated automated guide generation to request and enforce one merged final step per evidence image, avoiding duplicate image-attributed instructions.
- Verified backend and web suites locally.
