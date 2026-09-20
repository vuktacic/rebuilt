# Rebuilt API contract

The backend exposes the following JSON contract. `POST /jobs` accepts one
multipart field named `video` and returns `202 {"jobId":"..."}`. Clients poll
`GET /jobs/{jobId}` until the status is `ready` or `failed`, then use the guide
and supplied frame URLs. A complete guide can be saved with
`PUT /jobs/{jobId}/guide`.

`status` is one of `queued`, `extracting`, `annotating`, `pairing`, `analyzing`,
`generating`, `ready`, or `failed`. New jobs may select `mode: manual`; after
extraction they pause at `pairing` with server-owned `sourceIndex` and
`assemblyTimeSeconds` metadata on each frame. Manual pair saves use a monotonic
`revision`, stable pair IDs, and semantic before/after frame IDs. Historic jobs
without these fields default to the automated workflow. During automated analysis, `tracks` and `events` are populated when local
SAM3 finds stable attachment or detachment evidence. Each event references a
stable before/after frame pair and may carry an uncertainty reason. Frame IDs
are stable for the lifetime of a job. A guide step must reference an existing
selected frame and contain non-empty text. Error responses use
`{"error":{"code":"...","message":"..."}}`.

Manual review runs use `diffing` and persist one `manualReview.pairs` finding
per saved pair. Failed pairs remain retryable and never become guide steps.
Successful findings are handed to the text-only Luna drafting boundary; the
draft request contains compact differences and allowlisted frame IDs, but no
images, video bytes, or local filesystem paths. Findings that share an AFTER
frame are compacted into one final guide step.

The machine-readable schema is in [`job.schema.json`](job.schema.json). The
fixtures are intentionally small and are used by backend and frontend tests.
