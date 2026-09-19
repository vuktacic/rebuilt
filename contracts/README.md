# Rebuilt API contract

The backend exposes the following JSON contract. `POST /jobs` accepts one
multipart field named `video` and returns `202 {"jobId":"..."}`. Clients poll
`GET /jobs/{jobId}` until the status is `ready` or `failed`, then use the guide
and supplied frame URLs. A complete guide can be saved with
`PUT /jobs/{jobId}/guide`.

`status` is one of `queued`, `extracting`, `generating`, `ready`, or `failed`.
Frame IDs are stable for the lifetime of a job. A guide step must reference an
existing frame and contain non-empty text. Error responses use
`{"error":{"code":"...","message":"..."}}`.

The machine-readable schema is in [`job.schema.json`](job.schema.json). The
fixtures are intentionally small and are used by backend and frontend tests.
