# Rebuilt API contract

The backend exposes the following JSON contract. `POST /jobs` accepts one
multipart field named `video` and returns `202 {"jobId":"..."}`. Every new job
uses the `manual_pairs` workflow; stale multipart `pipeline` fields are ignored.
Clients poll
`GET /jobs/{jobId}` until the status is `ready` or `failed`, then use the guide
and supplied frame URLs. A complete guide can be saved with
`PUT /jobs/{jobId}/guide`.

`GET /capabilities` reports the manual workflow and missing OpenAI configuration
without exposing credentials or making inference calls. Its comparison and
writer model configuration is immutable job metadata.

Action-first jobs persist a source-order `timeline` with server-resolved frame
evidence, identity references, and a server-derived `assemblyActions` mapping.
Guide steps link back to source action IDs; unsupported or unresolved actions
remain review steps. Provider submissions and sanitized outputs are retained
under the job artifact directory for replay and audit without credentials or
remote file URIs.

When manual tracking is available, `POST /jobs/{jobId}/annotation-suggestions`
requests bounded, replaceable suggestions for a selected extracted frame.
Suggestions are never applied to `annotations` automatically. After a guide is
saved, `POST /jobs/{jobId}/verify` runs the bounded image-backed verification
request; verification results are optional and may be `unavailable` or `stale`.

`status` is one of `queued`, `extracting`, `suggesting`, `annotating`,
`analyzing`, `generating`, `verifying`, `correcting`, `ready`, or `failed`.
During analysis, `tracks` and `events` are populated when local
SAM3 finds stable attachment or detachment evidence. Each event references a
stable before/after frame pair and may carry an uncertainty reason. Frame IDs
are stable for the lifetime of a job. Annotation `partId` and guide `stepId`
values remain stable when names or text are edited. A guide step must reference
an existing selected frame and contain non-empty text. Error responses use
`{"error":{"code":"...","message":"..."}}`.

The machine-readable schema is in [`job.schema.json`](job.schema.json). The
fixtures are intentionally small and are used by backend and frontend tests.

`manual_pairs` is the browser demo workflow. It extracts orientation-corrected
frames at 5 FPS, saves source-ordered settled snapshots, compares adjacent
pairs with at most two requests in flight, and writes one text-only instruction
per reviewed included pair. Use `PUT /jobs/{jobId}/storyboard`,
`POST /jobs/{jobId}/compare`, `PUT /jobs/{jobId}/differences`, and
`POST /jobs/{jobId}/generate-guide`. The manual workflow requires OpenAI but
does not require SAM2, Gemini, or a GPU.

Legacy metadata may still contain older pipeline identifiers so saved jobs stay
readable, but those pipelines are no longer selectable for new jobs.
