# Rebuilt browser lane

The browser app is plain HTML, CSS, and JavaScript. The default mode uses the
live same-origin API. To rehearse the complete upload → processing → edit →
save flow without a backend, open the app with `?mock=1`; the banner makes the
mode visible and mock results are never used for live failures.

When the FastAPI app is running, open:

```text
http://localhost:8000/?mock=1
```

Use a filename containing `fail` in mock mode to rehearse a processing failure.
The API client polls every two seconds, keeps edits in the browser until the
explicit Save action, and restores a saved job from the `job` query parameter.

Run the dependency-free frontend tests with:

```bash
npm test
```
