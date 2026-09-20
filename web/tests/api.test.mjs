import test from "node:test";
import assert from "node:assert/strict";

import { ApiError, createApiClient } from "../src/api.js";

function response(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, async json() { return body; } };
}

const ready = (jobId = "job-1") => ({ jobId, status: "ready", frames: [], guide: { title: "Guide", steps: [] }, error: null });

test("live client uploads the video field and sends the complete guide on save", async () => {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url, options });
    if (options.method === "POST") return response({ jobId: "job-1" }, 202);
    if (options.method === "PUT") return response({ title: "Saved", steps: [] });
    return response(ready());
  };
  const client = createApiClient({ fetchImpl });
  const file = new File(["video"], "build.mp4", { type: "video/mp4" });
  assert.deepEqual(await client.uploadVideo(file), { jobId: "job-1" });
  const guide = { title: "Saved", steps: [] };
  assert.deepEqual(await client.saveGuide("job-1", guide), guide);
  assert.equal(calls[0].url, "/jobs");
  assert.equal(calls[0].options.body.get("video").name, "build.mp4");
  assert.equal(JSON.parse(calls[1].options.body).title, "Saved");
});

test("polling reports completion and stops at a failed job", async () => {
  let calls = 0;
  const client = createApiClient({
    fetchImpl: async () => {
      calls += 1;
      if (calls < 3) return response({ jobId: "job-1", status: calls === 1 ? "queued" : "generating", frames: [], guide: null, error: null });
      return response(ready());
    },
  });
  const updates = [];
  const complete = await client.pollJob("job-1", { intervalMs: 0, onUpdate: (job) => updates.push(job.status) });
  assert.equal(complete.status, "ready");
  assert.deepEqual(updates, ["queued", "generating", "ready"]);

  const failedClient = createApiClient({
    fetchImpl: async () => response({ jobId: "job-fail", status: "failed", frames: [], guide: null, error: { code: "PROCESSING_FAILED", message: "No video" } }),
  });
  const failed = await failedClient.pollJob("job-fail", { intervalMs: 0 });
  assert.equal(failed.status, "failed");
  assert.equal(failed.error.code, "PROCESSING_FAILED");
});

test("live HTTP errors stay errors instead of silently entering mock mode", async () => {
  const client = createApiClient({ fetchImpl: async () => response({ error: { code: "UPLOAD_TOO_LARGE", message: "Too big" } }, 413) });
  await assert.rejects(() => client.uploadVideo(new File(["x"], "big.mp4")), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.code, "UPLOAD_TOO_LARGE");
    assert.equal(error.status, 413);
    return true;
  });
});

test("live client requests suggestions for the selected frame", async () => {
  const calls = [];
  const client = createApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return response({ jobId: "job-1", status: "annotating", frames: [], annotationSuggestions: { status: "completed", frameIndex: 1, suggestions: [], message: null }, error: null }, 202);
    },
  });
  await client.suggestAnnotations("job-1", 1);
  assert.equal(calls[0].url, "/jobs/job-1/annotation-suggestions");
  assert.deepEqual(JSON.parse(calls[0].options.body), { frameIndex: 1 });
});

test("mock mode simulates progress and persists a successful save", async () => {
  const storage = new MapStorage();
  const client = createApiClient({ mock: true, storage });
  const { jobId } = await client.uploadVideo(new File(["x"], "build.mp4"));
  const final = await client.pollJob(jobId, { intervalMs: 0 });
  assert.equal(final.status, "ready");
  assert.equal(final.frames.length, 2);
  const saved = await client.saveGuide(jobId, final.guide);
  assert.equal(saved.title, "Small brick model");
  assert.ok(storage.getItem(`rebuilt:mock:${jobId}`));
});

class MapStorage {
  #values = new Map();
  getItem(key) { return this.#values.get(key) ?? null; }
  setItem(key, value) { this.#values.set(key, value); }
}
