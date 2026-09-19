import test from "node:test";
import assert from "node:assert/strict";

import { initialState, reduce, statusMessage, validateDraft } from "../src/state.js";

const frames = [
  { frameId: "frame-0001", timestampSeconds: 0, imageUrl: "/one.jpg" },
  { frameId: "frame-0002", timestampSeconds: 1, imageUrl: "/two.jpg" },
];
const guide = {
  title: "Build it",
  steps: [{ text: "Place the first piece.", frameId: "frame-0001", uncertainty: null }],
};

function readyState() {
  return reduce(initialState("live"), { type: "LOAD_JOB", job: { jobId: "job-1", status: "ready", frames, guide, error: null } });
}

test("editing actions update a local draft without changing the saved guide", () => {
  let state = readyState();
  state = reduce(state, { type: "EDIT_TITLE", value: "Edited build" });
  state = reduce(state, { type: "EDIT_STEP_TEXT", index: 0, value: "Put it down." });
  state = reduce(state, { type: "EDIT_STEP_FRAME", index: 0, frameId: "frame-0002" });
  state = reduce(state, { type: "EDIT_STEP_UNCERTAINTY", index: 0, value: "Partly hidden." });
  state = reduce(state, { type: "ADD_STEP", frameId: "frame-0001" });
  assert.equal(state.guide.title, "Build it");
  assert.equal(state.draftGuide.title, "Edited build");
  assert.equal(state.draftGuide.steps[0].frameId, "frame-0002");
  assert.equal(state.draftGuide.steps.length, 2);
  assert.equal(state.dirty, true);
  state = reduce(state, { type: "DELETE_STEP", index: 1 });
  assert.equal(state.draftGuide.steps.length, 1);
});

test("save failure preserves edits and save success clears the dirty state", () => {
  let state = readyState();
  state = reduce(state, { type: "EDIT_STEP_TEXT", index: 0, value: "Keep this local." });
  state = reduce(state, { type: "SAVE_STARTED" });
  state = reduce(state, { type: "SAVE_FAILED", error: { code: "REQUEST_FAILED", message: "No connection" } });
  assert.equal(state.draftGuide.steps[0].text, "Keep this local.");
  assert.equal(state.dirty, true);
  assert.equal(state.saveError.code, "REQUEST_FAILED");
  state = reduce(state, { type: "SAVE_SUCCEEDED", guide: state.draftGuide });
  assert.equal(state.dirty, false);
  assert.equal(state.guide.steps[0].text, "Keep this local.");
});

test("draft validation catches empty instructions and unknown frames", () => {
  assert.equal(validateDraft({ title: "", steps: guide.steps }, frames), "Add a guide title.");
  assert.equal(validateDraft({ title: "Build", steps: [{ text: " ", frameId: "frame-0001" }] }, frames), "Every step needs an instruction.");
  assert.equal(validateDraft({ title: "Build", steps: [{ text: "Text", frameId: "missing" }] }, frames), "Every step needs an extracted frame.");
  assert.equal(validateDraft(guide, frames), null);
});

test("status messages distinguish processing and unsaved review", () => {
  let state = reduce(initialState(), { type: "UPLOAD_STARTED", name: "build.mp4" });
  assert.equal(statusMessage(state), "Uploading your recording…");
  state = reduce(state, { type: "UPLOAD_ACCEPTED", jobId: "job-1" });
  state = reduce(state, { type: "JOB_UPDATE", job: { jobId: "job-1", status: "annotating", frames, events: [], guide: null, error: null } });
  assert.equal(statusMessage(state), "Name visible parts on the final frame…");
  state = reduce(state, { type: "JOB_UPDATE", job: { jobId: "job-1", status: "analyzing", frames, events: [{ eventId: "event-1" }], guide: null, error: null } });
  assert.equal(statusMessage(state), "Tracking named parts backward through the video…");
  assert.equal(state.events.length, 1);
  state = reduce(state, { type: "JOB_UPDATE", job: { jobId: "job-1", status: "generating", frames, guide: null, error: null } });
  assert.equal(statusMessage(state), "Writing the assembly guide…");
  state = reduce(state, { type: "JOB_UPDATE", job: { jobId: "job-1", status: "ready", frames, guide, error: null } });
  state = reduce(state, { type: "EDIT_TITLE", value: "Changed" });
  assert.equal(statusMessage(state), "Unsaved edits");
});
