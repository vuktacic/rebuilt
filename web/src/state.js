function copy(value) {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

export function initialState(mode = "live") {
  return {
    mode,
    phase: "idle",
    jobId: null,
    status: null,
    frames: [],
    tracks: [],
    events: [],
    analysis: null,
    guide: null,
    draftGuide: null,
    dirty: false,
    uploadName: "",
    isSaving: false,
    error: null,
    saveError: null,
  };
}

function withDraft(state, draftGuide) {
  return { ...state, draftGuide, dirty: true, saveError: null };
}

export function reduce(state, action) {
  switch (action.type) {
    case "UPLOAD_STARTED":
      return { ...initialState(state.mode), phase: "uploading", uploadName: action.name || "video" };
    case "UPLOAD_ACCEPTED":
      return { ...state, phase: "processing", jobId: action.jobId, status: "queued", error: null };
    case "JOB_UPDATE": {
      const job = action.job;
      const readyGuide = job.guide ? copy(job.guide) : state.guide;
      const keepDraft = state.dirty && state.jobId === job.jobId;
      return {
        ...state,
        phase: job.status === "ready" ? "ready" : job.status === "failed" ? "error" : "processing",
        jobId: job.jobId,
        status: job.status,
        frames: copy(job.frames || []),
        tracks: copy(job.tracks || []),
        events: copy(job.events || []),
        analysis: job.analysis ? copy(job.analysis) : null,
        guide: readyGuide,
        draftGuide: keepDraft ? state.draftGuide : job.guide ? copy(job.guide) : state.draftGuide,
        error: job.error || null,
      };
    }
    case "UPLOAD_FAILED":
      return { ...state, phase: "error", error: action.error, isSaving: false };
    case "LOAD_JOB": {
      const job = action.job;
      return {
        ...state,
        phase: job.status === "ready" ? "ready" : job.status === "failed" ? "error" : "processing",
        jobId: job.jobId,
        status: job.status,
        frames: copy(job.frames || []),
        tracks: copy(job.tracks || []),
        events: copy(job.events || []),
        analysis: job.analysis ? copy(job.analysis) : null,
        guide: job.guide ? copy(job.guide) : null,
        draftGuide: job.guide ? copy(job.guide) : null,
        dirty: false,
        error: job.error || null,
      };
    }
    case "EDIT_TITLE":
      return withDraft(state, { ...copy(state.draftGuide), title: action.value });
    case "EDIT_STEP_TEXT":
      return withDraft(state, updateStep(state.draftGuide, action.index, (step) => ({ ...step, text: action.value })));
    case "EDIT_STEP_FRAME":
      return withDraft(state, updateStep(state.draftGuide, action.index, (step) => ({ ...step, frameId: action.frameId })));
    case "EDIT_STEP_UNCERTAINTY":
      return withDraft(state, updateStep(state.draftGuide, action.index, (step) => ({ ...step, uncertainty: action.value || null })));
    case "ADD_STEP": {
      const frameId = action.frameId || state.frames[0]?.frameId || "";
      const guide = copy(state.draftGuide || { title: "", steps: [] });
      guide.steps.push({ text: "", frameId, uncertainty: null });
      return withDraft(state, guide);
    }
    case "DELETE_STEP": {
      const guide = copy(state.draftGuide);
      guide.steps.splice(action.index, 1);
      return withDraft(state, guide);
    }
    case "SAVE_STARTED":
      return { ...state, isSaving: true, saveError: null };
    case "SAVE_SUCCEEDED":
      return { ...state, guide: copy(action.guide), draftGuide: copy(action.guide), dirty: false, isSaving: false, saveError: null };
    case "SAVE_FAILED":
      return { ...state, isSaving: false, saveError: action.error };
    default:
      return state;
  }
}

function updateStep(guide, index, update) {
  const next = copy(guide);
  if (next?.steps?.[index]) next.steps[index] = update(next.steps[index]);
  return next;
}

export function validateDraft(guide, frames) {
  if (!guide || !guide.title.trim()) return "Add a guide title.";
  if (!guide.steps.length) return "Add at least one instruction step.";
  const frameIds = new Set(frames.map((frame) => frame.frameId));
  if (guide.steps.some((step) => !step.text.trim())) return "Every step needs an instruction.";
  if (guide.steps.some((step) => !frameIds.has(step.frameId))) return "Every step needs an extracted frame.";
  return null;
}

export function statusMessage(state) {
  if (state.phase === "uploading") return "Uploading your recording…";
  if (state.status === "queued") return "Waiting to start…";
  if (state.status === "extracting") return "Selecting useful frames…";
  if (state.status === "analyzing") return "Tracking pieces and detecting changes…";
  if (state.status === "generating") return "Writing the assembly guide…";
  if (state.phase === "ready") return state.dirty ? "Unsaved edits" : "Guide ready to review";
  if (state.phase === "error") return "Something needs attention";
  return "Choose a recording to begin";
}
