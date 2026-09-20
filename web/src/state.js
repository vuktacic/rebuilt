function copy(value) {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

export function initialState(mode = "live") {
  return {
    mode,
    pipeline: "manual_pairs",
    capabilities: [],
    phase: "idle",
    jobId: null,
    status: null,
    frames: [],
    tracks: [],
    annotations: [],
    annotationSuggestions: null,
    partTracks: [],
    trackingProgress: null,
    events: [],
    analysis: null,
    guide: null,
    guideRevision: 0,
    verification: null,
    timeline: null,
    pipelineProgress: null,
    inferenceMetrics: {},
    draftGuide: null,
    dirty: false,
    uploadName: "",
    isSaving: false,
    error: null,
    saveError: null,
    storyboard: null,
    manualSelection: [],
    manualContext: "",
    manualDraftDirty: false,
  };
}

function withDraft(state, draftGuide) {
  const verification = state.verification ? { ...state.verification, status: "stale", message: "Guide edits need a fresh verification." } : null;
  return { ...state, draftGuide, verification, dirty: true, saveError: null };
}

export function reduce(state, action) {
  switch (action.type) {
    case "UPLOAD_STARTED":
      return { ...initialState(state.mode), phase: "uploading", uploadName: action.name || "video" };
    case "UPLOAD_ACCEPTED":
      return { ...state, phase: "processing", jobId: action.jobId, status: "queued", pipeline: "manual_pairs", error: null, storyboard: null, manualSelection: [], manualContext: "", manualDraftDirty: false };
    case "CAPABILITIES_LOADED":
      return { ...state, capabilities: copy(action.capabilities?.pipelines || []) };
    case "JOB_UPDATE": {
      const job = action.job;
      const readyGuide = job.guide ? copy(job.guide) : state.guide;
      const keepDraft = state.dirty && state.jobId === job.jobId;
      return {
        ...state,
        phase: job.status === "ready" ? "ready" : job.status === "failed" ? "error" : "processing",
        jobId: job.jobId,
        status: job.status,
        pipeline: job.pipeline || state.pipeline,
        capabilities: state.capabilities,
        frames: copy(job.frames || []),
        tracks: copy(job.tracks || []),
        annotations: copy(job.annotations || []),
        annotationSuggestions: job.annotationSuggestions ? copy(job.annotationSuggestions) : null,
        partTracks: copy(job.partTracks || []),
        trackingProgress: job.trackingProgress ?? null,
        events: copy(job.events || []),
        analysis: job.analysis ? copy(job.analysis) : null,
        guide: readyGuide,
        guideRevision: job.guideRevision || 0,
        verification: job.verification ? copy(job.verification) : null,
        timeline: job.timeline ? copy(job.timeline) : null,
        pipelineProgress: job.pipelineProgress ? copy(job.pipelineProgress) : null,
        inferenceMetrics: copy(job.inferenceMetrics || {}),
        storyboard: job.storyboard ? copy(job.storyboard) : null,
        manualSelection: job.storyboard && !state.manualDraftDirty ? copy(job.storyboard.selectedFrameIds) : state.manualSelection,
        manualContext: job.storyboard && !state.manualDraftDirty ? job.storyboard.context || "" : state.manualContext,
        manualDraftDirty: job.storyboard ? false : state.manualDraftDirty,
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
        pipeline: job.pipeline || state.pipeline,
        frames: copy(job.frames || []),
        tracks: copy(job.tracks || []),
        annotations: copy(job.annotations || []),
        annotationSuggestions: job.annotationSuggestions ? copy(job.annotationSuggestions) : null,
        partTracks: copy(job.partTracks || []),
        trackingProgress: job.trackingProgress ?? null,
        events: copy(job.events || []),
        analysis: job.analysis ? copy(job.analysis) : null,
        guide: job.guide ? copy(job.guide) : null,
        guideRevision: job.guideRevision || 0,
        verification: job.verification ? copy(job.verification) : null,
        timeline: job.timeline ? copy(job.timeline) : null,
        pipelineProgress: job.pipelineProgress ? copy(job.pipelineProgress) : null,
        inferenceMetrics: copy(job.inferenceMetrics || {}),
        storyboard: job.storyboard ? copy(job.storyboard) : null,
        manualSelection: job.storyboard ? copy(job.storyboard.selectedFrameIds) : [],
        manualContext: job.storyboard?.context || "",
        manualDraftDirty: false,
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
    case "ACCEPT_SUGGESTIONS": {
      const existing = new Set(state.annotations.map((item) => item.partId));
      const additions = (state.annotationSuggestions?.suggestions || []).filter((item) => !existing.has(item.partId)).map((item) => ({
        partId: item.partId,
        name: item.name,
        frameIndex: item.frameIndex,
        points: [item.point],
        labels: [1],
        box: item.box || null,
      }));
      return { ...state, annotations: [...state.annotations, ...additions], annotationSuggestions: null };
    }
    case "EDIT_ANNOTATION_NAME":
      return { ...state, annotations: state.annotations.map((item) => item.partId === action.partId ? { ...item, name: action.value } : item) };
    case "EDIT_ANNOTATION_POINT":
      return { ...state, annotations: state.annotations.map((item) => item.partId === action.partId ? { ...item, points: [action.point], labels: [1] } : item) };
    case "DELETE_ANNOTATION":
      return { ...state, annotations: state.annotations.filter((item) => item.partId !== action.partId) };
    case "EDIT_MANUAL_SELECTION":
      return { ...state, manualSelection: copy(action.selectedFrameIds), manualDraftDirty: true, saveError: null };
    case "EDIT_MANUAL_CONTEXT":
      return { ...state, manualContext: action.value, manualDraftDirty: true, saveError: null };
    case "EDIT_PAIR_REVIEW": {
      if (!state.storyboard) return state;
      const pairs = state.storyboard.pairs.map((pair) => pair.pairId === action.pairId
        ? { ...pair, reviewedFinding: action.reviewedFinding === undefined ? pair.reviewedFinding : copy(action.reviewedFinding), disposition: action.disposition === undefined ? pair.disposition : action.disposition }
        : pair);
      return { ...state, storyboard: { ...state.storyboard, pairs }, manualDraftDirty: true, saveError: null };
    }
    case "STORYBOARD_SAVED":
      return { ...state, storyboard: copy(action.storyboard), manualSelection: copy(action.storyboard.selectedFrameIds), manualContext: action.storyboard.context || "", manualDraftDirty: false, error: null };
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

export function attachmentTargetName(track, events, partTracks) {
  const attachment = events.find((event) => (
    event.kind === "attach"
    && event.affectedTrackIds?.[0] === `part:${track.partId}`
    && event.affectedTrackIds.length > 1
  ));
  const targetId = attachment?.affectedTrackIds[1];
  return partTracks.find((candidate) => `part:${candidate.partId}` === targetId)?.name || null;
}

export function statusMessage(state) {
  if (state.phase === "uploading") return "Uploading your recording…";
  if (state.status === "queued") return "Waiting to start…";
  if (state.status === "extracting") return "Selecting useful frames…";
  if (state.status === "suggesting") return "Suggesting visible pieces…";
  if (state.status === "annotating") return state.pipeline === "manual_pairs" ? "Review snapshots and pair differences…" : "Name visible parts on the final frame…";
  if (state.status === "analyzing") return state.pipeline === "plan3" ? "Tracking named parts backward through the video…" : "Reconstructing the source-order action timeline…";
  if (state.status === "generating") return "Writing the assembly guide…";
  if (state.status === "verifying" || state.status === "correcting") return "Checking the guide against images…";
  if (state.phase === "ready") return state.dirty ? "Unsaved edits" : "Guide ready to review";
  if (state.phase === "error") return "Something needs attention";
  return "Choose a recording to begin";
}
