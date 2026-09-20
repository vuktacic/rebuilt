import { ApiError, createApiClient } from "./api.js";
import { initialState, reduce, statusMessage, validateDraft } from "./state.js";

const params = new URLSearchParams(window.location.search);
const mockEnabled = params.get("mock") === "1" || params.get("mock") === "true";
const api = createApiClient({ mock: mockEnabled });
let state = initialState(api.mode);
let pollController = null;
let selectedAnnotationPoint = null;

const elements = {
  modeBanner: document.querySelector("#mode-banner"),
  uploadForm: document.querySelector("#upload-form"),
  videoInput: document.querySelector("#video-input"),
  dropZone: document.querySelector("#drop-zone"),
  uploadButton: document.querySelector("#upload-button"),
  selectedFile: document.querySelector("#selected-file"),
  statusPill: document.querySelector("#status-pill"),
  statusMessage: document.querySelector("#status-message"),
  statusDetail: document.querySelector("#status-detail"),
  progressBar: document.querySelector("#progress-bar"),
  errorBox: document.querySelector("#error-box"),
  timelinePanel: document.querySelector("#timeline-panel"),
  timelineCount: document.querySelector("#timeline-count"),
  eventsList: document.querySelector("#events-list"),
  guidePanel: document.querySelector("#guide-panel"),
  dirtyLabel: document.querySelector("#dirty-label"),
  guideTitle: document.querySelector("#guide-title"),
  stepsList: document.querySelector("#steps-list"),
  saveButton: document.querySelector("#save-button"),
  saveError: document.querySelector("#save-error"),
  addStepButton: document.querySelector("#add-step-button"),
  resetButton: document.querySelector("#reset-button"),
  annotationPanel: document.querySelector("#annotation-panel"),
  annotationImage: document.querySelector("#annotation-image"),
  annotationMarkers: document.querySelector("#annotation-markers"),
  annotationFrame: document.querySelector("#annotation-frame"),
  annotationFrameLabel: document.querySelector("#annotation-frame-label"),
  annotationName: document.querySelector("#annotation-name"),
  annotationHint: document.querySelector("#annotation-hint"),
  suggestAnnotationsButton: document.querySelector("#suggest-annotations-button"),
  suggestionList: document.querySelector("#suggestion-list"),
  addAnnotationButton: document.querySelector("#add-annotation-button"),
  annotationList: document.querySelector("#annotation-list"),
  trackBackwardButton: document.querySelector("#track-backward-button"),
  trackingPreviewPanel: document.querySelector("#tracking-preview-panel"),
  trackingImage: document.querySelector("#tracking-image"),
  trackingMarkers: document.querySelector("#tracking-markers"),
  trackingFrame: document.querySelector("#tracking-frame"),
  trackingFrameLabel: document.querySelector("#tracking-frame-label"),
  trackingLegend: document.querySelector("#tracking-legend"),
};

if (api.mode === "mock") {
  elements.modeBanner.hidden = false;
  elements.modeBanner.textContent = "Mock mode is on · progress and saves are simulated locally. Live failures never fall back to mock results.";
}

function dispatch(action, preserveFocus = false) {
  const focused = preserveFocus ? document.activeElement?.dataset.field : null;
  const selectionStart = preserveFocus && document.activeElement?.selectionStart;
  state = reduce(state, action);
  render();
  if (focused) {
    const next = document.querySelector(`[data-field="${focused}"]`);
    if (next) {
      next.focus();
      if (typeof selectionStart === "number" && typeof next.setSelectionRange === "function") next.setSelectionRange(selectionStart, selectionStart);
    }
  }
}

function render() {
  const hasFile = Boolean(elements.videoInput.files?.[0]);
  elements.uploadButton.disabled = !hasFile || state.phase === "uploading" || state.phase === "processing";
  elements.videoInput.disabled = state.phase === "uploading" || state.phase === "processing";
  elements.uploadButton.textContent = state.phase === "uploading" ? "Uploading…" : "Begin analysis ↗";
  elements.selectedFile.hidden = !hasFile;
  elements.selectedFile.textContent = hasFile ? elements.videoInput.files[0].name : "";
  elements.dropZone.classList.toggle("has-file", hasFile);

  elements.statusMessage.textContent = statusMessage(state);
  elements.statusPill.textContent = pillText(state);
  elements.statusPill.className = `status-pill status-${state.phase}`;
  elements.progressBar.style.width = `${progressPercent(state)}%`;
  elements.statusDetail.textContent = detailText(state);
  showError(elements.errorBox, state.error);
  renderAnnotation();
  renderTrackingPreview();

  elements.timelinePanel.hidden = state.events.length === 0;
  if (state.events.length) renderEvents();

  const hasGuide = Boolean(state.draftGuide);
  elements.guidePanel.hidden = !hasGuide;
  elements.dirtyLabel.hidden = !state.dirty;
  if (hasGuide) {
    elements.guideTitle.value = state.draftGuide.title;
    elements.saveButton.disabled = state.isSaving || !state.dirty;
    elements.saveButton.textContent = state.isSaving ? "Saving…" : "Save guide ✓";
    renderSteps();
  }
  showError(elements.saveError, state.saveError);
}

function selectedFrame(frames, input) {
  return frames[Math.max(0, Math.min(frames.length - 1, Number(input.value) || 0))];
}

function renderAnnotation() {
  const active = state.status === "annotating";
  elements.annotationPanel.hidden = !active;
  if (!active || !state.frames.length) return;
  elements.annotationFrame.max = String(state.frames.length - 1);
  if (!elements.annotationFrame.dataset.initialized) {
    elements.annotationFrame.value = String(state.frames.length - 1);
    elements.annotationFrame.dataset.initialized = "true";
  }
  const frame = selectedFrame(state.frames, elements.annotationFrame);
  if (elements.annotationImage.dataset.frameId !== frame.frameId) {
    elements.annotationImage.src = frame.imageUrl;
    elements.annotationImage.dataset.frameId = frame.frameId;
  }
  elements.annotationFrameLabel.textContent = `${frame.frameId} · ${formatTime(frame.timestampSeconds)}`;
  elements.annotationMarkers.replaceChildren();
  const frameIndex = Number(elements.annotationFrame.value);
  const markers = [
    ...state.annotations.map((item) => ({ item, suggestion: false })),
    ...(state.annotationSuggestions?.suggestions || []).map((item) => ({ item: { ...item, points: [item.point] }, suggestion: true })),
    ...(selectedAnnotationPoint ? [{ item: { name: "new", frameIndex, points: [selectedAnnotationPoint] }, suggestion: false }] : []),
  ];
  markers.filter(({ item }) => item.frameIndex === frameIndex).forEach(({ item, suggestion }) => {
      const point = item.points[0];
      if (!point) return;
      const marker = document.createElement("i");
      marker.className = `annotation-point${suggestion ? " suggestion" : ""}`;
      marker.style.left = `${point.x / elements.annotationImage.naturalWidth * 100}%`;
      marker.style.top = `${point.y / elements.annotationImage.naturalHeight * 100}%`;
      elements.annotationMarkers.append(marker);
    });
  elements.suggestAnnotationsButton.disabled = false;
  elements.addAnnotationButton.disabled = !selectedAnnotationPoint || !elements.annotationName.value.trim();
  elements.trackBackwardButton.disabled = state.annotations.length === 0;
  elements.annotationList.replaceChildren(...state.annotations.map((item, index) => {
    const line = document.createElement("div");
    line.className = "annotation-list-row";
    const name = document.createElement("input");
    name.value = item.name;
    name.setAttribute("aria-label", `Name for part ${index + 1}`);
    name.addEventListener("input", (event) => dispatch({ type: "EDIT_ANNOTATION_NAME", index, value: event.target.value }));
    const frame = document.createElement("small");
    frame.textContent = `frame ${item.frameIndex + 1}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${item.name}`);
    remove.addEventListener("click", () => dispatch({ type: "DELETE_ANNOTATION", index }));
    line.append(name, frame, remove);
    return line;
  }));
  elements.suggestionList.replaceChildren();
  const suggestions = state.annotationSuggestions;
  if (suggestions?.status === "unavailable") {
    const line = document.createElement("p");
    line.textContent = suggestions.message || "Suggestions are unavailable; add points manually.";
    elements.suggestionList.append(line);
  } else if (suggestions?.suggestions?.length) {
    const line = document.createElement("p");
    line.textContent = `${suggestions.suggestions.length} suggested part${suggestions.suggestions.length === 1 ? "" : "s"}. Review the blue points before accepting.`;
    const accept = document.createElement("button");
    accept.type = "button";
    accept.className = "button button-secondary";
    accept.textContent = "Accept suggestions";
    accept.addEventListener("click", () => dispatch({ type: "ACCEPT_SUGGESTIONS" }));
    elements.suggestionList.append(line, accept);
    suggestions.suggestions.forEach((suggestion) => {
      const detail = document.createElement("p");
      detail.textContent = `${suggestion.name} · ${Math.round(suggestion.confidence * 100)}% · (${Math.round(suggestion.point.x)}, ${Math.round(suggestion.point.y)})`;
      elements.suggestionList.append(detail);
    });
  }
}

function renderTrackingPreview() {
  const active = state.partTracks.length > 0;
  elements.trackingPreviewPanel.hidden = !active;
  if (!active || !state.frames.length) return;
  elements.trackingFrame.max = String(state.frames.length - 1);
  const assemblyIndex = Number(elements.trackingFrame.value);
  const sourceIndex = state.frames.length - 1 - assemblyIndex;
  const frame = state.frames[sourceIndex];
  elements.trackingImage.src = frame.imageUrl;
  elements.trackingFrameLabel.textContent = `Assembly frame ${assemblyIndex + 1} · source ${frame.frameId} · ${formatTime(state.frames.at(-1).timestampSeconds - frame.timestampSeconds)}`;
  elements.trackingMarkers.replaceChildren();
  state.partTracks.forEach((track) => {
    const observation = track.observations.find((item) => item.frameIndex === sourceIndex);
    if (!observation?.visible || !observation.bbox || !elements.trackingImage.naturalWidth) return;
    const [x, y, width, height] = observation.bbox;
    const box = document.createElement("div");
    box.className = "annotation-box";
    box.style.left = `${x / elements.trackingImage.naturalWidth * 100}%`;
    box.style.top = `${y / elements.trackingImage.naturalHeight * 100}%`;
    box.style.width = `${width / elements.trackingImage.naturalWidth * 100}%`;
    box.style.height = `${height / elements.trackingImage.naturalHeight * 100}%`;
    box.innerHTML = `<span>${track.name}</span>`;
    elements.trackingMarkers.append(box);
  });
  elements.trackingLegend.replaceChildren(...state.partTracks.map((track) => {
    const line = document.createElement("p");
    line.textContent = track.attachmentStartFrame === null ? `${track.name} · no confident attachment range` : `${track.name} · joins near assembly frames ${state.frames.length - track.attachmentStartFrame}–${state.frames.length - track.attachmentEndFrame}`;
    return line;
  }));
}

function renderEvents() {
  elements.timelineCount.textContent = `${state.events.length} change${state.events.length === 1 ? "" : "s"}`;
  elements.eventsList.replaceChildren();
  state.events.forEach((event, index) => {
    const card = document.createElement("article");
    card.className = "event-card";
    const heading = document.createElement("div");
    heading.className = "event-card-heading";
    const title = document.createElement("h3");
    title.textContent = `${String(index + 1).padStart(2, "0")} · ${eventLabel(event.kind)}`;
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = `${formatTime(event.startTimestampSeconds)}–${formatTime(event.endTimestampSeconds)}`;
    heading.append(title, time);
    card.append(heading);

    const pair = document.createElement("div");
    pair.className = "event-pair";
    pair.append(eventFrame("Before", event.beforeFrameId), eventFrame("After", event.afterFrameId));
    card.append(pair);

    const evidence = document.createElement("p");
    evidence.className = "event-evidence";
    evidence.textContent = event.evidence;
    card.append(evidence);
    if (event.uncertainty) {
      const uncertainty = document.createElement("p");
      uncertainty.className = "event-uncertainty";
      uncertainty.textContent = `Uncertain: ${event.uncertainty}`;
      card.append(uncertainty);
    }
    elements.eventsList.append(card);
  });
}

function eventFrame(label, frameId) {
  const frame = state.frames.find((candidate) => candidate.frameId === frameId);
  const figure = document.createElement("figure");
  figure.className = "event-frame";
  const caption = document.createElement("figcaption");
  caption.textContent = frame ? `${label} · build time ${formatTime(state.frames.at(-1).timestampSeconds - frame.timestampSeconds)}` : label;
  if (frame) {
    const image = document.createElement("img");
    image.src = frame.imageUrl;
    image.alt = `${label} the detected assembly change`;
    figure.append(image);
  }
  figure.append(caption);
  return figure;
}

function eventLabel(kind) {
  return { attach: "Piece attached", detach: "Piece detached", uncertain_change: "Change needs review" }[kind] || "Assembly change";
}

function renderSteps() {
  elements.stepsList.replaceChildren();
  state.draftGuide.steps.forEach((step, index) => {
    const frame = state.frames.find((candidate) => candidate.frameId === step.frameId) || state.frames[0];
    const card = document.createElement("article");
    card.className = "step-card";
    card.innerHTML = `<div class="step-card-top"><span class="step-number">${String(index + 1).padStart(2, "0")}</span><span class="step-label">Assembly step</span><button class="icon-button delete-step" type="button" aria-label="Delete step ${index + 1}" data-index="${index}">×</button></div>`;

    const media = document.createElement("div");
    media.className = "step-media";
    if (frame) {
      const image = document.createElement("img");
      image.src = frame.imageUrl;
      image.alt = `Build frame at ${formatTime(frame.timestampSeconds)}`;
      media.append(image);
      const timestamp = document.createElement("span");
      timestamp.className = "timestamp";
      timestamp.textContent = `Build time ${formatTime(state.frames.at(-1).timestampSeconds - frame.timestampSeconds)}`;
      media.append(timestamp);
    } else {
      media.textContent = "Choose an extracted frame";
      media.classList.add("empty-media");
    }
    card.append(media);

    const content = document.createElement("div");
    content.className = "step-content";
    const instructionLabel = document.createElement("label");
    instructionLabel.textContent = "Instruction";
    const instruction = document.createElement("textarea");
    instruction.rows = 3;
    instruction.value = step.text;
    instruction.dataset.field = `step-${index}-text`;
    instruction.setAttribute("aria-label", `Instruction for step ${index + 1}`);
    instruction.addEventListener("input", (event) => dispatch({ type: "EDIT_STEP_TEXT", index, value: event.target.value }, true));
    instructionLabel.append(instruction);
    content.append(instructionLabel);

    const frameLabel = document.createElement("label");
    frameLabel.className = "frame-select";
    frameLabel.textContent = "Reference frame";
    const select = document.createElement("select");
    select.dataset.field = `step-${index}-frame`;
    select.setAttribute("aria-label", `Reference frame for step ${index + 1}`);
    state.frames.forEach((candidate) => {
      const option = document.createElement("option");
      option.value = candidate.frameId;
      option.textContent = `Frame at ${formatTime(candidate.timestampSeconds)}`;
      option.selected = candidate.frameId === step.frameId;
      select.append(option);
    });
    select.addEventListener("change", (event) => dispatch({ type: "EDIT_STEP_FRAME", index, frameId: event.target.value }));
    frameLabel.append(select);
    content.append(frameLabel);

    const uncertaintyLabel = document.createElement("label");
    uncertaintyLabel.className = "uncertainty-field";
    uncertaintyLabel.textContent = "Uncertainty note (optional)";
    const uncertainty = document.createElement("input");
    uncertainty.type = "text";
    uncertainty.value = step.uncertainty || "";
    uncertainty.dataset.field = `step-${index}-uncertainty`;
    uncertainty.placeholder = "What was partly obscured?";
    uncertainty.setAttribute("aria-label", `Uncertainty note for step ${index + 1}`);
    uncertainty.addEventListener("input", (event) => dispatch({ type: "EDIT_STEP_UNCERTAINTY", index, value: event.target.value }, true));
    uncertaintyLabel.append(uncertainty);
    content.append(uncertaintyLabel);
    card.append(content);
    elements.stepsList.append(card);
  });
  elements.stepsList.querySelectorAll(".delete-step").forEach((button) => button.addEventListener("click", () => dispatch({ type: "DELETE_STEP", index: Number(button.dataset.index) })));
}

function showError(element, error) {
  element.hidden = !error;
  element.textContent = error ? `${error.message}${error.code ? ` (${error.code})` : ""}` : "";
}

function pillText(current) {
  if (current.phase === "uploading") return "Uploading";
  if (current.phase === "processing") return current.status || "Processing";
  if (current.phase === "ready") return "Ready";
  if (current.phase === "error") return "Needs attention";
  return "Waiting";
}

function progressPercent(current) {
  if (current.phase === "uploading") return 18;
  if (current.status === "analyzing" && typeof current.trackingProgress === "number") return 55 + Math.round(current.trackingProgress * 29);
  return { queued: 20, extracting: 40, suggesting: 48, annotating: 52, analyzing: 68, generating: 84, ready: 100, failed: 100 }[current.status] || 0;
}

function detailText(current) {
  if (current.phase === "uploading") return `Uploading ${current.uploadName || "your video"}…`;
  if (current.status === "suggesting") return "GPT is reviewing the final disassembled frame for visible brick names and points.";
  if (current.status === "annotating") return "Click and name every clearly separated part. The final frame is selected by default; use an earlier frame only as a fallback.";
  if (current.status === "analyzing" && typeof current.trackingProgress === "number") return `SAM2 is propagating masks backward: ${Math.round(current.trackingProgress * 100)}% complete. You can leave this tab open.`;
  if (current.phase === "processing") return "This page checks for progress every two seconds. You can leave this tab open.";
  if (current.phase === "ready") return `${current.frames.length} extracted frame${current.frames.length === 1 ? "" : "s"} · review the wording before you save.`;
  if (current.phase === "error") return "Correct the issue or start another build. Live errors are shown as returned by the server.";
  return "Your original video stays on this device while it uploads.";
}

function formatTime(seconds) {
  const value = Number(seconds) || 0;
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
}

async function beginPolling(jobId) {
  pollController?.abort();
  pollController = new AbortController();
  try {
    await api.pollJob(jobId, {
      signal: pollController.signal,
      onUpdate: (job) => dispatch({ type: "JOB_UPDATE", job }),
    });
  } catch (error) {
    if (error.code !== "ABORTED") dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const file = elements.videoInput.files?.[0];
  if (!file) return;
  dispatch({ type: "UPLOAD_STARTED", name: file.name });
  try {
    const created = await api.uploadVideo(file);
    dispatch({ type: "UPLOAD_ACCEPTED", jobId: created.jobId });
    setJobUrl(created.jobId);
    await beginPolling(created.jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

async function handleSave() {
  const validation = validateDraft(state.draftGuide, state.frames);
  if (validation) {
    dispatch({ type: "SAVE_FAILED", error: { code: "INVALID_GUIDE", message: validation } });
    return;
  }
  dispatch({ type: "SAVE_STARTED" });
  try {
    const guide = await api.saveGuide(state.jobId, state.draftGuide);
    dispatch({ type: "SAVE_SUCCEEDED", guide });
  } catch (error) {
    dispatch({ type: "SAVE_FAILED", error: normalizeError(error) });
  }
}

async function handleSuggestAnnotations() {
  if (!state.jobId || state.status !== "annotating") return;
  elements.suggestAnnotationsButton.disabled = true;
  try {
    const result = await api.suggestAnnotations(state.jobId, Number(elements.annotationFrame.value));
    dispatch({ type: "JOB_UPDATE", job: result });
    if (result.status !== "annotating") await beginPolling(state.jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  } finally {
    elements.suggestAnnotationsButton.disabled = false;
  }
}

async function restoreJob(jobId) {
  state = { ...state, phase: "processing", jobId };
  render();
  try {
    const job = await api.getJob(jobId);
    dispatch({ type: "LOAD_JOB", job });
    if (job.status !== "ready" && job.status !== "failed") await beginPolling(jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

function setJobUrl(jobId) {
  const query = new URLSearchParams();
  if (mockEnabled) query.set("mock", "1");
  query.set("job", jobId);
  history.replaceState({}, "", `${location.pathname}?${query}`);
}

function normalizeError(error) {
  if (error instanceof ApiError) return { code: error.code, message: error.message };
  return { code: "REQUEST_FAILED", message: error?.message || "Something went wrong." };
}

function addAnnotation() {
  const name = elements.annotationName.value.trim();
  if (!selectedAnnotationPoint || !name) {
    elements.annotationHint.textContent = "Click a part and enter a name before adding it.";
    return;
  }
  const index = state.annotations.findIndex((item) => item.name.toLowerCase() === name.toLowerCase());
  const annotations = [...state.annotations];
  if (index >= 0) {
    annotations[index] = {
      ...annotations[index],
      points: [...annotations[index].points, selectedAnnotationPoint],
      labels: [...annotations[index].labels, 1],
    };
  } else {
    annotations.push({ name, frameIndex: Number(elements.annotationFrame.value), points: [selectedAnnotationPoint], labels: [1] });
  }
  state = { ...state, annotations };
  selectedAnnotationPoint = null;
  elements.annotationName.value = "";
  elements.annotationHint.textContent = index >= 0 ? "Point added to the existing named part." : "Part added. Click another part or scrub to an earlier fallback frame.";
  render();
}

async function trackBackward() {
  try {
    const saved = await api.saveAnnotations(state.jobId, state.annotations);
    dispatch({ type: "JOB_UPDATE", job: saved });
    const started = await api.trackBackward(state.jobId);
    dispatch({ type: "JOB_UPDATE", job: started });
    await beginPolling(state.jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

elements.uploadForm.addEventListener("submit", handleUpload);
elements.videoInput.addEventListener("change", render);
elements.dropZone.addEventListener("dragover", (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); });
elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("dragging"));
elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  if (event.dataTransfer.files.length) {
    elements.videoInput.files = event.dataTransfer.files;
    render();
  }
});
elements.guideTitle.addEventListener("input", (event) => dispatch({ type: "EDIT_TITLE", value: event.target.value }, true));
elements.saveButton.addEventListener("click", handleSave);
elements.addStepButton.addEventListener("click", () => dispatch({ type: "ADD_STEP" }));
elements.annotationFrame.addEventListener("input", () => { selectedAnnotationPoint = null; renderAnnotation(); });
elements.annotationName.addEventListener("input", renderAnnotation);
elements.annotationImage.addEventListener("load", renderAnnotation);
elements.annotationImage.addEventListener("click", (event) => {
  const bounds = elements.annotationImage.getBoundingClientRect();
  selectedAnnotationPoint = {
    x: (event.clientX - bounds.left) / bounds.width * elements.annotationImage.naturalWidth,
    y: (event.clientY - bounds.top) / bounds.height * elements.annotationImage.naturalHeight,
  };
  renderAnnotation();
});
elements.addAnnotationButton.addEventListener("click", addAnnotation);
elements.suggestAnnotationsButton.addEventListener("click", handleSuggestAnnotations);
elements.trackBackwardButton.addEventListener("click", trackBackward);
elements.trackingFrame.addEventListener("input", renderTrackingPreview);
elements.trackingImage.addEventListener("load", renderTrackingPreview);
elements.resetButton.addEventListener("click", () => {
  pollController?.abort();
  history.replaceState({}, "", mockEnabled ? "?mock=1" : location.pathname);
  elements.videoInput.value = "";
  state = initialState(api.mode);
  render();
});

render();
const restoredJobId = params.get("job");
if (restoredJobId) restoreJob(restoredJobId);
