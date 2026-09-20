import { ApiError, createApiClient } from "./api.js";
import { attachmentTargetName, initialState, reduce, statusMessage, validateDraft } from "./state.js";

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
  pipelineSelect: document.querySelector("#pipeline-select"),
  pipelineDisclosure: document.querySelector("#pipeline-disclosure"),
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
  verifyButton: document.querySelector("#verify-button"),
  verificationSummary: document.querySelector("#verification-summary"),
  addStepButton: document.querySelector("#add-step-button"),
  resetButton: document.querySelector("#reset-button"),
  annotationPanel: document.querySelector("#annotation-panel"),
  manualPanel: document.querySelector("#manual-panel"),
  manualImage: document.querySelector("#manual-image"),
  manualFrame: document.querySelector("#manual-frame"),
  manualFrameLabel: document.querySelector("#manual-frame-label"),
  manualPrev: document.querySelector("#manual-prev"),
  manualNext: document.querySelector("#manual-next"),
  manualAdd: document.querySelector("#manual-add"),
  manualSnapshotList: document.querySelector("#manual-snapshot-list"),
  manualContext: document.querySelector("#manual-context"),
  manualSave: document.querySelector("#manual-save"),
  manualCompare: document.querySelector("#manual-compare"),
  manualPairs: document.querySelector("#manual-pairs"),
  manualReviewSave: document.querySelector("#manual-review-save"),
  manualGenerate: document.querySelector("#manual-generate"),
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
  analyzePipelineButton: document.querySelector("#analyze-pipeline-button"),
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
  elements.pipelineSelect.value = state.pipeline;
  const selectedCapability = state.capabilities.find((item) => item.pipeline === state.pipeline);
  if (selectedCapability) {
    elements.pipelineDisclosure.textContent = selectedCapability.available ? selectedCapability.disclosure : `Unavailable: ${selectedCapability.missing.join(", ")}`;
    elements.uploadButton.disabled = elements.uploadButton.disabled || !selectedCapability.available;
  }

  elements.statusMessage.textContent = statusMessage(state);
  elements.statusPill.textContent = pillText(state);
  elements.statusPill.className = `status-pill status-${state.phase}`;
  elements.progressBar.style.width = `${progressPercent(state)}%`;
  elements.statusDetail.textContent = detailText(state);
  showError(elements.errorBox, state.error);
  renderAnnotation();
  renderManual();
  renderTrackingPreview();
  renderVerification();

  const timelineActions = [...(state.timeline?.actions || []), ...(state.timeline?.unresolvedIntervals || [])]
    .sort((left, right) => (left.startTimestampSeconds - right.startTimestampSeconds) || (left.endTimestampSeconds - right.endTimestampSeconds) || left.actionId.localeCompare(right.actionId));
  elements.timelinePanel.hidden = state.events.length === 0 && timelineActions.length === 0;
  if (state.events.length) renderEvents();
  else if (timelineActions.length) renderActionTimeline(timelineActions);

  const hasGuide = Boolean(state.draftGuide);
  elements.guidePanel.hidden = !hasGuide;
  elements.verifyButton.hidden = state.pipeline === "manual_pairs";
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

function renderManual() {
  const active = state.pipeline === "manual_pairs" && ["annotating", "analyzing", "generating", "ready"].includes(state.status);
  elements.manualPanel.hidden = !active;
  if (!active || !state.frames.length) return;
  elements.manualFrame.max = String(state.frames.length - 1);
  const index = Math.max(0, Math.min(state.frames.length - 1, Number(elements.manualFrame.value) || 0));
  elements.manualFrame.value = String(index);
  const frame = state.frames[index];
  elements.manualImage.src = frame.imageUrl;
  elements.manualImage.alt = `State snapshot at ${formatSourceTime(frame.timestampSeconds)}`;
  elements.manualFrameLabel.textContent = `${frame.frameId} · ${formatTime(frame.timestampSeconds)}`;
  elements.manualContext.value = state.manualContext;
  elements.manualContext.dataset.field = "manual-context";
  elements.manualPrev.disabled = index === 0;
  elements.manualNext.disabled = index === state.frames.length - 1;
  elements.manualAdd.disabled = state.manualSelection.includes(frame.frameId);
  elements.manualSnapshotList.replaceChildren(...state.manualSelection.map((frameId) => {
    const selected = state.frames.find((candidate) => candidate.frameId === frameId);
    const line = document.createElement("div");
    line.className = "annotation-list-row";
    const label = document.createElement("span");
    label.textContent = selected ? `${formatSourceTime(selected.timestampSeconds)} · ${selected.frameId}` : frameId;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove snapshot ${frameId}`);
    remove.addEventListener("click", () => dispatch({ type: "EDIT_MANUAL_SELECTION", selectedFrameIds: state.manualSelection.filter((id) => id !== frameId) }));
    line.append(label, remove);
    return line;
  }));
  const pairs = state.storyboard?.pairs || [];
  elements.manualCompare.disabled = !state.storyboard || pairs.some((pair) => pair.status === "pending" || pair.status === "failed") === false || state.status !== "annotating";
  elements.manualSave.disabled = state.manualSelection.length < 2 || !state.manualDraftDirty || state.status !== "annotating";
  elements.manualReviewSave.disabled = !pairs.length || pairs.some((pair) => pair.disposition === null || pair.disposition === undefined) || state.status !== "annotating";
  elements.manualGenerate.disabled = !pairs.length || pairs.some((pair) => pair.disposition === null || pair.disposition === undefined) || state.status !== "annotating";
  elements.manualPairs.replaceChildren(...pairs.map((pair) => renderManualPair(pair)));
}

function renderManualPair(pair) {
  const card = document.createElement("article");
  card.className = "event-card";
  card.append(eventFrame(`Before · ${pair.pairId}`, pair.beforeFrameId), eventFrame("After", pair.afterFrameId));
  const finding = pair.reviewedFinding || pair.rawFinding;
  const status = document.createElement("select");
  ["change", "no_change", "unclear"].forEach((value) => {
    const option = document.createElement("option"); option.value = value; option.textContent = value.replace("_", " "); option.selected = finding?.status === value; status.append(option);
  });
  status.addEventListener("change", () => updatePairFinding(pair, { ...(finding || emptyFinding()), status: status.value }));
  const difference = document.createElement("textarea");
  difference.rows = 2; difference.value = finding?.beforeAfterDifference || ""; difference.placeholder = "Describe the visible before/after difference"; difference.dataset.field = `pair-${pair.pairId}-difference`;
  difference.addEventListener("input", () => updatePairFinding(pair, { ...(finding || emptyFinding()), beforeAfterDifference: difference.value }));
  const uncertainty = document.createElement("input"); uncertainty.value = finding?.uncertainty || ""; uncertainty.placeholder = "Uncertainty note (optional)"; uncertainty.dataset.field = `pair-${pair.pairId}-uncertainty`;
  uncertainty.addEventListener("input", () => updatePairFinding(pair, { ...(finding || emptyFinding()), uncertainty: uncertainty.value || null }));
  const disposition = document.createElement("select");
  [["", "Choose disposition"], ["include", "Include in guide"], ["skip", "Explicitly skip"]].forEach(([value, label]) => { const option = document.createElement("option"); option.value = value; option.textContent = label; option.selected = (pair.disposition || "") === value; disposition.append(option); });
  disposition.addEventListener("change", () => dispatch({ type: "EDIT_PAIR_REVIEW", pairId: pair.pairId, disposition: disposition.value || null }));
  const text = document.createElement("p"); text.textContent = pair.error?.message || (pair.status === "completed" ? "Review the finding before saving." : "Comparison pending.");
  card.append(status, difference, uncertainty, disposition, text);
  if (pair.status === "failed") {
    const retry = document.createElement("button"); retry.type = "button"; retry.className = "button button-secondary"; retry.textContent = "Retry pair"; retry.addEventListener("click", () => compareManualPair(pair.pairId)); card.append(retry);
  }
  return card;
}

function emptyFinding() {
  return { status: "unclear", beforeAfterDifference: "", changedPieceDescription: null, receivingPieceDescription: null, receivingLocation: null, supportedPlacement: null, uncertainty: null, reason: null, suggestion: null };
}

function updatePairFinding(pair, reviewedFinding) {
  dispatch({ type: "EDIT_PAIR_REVIEW", pairId: pair.pairId, reviewedFinding }, true);
}

function renderAnnotation() {
  const active = state.status === "annotating" && state.pipeline !== "manual_pairs";
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
  [...state.annotations, ...(selectedAnnotationPoint ? [{ name: "new", frameIndex: Number(elements.annotationFrame.value), points: [selectedAnnotationPoint] }] : [])]
    .filter((item) => item.frameIndex === Number(elements.annotationFrame.value)).forEach((item) => {
      const point = item.points[0];
      if (!point) return;
      const marker = document.createElement("i");
      marker.className = "annotation-point";
      marker.style.left = `${point.x / elements.annotationImage.naturalWidth * 100}%`;
      marker.style.top = `${point.y / elements.annotationImage.naturalHeight * 100}%`;
      elements.annotationMarkers.append(marker);
    });
  elements.addAnnotationButton.disabled = !selectedAnnotationPoint || !elements.annotationName.value.trim();
  const actionFirst = state.pipeline !== "plan3";
  elements.trackBackwardButton.hidden = actionFirst;
  elements.analyzePipelineButton.hidden = !actionFirst;
  elements.trackBackwardButton.disabled = state.annotations.length === 0;
  elements.analyzePipelineButton.disabled = state.annotations.length === 0;
  elements.annotationList.replaceChildren(...state.annotations.map((item) => {
    const line = document.createElement("div");
    const name = document.createElement("input");
    name.value = item.name;
    name.setAttribute("aria-label", `Name for part ${item.partId}`);
    name.addEventListener("input", (event) => dispatch({ type: "EDIT_ANNOTATION_NAME", partId: item.partId, value: event.target.value }));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${item.name}`);
    remove.addEventListener("click", () => dispatch({ type: "DELETE_ANNOTATION", partId: item.partId }));
    const frame = document.createElement("small");
    frame.textContent = `frame ${item.frameIndex + 1}`;
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
    line.textContent = `${suggestions.suggestions.length} suggested part${suggestions.suggestions.length === 1 ? "" : "s"}. Review before accepting.`;
    const accept = document.createElement("button");
    accept.type = "button";
    accept.className = "button button-secondary";
    accept.textContent = "Accept suggestions";
    accept.addEventListener("click", () => dispatch({ type: "ACCEPT_SUGGESTIONS" }));
    elements.suggestionList.append(line, accept);
  }
}

function renderVerification() {
  const verification = state.verification;
  elements.verificationSummary.hidden = !verification;
  if (!verification) return;
  elements.verificationSummary.replaceChildren();
  const heading = document.createElement("strong");
  const result = verification.resultStatus;
  heading.textContent = verification.status === "stale"
    ? "Image check is stale"
    : result === "unresolved"
      ? "Image review left unresolved actions"
      : result === "needs_review"
        ? "Image review needs attention"
        : verification.status === "completed"
          ? `Image check complete · ${Math.round((verification.coverage || 0) * 100)}% covered`
          : "Image check unavailable";
  const detail = document.createElement("p");
  const counts = `Supported ${verification.supportedCount || 0} · rejected ${verification.rejectedCount || 0} · unresolved ${verification.unresolvedCount || 0}`;
  detail.textContent = verification.message ? `${verification.message} ${counts}` : `${counts}.`;
  elements.verificationSummary.append(heading, detail);
  (verification.reviewPasses || []).forEach((pass) => {
    const item = document.createElement("p");
    item.textContent = `${pass.kind} review · ${Math.round((pass.coverage || 0) * 100)}% covered`;
    elements.verificationSummary.append(item);
  });
  (verification.findings || []).forEach((finding) => {
    const item = document.createElement("p");
    item.textContent = `${finding.kind}: ${finding.rationale}`;
    elements.verificationSummary.append(item);
  });
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
    const targetName = attachmentTargetName(track, state.events, state.partTracks);
    if (targetName) {
      line.textContent = `${track.name} · joins ${targetName} near assembly frames ${state.frames.length - track.attachmentStartFrame}–${state.frames.length - track.attachmentEndFrame}`;
    } else {
      line.textContent = track.attachmentStartFrame === null ? `${track.name} · no confident attachment range` : `${track.name} · joins another tracked part near assembly frames ${state.frames.length - track.attachmentStartFrame}–${state.frames.length - track.attachmentEndFrame}`;
    }
    return line;
  }));
}

function renderEvents() {
  elements.eventsList.replaceChildren();
  elements.timelineCount.textContent = `${state.events.length} event${state.events.length === 1 ? "" : "s"}`;
  state.events.forEach((event, index) => {
    const card = document.createElement("article");
    card.className = "event-card";
    const heading = document.createElement("div");
    heading.className = "event-card-heading";
    const title = document.createElement("h3");
    title.textContent = `${String(index + 1).padStart(2, "0")} · ${eventLabel(event.kind)}`;
    const review = document.createElement("span");
    review.className = "review-badge";
    review.textContent = eventDisposition(event.eventId);
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = `${formatSourceTime(event.startTimestampSeconds)}–${formatSourceTime(event.endTimestampSeconds)}`;
    heading.append(title, review, time);
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

function renderActionTimeline(actions) {
  elements.eventsList.replaceChildren();
  elements.timelineCount.textContent = `${actions.length} action${actions.length === 1 ? "" : "s"}`;
  actions.forEach((action, index) => {
    const card = document.createElement("article");
    card.className = "event-card";
    const heading = document.createElement("div");
    heading.className = "event-card-heading";
    const title = document.createElement("h3");
    title.textContent = `${String(index + 1).padStart(2, "0")} · ${actionLabel(action.actionType)}`;
    const review = document.createElement("span");
    review.className = "review-badge";
    review.textContent = actionDisposition(action);
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = `${formatSourceTime(action.startTimestampSeconds)}–${formatSourceTime(action.endTimestampSeconds)}`;
    heading.append(title, review, time);
    card.append(heading);
    const participants = participantText(action.movingPartId, action.receivingPartId);
    if (participants) {
      const participantLine = document.createElement("p");
      participantLine.className = "event-evidence";
      participantLine.textContent = participants;
      card.append(participantLine);
    }
    const pair = document.createElement("div");
    pair.className = "event-pair";
    if (action.beforeFrameId || action.afterFrameId) pair.append(eventFrame("Before", action.beforeFrameId), eventFrame("After", action.afterFrameId));
    card.append(pair);
    const evidence = document.createElement("p");
    evidence.className = "event-evidence";
    evidence.textContent = action.evidence;
    card.append(evidence);
    if (action.uncertainty) {
      const uncertainty = document.createElement("p");
      uncertainty.className = "event-uncertainty";
      uncertainty.textContent = `Uncertain: ${action.uncertainty}`;
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
  caption.textContent = frame ? `${label} · Source video ${formatSourceTime(frame.timestampSeconds)}` : label;
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

function actionLabel(kind) {
  return { attach: "Piece attached", detach: "Piece detached", move: "Piece moved", separate: "Pieces separated", unknown: "Action needs review" }[kind] || "Assembly action";
}

function renderSteps() {
  elements.stepsList.replaceChildren();
  state.draftGuide.steps.forEach((step, index) => {
    const frame = state.frames.find((candidate) => candidate.frameId === step.frameId) || state.frames[0];
    const card = document.createElement("article");
    card.className = "step-card";
    const label = step.kind === "review" || step.reviewStatus === "unresolved" ? "Review required" : "Assembly step";
    card.innerHTML = `<div class="step-card-top"><span class="step-number">${String(index + 1).padStart(2, "0")}</span><span class="step-label">${label}</span><button class="icon-button delete-step" type="button" aria-label="Delete step ${index + 1}" data-index="${index}">×</button></div>`;

    const media = document.createElement("div");
    media.className = "step-media";
    if (frame) {
      const image = document.createElement("img");
      image.src = frame.imageUrl;
      image.alt = `Source video frame at ${formatSourceTime(frame.timestampSeconds)}`;
      media.append(image);
      const timestamp = document.createElement("span");
      timestamp.className = "timestamp";
      timestamp.textContent = `Source video ${formatSourceTime(frame.timestampSeconds)}`;
      media.append(timestamp);
    } else {
      media.textContent = "Choose an extracted frame";
      media.classList.add("empty-media");
    }
    card.append(media);

    const content = document.createElement("div");
    content.className = "step-content";
    const participants = participantText(step.movingPartId, step.receivingPartId);
    if (participants) {
      const participantLine = document.createElement("p");
      participantLine.className = "step-participants";
      participantLine.textContent = participants;
      content.append(participantLine);
    }
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
      option.textContent = `Source video ${formatSourceTime(candidate.timestampSeconds)}`;
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
  if (current.pipeline !== "plan3" && current.pipelineProgress && current.phase === "processing") return Math.round(current.pipelineProgress.progress * 100);
  if (current.status === "analyzing" && typeof current.trackingProgress === "number") return 55 + Math.round(current.trackingProgress * 29);
  return { queued: 20, extracting: 40, suggesting: 48, annotating: 52, analyzing: 68, generating: 84, verifying: 92, correcting: 94, ready: 100, failed: 100 }[current.status] || 0;
}

function detailText(current) {
  if (current.phase === "uploading") return `Uploading ${current.uploadName || "your video"}…`;
  if (current.status === "suggesting") return "Finding visual piece suggestions for your review…";
  if (current.status === "annotating") return current.pipeline === "manual_pairs" ? "Select settled snapshots, compare each adjacent pair, and explicitly include or skip every difference." : "Click and name every clearly separated part. The final frame is selected by default; use an earlier frame only as a fallback.";
  if (current.pipeline !== "plan3" && current.pipelineProgress?.message) return current.pipelineProgress.message;
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

function formatSourceTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${value.toFixed(1).replace(/\.0$/, "")}s`;
}

function annotationName(partId) {
  return state.annotations.find((annotation) => annotation.partId === partId)?.name || `part ${partId}`;
}

function participantText(movingPartId, receivingPartId) {
  const participants = [];
  if (movingPartId) participants.push(`moving: ${annotationName(movingPartId)}`);
  if (receivingPartId) participants.push(`receiving: ${annotationName(receivingPartId)}`);
  return participants.join(" · ");
}

function actionDisposition(action) {
  if (action.uncertainty) return "Needs review";
  const mapping = state.timeline?.assemblyActions?.find((candidate) => candidate.sourceActionIds?.includes(action.actionId));
  if (mapping?.disposition === "review") return "Needs review";
  const findings = state.verification?.findings || [];
  const relevant = findings.filter((finding) => finding.intervalId === action.actionId);
  if (relevant.some((finding) => ["rejected", "unresolved"].includes(finding.disposition))) return "Needs review";
  if (relevant.some((finding) => ["supported", "merged"].includes(finding.disposition))) return "Supported";
  return "Action hypothesis";
}

function eventDisposition(eventId) {
  const relevant = (state.verification?.findings || []).filter((finding) => finding.intervalId === eventId);
  if (relevant.some((finding) => ["rejected", "unresolved"].includes(finding.disposition))) return "Needs review";
  if (relevant.some((finding) => ["supported", "merged"].includes(finding.disposition))) return "Supported";
  return state.verification?.status === "stale" ? "Check stale" : "Local evidence";
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
    dispatch({ type: "UPLOAD_ACCEPTED", jobId: created.jobId, pipeline: state.pipeline });
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
  if (!state.jobId) return;
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

async function handleVerify() {
  const validation = validateDraft(state.draftGuide, state.frames);
  if (validation) {
    dispatch({ type: "SAVE_FAILED", error: { code: "INVALID_GUIDE", message: validation } });
    return;
  }
  try {
    const result = await api.verifyGuide(state.jobId, state.draftGuide, state.guideRevision);
    dispatch({ type: "JOB_UPDATE", job: result });
    await beginPolling(state.jobId);
  } catch (error) {
    dispatch({ type: "SAVE_FAILED", error: normalizeError(error) });
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
    const sameFrame = annotations[index].frameIndex === Number(elements.annotationFrame.value);
    annotations[index] = {
      ...annotations[index],
      frameIndex: Number(elements.annotationFrame.value),
      points: sameFrame ? [...annotations[index].points, selectedAnnotationPoint] : [selectedAnnotationPoint],
      labels: sameFrame ? [...annotations[index].labels, 1] : [1],
    };
  } else {
    const partId = Math.max(0, ...annotations.map((item) => item.partId || 0)) + 1;
    annotations.push({ partId, name, frameIndex: Number(elements.annotationFrame.value), points: [selectedAnnotationPoint], labels: [1] });
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

async function analyzePipeline() {
  try {
    const saved = await api.saveAnnotations(state.jobId, state.annotations);
    dispatch({ type: "JOB_UPDATE", job: saved });
    const started = await api.analyzeJob(state.jobId);
    dispatch({ type: "JOB_UPDATE", job: started });
    await beginPolling(state.jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

async function saveManualStoryboard() {
  try {
    const saved = await api.saveStoryboard(state.jobId, state.manualSelection, state.manualContext, state.storyboard?.revision ?? null);
    dispatch({ type: "JOB_UPDATE", job: saved });
  } catch (error) {
    dispatch({ type: "SAVE_FAILED", error: normalizeError(error) });
  }
}

async function compareManualPair(pairId = null) {
  try {
    const started = await api.comparePairs(state.jobId, pairId);
    dispatch({ type: "JOB_UPDATE", job: started });
    await beginPolling(state.jobId);
  } catch (error) {
    dispatch({ type: "UPLOAD_FAILED", error: normalizeError(error) });
  }
}

async function saveManualDifferences() {
  try {
    const saved = await api.saveDifferences(state.jobId, state.storyboard);
    dispatch({ type: "JOB_UPDATE", job: saved });
  } catch (error) {
    dispatch({ type: "SAVE_FAILED", error: normalizeError(error) });
  }
}

async function generateManualGuide() {
  try {
    const started = await api.generateGuide(state.jobId, state.storyboard?.revision ?? null);
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
elements.verifyButton.addEventListener("click", handleVerify);
elements.addStepButton.addEventListener("click", () => dispatch({ type: "ADD_STEP" }));
elements.annotationFrame.addEventListener("input", () => { selectedAnnotationPoint = null; renderAnnotation(); });
elements.annotationName.addEventListener("input", renderAnnotation);
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
elements.analyzePipelineButton.addEventListener("click", analyzePipeline);
elements.manualFrame.addEventListener("input", renderManual);
elements.manualPrev.addEventListener("click", () => { elements.manualFrame.value = String(Math.max(0, Number(elements.manualFrame.value) - 1)); renderManual(); });
elements.manualNext.addEventListener("click", () => { elements.manualFrame.value = String(Math.min(state.frames.length - 1, Number(elements.manualFrame.value) + 1)); renderManual(); });
elements.manualAdd.addEventListener("click", () => {
  const frame = selectedFrame(state.frames, elements.manualFrame);
  if (frame && !state.manualSelection.includes(frame.frameId)) dispatch({ type: "EDIT_MANUAL_SELECTION", selectedFrameIds: [...state.manualSelection, frame.frameId].sort((left, right) => state.frames.find((item) => item.frameId === left).timestampSeconds - state.frames.find((item) => item.frameId === right).timestampSeconds) });
});
elements.manualContext.addEventListener("input", (event) => dispatch({ type: "EDIT_MANUAL_CONTEXT", value: event.target.value }, true));
elements.manualSave.addEventListener("click", saveManualStoryboard);
elements.manualCompare.addEventListener("click", () => compareManualPair());
elements.manualReviewSave.addEventListener("click", saveManualDifferences);
elements.manualGenerate.addEventListener("click", generateManualGuide);
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
api.getCapabilities().then((result) => {
  dispatch({ type: "CAPABILITIES_LOADED", capabilities: result });
  const selected = result.pipelines.find((item) => item.pipeline === state.pipeline) || result.pipelines[0];
  if (selected) {
    state = { ...state, pipeline: selected.pipeline };
    elements.pipelineSelect.value = selected.pipeline;
    elements.pipelineDisclosure.textContent = selected.available ? selected.disclosure : `Unavailable: ${selected.missing.join(", ")}`;
    render();
  }
}).catch(() => {
  elements.pipelineDisclosure.textContent = "Capabilities unavailable; the server will validate the selected pipeline.";
});
const restoredJobId = params.get("job");
if (restoredJobId) restoreJob(restoredJobId);
