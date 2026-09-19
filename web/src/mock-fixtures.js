// These are the browser harness counterparts of contracts/fixtures/*.json.
// Frame URLs are replaced with local SVG data URLs by api.js so mock mode works offline.
export const PROCESSING_FIXTURE = {
  jobId: "job-processing",
  status: "extracting",
  frames: [],
  guide: null,
  error: null,
};

export const SUCCESSFUL_GUIDE_FIXTURE = {
  jobId: "job-ready",
  status: "ready",
  frames: [
    { frameId: "frame-0001", timestampSeconds: 0, imageUrl: "/jobs/job-ready/frames/frame-0001" },
    { frameId: "frame-0002", timestampSeconds: 1, imageUrl: "/jobs/job-ready/frames/frame-0002" },
  ],
  tracks: [
    { trackId: "LEGO piece:track-0001", concept: "LEGO piece", firstTimestampSeconds: 0, lastTimestampSeconds: 1, visibility: "occluded", membership: "attached" },
  ],
  events: [
    {
      eventId: "event-0001",
      kind: "uncertain_change",
      startTimestampSeconds: 0,
      endTimestampSeconds: 1,
      affectedTrackIds: ["LEGO piece:track-0001"],
      beforeFrameId: "frame-0001",
      afterFrameId: "frame-0002",
      evidenceStrength: 0.55,
      uncertainty: "The hand briefly covered the connection point.",
      evidence: "The blue piece remained beside the base, then stayed attached.",
    },
  ],
  analysis: { backend: "sam3", modelVersion: "facebook/sam3:bf16", configVersion: "v2", durationSeconds: 18.4, metrics: {} },
  guide: {
    title: "Small brick model",
    steps: [
      { text: "Place the red brick on the base.", frameId: "frame-0001", uncertainty: null },
      { text: "Add the blue brick on the right side.", frameId: "frame-0002", uncertainty: "The hand obscures the final alignment." },
    ],
  },
  error: null,
};

export const UNCERTAIN_STEP_FIXTURE = SUCCESSFUL_GUIDE_FIXTURE.guide.steps[1];

export const FAILED_JOB_FIXTURE = {
  jobId: "job-failed",
  status: "failed",
  frames: [],
  tracks: [],
  events: [],
  analysis: null,
  guide: null,
  error: { code: "PROCESSING_FAILED", message: "The video could not be processed." },
};

export const SAVE_VALIDATION_ERROR_FIXTURE = {
  error: { code: "INVALID_GUIDE", message: "Each step must contain non-empty text and reference an extracted frame." },
};
