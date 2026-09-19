import { FAILED_JOB_FIXTURE, SAVE_VALIDATION_ERROR_FIXTURE, SUCCESSFUL_GUIDE_FIXTURE } from "./mock-fixtures.js";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "REQUEST_FAILED" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

function copy(value) {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

async function readResponse(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(error?.message || `Request failed with HTTP ${response.status}.`, {
      status: response.status,
      code: error?.code || "REQUEST_FAILED",
    });
  }
  return body;
}

function requestFailure(error) {
  if (error instanceof ApiError) return error;
  return new ApiError(error?.message || "The server could not be reached.", { code: "NETWORK_ERROR" });
}

function liveClient(fetchImpl) {
  const request = async (url, options = {}) => {
    try {
      return await readResponse(await fetchImpl(url, options));
    } catch (error) {
      throw requestFailure(error);
    }
  };

  return {
    mode: "live",

    async uploadVideo(file) {
      const form = new FormData();
      form.append("video", file, file.name);
      return request("/jobs", { method: "POST", body: form });
    },

    getJob(jobId) {
      return request(`/jobs/${encodeURIComponent(jobId)}`);
    },

    saveGuide(jobId, guide) {
      return request(`/jobs/${encodeURIComponent(jobId)}/guide`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(guide),
      });
    },

    async pollJob(jobId, { onUpdate, signal, intervalMs = 2000 } = {}) {
      while (true) {
        if (signal?.aborted) throw new ApiError("Polling was cancelled.", { code: "ABORTED" });
        const job = await this.getJob(jobId);
        onUpdate?.(job);
        if (job.status === "ready" || job.status === "failed") return job;
        await wait(intervalMs, signal);
      }
    },
  };
}

function wait(milliseconds, signal) {
  if (milliseconds <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new ApiError("Polling was cancelled.", { code: "ABORTED" }));
    }, { once: true });
  });
}

function mockClient(options = {}) {
  const jobs = options.jobs || new Map();
  const storage = options.storage || globalThis.localStorage;
  const failSave = options.failSave === true;
  let nextId = options.nextId || 1;

  const remember = (job) => {
    jobs.set(job.jobId, copy(job));
    try {
      storage?.setItem(`rebuilt:mock:${job.jobId}`, JSON.stringify(job));
    } catch {
      // Private browsing and test storage can reject writes; the in-memory mock still works.
    }
  };

  const restore = (jobId) => {
    if (jobs.has(jobId)) return jobs.get(jobId);
    try {
      const saved = storage?.getItem(`rebuilt:mock:${jobId}`);
      if (saved) {
        const job = JSON.parse(saved);
        jobs.set(jobId, job);
        return job;
      }
    } catch {
      return null;
    }
    return null;
  };

  const baseFrames = () => [
    { frameId: "frame-0001", timestampSeconds: 0, imageUrl: svgFrame("1", "#e5b94d") },
    { frameId: "frame-0002", timestampSeconds: 1, imageUrl: svgFrame("2", "#6f9fc8") },
  ];

  const getJob = async (jobId) => {
    const job = restore(jobId);
    if (!job) throw new ApiError("The requested job does not exist.", { status: 404, code: "JOB_NOT_FOUND" });
    if (job.status === "ready" || job.status === "failed") return copy(job);
    job.polls = (job.polls || 0) + 1;
    const failed = job.shouldFail;
    if (failed && job.polls >= 2) {
      job.status = "failed";
      job.error = { ...FAILED_JOB_FIXTURE.error, message: "The mock processor could not read this video." };
    } else if (job.polls === 1) {
      job.status = "queued";
    } else if (job.polls === 2) {
      job.status = "extracting";
      job.frames = baseFrames();
    } else if (job.polls === 3) {
      job.status = "generating";
    } else {
      job.status = "ready";
      job.guide = copy(SUCCESSFUL_GUIDE_FIXTURE.guide);
    }
    remember(job);
    return copy(job);
  };

  return {
    mode: "mock",

    async uploadVideo(file) {
      const jobId = `mock-job-${nextId++}`;
      const job = {
        jobId,
        status: "queued",
        frames: [],
        guide: null,
        error: null,
        polls: 0,
        shouldFail: file?.name?.toLowerCase().includes("fail"),
      };
      remember(job);
      return { jobId };
    },

    getJob,

    async saveGuide(jobId, guide) {
      const job = restore(jobId);
      if (!job) throw new ApiError("The requested job does not exist.", { status: 404, code: "JOB_NOT_FOUND" });
      if (failSave) throw new ApiError("The mock save failed; your edits remain local.", { status: 503, code: "MOCK_SAVE_FAILED" });
      const frameIds = new Set(job.frames.map((frame) => frame.frameId));
      if (!guide.title.trim() || !guide.steps.length || guide.steps.some((step) => !step.text.trim() || !frameIds.has(step.frameId))) {
        throw new ApiError(SAVE_VALIDATION_ERROR_FIXTURE.error.message, { status: 422, code: "INVALID_GUIDE" });
      }
      job.guide = copy(guide);
      job.status = "ready";
      remember(job);
      return copy(job.guide);
    },

    async pollJob(jobId, { onUpdate, signal, intervalMs = 2000 } = {}) {
      while (true) {
        if (signal?.aborted) throw new ApiError("Polling was cancelled.", { code: "ABORTED" });
        const job = await getJob(jobId);
        onUpdate?.(job);
        if (job.status === "ready" || job.status === "failed") return job;
        await wait(intervalMs, signal);
      }
    },
  };
}

function svgFrame(label, color) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360"><rect width="640" height="360" fill="#f5f0e6"/><rect x="110" y="160" width="420" height="90" rx="12" fill="${color}"/><circle cx="220" cy="145" r="22" fill="#d19b3d"/><circle cx="320" cy="145" r="22" fill="#d19b3d"/><circle cx="420" cy="145" r="22" fill="#d19b3d"/><text x="320" y="315" text-anchor="middle" font-family="sans-serif" font-size="32" fill="#273444">Mock frame ${label}</text></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

export function createApiClient({ mock = false, fetchImpl = globalThis.fetch, ...options } = {}) {
  if (mock) return mockClient(options);
  if (typeof fetchImpl !== "function") throw new Error("Live mode requires fetch.");
  return liveClient(fetchImpl);
}
