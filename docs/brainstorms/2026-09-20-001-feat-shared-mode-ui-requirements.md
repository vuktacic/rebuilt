---
title: "feat: Revamp the shared Rebuilt UI for automated and manual workflows"
type: feat
status: requirements
created: 2026-09-20
---

# feat: Revamp the shared Rebuilt UI for automated and manual workflows

## Problem Frame

Rebuilt currently presents one upload-to-guide workflow. The automated path extracts frames, pauses for named prompts when backward tracking is selected, derives candidate assembly changes, and presents an editable guide. The next backend iteration adds a second path in which a reviewer scrubs the extracted video, explicitly selects before/after frame pairs, asks a pairwise vision model to describe only the visible difference, and sends those compact findings to a lower-cost drafting model.

Both paths must remain understandable as one product. The browser must not lose manual selections when the server is idle in an interactive state, overwrite an unsaved pair list during polling, or hide whether a guide came from automated tracking, manual pairs, or a degraded result. A parallel UI agent needs a stable behavioral and contract specification while backend processing is reorganized.

## Actors

- **A1. Builder/reviewer:** uploads a disassembly recording, chooses a workflow, reviews evidence, and edits or saves the generated guide.
- **A2. UI implementer:** rebuilds the plain HTML/CSS/JavaScript interface against the shared job and review contract while backend work proceeds.
- **A3. Backend/model pipeline:** extracts stable frame references, persists user selections, runs the selected analysis path, validates model output, and exposes reviewable results.

## Key Flows

### F1. Choose a workflow and upload a recording

1. The user chooses **Automated tracking** or **Manual pairs** before starting analysis.
2. The browser uploads the video with the selected mode and shows the same upload, progress, error, and reset shell for both paths.
3. The created job records its mode as immutable job metadata. Historic jobs without a mode continue to render as automated jobs.

### F2. Automated tracking and review

1. The job extracts stable frame IDs and enters the existing automated path.
2. If the configured local analyzer supports backward tracking, the job pauses in an annotation state where the user names visible parts and saves prompts.
3. The analyzer tracks prompts backward, exposes a review overlay, and derives event pairs in assembly chronology.
4. Existing automated LLM review and guide editing remain available without requiring the manual-pair controls.

### F3. Manual pair selection and two-stage generation

1. After extraction, the job enters an interactive **pairing** state rather than starting a local tracker.
2. The user scrubs the extracted timeline, sets a **before** frame and an **after** frame for one physical change, and adds the pair to an ordered list.
3. The UI labels both raw source time and build/assembly time. Pair roles are semantic: `beforeFrameId` is the separated state before the assembly action and `afterFrameId` is the attached state after it, even when the source footage runs in disassembly order.
4. The user can replace, remove, reorder, and save pairs without sending images to a model. The browser keeps unsaved pair edits separate from the last server snapshot.
5. An explicit **Analyze selected changes** action starts bounded pairwise visual review. Astra receives only the selected pair images plus stable labels and returns one compact, structured difference per pair.
6. Luna receives the compact validated differences and pair chronology, not the original images or whole video, and returns an editable guide draft whose reference frames are drawn from the selected pairs.
7. The UI shows pair-level review status, uncertainty, and model failure without hiding the saved pair list or blocking manual guide editing when a degraded draft is available.

### F4. Resume, failure, and reset

1. Polling stops at interactive states (`annotating` and `pairing`) so the browser does not dispatch stale server snapshots over unsaved inputs.
2. Reloading a saved job restores the mode, extracted frames, saved pairs/annotations, review status, and guide draft from the server.
3. A provider failure preserves local evidence and the last completed stage. The user can retry the failed stage or edit the guide manually; live errors never silently switch to mock data.
4. A reset abandons the current client draft, aborts polling, and starts a new analysis without mutating the prior job.

## Requirements

### Shared job and timeline behavior

- **R1. Mode identity:** Every new job records `mode: automated | manual`. The mode is visible in the workspace and controls which actions are available; it cannot change after processing begins.
- **R2. Stable frame references:** Every frame shown or submitted by the UI uses the server-provided `frameId`, image URL, raw source timestamp, and canonical assembly/build time. The UI must not substitute array position for frame identity.
- **R3. Explicit chronology:** The contract exposes enough metadata for the UI to distinguish source/disassembly chronology from assembly chronology. The browser consumes the server's conversion instead of independently reversing timestamps.
- **R4. Shared guide editing:** Both modes project into the existing editable `Guide` shape and use the same dirty-state, validation, save, and reset behavior.
- **R5. Backward compatibility:** Historic ready jobs, existing automated fixtures, and clients that omit `mode` or manual fields continue to render without errors.

### Manual pair workspace

- **R6. Pair draft:** A manual pair contains a stable `pairId`, an ordered sequence, `beforeFrameId`, and `afterFrameId`. It may include server-generated status and uncertainty, but the first version does not accept arbitrary free-text notes into model prompts.
- **R7. Pair validation:** The server rejects unknown frame IDs, duplicate pair IDs, empty pairs, invalid role references, and pairs that violate the supported assembly chronology. The UI shows the returned field-level or pair-level error without discarding other valid selections.
- **R8. Non-destructive editing:** The UI supports setting, replacing, removing, and reordering pairs. A server poll or unrelated job update must not erase a dirty client pair draft.
- **R9. Explicit start:** Saving pairs and running pair analysis are separate actions. The run action is disabled until at least one valid pair is saved and the job is in `pairing`.
- **R10. Bounded visual diff:** Astra receives only labelled before/after images for the submitted pairs, with a configured pair/image cap and intentional image detail setting. Adjacent pairs may share an image only when the payload preserves each pair/role label.
- **R11. Compact handoff:** Luna receives only validated compact pair findings, chronology, and allowable frame references. It does not receive the original video, arbitrary filesystem paths, or the pair images in the drafting stage.
- **R12. Reviewability:** Each submitted pair has a visible `pending`, `completed`, `needs_review`, or `failed` result. The UI shows the compact difference and uncertainty tied to that pair.
- **R13. Degraded completion:** If Astra fails, completed local pair selections remain editable and the UI offers a retry. If Astra succeeds but Luna fails, the pair findings remain visible and the user can create or edit a guide manually. No failed or uncertain result is presented as a verified instruction.

### Automated mode parity

- **R14. Existing path remains intact:** Automated tracking retains its prompt annotation, backward propagation, overlay review, event timeline, and existing LLM-review presentation. Manual pair controls are not shown as required steps in automated mode.
- **R15. Shared evidence language:** Automated events and manual pairs both display before/after evidence, build chronology, uncertainty, and model provenance using the same visual vocabulary.
- **R16. Shared accessibility floor:** New controls use semantic buttons, labels, keyboard-accessible timeline controls, visible focus states, status announcements, and text alternatives for evidence images. Color is not the only distinction between state badges.

## Contract Shape for Parallel UI Work

The following is the behavioral contract the UI agent may target. Exact field casing should follow the existing camelCase API convention.

```json
{
  "jobId": "job-123",
  "mode": "manual",
  "status": "pairing",
  "frames": [
    {
      "frameId": "frame-0012",
      "sourceIndex": 12,
      "timestampSeconds": 6.0,
      "assemblyTimeSeconds": 4.5,
      "imageUrl": "/jobs/job-123/frames/frame-0012"
    }
  ],
  "manualPairs": [
    {
      "pairId": "pair-0001",
      "sequence": 1,
      "beforeFrameId": "frame-0012",
      "afterFrameId": "frame-0015"
    }
  ],
  "manualReview": {
    "status": "not_started",
    "pairs": [],
    "guideStatus": "not_started"
  },
  "guide": null,
  "error": null
}
```

The automated path may keep `manualPairs` and `manualReview` null or empty. The UI must tolerate absent optional fields, but a manual job must receive an explicit pairing/review status rather than inferring it from an empty array.

## Acceptance Examples

- **AE1. Pair draft survives an interactive poll:** While a manual job is `pairing`, the user selects three pairs and changes the second pair locally. A polling update with unchanged server pairs does not revert the dirty draft or move the UI out of the pairing workspace.
- **AE2. Assembly semantics are visible:** For disassembly footage, a pair whose raw source indices decrease from separated to attached is still labelled with the semantic assembly roles “Before” and “After,” and the guide uses the server-provided assembly time rather than raw source order.
- **AE3. Two-stage payload boundary:** A manual run sends Astra exactly the bounded labelled image pairs and receives one validated result per pair. The Luna request contains the compact results and frame references but no image blocks or local filesystem paths.
- **AE4. Partial provider failure is reviewable:** If Astra completes two of three pairs and fails on the third, the UI preserves all three saved pairs, shows two completed results and one retryable failure, and does not create a falsely verified guide step for the failed pair.
- **AE5. Automated compatibility:** A historic automated ready fixture with no `mode`, `manualPairs`, or `manualReview` renders the existing event timeline and editable guide without console errors or manual controls.
- **AE6. Shared guide save:** A guide drafted from either mode uses the existing guide save contract, rejects an unknown frame reference client/server-side, and preserves unsaved edits when the save request fails.
- **AE7. Reset isolation:** Resetting a manual job aborts its polling and clears only client state; reloading the previous job URL restores its persisted pairs and review results.

## Key Decisions

| Decision | Rationale |
|---|---|
| Choose the mode before upload and persist it on the job. | The backend can select the correct lifecycle from the start, while historic jobs remain compatible through an automated default. |
| Give manual selection its own interactive `pairing` state. | It is not local object annotation and should not reuse automated tracking controls or polling semantics. |
| Use explicit semantic before/after fields plus canonical assembly time. | The product reverses disassembly footage; numeric source order alone is unsafe for instructions. |
| Separate `manualPairs`/`manualReview` from automated `events` and local tracks. | User-selected evidence and model findings are different provenance layers and should remain inspectable. |
| Keep Astra and Luna as configurable provider roles with initial defaults `gpt-6-astra` and `gpt-5.6-luna`. | The product contract describes capabilities, while model IDs, limits, and deployment access can change without a UI rewrite. |
| Require strict structured output and server-side reference validation. | Model output can be syntactically valid while still naming an unknown pair, frame, or unsupported claim. |
| Preserve the existing plain HTML/CSS/JavaScript design language. | The current app already has a restrained editorial visual system; a second component framework would slow parallel work and create inconsistent states. |

## Scope Boundaries

### In scope

- A shared mode chooser and mode-aware workspace states.
- A manual pair scrubber with stable frame identity and assembly chronology.
- Pair save/edit/reorder/run states and review badges.
- Shared guide editing, save, reset, error, and degraded-result behavior.
- API fixtures and state contracts that let the UI agent work before live model calls exist.

### Deferred for later

- Automatic suggestions for where a physical change occurred.
- Fine-grained LEGO part, color, stud-count, or orientation recognition.
- Whole-video multimodal prompting or video uploads directly to a model.
- Collaborative multi-user editing, comments, or live presence.
- Automatic re-running after every guide edit.

### Outside this product's identity

- A general-purpose video editor.
- A freeform image annotation or computer-vision labeling platform.
- A fully autonomous instruction publisher that bypasses human review.

## Dependencies and Assumptions

- The existing frame extractor continues to produce stable frame IDs and JPEG evidence URLs.
- The backend remains one-active-job-at-a-time for the initial manual mode.
- The existing `Guide` save contract remains the compatibility boundary for the editor.
- The first manual version uses image-pair evidence only; no model receives the entire uploaded video.
- Model availability and exact account limits are checked during backend implementation and represented as sanitized configuration/readiness errors.

## Open Questions

- Whether the first UI should expose a low/high image-detail toggle or keep it as an operator setting until pairwise fixtures establish a quality/cost baseline.
- Whether a partial Astra result should allow Luna to draft immediately from completed pairs or wait for an explicit user choice to omit failed pairs.
- Whether the UI should allow a pair to be marked “needs review” before model analysis, or reserve that state for validated model output.

## Sources and Existing Patterns

- Current job lifecycle, annotation pause, and frame endpoint: `backend/app/main.py`.
- Current durable atomic metadata and interrupted-job handling: `backend/app/storage.py`.
- Current frame/event/guide models: `backend/app/models.py`.
- Current automated processing and OpenAI Responses transport: `backend/app/processing.py`.
- Current browser reducer, dirty-guide preservation, and polling: `web/src/state.js` and `web/src/api.js`.
- Current evidence and annotation presentation: `web/src/app.js` and `web/index.html`.
- Current API compatibility contract and fixtures: `contracts/job.schema.json`, `contracts/README.md`, and `contracts/fixtures/`.
- Related automated post-processing design: `docs/plans/2026-09-19-002-feat-llm-detection-review-plan.md`.
