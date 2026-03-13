import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { createLogger } from "../utils/logger.js";
import type { OpenAiToolCall } from "../proxy/tool-loop.js";

const log = createLogger("edit-file-resolver");

/**
 * Cursor's edit_file tool sends a "sketch" in streamContent: partial file
 * content with elision markers (e.g. `// ... existing code ...`) representing
 * unchanged regions. A secondary "apply model" inside Cursor normally resolves
 * these into real edits. Since we don't have that model, this module
 * reconstructs the full file content by merging the sketch with the original
 * file, then diffs the result against the original to produce a surgical
 * `edit(oldString, newString)` call.
 *
 * Resolution paths:
 *  1. New or empty file -> `write` (nothing to diff against)
 *  2. Sketch with elision markers -> merge with original, diff, -> `edit`
 *  3. Full-file content (starts like the original, >50% length) -> diff, -> `edit`
 *  4. Unresolvable fragment -> return null (caller emits error hint)
 */

// Matches common elision markers across languages:
//   // ... existing code ...
//   # ... existing code ...
//   <!-- ... existing content ... -->
//   /* ... existing code ... */
//   -- ... existing code ...
//   ; ... existing code ...
// The marker must occupy the entire line (ignoring whitespace).
const ELISION_PATTERN =
  /^\s*(?:\/\/|#|<!--|\/\*|--|;)\s*\.{2,}\s*existing\s+\w+\s*\.{2,}\s*(?:-->|\*\/)?\s*$/i;

export interface EditFileResolution {
  action: "write" | "edit";
  toolCall: OpenAiToolCall;
}

/**
 * Attempt to resolve a Cursor edit_file call into an OpenCode edit or write.
 *
 * For existing files, produces an `edit(oldString, newString)` by diffing the
 * merged content against the original. For new/empty files, produces a `write`.
 */
export function resolveEditFile(
  filePath: string,
  content: string,
  baseToolCall: OpenAiToolCall,
  allowedToolNames: Set<string>,
): EditFileResolution | null {
  if (!allowedToolNames.has("edit")) {
    log.debug("edit tool not available; cannot resolve edit_file", { filePath });
    return null;
  }

  // New file: ensure it exists on disk so OpenCode's edit tool can operate on
  // it, then emit edit(oldString="", newString=content).
  if (!existsSync(filePath)) {
    try {
      mkdirSync(dirname(filePath), { recursive: true });
      writeFileSync(filePath, "", "utf-8");
    } catch (err) {
      log.warn("Failed to create file for edit_file resolution", { filePath, err });
      return null;
    }
    log.info("Created new file; emitting edit to populate", { filePath });
    return makeEdit(baseToolCall, filePath, "", content);
  }

  let original: string;
  try {
    original = readFileSync(filePath, "utf-8");
  } catch (err) {
    log.warn("Failed to read original file for edit_file resolution", { filePath, err });
    return null;
  }

  // Empty file: edit(oldString="", newString=content).
  if (original.trim().length === 0) {
    log.info("Target file is empty; emitting edit to populate", { filePath });
    return makeEdit(baseToolCall, filePath, "", content);
  }

  // Check for elision markers in the sketch.
  const sketchLines = content.split("\n");
  const markerIndices = findElisionMarkers(sketchLines);

  if (markerIndices.length > 0) {
    const merged = mergeSketch(original, sketchLines, markerIndices);
    if (merged !== null) {
      log.info("Resolved edit_file sketch via marker merge", {
        filePath,
        markers: markerIndices.length,
      });
      return diffToEdit(baseToolCall, filePath, original, merged);
    }
    log.debug("Marker merge failed; falling through to full-content check", { filePath });
  }

  // No markers (or merge failed). Check if this looks like full file content.
  if (looksLikeFullFile(original, content)) {
    log.info("edit_file content looks like full file replacement", { filePath });
    return diffToEdit(baseToolCall, filePath, original, content);
  }

  // Try context overlap: find lines in the fragment that match the original
  // file and use them as anchors to determine the replacement region.
  const overlapResult = resolveByContextOverlap(original, content);
  if (overlapResult !== null) {
    log.info("Resolved edit_file fragment via context overlap", {
      filePath,
      oldLen: overlapResult.oldString.length,
      newLen: overlapResult.newString.length,
    });
    return makeEdit(baseToolCall, filePath, overlapResult.oldString, overlapResult.newString);
  }

  // Unresolvable fragment.
  log.info("Cannot resolve edit_file sketch; content is an unresolvable fragment", {
    filePath,
    originalLines: original.split("\n").length,
    sketchLines: sketchLines.length,
  });
  return null;
}

// ---------------------------------------------------------------------------
// Sketch merging
// ---------------------------------------------------------------------------

function findElisionMarkers(lines: string[]): number[] {
  const indices: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (ELISION_PATTERN.test(lines[i])) {
      indices.push(i);
    }
  }
  return indices;
}

/**
 * Merge a sketch (with elision markers) against the original file content.
 *
 * Algorithm:
 *   - Split the sketch into segments separated by markers.
 *   - Each segment is a run of concrete (non-marker) lines.
 *   - For each segment, find its location in the original file by matching
 *     leading and trailing context lines.
 *   - Stitch together: original content before the first segment, then
 *     alternating (segment content, original gap between segments).
 */
function mergeSketch(
  original: string,
  sketchLines: string[],
  markerIndices: number[],
): string | null {
  const originalLines = original.split("\n");
  const segments = extractSegments(sketchLines, markerIndices);

  if (segments.length === 0) {
    return null;
  }

  // For each segment, find where it anchors in the original file.
  const anchored = anchorSegments(segments, originalLines);
  if (anchored === null) {
    return null;
  }

  // Reconstruct the full file from anchored segments.
  return reconstruct(originalLines, anchored);
}

interface Segment {
  lines: string[];
  /** Whether this segment is preceded by a marker (true) or is the start of the sketch (false). */
  precededByMarker: boolean;
  /** Whether this segment is followed by a marker (true) or is the end of the sketch (false). */
  followedByMarker: boolean;
}

interface AnchoredSegment {
  /** Start line in the original file that this segment replaces (inclusive). */
  originalStart: number;
  /** End line in the original file that this segment replaces (exclusive). */
  originalEnd: number;
  /** The replacement lines. */
  lines: string[];
}

function extractSegments(sketchLines: string[], markerIndices: number[]): Segment[] {
  const markerSet = new Set(markerIndices);
  const segments: Segment[] = [];
  let currentLines: string[] = [];
  let precededByMarker = false;

  for (let i = 0; i < sketchLines.length; i++) {
    if (markerSet.has(i)) {
      if (currentLines.length > 0) {
        segments.push({
          lines: currentLines,
          precededByMarker,
          followedByMarker: true,
        });
        currentLines = [];
      }
      precededByMarker = true;
    } else {
      currentLines.push(sketchLines[i]);
    }
  }

  if (currentLines.length > 0) {
    segments.push({
      lines: currentLines,
      precededByMarker,
      followedByMarker: false,
    });
  }

  return segments;
}

function anchorSegments(
  segments: Segment[],
  originalLines: string[],
): AnchoredSegment[] | null {
  const anchored: AnchoredSegment[] = [];
  let searchStart = 0;

  for (const segment of segments) {
    const anchor = anchorOneSegment(segment, originalLines, searchStart);
    if (anchor === null) {
      return null;
    }
    anchored.push(anchor);
    searchStart = anchor.originalEnd;
  }

  return anchored;
}

/**
 * Find where a single sketch segment anchors in the original file.
 *
 * Strategy: use the first few lines and last few lines of the segment as
 * context anchors. Find a region in the original file where either the
 * leading context or trailing context matches. The segment replaces that
 * region.
 */
function anchorOneSegment(
  segment: Segment,
  originalLines: string[],
  searchStart: number,
): AnchoredSegment | null {
  const { lines, precededByMarker, followedByMarker } = segment;

  // If segment is at the start of the sketch (no preceding marker),
  // it anchors at line 0 of the original.
  if (!precededByMarker) {
    // Find where the segment's last line appears in the original to determine
    // how far into the original file this segment extends.
    const endAnchor = followedByMarker
      ? findContextEnd(lines, originalLines, searchStart)
      : originalLines.length;

    return {
      originalStart: 0,
      originalEnd: endAnchor,
      lines,
    };
  }

  // If segment is at the end of the sketch (no following marker),
  // it anchors at the end of the original.
  if (!followedByMarker) {
    const startAnchor = findContextStart(lines, originalLines, searchStart);
    if (startAnchor === null) return null;

    return {
      originalStart: startAnchor,
      originalEnd: originalLines.length,
      lines,
    };
  }

  // Middle segment: both preceded and followed by markers.
  // Use leading context to find the start and trailing context to find the end.
  const startAnchor = findContextStart(lines, originalLines, searchStart);
  if (startAnchor === null) return null;

  const endAnchor = findContextEnd(lines, originalLines, startAnchor);

  return {
    originalStart: startAnchor,
    originalEnd: endAnchor,
    lines,
  };
}

/** Number of context lines to use for anchoring. */
const CONTEXT_LINES = 3;

/**
 * Find where the segment starts in the original file by matching the first
 * few lines of the segment against consecutive lines in the original.
 */
function findContextStart(
  segmentLines: string[],
  originalLines: string[],
  searchStart: number,
): number | null {
  const contextCount = Math.min(CONTEXT_LINES, segmentLines.length);
  const context = segmentLines.slice(0, contextCount);

  for (let i = searchStart; i <= originalLines.length - contextCount; i++) {
    if (linesMatch(context, originalLines, i)) {
      return i;
    }
  }

  // Fallback: try matching just the first line.
  if (contextCount > 1) {
    for (let i = searchStart; i < originalLines.length; i++) {
      if (trimmedEqual(segmentLines[0], originalLines[i])) {
        return i;
      }
    }
  }

  return null;
}

/**
 * Find the end boundary in the original file for a segment by matching the
 * last few lines of the segment against the original.
 */
function findContextEnd(
  segmentLines: string[],
  originalLines: string[],
  searchStart: number,
): number {
  const contextCount = Math.min(CONTEXT_LINES, segmentLines.length);
  const context = segmentLines.slice(segmentLines.length - contextCount);

  for (let i = searchStart; i <= originalLines.length - contextCount; i++) {
    if (linesMatch(context, originalLines, i)) {
      // The end is after the matched context lines.
      return i + contextCount;
    }
  }

  // If we can't find the end context, assume the segment extends to where
  // the next segment would start (caller handles this).
  return searchStart + segmentLines.length;
}

function linesMatch(
  context: string[],
  originalLines: string[],
  startIndex: number,
): boolean {
  for (let i = 0; i < context.length; i++) {
    if (!trimmedEqual(context[i], originalLines[startIndex + i])) {
      return false;
    }
  }
  return true;
}

function trimmedEqual(a: string, b: string): boolean {
  return a.trimEnd() === b.trimEnd();
}

function reconstruct(
  originalLines: string[],
  anchored: AnchoredSegment[],
): string {
  const result: string[] = [];
  let pos = 0;

  for (const seg of anchored) {
    // Copy original lines before this segment.
    if (seg.originalStart > pos) {
      result.push(...originalLines.slice(pos, seg.originalStart));
    }
    // Insert the segment's lines.
    result.push(...seg.lines);
    pos = seg.originalEnd;
  }

  // Copy any remaining original lines after the last segment.
  if (pos < originalLines.length) {
    result.push(...originalLines.slice(pos));
  }

  return result.join("\n");
}

// ---------------------------------------------------------------------------
// Context overlap resolution (no markers, partial fragment)
// ---------------------------------------------------------------------------

/**
 * Resolve a fragment by finding lines that overlap with the original file.
 *
 * Strategy:
 *   1. Find a run of consecutive lines at the START of the fragment that match
 *      consecutive lines in the original (leading context).
 *   2. Find a run of consecutive lines at the END of the fragment that match
 *      consecutive lines in the original (trailing context).
 *   3. Use the matched positions to determine the original region being replaced.
 *   4. Return oldString (original region) and newString (fragment).
 *
 * Handles these patterns:
 *   - Fragment starts with existing code + appends new code (trailing insert)
 *   - Fragment ends with existing code + prepends new code (leading insert)
 *   - Fragment has existing code on both sides with changes in the middle
 */
function resolveByContextOverlap(
  original: string,
  fragment: string,
): { oldString: string; newString: string } | null {
  const origLines = original.split("\n");
  const fragLines = fragment.split("\n");

  if (fragLines.length === 0) return null;

  // Find leading overlap: how many lines at the start of the fragment match
  // consecutive lines in the original?
  const leadingMatch = findLeadingOverlap(fragLines, origLines);

  // Find trailing overlap: constrain to positions AFTER the leading match to
  // avoid false matches on generic lines like "}". Require at least 2 matching
  // lines to reduce false positives.
  const trailingSearchStart = leadingMatch !== null ? leadingMatch.origEnd : 0;
  const trailingMatch = findTrailingOverlap(fragLines, origLines, trailingSearchStart, 2);

  // Need at least one anchor point.
  if (leadingMatch === null && trailingMatch === null) {
    return null;
  }

  // Determine the original region being replaced.
  let origStart: number;
  let origEnd: number;

  if (leadingMatch !== null && trailingMatch !== null && trailingMatch.origEnd > leadingMatch.origStart) {
    // Both anchors found in valid order.
    origStart = leadingMatch.origStart;
    origEnd = trailingMatch.origEnd;
  } else if (leadingMatch !== null) {
    // Leading anchor only: fragment starts with known lines, extends to end of file.
    origStart = leadingMatch.origStart;
    origEnd = origLines.length;
  } else {
    // Trailing anchor only.
    origStart = trailingMatch!.origStart;
    origEnd = trailingMatch!.origEnd;
  }

  if (origStart > origEnd) return null;

  const oldString = origLines.slice(origStart, origEnd).join("\n");
  const newString = fragment;

  // Reject if old and new are identical (no actual change).
  if (oldString === newString) return null;

  log.info("Context overlap resolved", {
    origStart,
    origEnd,
    oldLen: oldString.length,
    newLen: newString.length,
    leadingLines: leadingMatch?.matchCount ?? 0,
    trailingLines: trailingMatch?.matchCount ?? 0,
  });

  return { oldString, newString };
}

interface OverlapMatch {
  /** Index in the original file where the matching run starts. */
  origStart: number;
  /** Index in the original file where the matching run ends (exclusive). */
  origEnd: number;
  /** Number of lines that matched. */
  matchCount: number;
}

/**
 * Find the longest run of consecutive lines at the START of the fragment
 * that match consecutive lines somewhere in the original file.
 */
function findLeadingOverlap(
  fragLines: string[],
  origLines: string[],
): OverlapMatch | null {
  // Try matching decreasing numbers of leading lines.
  const maxTry = Math.min(fragLines.length, origLines.length);
  for (let count = maxTry; count >= 1; count--) {
    const context = fragLines.slice(0, count);
    for (let i = 0; i <= origLines.length - count; i++) {
      if (linesMatch(context, origLines, i)) {
        return { origStart: i, origEnd: i + count, matchCount: count };
      }
    }
  }
  return null;
}

/**
 * Find the longest run of consecutive lines at the END of the fragment
 * that match consecutive lines somewhere in the original file.
 *
 * @param searchStart  Only consider matches at or after this index in origLines
 * @param minCount     Minimum number of matching lines required (avoids false
 *                     positives on generic lines like "}")
 */
function findTrailingOverlap(
  fragLines: string[],
  origLines: string[],
  searchStart: number = 0,
  minCount: number = 1,
): OverlapMatch | null {
  const maxTry = Math.min(fragLines.length, origLines.length - searchStart);
  for (let count = maxTry; count >= minCount; count--) {
    const context = fragLines.slice(fragLines.length - count);
    for (let i = searchStart; i <= origLines.length - count; i++) {
      if (linesMatch(context, origLines, i)) {
        return { origStart: i, origEnd: i + count, matchCount: count };
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Full-file detection
// ---------------------------------------------------------------------------

/**
 * Heuristic: does the content look like it's a full replacement of the file?
 *
 * Checks:
 *  - Content starts with the same first non-empty line as the original
 *  - Content is at least 50% of the original length
 */
function looksLikeFullFile(original: string, content: string): boolean {
  const originalFirstLine = firstNonEmptyLine(original);
  const contentFirstLine = firstNonEmptyLine(content);

  if (originalFirstLine && contentFirstLine && trimmedEqual(originalFirstLine, contentFirstLine)) {
    // Starts the same way and is a substantial portion of the file.
    return content.length >= original.length * 0.5;
  }

  return false;
}

function firstNonEmptyLine(text: string): string | null {
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.length > 0) return trimmed;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert a `write` tool call on an existing file to an equivalent `edit` call.
 * This bypasses tool-guards plugins that block `write` on existing files.
 *
 * - Non-existent file -> returns null (keep as write; new file creation)
 * - Empty file -> edit(oldString="", newString=content)
 * - Non-empty file -> edit(oldString=<original>, newString=content)
 */
export function maybeConvertWriteToEdit(
  toolCall: OpenAiToolCall,
  allowedToolNames: Set<string>,
): OpenAiToolCall | null {
  if (toolCall.function.name.toLowerCase() !== "write") {
    return null;
  }
  if (!allowedToolNames.has("edit")) {
    return null;
  }

  let args: Record<string, unknown>;
  try {
    args = JSON.parse(toolCall.function.arguments);
  } catch {
    return null;
  }

  const filePath = typeof args.filePath === "string" ? args.filePath : null;
  const content = typeof args.content === "string" ? args.content : null;
  if (!filePath || content === null) {
    return null;
  }

  if (!existsSync(filePath)) {
    return null;
  }

  let original: string;
  try {
    original = readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }

  // Empty file: edit with oldString="" replaces everything (equivalent to write).
  const oldString = original.trim().length === 0 ? "" : original;

  log.debug("Converting write to edit for existing file", {
    filePath,
    originalLen: original.length,
  });

  return {
    ...toolCall,
    function: {
      name: "edit",
      arguments: JSON.stringify({ filePath, oldString, newString: content }),
    },
  };
}

function makeEdit(
  baseToolCall: OpenAiToolCall,
  filePath: string,
  oldString: string,
  newString: string,
): EditFileResolution {
  return {
    action: "edit",
    toolCall: {
      ...baseToolCall,
      function: {
        name: "edit",
        arguments: JSON.stringify({ filePath, oldString, newString }),
      },
    },
  };
}

function makeWrite(
  baseToolCall: OpenAiToolCall,
  filePath: string,
  content: string,
): EditFileResolution {
  return {
    action: "write",
    toolCall: {
      ...baseToolCall,
      function: {
        name: "write",
        arguments: JSON.stringify({ filePath, content }),
      },
    },
  };
}

/**
 * Diff the original file against the new (merged) content and produce a
 * single edit(oldString, newString) covering the changed region.
 *
 * Finds the first and last differing lines, then uses the original lines in
 * that range as oldString and the merged lines as newString.
 */
function diffToEdit(
  baseToolCall: OpenAiToolCall,
  filePath: string,
  original: string,
  merged: string,
): EditFileResolution | null {
  if (original === merged) {
    log.debug("No differences between original and merged; nothing to edit", { filePath });
    return null;
  }

  const origLines = original.split("\n");
  const newLines = merged.split("\n");

  // Find the first line that differs.
  let firstDiff = 0;
  while (
    firstDiff < origLines.length
    && firstDiff < newLines.length
    && trimmedEqual(origLines[firstDiff], newLines[firstDiff])
  ) {
    firstDiff++;
  }

  // Find the last line that differs (counting from the end).
  let origEnd = origLines.length - 1;
  let newEnd = newLines.length - 1;
  while (
    origEnd > firstDiff
    && newEnd > firstDiff
    && trimmedEqual(origLines[origEnd], newLines[newEnd])
  ) {
    origEnd--;
    newEnd--;
  }

  // Extract the differing regions. Use the exact original text (preserving
  // whitespace) for oldString so OpenCode's string match succeeds.
  const oldString = origLines.slice(firstDiff, origEnd + 1).join("\n");
  const newString = newLines.slice(firstDiff, newEnd + 1).join("\n");

  if (oldString === newString) {
    log.debug("Diff produced identical old/new strings; no edit needed", { filePath });
    return null;
  }

  log.debug("Computed edit from diff", {
    filePath,
    firstDiffLine: firstDiff,
    origRange: `${firstDiff}-${origEnd}`,
    newRange: `${firstDiff}-${newEnd}`,
    oldLen: oldString.length,
    newLen: newString.length,
  });

  return {
    action: "edit",
    toolCall: {
      ...baseToolCall,
      function: {
        name: "edit",
        arguments: JSON.stringify({ filePath, oldString, newString }),
      },
    },
  };
}