import type { ToolUpdate, ToolMapper } from "../acp/tools.js";
import { extractOpenAiToolCall, type OpenAiToolCall, type ToolCallExtractionResult } from "../proxy/tool-loop.js";
import type { StreamJsonToolCallEvent } from "../streaming/types.js";
import type { ToolRouter } from "../tools/router.js";
import { createLogger } from "../utils/logger.js";
import { applyToolSchemaCompat, type ToolSchemaValidationResult } from "./tool-schema-compat.js";
import { resolveEditFile, maybeConvertWriteToEdit, type EditFileResolution } from "./edit-file-resolver.js";
import type { ToolLoopGuard } from "./tool-loop-guard.js";
import type { ProviderBoundaryMode, ToolLoopMode } from "./boundary.js";
import type { ProviderBoundary } from "./boundary.js";
import type { PassThroughTracker } from "./passthrough-tracker.js";

const log = createLogger("runtime-interception");

log.info("runtime-interception module loaded");

interface HandleToolLoopEventBaseOptions {
  event: StreamJsonToolCallEvent;
  toolLoopMode: ToolLoopMode;
  allowedToolNames: Set<string>;
  toolSchemaMap: Map<string, unknown>;
  toolLoopGuard: ToolLoopGuard;
  toolMapper: ToolMapper;
  toolSessionId: string;
  shouldEmitToolUpdates: boolean;
  proxyExecuteToolCalls: boolean;
  suppressConverterToolEvents: boolean;
  toolRouter?: ToolRouter;
  responseMeta: { id: string; created: number; model: string };
  onToolUpdate: (update: ToolUpdate) => Promise<void> | void;
  onToolResult: (toolResult: any) => Promise<void> | void;
  onInterceptedToolCall: (toolCall: OpenAiToolCall) => Promise<void> | void;
  passThroughTracker?: PassThroughTracker;
}

export interface HandleToolLoopEventLegacyOptions extends HandleToolLoopEventBaseOptions {}

export interface HandleToolLoopEventV1Options extends HandleToolLoopEventBaseOptions {
  boundary: ProviderBoundary;
  schemaValidationFailureMode?: "pass_through" | "terminate";
}

export interface HandleToolLoopEventWithFallbackOptions
  extends HandleToolLoopEventV1Options {
  boundaryMode: ProviderBoundaryMode;
  autoFallbackToLegacy: boolean;
  onFallbackToLegacy?: (error: unknown) => void;
}

export interface HandleToolLoopEventResult {
  intercepted: boolean;
  skipConverter: boolean;
  terminate?: ToolLoopTermination;
}

export interface ToolLoopGuardTermination {
  reason: "loop_guard";
  message: string;
  tool: string;
  fingerprint: string;
  repeatCount: number;
  maxRepeat: number;
  errorClass: string;
  silent?: boolean;
}

export interface ToolSchemaValidationTermination {
  reason: "schema_validation";
  message: string;
  tool: string;
  errorClass: "validation";
  repairHint?: string;
  missing: string[];
  unexpected: string[];
  typeErrors: string[];
}

export type ToolLoopTermination = ToolLoopGuardTermination | ToolSchemaValidationTermination;

interface NonFatalSchemaValidationResultChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role: "assistant";
      content: string;
    };
    finish_reason: null;
  }>;
}

export class ToolBoundaryExtractionError extends Error {
  cause?: unknown;
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "ToolBoundaryExtractionError";
    this.cause = cause;
  }
}

export async function handleToolLoopEventLegacy(
  options: HandleToolLoopEventLegacyOptions,
): Promise<HandleToolLoopEventResult> {
  log.info("handleToolLoopEventLegacy ENTRY", { toolLoopMode: options.toolLoopMode });
  const {
    event,
    toolLoopMode,
    allowedToolNames,
    toolSchemaMap,
    toolLoopGuard,
    toolMapper,
    toolSessionId,
    shouldEmitToolUpdates,
    proxyExecuteToolCalls,
    suppressConverterToolEvents,
    toolRouter,
    responseMeta,
    onToolUpdate,
    onToolResult,
    onInterceptedToolCall,
    passThroughTracker,
  } = options;

  const extraction =
    toolLoopMode === "opencode"
      ? extractOpenAiToolCall(event as any, allowedToolNames)
      : { action: "skip" as const, skipReason: "tool_loop_mode_not_opencode" };

  log.info("legacy extraction result", { action: extraction.action, skipReason: (extraction as any).skipReason, toolName: extraction.toolCall?.function?.name });

  // Handle pass-through: in opencode mode, unknown tools (e.g. Cursor-internal
  // tools like webSearch) have no handler in OpenCode, so skip them silently.
  // In proxy-exec mode this branch is unreachable (extraction is always "skip").
  if (extraction.action === "passthrough") {
    passThroughTracker?.trackTool(extraction.passthroughName!);
    if (toolLoopMode === "opencode") {
      log.debug("Suppressing Cursor-internal tool in opencode mode", {
        tool: extraction.passthroughName,
      });
      return { intercepted: false, skipConverter: true };
    }
    log.debug("MCP tool passed through to cursor-agent (legacy)", {
      tool: extraction.passthroughName,
    });
    return { intercepted: false, skipConverter: false };
  }

  // Handle skip: no tool to intercept
  if (extraction.action === "skip" || !extraction.toolCall) {
    const updates = await toolMapper.mapCursorEventToAcp(
      event,
      event.session_id ?? toolSessionId,
    );

    if (shouldEmitToolUpdates) {
      for (const update of updates) {
        await onToolUpdate(update);
      }
    }

    if (toolRouter && proxyExecuteToolCalls) {
      const toolResult = await toolRouter.handleToolCall(event as any, responseMeta);
      if (toolResult) {
        await onToolResult(toolResult);
      }
    }

    return { intercepted: false, skipConverter: suppressConverterToolEvents };
  }

  // Handle intercept: known OpenCode tool
  const interceptedToolCall = extraction.toolCall;
  if (interceptedToolCall) {
    const compat = applyToolSchemaCompat(interceptedToolCall, toolSchemaMap);
    let normalizedToolCall = compat.toolCall;
    log.debug("Applied tool schema compatibility (legacy)", {
      tool: normalizedToolCall.function.name,
      originalArgKeys: compat.originalArgKeys,
      normalizedArgKeys: compat.normalizedArgKeys,
      collisionKeys: compat.collisionKeys,
      validationOk: compat.validation.ok,
    });

    if (compat.validation.hasSchema && !compat.validation.ok) {
      const validationTermination = evaluateSchemaValidationLoopGuard(
        toolLoopGuard,
        normalizedToolCall,
        compat.validation,
      );
      if (validationTermination) {
        return { intercepted: false, skipConverter: true, terminate: validationTermination };
      }

      const editFileResolution = tryResolveEditFile(
        normalizedToolCall,
        compat.preSanitizationArgs,
        allowedToolNames,
      );
      if (editFileResolution) {
        log.info("Resolved edit_file sketch (legacy)", {
          action: editFileResolution.action,
          tool: editFileResolution.toolCall.function.name,
        });
        normalizedToolCall = editFileResolution.toolCall;
      } else if (shouldEmitNonFatalSchemaValidationHint(normalizedToolCall, compat.validation)) {
        const hintChunk = createNonFatalSchemaValidationHintChunk(
          responseMeta,
          normalizedToolCall,
          compat.validation,
        );
        log.debug("Emitting non-fatal schema validation hint in legacy and skipping malformed tool execution", {
          tool: normalizedToolCall.function.name,
          missing: compat.validation.missing,
          typeErrors: compat.validation.typeErrors,
        });
        await onToolResult(hintChunk);
        return { intercepted: false, skipConverter: true };
      }
    }

    const termination = evaluateToolLoopGuard(toolLoopGuard, normalizedToolCall);
    if (termination) {
      return { intercepted: false, skipConverter: true, terminate: termination };
    }

    // Convert write-on-existing-file to edit to bypass tool-guards restrictions.
    const writeConverted = maybeConvertWriteToEdit(normalizedToolCall, allowedToolNames);
    if (writeConverted) {
      await onInterceptedToolCall(writeConverted);
    } else {
      await onInterceptedToolCall(normalizedToolCall);
    }
    return { intercepted: true, skipConverter: true };
  }

  const updates = await toolMapper.mapCursorEventToAcp(
    event,
    event.session_id ?? toolSessionId,
  );

  if (shouldEmitToolUpdates) {
    for (const update of updates) {
      await onToolUpdate(update);
    }
  }

  if (toolRouter && proxyExecuteToolCalls) {
    const toolResult = await toolRouter.handleToolCall(event as any, responseMeta);
    if (toolResult) {
      await onToolResult(toolResult);
    }
  }

  return {
    intercepted: false,
    skipConverter: suppressConverterToolEvents,
  };
}

export async function handleToolLoopEventV1(
  options: HandleToolLoopEventV1Options,
): Promise<HandleToolLoopEventResult> {
  const {
    event,
    boundary,
    schemaValidationFailureMode = "pass_through",
    toolLoopMode,
    allowedToolNames,
    toolSchemaMap,
    toolLoopGuard,
    toolMapper,
    toolSessionId,
    shouldEmitToolUpdates,
    proxyExecuteToolCalls,
    suppressConverterToolEvents,
    toolRouter,
    responseMeta,
    onToolUpdate,
    onToolResult,
    onInterceptedToolCall,
    passThroughTracker,
  } = options;

  log.info("handleToolLoopEventV1 ENTRY", { toolLoopMode, boundaryMode: boundary.mode });

  let extraction: ToolCallExtractionResult;
  try {
    extraction = boundary.maybeExtractToolCall(
      event,
      allowedToolNames,
      toolLoopMode,
    );
  } catch (error) {
    log.info("v1 extraction THREW", { error: String(error) });
    throw new ToolBoundaryExtractionError("Boundary tool extraction failed", error);
  }

  log.info("v1 extraction result", { action: extraction.action, skipReason: extraction.skipReason, toolName: extraction.toolCall?.function?.name });

  // Pass-through: unreachable in opencode mode (boundary converts to skip),
  // but reachable in other modes for Cursor-internal tools.
  if (extraction.action === "passthrough") {
    passThroughTracker?.trackTool(extraction.passthroughName!);
    log.debug("Tool passed through to cursor-agent (v1)", {
      tool: extraction.passthroughName,
    });
    return { intercepted: false, skipConverter: false };
  }

  // Handle skip: no tool to intercept
  if (extraction.action === "skip" || !extraction.toolCall) {
    // Cursor-internal tools (e.g. webSearch) have no handler in OpenCode;
    // suppress them so they don't surface as "invalid tool" errors.
    if (extraction.skipReason === "cursor_internal_tool") {
      return { intercepted: false, skipConverter: true };
    }

    const updates = await toolMapper.mapCursorEventToAcp(
      event,
      event.session_id ?? toolSessionId,
    );

    if (shouldEmitToolUpdates) {
      for (const update of updates) {
        await onToolUpdate(update);
      }
    }

    if (toolRouter && proxyExecuteToolCalls) {
      const toolResult = await toolRouter.handleToolCall(event as any, responseMeta);
      if (toolResult) {
        await onToolResult(toolResult);
      }
    }

    return { intercepted: false, skipConverter: suppressConverterToolEvents };
  }

  // Handle intercept: known OpenCode tool
  const interceptedToolCall = extraction.toolCall;
  log.info("v1 about to call applyToolSchemaCompat", { tool: interceptedToolCall.function.name, args: interceptedToolCall.function.arguments.slice(0, 200) });
  const compat = applyToolSchemaCompat(interceptedToolCall, toolSchemaMap);
  log.info("v1 applyToolSchemaCompat returned", { tool: compat.toolCall.function.name, validationOk: compat.validation.ok, outKeys: compat.normalizedArgKeys });
  let normalizedToolCall = compat.toolCall;
  const editDiag =
    normalizedToolCall.function.name.toLowerCase() === "edit"
      ? {
          rawArgs: safeArgTypeSummary(event),
          normalizedArgs: compat.normalizedArgs,
        }
      : undefined;
  log.debug("Applied tool schema compatibility", {
    tool: normalizedToolCall.function.name,
    originalArgKeys: compat.originalArgKeys,
    normalizedArgKeys: compat.normalizedArgKeys,
    collisionKeys: compat.collisionKeys,
    validationOk: compat.validation.ok,
    ...(editDiag ? { editDiag } : {}),
  });
  if (compat.validation.hasSchema && !compat.validation.ok) {
    log.debug("Tool schema compatibility validation failed", {
      tool: normalizedToolCall.function.name,
      missing: compat.validation.missing,
      unexpected: compat.validation.unexpected,
      typeErrors: compat.validation.typeErrors,
      repairHint: compat.validation.repairHint,
    });
    const validationTermination = evaluateSchemaValidationLoopGuard(
      toolLoopGuard,
      normalizedToolCall,
      compat.validation,
    );
    if (validationTermination) {
      return { intercepted: false, skipConverter: true, terminate: validationTermination };
    }
    const termination = evaluateToolLoopGuard(toolLoopGuard, normalizedToolCall);
    if (termination) {
      return { intercepted: false, skipConverter: true, terminate: termination };
    }
    const editFileResolution = tryResolveEditFile(
      normalizedToolCall,
      compat.preSanitizationArgs,
      allowedToolNames,
    );
    if (editFileResolution) {
      log.info("Resolved edit_file sketch (v1)", {
        action: editFileResolution.action,
        tool: editFileResolution.toolCall.function.name,
      });
      await onInterceptedToolCall(editFileResolution.toolCall);
      return {
        intercepted: true,
        skipConverter: true,
      };
    }
    if (
      schemaValidationFailureMode === "pass_through"
      && shouldTerminateOnSchemaValidation(normalizedToolCall, compat.validation)
    ) {
      return {
        intercepted: false,
        skipConverter: true,
        terminate: createSchemaValidationTermination(normalizedToolCall, compat.validation),
      };
    }
    if (
      schemaValidationFailureMode === "pass_through"
      && shouldEmitNonFatalSchemaValidationHint(normalizedToolCall, compat.validation)
    ) {
      const hintChunk = createNonFatalSchemaValidationHintChunk(
        responseMeta,
        normalizedToolCall,
        compat.validation,
      );
      log.debug("Emitting non-fatal schema validation hint and skipping malformed tool execution", {
        tool: normalizedToolCall.function.name,
        missing: compat.validation.missing,
        typeErrors: compat.validation.typeErrors,
      });
      await onToolResult(hintChunk);
      return {
        intercepted: false,
        skipConverter: true,
      };
    }
    if (schemaValidationFailureMode === "terminate") {
      return {
        intercepted: false,
        skipConverter: true,
        terminate: createSchemaValidationTermination(normalizedToolCall, compat.validation),
      };
    }
    log.debug("Forwarding schema-invalid tool call to OpenCode loop", {
      tool: normalizedToolCall.function.name,
      repairHint: compat.validation.repairHint,
    });
    await onInterceptedToolCall(normalizedToolCall);
    return {
      intercepted: true,
      skipConverter: true,
    };
  }

  const termination = evaluateToolLoopGuard(toolLoopGuard, normalizedToolCall);
  if (termination) {
    return { intercepted: false, skipConverter: true, terminate: termination };
  }

  // Convert write-on-existing-file to edit to bypass tool-guards restrictions.
  const writeConverted = maybeConvertWriteToEdit(normalizedToolCall, allowedToolNames);
  if (writeConverted) {
    await onInterceptedToolCall(writeConverted);
  } else {
    await onInterceptedToolCall(normalizedToolCall);
  }
  return { intercepted: true, skipConverter: true };
}

export async function handleToolLoopEventWithFallback(
  options: HandleToolLoopEventWithFallbackOptions,
): Promise<HandleToolLoopEventResult> {
  const {
    boundaryMode,
    autoFallbackToLegacy,
    onFallbackToLegacy,
    ...shared
  } = options;

  log.warn("handleToolLoopEventWithFallback ENTRY", { boundaryMode, autoFallbackToLegacy });

  if (boundaryMode === "legacy") {
    log.info("withFallback: using legacy directly");
    return handleToolLoopEventLegacy(shared);
  }

  try {
    const schemaValidationFailureMode: "pass_through" | "terminate" =
      autoFallbackToLegacy
      && boundaryMode === "v1"
      && !shouldUsePassThroughForEditSchema(shared.event)
        ? "terminate"
        : "pass_through";
    log.info("withFallback: calling v1", { schemaValidationFailureMode });
    const result = await handleToolLoopEventV1({
      ...shared,
      schemaValidationFailureMode,
    });
    log.info("withFallback: v1 returned", { intercepted: result.intercepted, skipConverter: result.skipConverter, terminate: result.terminate ? { reason: result.terminate.reason, tool: result.terminate.tool } : null });
    if (
      result.terminate
      && autoFallbackToLegacy
      && boundaryMode === "v1"
      && (result.terminate.reason === "loop_guard" || result.terminate.reason === "schema_validation")
    ) {
      if (result.terminate.reason === "loop_guard") {
        if (result.terminate.errorClass === "validation" || result.terminate.errorClass === "success") {
          return result;
        }
        shared.toolLoopGuard.resetFingerprint(result.terminate.fingerprint);
        onFallbackToLegacy?.(new Error(`loop guard: ${result.terminate.fingerprint}`));
      } else {
        log.info("withFallback: falling back to legacy due to schema validation", { tool: result.terminate.tool });
        onFallbackToLegacy?.(new Error(`schema validation: ${result.terminate.tool}`));
      }
      return handleToolLoopEventLegacy(shared);
    }
    return result;
  } catch (error) {
    if (
      !autoFallbackToLegacy
      || boundaryMode !== "v1"
      || !(error instanceof ToolBoundaryExtractionError)
    ) {
      throw error;
    }
    onFallbackToLegacy?.(error.cause ?? error);
    return handleToolLoopEventLegacy(shared);
  }
}

function evaluateToolLoopGuard(
  toolLoopGuard: ToolLoopGuard,
  toolCall: OpenAiToolCall,
): ToolLoopTermination | null {
  const decision = toolLoopGuard.evaluate(toolCall);
  if (!decision.tracked) {
    return null;
  }
  if (!decision.triggered) {
    return null;
  }

  log.debug("Tool loop guard triggered", {
    tool: toolCall.function.name,
    fingerprint: decision.fingerprint,
    repeatCount: decision.repeatCount,
    maxRepeat: decision.maxRepeat,
    errorClass: decision.errorClass,
  });

  // For success loops, terminate silently without emitting an error message to the user.
  // The tool has already succeeded; we just need to stop the loop.
  if (decision.errorClass === "success") {
    return {
      reason: "loop_guard",
      message: "",
      tool: toolCall.function.name,
      fingerprint: decision.fingerprint,
      repeatCount: decision.repeatCount,
      maxRepeat: decision.maxRepeat,
      errorClass: decision.errorClass,
      silent: true,
    };
  }

  return {
    reason: "loop_guard",
    message: `Tool loop guard stopped repeated failing calls to "${toolCall.function.name}" `
      + `after ${decision.repeatCount} attempts (limit ${decision.maxRepeat}). `
      + "Adjust tool arguments and retry.",
    tool: toolCall.function.name,
    fingerprint: decision.fingerprint,
    repeatCount: decision.repeatCount,
    maxRepeat: decision.maxRepeat,
    errorClass: decision.errorClass,
  };
}

function createSchemaValidationTermination(
  toolCall: OpenAiToolCall,
  validation: ToolSchemaValidationResult,
): ToolSchemaValidationTermination {
  const reasonParts: string[] = [];
  if (validation.missing.length > 0) {
    reasonParts.push(`missing required: ${validation.missing.join(", ")}`);
  }
  if (validation.unexpected.length > 0) {
    reasonParts.push(`unsupported fields: ${validation.unexpected.join(", ")}`);
  }
  if (validation.typeErrors.length > 0) {
    reasonParts.push(`type errors: ${validation.typeErrors.join("; ")}`);
  }

  const reasonText = reasonParts.length > 0 ? reasonParts.join(" | ") : "arguments did not match schema";
  const repairHint = validation.repairHint
    ? ` ${validation.repairHint}`
    : "";
  return {
    reason: "schema_validation",
    message: `Invalid arguments for tool "${toolCall.function.name}": ${reasonText}.${repairHint}`.trim(),
    tool: toolCall.function.name,
    errorClass: "validation",
    repairHint: validation.repairHint,
    missing: validation.missing,
    unexpected: validation.unexpected,
    typeErrors: validation.typeErrors,
  };
}

function evaluateSchemaValidationLoopGuard(
  toolLoopGuard: ToolLoopGuard,
  toolCall: OpenAiToolCall,
  validation: ToolSchemaValidationResult,
): ToolLoopTermination | null {
  const validationSignature = buildValidationSignature(validation);
  const decision = toolLoopGuard.evaluateValidation(toolCall, validationSignature);
  if (!decision.tracked || !decision.triggered) {
    return null;
  }

  log.warn("Tool loop guard triggered on schema validation", {
    tool: toolCall.function.name,
    fingerprint: decision.fingerprint,
    repeatCount: decision.repeatCount,
    maxRepeat: decision.maxRepeat,
    validationSignature,
  });
  return {
    reason: "loop_guard",
    message:
      `Tool loop guard stopped repeated schema-invalid calls to "${toolCall.function.name}" `
      + `after ${decision.repeatCount} attempts (limit ${decision.maxRepeat}). `
      + "Adjust tool arguments and retry.",
    tool: toolCall.function.name,
    fingerprint: decision.fingerprint,
    repeatCount: decision.repeatCount,
    maxRepeat: decision.maxRepeat,
    errorClass: decision.errorClass,
  };
}

function buildValidationSignature(validation: ToolSchemaValidationResult): string {
  const parts: string[] = [];
  if (validation.missing.length > 0) {
    const sortedMissing = [...validation.missing].sort();
    parts.push(`missing:${sortedMissing.join(",")}`);
  }
  if (validation.typeErrors.length > 0) {
    const sortedTypeErrors = [...validation.typeErrors].sort();
    parts.push(`type:${sortedTypeErrors.join(",")}`);
  }
  if (parts.length === 0) {
    return "invalid";
  }
  return parts.join("|");
}

function shouldEmitNonFatalSchemaValidationHint(
  toolCall: OpenAiToolCall,
  validation: ToolSchemaValidationResult,
): boolean {
  if (toolCall.function.name.toLowerCase() !== "edit") {
    return false;
  }
  if (validation.typeErrors.length > 0) {
    return false;
  }
  const missing = new Set(validation.missing);
  return missing.has("oldString") || missing.has("newString") || missing.has("filePath");
}

function shouldTerminateOnSchemaValidation(
  toolCall: OpenAiToolCall,
  validation: ToolSchemaValidationResult,
): boolean {
  if (toolCall.function.name.toLowerCase() !== "edit") {
    return false;
  }
  if (validation.typeErrors.length > 0) {
    return true;
  }
  return false;
}

function createNonFatalSchemaValidationHintChunk(
  meta: { id: string; created: number; model: string },
  toolCall: OpenAiToolCall,
  validation: ToolSchemaValidationResult,
): NonFatalSchemaValidationResultChunk {
  const termination = createSchemaValidationTermination(toolCall, validation);
  const hint =
    termination.repairHint
    || "Use write for full-file replacement, or provide filePath, oldString, and newString for edit.";
  const content =
    `Skipped malformed tool call "${toolCall.function.name}": ${termination.message} ${hint}`.trim();
  return {
    id: meta.id,
    object: "chat.completion.chunk",
    created: meta.created,
    model: meta.model,
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content,
        },
        finish_reason: null,
      },
    ],
  };
}

function safeArgTypeSummary(event: StreamJsonToolCallEvent): Record<string, string> {
  try {
    let raw: unknown;
    const toolCallPayload = (event as any)?.tool_call;
    if (isRecord(toolCallPayload)) {
      const entries = Object.entries(toolCallPayload);
      if (entries.length > 0) {
        const [, payload] = entries[0];
        if (isRecord(payload)) {
          raw = payload.args;
          if (raw === undefined) {
            const { result: _result, ...rest } = payload;
            if (Object.keys(rest).length > 0) {
              raw = rest;
            }
          }
        }
      }
    }
    if (raw === undefined) {
      raw = (event as any)?.function?.arguments ?? (event as any)?.arguments;
    }
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (typeof parsed !== "object" || parsed === null) {
      return { _raw: typeof parsed };
    }
    const summary: Record<string, string> = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (v === null) {
        summary[k] = "null";
      } else if (Array.isArray(v)) {
        summary[k] = `array[${v.length}]`;
      } else {
        summary[k] = typeof v;
      }
    }
    return summary;
  } catch {
    return { _error: "parse_failed" };
  }
}

function shouldUsePassThroughForEditSchema(event: StreamJsonToolCallEvent): boolean {
  const toolCallPayload = (event as any)?.tool_call;
  if (!isRecord(toolCallPayload)) {
    return false;
  }
  const keys = Object.keys(toolCallPayload);
  if (keys.length === 0) {
    return false;
  }
  const rawName = keys[0];
  const normalizedName = rawName.endsWith("ToolCall")
    ? rawName.slice(0, -"ToolCall".length)
    : rawName;
  return normalizedName.toLowerCase() === "edit";
}

/**
 * Detect an edit_file-style call (edit with content/streamContent but no
 * oldString/newString) and resolve it via the sketch merger or full-file
 * write path. Returns null if this is a normal edit call or if the sketch
 * cannot be resolved.
 */
function tryResolveEditFile(
  toolCall: OpenAiToolCall,
  preSanitizationArgs: Record<string, unknown>,
  allowedToolNames: Set<string>,
): EditFileResolution | null {
  if (toolCall.function.name.toLowerCase() !== "edit") {
    return null;
  }

  // If the call has a non-empty oldString, it's a proper search_replace-style
  // edit, not an edit_file sketch. Let it through normally.
  const oldString = preSanitizationArgs.oldString;
  if (typeof oldString === "string" && oldString.length > 0) {
    return null;
  }

  const filePath =
    (typeof preSanitizationArgs.filePath === "string" && preSanitizationArgs.filePath.length > 0
      ? preSanitizationArgs.filePath
      : null)
    ?? (typeof preSanitizationArgs.path === "string" && preSanitizationArgs.path.length > 0
      ? preSanitizationArgs.path
      : null);
  if (!filePath) {
    return null;
  }

  // The sketch content can arrive as `content` (after global alias resolution
  // of streamContent) or as `newString` (after the edit repair logic).
  const content =
    typeof preSanitizationArgs.content === "string"
      ? preSanitizationArgs.content
      : typeof preSanitizationArgs.newString === "string"
        ? preSanitizationArgs.newString
        : null;
  if (content === null || content.length === 0) {
    return null;
  }

  return resolveEditFile(filePath, content, toolCall, allowedToolNames);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
