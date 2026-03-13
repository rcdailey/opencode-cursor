import type { OpenAiToolCall } from "../proxy/tool-loop.js";
import { createLogger } from "../utils/logger.js";

const log = createLogger("tool-schema-compat");

type JsonRecord = Record<string, unknown>;

// Global argument key aliases. These apply to all tools before tool-specific
// normalization runs. Keys that are tool-dependent (e.g. filepath -> filePath
// vs filepath -> path) should NOT go here; use normalizeToolSpecificArgs.
const ARG_KEY_ALIASES = new Map<string, string>([
  ["globpattern", "pattern"],
  ["filepattern", "pattern"],
  ["searchpattern", "pattern"],
  ["includepattern", "include"],
  ["workingdirectory", "cwd"],
  ["workdir", "cwd"],
  ["currentdirectory", "cwd"],
  ["cmd", "command"],
  ["script", "command"],
  ["shellcommand", "command"],
  ["terminalcommand", "command"],
  ["contents", "content"],
  ["text", "content"],
  ["body", "content"],
  ["data", "content"],
  ["payload", "content"],
  ["streamcontent", "content"],
  ["recursive", "force"],
  ["oldstring", "oldString"],
  ["newstring", "newString"],
]);

export interface ToolSchemaValidationResult {
  hasSchema: boolean;
  ok: boolean;
  missing: string[];
  unexpected: string[];
  typeErrors: string[];
  repairHint?: string;
}

export interface ToolSchemaCompatResult {
  toolCall: OpenAiToolCall;
  normalizedArgs: JsonRecord;
  originalArgKeys: string[];
  normalizedArgKeys: string[];
  collisionKeys: string[];
  validation: ToolSchemaValidationResult;
  preSanitizationArgs: JsonRecord;
}

export function buildToolSchemaMap(tools: Array<unknown>): Map<string, unknown> {
  const schemas = new Map<string, unknown>();
  for (const rawTool of tools) {
    const tool = isRecord(rawTool) ? rawTool : null;
    if (!tool) {
      continue;
    }
    const fn = isRecord(tool.function) ? tool.function : tool;
    const name = typeof fn.name === "string" ? fn.name.trim() : "";
    if (!name) {
      continue;
    }
    if (fn.parameters !== undefined) {
      schemas.set(name, fn.parameters);
    }
  }
  return schemas;
}

export function applyToolSchemaCompat(
  toolCall: OpenAiToolCall,
  toolSchemaMap: Map<string, unknown>,
): ToolSchemaCompatResult {
  log.info("applyToolSchemaCompat entry", {
    tool: toolCall.function.name,
    rawArgs: toolCall.function.arguments.slice(0, 200),
  });
  const parsedArgs = parseArguments(toolCall.function.arguments);
  const originalArgKeys = Object.keys(parsedArgs);
  const { normalizedArgs, collisionKeys } = normalizeArgumentKeys(parsedArgs);
  const toolSpecificArgs = normalizeToolSpecificArgs(toolCall.function.name, normalizedArgs);
  const schema = toolSchemaMap.get(toolCall.function.name);
  const sanitization = sanitizeArgumentsForSchema(toolSpecificArgs, schema);
  const validation = validateToolArguments(
    toolCall.function.name,
    sanitization.args,
    schema,
    sanitization.unexpected,
  );

  const normalizedToolCall: OpenAiToolCall = {
    ...toolCall,
    function: {
      ...toolCall.function,
      arguments: JSON.stringify(sanitization.args),
    },
  };

  const normalizedArgKeys = Object.keys(sanitization.args);
  log.info("Tool schema compat", {
    tool: toolCall.function.name,
    incoming: originalArgKeys,
    outgoing: normalizedArgKeys,
    validationOk: validation.ok,
  });

  return {
    toolCall: normalizedToolCall,
    normalizedArgs: sanitization.args,
    preSanitizationArgs: toolSpecificArgs,
    originalArgKeys,
    normalizedArgKeys,
    collisionKeys,
    validation,
  };
}

function parseArguments(rawArguments: string): JsonRecord {
  try {
    const parsed = JSON.parse(rawArguments);
    if (isRecord(parsed)) {
      return parsed;
    }
    return { value: parsed };
  } catch {
    return { value: rawArguments };
  }
}

function normalizeArgumentKeys(args: JsonRecord): {
  normalizedArgs: JsonRecord;
  collisionKeys: string[];
} {
  const normalizedArgs: JsonRecord = { ...args };
  const collisionKeys: string[] = [];

  for (const [rawKey, rawValue] of Object.entries(args)) {
    const canonicalKey = resolveCanonicalArgKey(rawKey);
    if (!canonicalKey || canonicalKey === rawKey) {
      continue;
    }

    const canonicalInOriginal = hasOwn(args, canonicalKey);
    const canonicalInNormalized = hasOwn(normalizedArgs, canonicalKey);
    if (canonicalInOriginal || canonicalInNormalized) {
      collisionKeys.push(rawKey);
      delete normalizedArgs[rawKey];
      continue;
    }

    normalizedArgs[canonicalKey] = rawValue;
    delete normalizedArgs[rawKey];
  }

  return { normalizedArgs, collisionKeys };
}

function resolveCanonicalArgKey(rawKey: string): string | null {
  const token = rawKey.toLowerCase().replace(/[^a-z0-9]/g, "");
  return ARG_KEY_ALIASES.get(token) ?? null;
}

function asInt(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.floor(value);
  }
  if (typeof value === "string") {
    const n = parseInt(value, 10);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

// Resolve various file/path argument names to the canonical key for a tool.
// Tools like read/edit/write use "filePath"; tools like grep/glob use "path".
function resolveFilePathArg(args: JsonRecord, canonicalKey: string): void {
  if (args[canonicalKey] !== undefined) return;
  for (const alias of [
    "target_file", "targetfile", "filepath", "filePath", "file_path",
    "filename", "file", "path", "targetpath",
    "relative_workspace_path", "relativeworkspacepath",
    "dir", "folder", "directory", "directorypath", "targetdirectory",
  ]) {
    if (alias === canonicalKey) continue;
    if (typeof args[alias] === "string" && args[alias].toString().trim().length > 0) {
      args[canonicalKey] = args[alias];
      delete args[alias];
      return;
    }
  }
}

function normalizeToolSpecificArgs(toolName: string, args: JsonRecord): JsonRecord {
  const normalizedToolName = toolName.toLowerCase();

  // Tools that use "filePath" as the canonical file argument
  if (normalizedToolName === "read" || normalizedToolName === "edit" || normalizedToolName === "write") {
    const normalized: JsonRecord = { ...args };
    resolveFilePathArg(normalized, "filePath");

    if (normalizedToolName === "read") {
      // Convert Cursor's inclusive line range to offset/limit
      const start = asInt(normalized["start_line_one_indexed"]);
      const end = asInt(normalized["end_line_one_indexed_inclusive"]);
      if (start != null && normalized.offset === undefined) {
        normalized.offset = start;
      }
      if (start != null && end != null && end >= start && normalized.limit === undefined) {
        normalized.limit = end - start + 1;
      }
      delete normalized["start_line_one_indexed"];
      delete normalized["end_line_one_indexed_inclusive"];
      delete normalized["should_read_entire_file"];
    }

    // Strip Cursor-specific params that have no OpenCode equivalent
    delete normalized["explanation"];
    delete normalized["instructions"];

    // Fall through to write/edit-specific handling below for those tools
    if (normalizedToolName === "read") return normalized;
    args = normalized;
  }

  // Tools that use "path" as the canonical file/directory argument
  if (normalizedToolName === "grep" || normalizedToolName === "glob") {
    const normalized: JsonRecord = { ...args };
    resolveFilePathArg(normalized, "path");

    // Cursor's grep_search uses "query"; OpenCode grep uses "pattern"
    if (normalized.pattern === undefined && typeof normalized.query === "string") {
      normalized.pattern = normalized.query;
      delete normalized.query;
    }

    // Cursor's include_pattern -> OpenCode's include (already handled by global alias)
    delete normalized["exclude_pattern"];
    delete normalized["case_sensitive"];
    delete normalized["explanation"];
    return normalized;
  }

  if (normalizedToolName === "bash") {
    const normalized: JsonRecord = { ...args };
    const normalizedCommand = normalizeBashCommand(normalized.command);
    if (typeof normalizedCommand === "string" && normalizedCommand.trim().length > 0) {
      normalized.command = normalizedCommand;
    }
    if (
      normalized.cwd === undefined
      && typeof normalized.path === "string"
      && normalized.path.trim().length > 0
    ) {
      normalized.cwd = normalized.path;
    }
    // Cursor-specific params with no OpenCode equivalent
    delete normalized["is_background"];
    delete normalized["require_user_approval"];
    delete normalized["explanation"];
    return normalized;
  }

  if (normalizedToolName === "webfetch") {
    const normalized: JsonRecord = { ...args };
    if (normalized.format === undefined) {
      normalized.format = "markdown";
    }
    delete normalized["explanation"];
    return normalized;
  }

  if (normalizedToolName === "rm") {
    const normalized: JsonRecord = { ...args };
    if (typeof normalized.force === "string") {
      const lowered = normalized.force.trim().toLowerCase();
      if (lowered === "true" || lowered === "1" || lowered === "yes") {
        normalized.force = true;
      } else if (lowered === "false" || lowered === "0" || lowered === "no") {
        normalized.force = false;
      }
    }
    return normalized;
  }

  if (normalizedToolName === "todowrite") {
    if (!Array.isArray(args.todos)) {
      return args;
    }

    const todos = args.todos.map((entry) => {
      if (!isRecord(entry)) {
        return entry;
      }

      const todo: JsonRecord = { ...entry };
      if (typeof todo.status === "string") {
        todo.status = normalizeTodoStatus(todo.status);
      }
      if (
        todo.priority === undefined
        || todo.priority === null
        || (typeof todo.priority === "string" && todo.priority.trim().length === 0)
      ) {
        todo.priority = "medium";
      }
      return todo;
    });

    return {
      ...args,
      todos,
    };
  }

  if (normalizedToolName === "write") {
    const normalized: JsonRecord = { ...args };

    // Some model variants confuse write/edit and send edit-style payload keys.
    // Map them into canonical write arguments before schema validation/sanitization.
    if (normalized.content === undefined && normalized.newString !== undefined) {
      const coerced = coerceToString(normalized.newString);
      if (coerced !== null) {
        normalized.content = coerced;
      }
      delete normalized.newString;
    }

    if (normalized.content !== undefined && typeof normalized.content !== "string") {
      const coerced = coerceToString(normalized.content);
      if (coerced !== null) {
        normalized.content = coerced;
      }
    }

    return normalized;
  }

  // For edit: coerce non-string content but do NOT synthesize oldString/newString.
  // edit_file-style calls (content without oldString) are handled by the
  // edit-file-resolver in runtime-interception, not by repair logic here.
  if (normalizedToolName === "edit") {
    const repaired: JsonRecord = { ...args };
    if (repaired.content !== undefined && typeof repaired.content !== "string") {
      const coerced = coerceToString(repaired.content);
      if (coerced !== null) {
        repaired.content = coerced;
      }
    }
    return repaired;
  }

  return args;
}

function normalizeBashCommand(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((entry) => (typeof entry === "string" ? entry : coerceToString(entry)))
      .filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
    return parts.length > 0 ? parts.join(" ") : null;
  }
  if (isRecord(value)) {
    const command = typeof value.command === "string" ? value.command : null;
    const args = Array.isArray(value.args)
      ? value.args
          .map((entry) => (typeof entry === "string" ? entry : coerceToString(entry)))
          .filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
      : [];
    if (command && args.length > 0) {
      return [command, ...args].join(" ");
    }
    if (command) {
      return command;
    }
  }
  return null;
}

function normalizeTodoStatus(status: string): string {
  const normalized = status.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (normalized === "todo_status_pending") {
    return "pending";
  }
  if (normalized === "todo_status_inprogress" || normalized === "todo_status_in_progress") {
    return "in_progress";
  }
  if (
    normalized === "todo_status_done"
    || normalized === "todo_status_complete"
    || normalized === "todo_status_completed"
  ) {
    return "completed";
  }
  if (normalized === "todo" || normalized === "pending") {
    return "pending";
  }
  if (normalized === "inprogress" || normalized === "in_progress") {
    return "in_progress";
  }
  if (normalized === "done" || normalized === "complete" || normalized === "completed") {
    return "completed";
  }
  return status;
}

function sanitizeArgumentsForSchema(
  args: JsonRecord,
  schema: unknown,
): { args: JsonRecord; unexpected: string[] } {
  if (!isRecord(schema)) {
    return { args, unexpected: [] };
  }

  if (schema.additionalProperties !== false) {
    return { args, unexpected: [] };
  }

  const properties = isRecord(schema.properties) ? schema.properties : {};
  const propertyNames = new Set(Object.keys(properties));
  const sanitized: JsonRecord = {};
  const unexpected: string[] = [];

  for (const [key, value] of Object.entries(args)) {
    if (propertyNames.has(key)) {
      sanitized[key] = value;
      continue;
    }
    unexpected.push(key);
  }

  return { args: sanitized, unexpected };
}

function validateToolArguments(
  toolName: string,
  args: JsonRecord,
  schema: unknown,
  unexpected: string[],
): ToolSchemaValidationResult {
  if (!isRecord(schema)) {
    return {
      hasSchema: false,
      ok: true,
      missing: [],
      unexpected: [],
      typeErrors: [],
    };
  }

  const properties = isRecord(schema.properties) ? schema.properties : {};
  const required = Array.isArray(schema.required)
    ? schema.required.filter((value): value is string => typeof value === "string")
    : [];
  const missing = required.filter((key) => !hasOwn(args, key));

  const typeErrors: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    const propertySchema = properties[key];
    if (!isRecord(propertySchema)) {
      continue;
    }
    if (!matchesType(value, propertySchema.type)) {
      if (propertySchema.type !== undefined) {
        typeErrors.push(`${key}: expected ${String(propertySchema.type)}`);
      }
      continue;
    }
    if (
      Array.isArray(propertySchema.enum)
      && !propertySchema.enum.some((candidate) => Object.is(candidate, value))
    ) {
      typeErrors.push(`${key}: expected enum ${JSON.stringify(propertySchema.enum)}`);
    }
  }

  const ok = missing.length === 0 && typeErrors.length === 0;
  return {
    hasSchema: true,
    ok,
    missing,
    unexpected,
    typeErrors,
    repairHint: ok ? undefined : buildRepairHint(toolName, missing, unexpected, typeErrors),
  };
}

function buildRepairHint(
  toolName: string,
  missing: string[],
  unexpected: string[],
  typeErrors: string[],
): string {
  const hints: string[] = [];
  if (missing.length > 0) {
    hints.push(`missing required: ${missing.join(", ")}`);
  }
  if (unexpected.length > 0) {
    hints.push(`remove unsupported fields: ${unexpected.join(", ")}`);
  }
  if (typeErrors.length > 0) {
    hints.push(`fix type errors: ${typeErrors.join("; ")}`);
  }
  if (
    toolName.toLowerCase() === "edit"
    && (missing.includes("oldString") || missing.includes("newString"))
  ) {
    hints.push("edit requires filePath, oldString, and newString");
  }
  return hints.join(" | ");
}

function matchesType(value: unknown, schemaType: unknown): boolean {
  if (schemaType === undefined) {
    return true;
  }
  if (Array.isArray(schemaType)) {
    return schemaType.some((entry) => matchesType(value, entry));
  }
  if (typeof schemaType !== "string") {
    return true;
  }
  switch (schemaType) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "object":
      return isRecord(value);
    case "array":
      return Array.isArray(value);
    case "null":
      return value === null;
    default:
      return true;
  }
}

function coerceToString(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return null;
  }
  if (Array.isArray(value)) {
    const parts: string[] = [];
    for (const item of value) {
      if (typeof item === "string") {
        parts.push(item);
      } else if (isRecord(item)) {
        const text = typeof item.text === "string"
          ? item.text
          : typeof item.content === "string"
            ? item.content
            : typeof item.value === "string"
              ? item.value
              : null;
        if (text !== null) {
          parts.push(text);
        } else {
          parts.push(JSON.stringify(item));
        }
      } else {
        parts.push(String(item));
      }
    }
    return parts.length > 0 ? parts.join("") : null;
  }
  if (isRecord(value)) {
    if (typeof value.text === "string") {
      return value.text;
    }
    if (typeof value.content === "string") {
      return value.content;
    }
    if (typeof value.value === "string") {
      return value.value;
    }
    return JSON.stringify(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return null;
}

function hasOwn(record: JsonRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
