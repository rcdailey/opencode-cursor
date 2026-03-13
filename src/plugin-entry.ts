/**
 * OpenCode-only entrypoint.
 *
 * When cursor-acp is removed from the `plugin` array in opencode.json,
 * this entrypoint turns into a no-op so users can disable the plugin
 * without deleting the symlink file.
 */
import type { Plugin } from "@opencode-ai/plugin";
import { shouldEnableCursorPlugin } from "./plugin-toggle.js";
import { createLogger } from "./utils/logger.js";

const log = createLogger("plugin-entry");

log.info("Loading cursor-acp plugin from source", { path: import.meta.url });

const CursorPluginEntry: Plugin = async (input) => {
  const state = shouldEnableCursorPlugin();
  if (!state.enabled) {
    log.info("Plugin disabled in OpenCode config; skipping initialization", {
      configPath: state.configPath,
      reason: state.reason,
    });
    return {};
  }

  const mod = await import("./plugin.js");
  return mod.CursorPlugin(input);
};

export default CursorPluginEntry;
