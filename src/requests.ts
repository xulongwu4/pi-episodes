import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ModelsSimpleStreamOptions, Usage } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import { appliedFolds } from './archive.js';
import { anchorBreakpoint, cacheAnchor } from './cache.js';
import {
  branchHasMutatedHistory,
  branchRetiredResponsesCallIds,
  sanitizeProviderPayload
} from './fold.js';
import { episodesSettings } from './settings.js';

export interface ExtraRequestState {
  retiredCallIds?: ReadonlySet<string> | (() => ReadonlySet<string>) | undefined;
  extraApplied?: ReadonlySet<string> | (() => ReadonlySet<string>) | undefined;
  hasMutatedHistory?: boolean | (() => boolean) | undefined;
  /** Whether to union active folds and retired ids from `ctx.sessionManager.getBranch()`. Default true. */
  includeBranch?: boolean | undefined;
}

function isReadonlySet(value: unknown): value is ReadonlySet<string> {
  return (
    typeof value === 'object' &&
    value !== null &&
    'has' in value &&
    typeof (value as ReadonlySet<string>).has === 'function'
  );
}

function resolveSet(
  value: ReadonlySet<string> | (() => ReadonlySet<string>) | undefined
): ReadonlySet<string> | undefined {
  return typeof value === 'function' ? value() : value;
}

/**
 * Options for requests made outside pi's agent loop (ask, fork) that should look like the
 * main thread's own requests to the provider: same session id for cache routing, same
 * reasoning settings, same breakpoint placement, and same wire replay sanitization.
 */
export function requestOptions(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  /** The messages being sent, or a way to read them per request when they keep changing. */
  messages: readonly AgentMessage[] | (() => readonly AgentMessage[]),
  signal: AbortSignal | undefined,
  extra?: ReadonlySet<string> | (() => ReadonlySet<string>) | ExtraRequestState
): ModelsSimpleStreamOptions {
  const current = typeof messages === 'function' ? messages : () => messages;
  const state: ExtraRequestState =
    extra === undefined
      ? {}
      : typeof extra === 'function' || isReadonlySet(extra)
        ? { retiredCallIds: extra }
        : extra;
  const includeBranch = state.includeBranch ?? true;
  const options: ModelsSimpleStreamOptions = { sessionId: ctx.sessionManager.getSessionId() };
  const settings = pi.getSettings();
  const useAnchor = episodesSettings(settings).cacheAnchor;
  options.onPayload = (payload) => {
    const branch = includeBranch ? ctx.sessionManager.getBranch() : [];
    const projectionEntries = includeBranch
      ? ctx.sessionManager.buildSessionProjection().entries
      : [];
    const retired = includeBranch
      ? branchRetiredResponsesCallIds(branch, projectionEntries)
      : new Set<string>();
    const extraRetired = resolveSet(state.retiredCallIds);
    if (extraRetired) {
      for (const id of extraRetired) retired.add(id);
    }
    const extraMutated =
      typeof state.hasMutatedHistory === 'function'
        ? state.hasMutatedHistory()
        : Boolean(state.hasMutatedHistory);
    const hasMutatedHistory =
      extraMutated ||
      retired.size > 0 ||
      (includeBranch && branchHasMutatedHistory(branch, projectionEntries));
    const sanitized = sanitizeProviderPayload(payload, {
      retiredCallIds: retired,
      hasMutatedHistory
    });
    let applied: ReadonlySet<string> | undefined;
    if (useAnchor) {
      const merged = includeBranch ? appliedFolds(branch) : new Set<string>();
      const childApplied = resolveSet(state.extraApplied);
      if (childApplied) {
        for (const handle of childApplied) merged.add(handle);
      }
      applied = merged;
    }
    const anchored = useAnchor
      ? anchorBreakpoint(sanitized, cacheAnchor(current(), applied))
      : undefined;
    return anchored ?? (sanitized !== payload ? sanitized : undefined);
  };
  const level = ctx.thinkingLevel ?? pi.getThinkingLevel();
  if (level !== 'off') options.reasoning = level;
  if (settings.thinkingBudgets) options.thinkingBudgets = settings.thinkingBudgets;
  // The main thread's provider-level retries and timeouts; pi's own run-level retry and
  // compaction do not exist outside its agent loop.
  const retry = settings.retry?.provider;
  if (retry?.timeoutMs !== undefined) options.timeoutMs = retry.timeoutMs;
  if (retry?.maxRetries !== undefined) options.maxRetries = retry.maxRetries;
  if (retry?.maxRetryDelayMs !== undefined) options.maxRetryDelayMs = retry.maxRetryDelayMs;
  if (signal) options.signal = signal;
  return options;
}

export function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
  };
}

export function addUsage(total: Usage, usage: Usage): void {
  total.input += usage.input;
  total.output += usage.output;
  total.cacheRead += usage.cacheRead;
  total.cacheWrite += usage.cacheWrite;
  total.totalTokens += usage.totalTokens;
  total.cost.input += usage.cost.input;
  total.cost.output += usage.cost.output;
  total.cost.cacheRead += usage.cost.cacheRead;
  total.cost.cacheWrite += usage.cost.cacheWrite;
  total.cost.total += usage.cost.total;
}
