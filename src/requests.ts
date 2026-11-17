import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ModelsSimpleStreamOptions, Usage } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import { appliedFolds, RUN_END_FOLD_TYPE } from './archive.js';
import { anchorBreakpoint, cacheAnchor } from './cache.js';
import { episodesSettings } from './settings.js';

export interface ExtraRequestState {
  /** Handles of folds in effect in the messages sent that the session's branch does not hold. */
  extraApplied?: ReadonlySet<string> | (() => ReadonlySet<string>) | undefined;
  /** Whether to add the folds in effect on `ctx.sessionManager.getBranch()`. Default true. */
  includeBranch?: boolean | undefined;
}

/**
 * What became of a request for a fold at the end of a run, from the messages after it. It is
 * answered when a response that was sent stands right behind it, whether that folded or not.
 * It is open while nothing does. It is stale when something else follows: the run was aborted
 * on it or the provider failed, both of which leave a response that is not sent again, the
 * process ended, or a message arrived that the run went on for.
 */
export function askState(after: Iterable<AgentMessage>): 'answered' | 'open' | 'stale' {
  for (const message of after) {
    // Pi's note on a change of tools or prompt, which the agent did not write and no one asked.
    if ((message.role as string) === 'system') continue;
    if (message.role !== 'assistant') return 'stale';
    if (message.stopReason !== 'error' && message.stopReason !== 'aborted') return 'answered';
  }
  return 'open';
}

/**
 * The messages without the requests for a fold that got no answer. Left in, such a request
 * would stand in front of the next request and tell the agent to fold and, under codemode, to
 * reply with one word. The next boundary removes it from the session (`registerNudge`); until
 * then, and for a thread whose context was taken before that, this is applied to whatever is
 * sent: the main agent's requests, the prefix a fork starts from, and what `ask` sends, so
 * that all three still begin alike.
 */
export function withoutStaleAsks(messages: readonly AgentMessage[]): AgentMessage[] {
  return messages.filter(
    (message, index) =>
      !(
        message.role === 'custom' &&
        message.customType === RUN_END_FOLD_TYPE &&
        askState(messages.slice(index + 1)) === 'stale'
      )
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
 * reasoning settings, same breakpoint placement.
 */
export function requestOptions(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  /** The messages being sent, or a way to read them per request when they keep changing. */
  messages: readonly AgentMessage[] | (() => readonly AgentMessage[]),
  signal: AbortSignal | undefined,
  extra: ExtraRequestState = {}
): ModelsSimpleStreamOptions {
  const current = typeof messages === 'function' ? messages : () => messages;
  const includeBranch = extra.includeBranch ?? true;
  const options: ModelsSimpleStreamOptions = { sessionId: ctx.sessionManager.getSessionId() };
  const settings = pi.getSettings();
  if (episodesSettings(settings).cacheAnchor) {
    options.onPayload = (payload) => {
      const applied = includeBranch
        ? appliedFolds(ctx.sessionManager.getBranch())
        : new Set<string>();
      for (const handle of resolveSet(extra.extraApplied) ?? []) applied.add(handle);
      return anchorBreakpoint(payload, cacheAnchor(current(), applied));
    };
  }
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
