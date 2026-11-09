import type { AssistantMessage } from '@earendil-works/pi-ai';
import type {
  ExtensionAPI,
  ExtensionContext,
  ProjectedSessionEntry
} from '@earendil-works/pi-coding-agent';
import { foldInstruction, foldStart, renderFoldAsk, USER_TEXT_CAP } from './fold.js';
import { estimateTokens } from './messages.js';
import { episodesSettings } from './settings.js';
import { stampLabel } from './stamps.js';

export const NUDGE_TYPE = 'episodes-nudge';

/** At most a quarter of the window, so small windows are nudged well before pi compacts. */
export function nudgeThreshold(tokens: number, contextWindow: number | undefined): number {
  return contextWindow ? Math.min(tokens, Math.floor(contextWindow / 4)) : tokens;
}

/**
 * Tokens a fold from the earliest stamp could remove that built up since the latest nudge,
 * read from the entries rather than kept in memory, so a resumed session or another branch is
 * measured the same way. The system prompt, tool declarations, results a fold cannot reach,
 * such as those of calls made next to a fold call, and what a fold keeps, such as the start of
 * each user message, are not counted.
 */
export function nudgeGrowth(entries: readonly ProjectedSessionEntry[]): number {
  const start = foldStart(entries);
  if (!start) return 0;
  const nudged = entries.findLastIndex(
    (entry) =>
      entry.sourceEntry.type === 'custom_message' && entry.sourceEntry.customType === NUDGE_TYPE
  );
  let tokens = 0;
  for (const entry of entries.slice(Math.max(start.index, nudged + 1))) {
    const source = entry.sourceEntry;
    const role = source.type === 'message' ? source.message.role : undefined;
    const size = estimateTokens(entry.messages);
    if (role === 'user') tokens += Math.max(0, size - USER_TEXT_CAP / 4);
    else if (source.type === 'custom_message' || role === 'assistant' || role === 'toolResult') {
      tokens += size;
    }
  }
  return tokens;
}

/** The room pi keeps free before it compacts, `compaction.reserveTokens`, read loosely. */
function reserveTokens(pi: ExtensionAPI, model: ExtensionContext['model']): number {
  type Reserve = { reserveTokens?: unknown };
  const compaction = (
    pi.getSettings() as { compaction?: Reserve & { modelOverrides?: Record<string, Reserve> } }
  ).compaction;
  const override = model && compaction?.modelOverrides?.[`${model.provider}/${model.id}`];
  const value = override?.reserveTokens ?? compaction?.reserveTokens;
  return typeof value === 'number' && value >= 0 ? value : 16_384;
}

/** A response pi follows with another request on its own: one that called tools. */
function continues(message: AssistantMessage): boolean {
  return (
    message.stopReason === 'toolUse' ||
    (message.stopReason === 'length' && message.content.some((block) => block.type === 'toolCall'))
  );
}

/**
 * The prompt leaves folding to the model, and in long sessions it rarely folds on its own.
 * Once `foldNudgeTokens` built up since the last fold, compaction or nudge, the next user prompt
 * gets one short message asking for a fold, and so does the next request of a run that grew by
 * twice as much on its own. Once less than `foldNudgeTokens` of room is left before pi compacts,
 * a quarter of it is enough. It names the earliest stamp a fold can start from: a fold from a
 * later stamp leaves the work before it behind the new note for good. It is appended, never
 * edited in, so the cached prefix is untouched.
 */
export function registerNudge(pi: ExtensionAPI): void {
  const nudge = (
    entries: readonly ProjectedSessionEntry[],
    ctx: ExtensionContext,
    factor: number,
    when: string,
    /** Tokens about to be added that the context usage does not hold yet. */
    incoming = 0
  ): string | undefined => {
    const settings = episodesSettings(pi.getSettings());
    if (!settings.foldNudge || !pi.getActiveTools().includes('fold')) return undefined;
    const earliest = foldStart(entries);
    if (!earliest) return undefined;
    // The window of the model that answers, which a routing model does not report itself.
    const usage = ctx.getContextUsage();
    const threshold = nudgeThreshold(settings.foldNudgeTokens, usage?.contextWindow);
    const pressed =
      usage?.tokens != null &&
      usage.tokens + incoming + threshold >= usage.contextWindow - reserveTokens(pi, ctx.model);
    const growth = nudgeGrowth(entries);
    if (growth < (pressed ? threshold / 4 : factor * threshold)) return undefined;
    const since = stampLabel(earliest.stamp);
    return (
      `About ${Math.round(growth / 1000)}k tokens of work built up since the last fold, compaction or message like this one. ` +
      `If the work since ${since} is finished and ${when} does not need its raw output, fold it first: ${foldInstruction(pi)}, with since "${since}", the earliest stamp you can fold from, and a note covering everything from there; then continue. ` +
      'Do not fold from a later stamp instead: the work before it would stay in your context for good. If that work is still needed, fold once it is done.'
    );
  };

  pi.on('before_agent_start', (event, ctx) => {
    // Extension messages, such as the /fold request, are not the user coming back to the task.
    const messageType = (event as { messageType?: string }).messageType;
    if (messageType !== undefined && messageType !== 'user') return undefined;
    // The prompt is not in the session yet: it adds to the context, but not to what can be folded.
    const prompt = (event as { prompt?: string }).prompt ?? '';
    const content = nudge(
      ctx.sessionManager.buildSessionProjection().entries,
      ctx,
      1,
      'this request',
      Math.ceil(prompt.length / 4)
    );
    return content ? { message: { customType: NUDGE_TYPE, content, display: true } } : undefined;
  });

  // A long run gets no user prompt to nudge on. The preview already holds this turn's folds.
  pi.on('turn_end', (event, ctx) => {
    if (event.message.role !== 'assistant' || !continues(event.message)) return undefined;
    const content = nudge(event.context.contextEntries, ctx, 2, 'the task');
    if (!content) return undefined;
    return {
      entries: [
        ...event.entries,
        { type: 'custom_message' as const, customType: NUDGE_TYPE, content, display: true }
      ]
    };
  });

  pi.registerMessageRenderer(NUDGE_TYPE, renderFoldAsk);
}
