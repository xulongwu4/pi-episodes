import type { AssistantMessage, ImageContent } from '@earendil-works/pi-ai';
import {
  type ExtensionAPI,
  type ExtensionContext,
  estimateTokens as estimateMessageTokens,
  type ProjectedSessionEntry
} from '@earendil-works/pi-coding-agent';
import {
  foldBoundary,
  foldInstruction,
  foldStart,
  isRemovedWork,
  renderFoldAsk,
  USER_TEXT_CAP,
  unfoldedBehindNote
} from './fold.js';
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
export function nudgeGrowth(entries: readonly ProjectedSessionEntry[], notes = true): number {
  const start = foldStart(entries, notes);
  if (!start) return 0;
  const nudged = entries.findLastIndex(
    (entry) =>
      entry.sourceEntry.type === 'custom_message' && entry.sourceEntry.customType === NUDGE_TYPE
  );
  return foldableTokens(entries.slice(Math.max(start.index, nudged + 1)));
}

/**
 * Tokens a fold from `start`, by default the earliest stamp after the latest compaction, could
 * remove that sit up to the latest note: earlier notes, and work a fold that started later
 * left behind them.
 */
export function tokensBehindNote(
  entries: readonly ProjectedSessionEntry[],
  start = foldStart(entries, false)
): number {
  const boundary = foldBoundary(entries);
  return start && start.index <= boundary
    ? foldableTokens(entries.slice(start.index, boundary + 1))
    : 0;
}

/**
 * Whether the nudge about work left behind the latest note is due. The agent last decided
 * about that work at the latest nudge or, inside a run, at a later fold that stopped at
 * `until`: its own choice to keep those turns for its next steps, which the next prompt asks
 * about. With no such point the nudge is due at once. After one, it is due once a fold has
 * removed work done since and left this work behind again, or once a quarter of the
 * threshold has built up since. A fold that only merges notes removes no work and does not
 * bring the nudge back: an agent that answered each nudge that way would be asked without end.
 */
function dueBehindNote(
  entries: readonly ProjectedSessionEntry[],
  left: { keptAt: number },
  inRun: boolean,
  quarter: number
): boolean {
  const nudged = entries.findLastIndex(
    (entry) =>
      entry.sourceEntry.type === 'custom_message' && entry.sourceEntry.customType === NUDGE_TYPE
  );
  const decided = Math.max(nudged, inRun ? left.keptAt : -1);
  if (decided < 0) return true;
  const since = entries.slice(decided + 1);
  return since.some(isRemovedWork) || foldableTokens(since) >= quarter;
}

function foldableTokens(entries: readonly ProjectedSessionEntry[]): number {
  let tokens = 0;
  for (const entry of entries) {
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
 * a quarter of it is enough. It names the earliest stamp after the latest note: a fold from a
 * later stamp leaves the work before it in context. A fold that reaches back past notes merges
 * them, and what a later fold left behind them, but rewrites the cache from its start. Two
 * things make that pay. Work that sits behind the latest note unfolded, left there by folds
 * from later stamps or kept by `until`, is sent again with every request: once it reaches
 * `foldNudgeTokens`, or a quarter of that close to compaction, the nudge names the stamp it
 * starts at; `dueBehindNote` says when it comes again.
 * A model that folds only its last request would otherwise not be asked about the work
 * before it until pi compacts; and asked then to fold from the first stamp of a long session,
 * one model did not, where it did fold from the stamp its unfolded work started at. Notes
 * alone are merged only close to compaction, which loses the cache too: once a quarter of
 * the threshold sits up to the latest note and no such work does, the nudge names the
 * earliest stamp after the compaction. It is appended, never edited in, so the cached prefix
 * is untouched.
 */
export function registerNudge(pi: ExtensionAPI): void {
  const nudge = (
    entries: readonly ProjectedSessionEntry[],
    ctx: ExtensionContext,
    /** Between two requests of a run, rather than at a prompt of the user. */
    inRun: boolean,
    /** Tokens about to be added that the context usage does not hold yet. */
    incoming = 0
  ): string | undefined => {
    const settings = episodesSettings(pi.getSettings());
    if (!settings.foldNudge || !pi.getActiveTools().includes('fold')) return undefined;
    // The window of the model that answers, which a routing model does not report itself.
    const usage = ctx.getContextUsage();
    const threshold = nudgeThreshold(settings.foldNudgeTokens, usage?.contextWindow);
    const pressed =
      usage?.tokens != null &&
      usage.tokens + incoming + threshold >= usage.contextWindow - reserveTokens(pi, ctx.model);
    const left = unfoldedBehindNote(entries);
    const leftTokens = left ? foldableTokens(left.work) : 0;
    const stranded = leftTokens >= (pressed ? threshold / 4 : threshold) ? left : undefined;
    const behind = pressed || stranded ? tokensBehindNote(entries, stranded) : 0;
    const merge = stranded !== undefined || behind >= threshold / 4;
    const growth = nudgeGrowth(entries, !merge);
    const due = stranded
      ? dueBehindNote(entries, stranded, inRun, threshold / 4)
      : growth >= (pressed ? threshold / 4 : (inRun ? 2 : 1) * threshold);
    if (!due) return undefined;
    const when = inRun ? 'the task' : 'this request';
    const earliest = stranded ?? foldStart(entries, !merge);
    if (!earliest) return undefined;
    const since = stampLabel(earliest.stamp);
    const k = (tokens: number) => Math.round(tokens / 1000);
    const lead = stranded
      ? `About ${k(leftTokens)}k tokens of work that later folds did not cover are still in your context behind their notes, from ${since} on, and are sent again with every request. `
      : `About ${k(growth)}k tokens of work built up since the last fold, compaction or message like this one. `;
    const start = stranded
      ? `with since "${since}"`
      : `with since "${since}", the earliest stamp you can fold from,`;
    const covering = merge
      ? `a note covering everything from there, including the notes of ${stranded ? 'the folds made since' : 'earlier folds there'} (about ${k(behind)}k tokens with what they left behind), which it merges into one${pressed ? '; pi would compact soon otherwise' : ''}`
      : 'a note covering everything from there';
    return (
      lead +
      `If the work since ${since} is finished and ${when} does not need its raw output, fold it first: ${foldInstruction(pi)}, ${start} and ${covering}; then continue. ` +
      'Do not fold from a later stamp instead: the work before it would stay in your context. If that work is still needed, fold once it is done.'
    );
  };

  pi.on('before_agent_start', (event, ctx) => {
    // Extension messages, such as the /fold request, are not the user coming back to the task.
    const messageType = (event as { messageType?: string }).messageType;
    if (messageType !== undefined && messageType !== 'user') return undefined;
    // The prompt is not in the session yet: it adds to the context, but not to what can be folded.
    const { prompt = '', images = [] } = event as { prompt?: string; images?: ImageContent[] };
    const content = nudge(
      ctx.sessionManager.buildSessionProjection().entries,
      ctx,
      false,
      estimateMessageTokens({
        role: 'user',
        content: [{ type: 'text', text: prompt }, ...images],
        timestamp: Date.now()
      })
    );
    return content ? { message: { customType: NUDGE_TYPE, content, display: true } } : undefined;
  });

  // A long run gets no user prompt to nudge on. The preview already holds this turn's folds.
  pi.on('turn_end', (event, ctx) => {
    if (event.message.role !== 'assistant' || !continues(event.message)) return undefined;
    const content = nudge(event.context.contextEntries, ctx, true);
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
