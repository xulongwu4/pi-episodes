import type { ExtensionAPI, ProjectedSessionEntry } from '@earendil-works/pi-coding-agent';
import { foldableTurns, foldBoundary, foldInstruction, renderFoldAsk } from './fold.js';
import { estimateTokens } from './messages.js';
import { episodesSettings } from './settings.js';
import { stampLabel } from './stamps.js';

export const NUDGE_TYPE = 'episodes-nudge';

/** At most a quarter of the window, so small windows are nudged well before pi compacts. */
export function nudgeThreshold(tokens: number, contextWindow: number | undefined): number {
  return contextWindow ? Math.min(tokens, Math.floor(contextWindow / 4)) : tokens;
}

/**
 * Tokens a fold could remove that built up since the latest fold note, compaction or nudge,
 * read from the entries rather than kept in memory, so a resumed session or another branch is
 * measured the same way. The system prompt and tool declarations are not counted.
 */
export function nudgeGrowth(entries: readonly ProjectedSessionEntry[]): number {
  const nudged = entries.findLastIndex(
    (entry) =>
      entry.sourceEntry.type === 'custom_message' && entry.sourceEntry.customType === NUDGE_TYPE
  );
  const from = Math.max(foldBoundary(entries), nudged) + 1;
  return estimateTokens(entries.slice(from).flatMap((entry) => entry.messages));
}

/**
 * The prompt leaves folding to the model, and in long sessions it rarely folds on its own.
 * Once `foldNudgeTokens` built up since the last fold, compaction or nudge, the next user prompt
 * gets one short message asking for a fold, and so does the next request of a run that grew by
 * twice as much on its own. It names the earliest stamp a fold can start from: a fold from a
 * later stamp leaves the work before it behind the new note for good. It is appended, never
 * edited in, so the cached prefix is untouched.
 */
export function registerNudge(pi: ExtensionAPI): void {
  const nudge = (
    entries: readonly ProjectedSessionEntry[],
    contextWindow: number | undefined,
    factor: number,
    when: string
  ): string | undefined => {
    const settings = episodesSettings(pi.getSettings());
    if (!settings.foldNudge || !pi.getActiveTools().includes('fold')) return undefined;
    const growth = nudgeGrowth(entries);
    if (growth < factor * nudgeThreshold(settings.foldNudgeTokens, contextWindow)) return undefined;
    const earliest = foldableTurns(entries).at(-1);
    if (!earliest) return undefined;
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
    const content = nudge(
      ctx.sessionManager.buildSessionProjection().entries,
      ctx.model?.contextWindow,
      1,
      'this request'
    );
    return content ? { message: { customType: NUDGE_TYPE, content, display: true } } : undefined;
  });

  // A long run gets no user prompt to nudge on. The preview already holds this turn's folds.
  pi.on('turn_end', (event, ctx) => {
    if (event.message.role !== 'assistant' || event.message.stopReason !== 'toolUse') {
      return undefined;
    }
    const content = nudge(event.context.contextEntries, ctx.model?.contextWindow, 2, 'the task');
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
