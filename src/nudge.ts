import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { Text } from '@earendil-works/pi-tui';
import { foldableTurns, foldInstruction } from './fold.js';
import { episodesSettings } from './settings.js';
import { stampLabel } from './stamps.js';

export const NUDGE_TYPE = 'episodes-nudge';

/**
 * The prompt leaves folding to the model, and in long sessions it rarely folds on its own.
 * Once the context grew by `foldNudgeTokens` since it last shrank, the next user prompt gets one
 * short message asking for a fold. It names the earliest stamp a fold can start from: a fold
 * from a later stamp leaves the work before it behind the new note for good. It is appended at
 * the start of a request, where the prompt already says to fold the request before, never
 * edited in, so the cached prefix is untouched.
 */
export function registerNudge(pi: ExtensionAPI): void {
  /** Context size the growth is measured from, and the size at the previous prompt. */
  let floor: number | undefined;
  let last: number | undefined;

  pi.on('session_start', () => {
    floor = undefined;
    last = undefined;
  });

  pi.on('before_agent_start', (event, ctx) => {
    const messageType = (event as { messageType?: string }).messageType;
    if (messageType !== undefined && messageType !== 'user') return undefined;
    const settings = episodesSettings(pi.getSettings());
    if (!settings.foldNudge || !pi.getActiveTools().includes('fold')) return undefined;
    const tokens = ctx.getContextUsage()?.tokens;
    if (tokens == null) return undefined;

    // A smaller context means a fold or compaction; growth counts from there.
    if (floor === undefined || (last !== undefined && tokens < last)) floor = tokens;
    last = tokens;
    const growth = tokens - floor;
    if (growth < settings.foldNudgeTokens) return undefined;
    // Ignored, it comes back only after as much growth again.
    floor = tokens;

    const earliest = foldableTurns(ctx.sessionManager.buildSessionProjection().entries).at(-1);
    if (!earliest) return undefined;
    const k = (n: number) => `${Math.round(n / 1000)}k`;
    return {
      message: {
        customType: NUDGE_TYPE,
        display: true,
        content:
          `Your context grew by about ${k(growth)} tokens since the last fold or compaction (now ${k(tokens)}). ` +
          `Before working on this request, fold the finished episodes: ${foldInstruction(pi)}, ` +
          `with since "${stampLabel(earliest.stamp)}", the earliest stamp you can fold from, so no finished work is left behind, and a note covering everything from there. ` +
          'Skip this only if nothing finished is worth more than about 2k tokens.'
      }
    };
  });

  pi.registerMessageRenderer(NUDGE_TYPE, (message, _options, theme) => {
    const text = typeof message.content === 'string' ? message.content : '';
    const since = /since "(t\d+)"/.exec(text)?.[1];
    return new Text(
      theme.fg('dim', `episodes: asked the agent to fold${since ? ` from ${since}` : ''}`),
      0,
      0
    );
  });
}
