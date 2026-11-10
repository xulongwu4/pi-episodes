/**
 * The `episodes` key in pi's settings.json. Pi's settings type has no slot for extension keys,
 * so the object is read loosely and every value falls back to its default.
 */
export interface EpisodesSettings {
  /** Stop a fork after this long; its work so far stays archived. */
  forkTimeoutMs: number;
  /** Remind the agent of open items from notes folded since the last reminder. */
  openReminder: boolean;
  /** Upper bound on the reminder's size. */
  openReminderMaxTokens: number;
  /** After a compaction, tell the agent the compacted turns are archived and recallable. */
  compactionNotice: boolean;
  /** Keep a cache breakpoint where the next fold can start. Off only to measure what it saves. */
  cacheAnchor: boolean;
  /**
   * Ask the agent to fold once `foldNudgeTokens` (at most a quarter of the context window)
   * built up since the last fold, compaction or nudge, naming the earliest stamp it can fold from.
   */
  foldNudge: boolean;
  foldNudgeTokens: number;
  /**
   * Cut a tool result of `outputCapChars` characters or more down to about
   * `outputCapKeepChars` before it enters the context, and keep all of it in a file next to
   * the session.
   */
  outputCap: boolean;
  outputCapChars: number;
  outputCapKeepChars: number;
  /**
   * Snapshot the working tree at every stamped message, for the disk changes in a note and
   * the staleness of a thread. Off where `git add -A` over the tree is too slow or unwanted.
   */
  snapshots: boolean;
}

const DEFAULTS: EpisodesSettings = {
  forkTimeoutMs: 30 * 60 * 1000,
  openReminder: true,
  openReminderMaxTokens: 300,
  compactionNotice: true,
  cacheAnchor: true,
  foldNudge: true,
  foldNudgeTokens: 50_000,
  outputCap: true,
  outputCapChars: 16_000,
  outputCapKeepChars: 6_000,
  snapshots: true
};

function flag(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function positive(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
}

export function episodesSettings(settings: object): EpisodesSettings {
  const raw = (settings as { episodes?: Record<string, unknown> }).episodes ?? {};
  return {
    forkTimeoutMs: positive(raw.forkTimeoutMs, DEFAULTS.forkTimeoutMs),
    openReminder: flag(raw.openReminder, DEFAULTS.openReminder),
    openReminderMaxTokens: positive(raw.openReminderMaxTokens, DEFAULTS.openReminderMaxTokens),
    compactionNotice: flag(raw.compactionNotice, DEFAULTS.compactionNotice),
    cacheAnchor: flag(raw.cacheAnchor, DEFAULTS.cacheAnchor),
    foldNudge: flag(raw.foldNudge, DEFAULTS.foldNudge),
    foldNudgeTokens: positive(raw.foldNudgeTokens, DEFAULTS.foldNudgeTokens),
    outputCap: flag(raw.outputCap, DEFAULTS.outputCap),
    outputCapChars: positive(raw.outputCapChars, DEFAULTS.outputCapChars),
    outputCapKeepChars: positive(raw.outputCapKeepChars, DEFAULTS.outputCapKeepChars),
    snapshots: flag(raw.snapshots, DEFAULTS.snapshots)
  };
}
