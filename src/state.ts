import { realpathSync } from 'node:fs';
import type { ContextEditEntryDraft, ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { FoldDetails, ForkArchive } from './archive.js';
import type { RecallIndex } from './search.js';
import { Snapshots } from './snapshots.js';
import { readStamp } from './stamps.js';

export interface PendingFold {
  handle: string;
  messages: number;
  tokens: number;
  edits: ContextEditEntryDraft[];
  /** The notes the fold covers, as its result holds them under its own note. */
  carried?: string;
  /**
   * Set when a codemode script made the fold call: the pending fold is keyed by the script's
   * call instead, and its result carries the details, the note if the script did not print it,
   * and the notes the fold covers.
   */
  script?: {
    foldCallId: string;
    note: string;
    details: FoldDetails;
    /** A hook changed the fold's result before the script got it. */
    changed?: boolean;
  };
}

/** Per-session runtime state. Everything durable lives in the session file instead. */
export class EpisodeState {
  nextStamp = 1;
  /**
   * The working directory with symlinks resolved, so paths from file tools compare equal to
   * the ones git reports (on macOS the temp directory alone is behind a symlink).
   */
  cwd = process.cwd();
  snapshots: Snapshots | undefined;
  /** Context edits computed by a fold call, emitted at the end of its turn. */
  readonly pendingFolds = new Map<string, PendingFold>();
  /** Response entry ids that already made a fold call in the current run, with that call. */
  readonly foldResponses = new Map<string, string>();
  /** The calling tool of each nested call seen in the current run, from `tool_call` events. */
  readonly parentCalls = new Map<string, string>();
  /** Counts settled runs, so a fold that finishes after its run ended can tell. */
  run = 0;
  /** `handle:path` pairs already surfaced, so a folded thread is pointed at once per file. */
  readonly surfaced = new Set<string>();
  /** Fork transcripts for sessions that have no file to write them next to. */
  readonly forkArchives = new Map<string, ForkArchive>();
  /**
   * Set by /fold so the requested fold ends the run instead of continuing the task: `since`
   * when the user picked where the span starts, `request` when the agent picks.
   */
  manualFold: 'since' | 'request' | undefined;
  /** Whether the cache anchor was placed on the last main-thread request; unknown off Anthropic. */
  anchorPlaced: boolean | undefined;
  /** Forks in progress, for the footer status. */
  runningForks = 0;
  /** Built on the first ranked recall; rebuilt per session rather than stored, it is derived. */
  recallIndex: RecallIndex | undefined;

  reset(ctx: ExtensionContext, snapshots = true): void {
    void this.snapshots?.dispose();
    this.cwd = realpathOr(ctx.cwd);
    this.snapshots = snapshots
      ? new Snapshots(this.cwd, ctx.sessionManager.getSessionId())
      : undefined;
    this.pendingFolds.clear();
    this.foldResponses.clear();
    this.parentCalls.clear();
    this.run++;
    this.surfaced.clear();
    this.forkArchives.clear();
    this.manualFold = undefined;
    this.anchorPlaced = undefined;
    this.runningForks = 0;
    this.recallIndex = undefined;

    // Stamps must stay unique across every branch of the session file, so `since` always
    // resolves to exactly one message.
    let max = 0;
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type !== 'message') continue;
      max = Math.max(max, readStamp(entry.message) ?? 0);
    }
    this.nextStamp = max + 1;
  }

  /** The call in the transcript that a nested call descends from; the call itself otherwise. */
  rootCall(toolCallId: string): string {
    let id = toolCallId;
    for (let parent = this.parentCalls.get(id); parent; parent = this.parentCalls.get(id)) {
      id = parent;
    }
    return id;
  }
  async dispose(): Promise<void> {
    await this.snapshots?.dispose();
    this.snapshots = undefined;
  }
}

function realpathOr(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return path;
  }
}
