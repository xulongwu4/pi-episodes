import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, join } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { Usage } from '@earendil-works/pi-ai';
import {
  buildSessionProjection,
  type CompactionEntry,
  type ExtensionContext,
  type SessionEntry,
  sessionEntryToContextMessages
} from '@earendil-works/pi-coding-agent';
import { readStamp, stampLabel } from './stamps.js';

export type SessionView = ExtensionContext['sessionManager'];

export const FOLD_TOOL = 'fold';
export const FORK_TOOL = 'fork';
/** The message /fold sends to ask for a fold. */
export const FOLD_REQUEST_TYPE = 'episodes-fold-request';
/**
 * Appended after a fold's context edits, in the same boundary batch. Pi checks the batch as a
 * whole but writes it one entry at a time, so a fold result without the marker either never took
 * effect (the batch was refused, or a crash came before the turn end) or was cut short partway
 * through its edits; the next turn end completes the latter.
 */
export const FOLD_APPLIED_TYPE = 'episodes-fold-applied';
/**
 * Written first in a fold's batch, with the fold's handle and the edits that follow, in order.
 * It tells a fold cut short from one that never took effect, and holds the edits still to write.
 */
export const FOLD_STARTED_TYPE = 'episodes-fold-started';
/**
 * Key under which a fold called from a codemode script stores its FoldDetails on the script's
 * tool result. That result stays in context as the fold's note, as a fold result does for a
 * top-level call; the script's own details stay alongside.
 */
export const FOLD_DETAILS_KEY = 'episodesFold';

/** Handles of folds whose edits are among these entries. */
export function appliedFolds(entries: readonly SessionEntry[]): Set<string> {
  const handles = new Set<string>();
  for (const entry of entries) {
    if (entry.type !== 'custom' || entry.customType !== FOLD_APPLIED_TYPE) continue;
    const handle = (entry.data as { handle?: unknown } | undefined)?.handle;
    if (typeof handle === 'string') handles.add(handle);
  }
  return handles;
}

/** Stored on the fold tool result, so the record follows the branch it belongs to. */
export interface FoldDetails {
  kind: 'fold';
  handle: string;
  since: string;
  firstEntryId: string;
  lastEntryId: string;
  /** Last stamp inside the span, so the range it covered can be shown. */
  until?: string;
  messages: number;
  tokens: number;
  headline: string;
  /** The note's open items, for the reminder after the fold. */
  open?: string;
  keptUserTurns: string[];
  /** First stamp kept after the span, when the fold stopped before the call that made it. */
  keptFrom?: string;
  /**
   * Handles of the earlier folds whose notes this fold's note stands for, oldest first: those
   * inside the span and those kept under each of them.
   */
  covers?: string[];
  /**
   * The note as the agent wrote it, which a later fold that covers this one keeps under its
   * own while the note's result still holds it. Absent for a fold made before notes were
   * recorded.
   */
  note?: string;
  /** Absolute paths touched or changed inside the span. */
  files: string[];
  /** Working tree at fold time, to tell later readers which of `files` changed since. */
  treeAfter?: string;
  /** Files the span only read, not changed. */
  readOnly?: string[];
  /** Estimated tokens of the note the agent wrote for the span. */
  noteTokens?: number;
  /** Estimated tokens of the earlier notes kept under it, which the fold did not free. */
  keptTokens?: number;
  /**
   * Length and hash of the text in which the fold wrote those notes under its own. A later
   * fold that covers this one takes that text apart only if it still stands as written.
   */
  kept?: { chars: number; sha256: string };
  /**
   * Set when the user asked for the fold with /fold: `since` when they picked where the span
   * starts, `request` when the agent picked. Absent for folds the agent chose on its own.
   */
  manual?: 'since' | 'request';
  /**
   * The cache anchor on the request that called fold: whether it was placed (absent for
   * providers without breakpoints), and the tokens between it and the span start, which the
   * next request writes to the cache again. No gap when the anchor was inside the span.
   */
  anchor?: { placed?: boolean; gapTokens?: number };
}

export type ForkStatus = 'done' | 'failed' | 'timeout' | 'aborted';

export interface ForkDetails {
  kind: 'fork';
  handle: string;
  task: string;
  /** The assistant entry that called fork; the child started from the view at that entry. */
  parentEntryId: string;
  turns: number;
  headline: string;
  /** The final note's open items, for the reminder after the fork. */
  open?: string;
  /** Absolute paths touched or changed while the fork ran. */
  files: string[];
  /** Working tree when the fork finished. */
  treeAfter?: string;
  /** Child transcript file, absent for in-memory sessions. */
  archive?: string;
  /** The child's own model calls. */
  usage?: Usage;
  /** Files the fork only read, not changed. */
  readOnly?: string[];
  status?: ForkStatus;
  /** Folds the child made of its own work. */
  folds?: number;
}

/**
 * The turns a pi compaction summarized away, as one more archived thread. Derived from the
 * compaction entry itself; nothing is stored for it.
 */
export interface CompactionRecord {
  kind: 'compaction';
  handle: string;
  entryId: string;
  headline: string;
  tokens: number;
  /** Files pi's default compaction recorded as modified, as recorded (possibly relative). */
  files: string[];
  /** Never set; present so every record can be read the same way. */
  open?: string;
  /** Never set; there is no snapshot from compaction time. */
  treeAfter?: string;
}

export type ThreadRecord = FoldDetails | ForkDetails | CompactionRecord;

export function compactionHandle(entryId: string): string {
  return `cp-${entryId.slice(0, 8)}`;
}

function compactionRecord(entry: CompactionEntry): CompactionRecord {
  // Only modified files: compactions list every file read, which would make the file-touch
  // cue fire on nearly everything.
  const modified = (entry.details as { modifiedFiles?: unknown } | undefined)?.modifiedFiles;
  return {
    kind: 'compaction',
    handle: compactionHandle(entry.id),
    entryId: entry.id,
    headline: `turns compacted at ${entry.timestamp.slice(0, 16).replace('T', ' ')}`,
    tokens: entry.tokensBefore,
    files: Array.isArray(modified)
      ? modified.filter((path): path is string => typeof path === 'string')
      : []
  };
}

/** A fold made inside a fork. Indices point into the fork archive's `messages`. */
export interface NestedFold {
  record: FoldDetails;
  first: number;
  last: number;
  /** Messages the fold hid from the child's view. */
  omit: number[];
}

/** Messages left after hiding what the given nested folds omitted. */
function withoutFolded(
  messages: readonly AgentMessage[],
  folds: readonly NestedFold[]
): AgentMessage[] {
  const omitted = new Set(folds.flatMap((fold) => fold.omit));
  return messages.filter((_, index) => !omitted.has(index));
}

export interface ForkArchive {
  handle: string;
  task: string;
  parentEntryId: string;
  /** Results that closed sibling fork calls in the child's prefix, so `ask` rebuilds it as it was. */
  siblings?: AgentMessage[];
  /** Messages the child added after the parent prefix, starting with the task. */
  messages: AgentMessage[];
  /** Folds the child made of its own work, in order. */
  folds?: NestedFold[];
}

export interface Thread {
  handle: string;
  kind: ThreadRecord['kind'];
  label: string;
  /** What the thread itself held, unedited. */
  messages: AgentMessage[];
  /**
   * The full context the thread ran with: everything before it plus the thread. Absent for
   * compacted turns, which filled the window when they were compacted; `ask` cannot resume them.
   */
  context?: () => AgentMessage[];
  record: ThreadRecord;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function isFoldDetails(value: unknown): value is FoldDetails {
  return isRecord(value) && value.kind === 'fold' && typeof value.handle === 'string';
}

export function isForkDetails(value: unknown): value is ForkDetails {
  return isRecord(value) && value.kind === 'fork' && typeof value.handle === 'string';
}

/**
 * The fold a tool result stands for: a successful fold result, or the result of a codemode
 * script whose fold call was applied there. Whether the fold took effect is a separate
 * question, answered by {@link appliedFolds}.
 */
export function foldNoteDetails(message: AgentMessage | undefined): FoldDetails | undefined {
  if (message?.role !== 'toolResult' || message.isError) return undefined;
  if (message.toolName === FOLD_TOOL) {
    return isFoldDetails(message.details) ? message.details : undefined;
  }
  const nested = isRecord(message.details) ? message.details[FOLD_DETAILS_KEY] : undefined;
  return isFoldDetails(nested) ? nested : undefined;
}

function toolDetails(entry: SessionEntry, toolName: string): unknown {
  if (entry.type !== 'message' || entry.message.role !== 'toolResult') return undefined;
  if (entry.message.toolName !== toolName) return undefined;
  // A failed fork still did work worth recalling.
  return entry.message.details;
}

export function foldEntryDetails(entry: SessionEntry): FoldDetails | undefined {
  return entry.type === 'message' ? foldNoteDetails(entry.message) : undefined;
}

export function foldRecords(entries: readonly SessionEntry[]): FoldDetails[] {
  const applied = appliedFolds(entries);
  return entries
    .map(foldEntryDetails)
    .filter((record): record is FoldDetails => record !== undefined && applied.has(record.handle));
}

export function forkRecords(entries: readonly SessionEntry[]): ForkDetails[] {
  return entries.map((entry) => toolDetails(entry, FORK_TOOL)).filter(isForkDetails);
}

/** Folds, forks and compactions on these entries, in order. */
export function threadRecords(entries: readonly SessionEntry[]): ThreadRecord[] {
  const applied = appliedFolds(entries);
  return entries.flatMap((entry): ThreadRecord[] => {
    if (entry.type === 'compaction') return [compactionRecord(entry)];
    const fold = foldEntryDetails(entry);
    if (fold) return applied.has(fold.handle) ? [fold] : [];
    const record = toolDetails(entry, FORK_TOOL);
    return isForkDetails(record) ? [record] : [];
  });
}

/** Directory next to the session file where fork transcripts are kept. */
export function archiveDir(sessions: SessionView): string | undefined {
  const file = sessions.getSessionFile();
  if (!file) return undefined;
  return join(dirname(file), `${basename(file, extname(file))}.episodes`);
}

export async function writeForkArchive(
  dir: string | undefined,
  archive: ForkArchive
): Promise<string | undefined> {
  if (!dir) return undefined;
  await mkdir(dir, { recursive: true });
  const file = join(dir, `${archive.handle}.json`);
  // Written after every turn; a crash mid-write must not leave a truncated transcript.
  const temporary = `${file}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(archive));
  await rename(temporary, file);
  return file;
}

export class ThreadIndex {
  private readonly sessions: SessionView;
  private readonly memory: ReadonlyMap<string, ForkArchive>;

  constructor(sessions: SessionView, memory: ReadonlyMap<string, ForkArchive>) {
    this.sessions = sessions;
    this.memory = memory;
  }

  /** Threads archived on the current branch, oldest first. */
  async onBranch(): Promise<Thread[]> {
    const threads: Thread[] = [];
    for (const record of threadRecords(this.sessions.getBranch())) {
      if (record.kind === 'fork') {
        threads.push(...(await this.forkThreads(record)));
        continue;
      }
      const thread =
        record.kind === 'fold' ? this.foldThread(record) : this.compactionThread(record);
      if (thread) threads.push(thread);
    }
    return threads;
  }

  /** Any archived thread in the session, on this branch or an abandoned one. */
  async find(handle: string): Promise<Thread | undefined> {
    const entries = this.sessions.getEntries();
    const fold = foldRecords(entries).find((record) => record.handle === handle);
    if (fold) return this.foldThread(fold);
    // A fork's handle, or a fold made inside one: the fork's archive holds both.
    for (const fork of forkRecords(entries)) {
      const thread = (await this.forkThreads(fork)).find((found) => found.handle === handle);
      if (thread) return thread;
    }
    const compaction = entries.find(
      (entry): entry is CompactionEntry =>
        entry.type === 'compaction' && compactionHandle(entry.id) === handle
    );
    return compaction ? this.compactionThread(compactionRecord(compaction)) : undefined;
  }

  /**
   * What the compaction summarized: the view just before it (folds applied, so folded spans
   * stay in their own threads), from after the previous summary up to the first kept entry.
   */
  private compactionThread(record: CompactionRecord): Thread | undefined {
    const entries = this.sessions.getEntries();
    const entry = entries.find((candidate) => candidate.id === record.entryId);
    if (entry?.type !== 'compaction' || !entry.parentId) return undefined;

    const before = buildSessionProjection(entries, entry.parentId).entries;
    const start = before[0]?.sourceEntry.type === 'compaction' ? 1 : 0;
    const kept = before.findIndex(
      (candidate) => candidate.sourceEntry.id === entry.firstKeptEntryId
    );
    const messages = before
      .slice(start, kept < 0 ? before.length : kept)
      .flatMap((candidate) => candidate.messages)
      .filter((message) => message.role !== 'system');
    if (messages.length === 0) return undefined;

    const stamps = messages.map(readStamp).filter((stamp) => stamp !== undefined);
    const first = stamps[0];
    const last = stamps.at(-1);
    return {
      handle: record.handle,
      kind: 'compaction',
      label:
        first !== undefined && last !== undefined
          ? `turns ${stampLabel(first)}-${stampLabel(last)}, compacted`
          : 'compacted turns',
      messages,
      record
    };
  }

  private foldThread(fold: FoldDetails): Thread | undefined {
    const path = this.sessions.getBranch(fold.lastEntryId);
    const start = path.findIndex((entry) => entry.id === fold.firstEntryId);
    if (start < 0) return undefined;
    // The spans of folds applied before this one are already archived under their own
    // handles: the folds this one covers, whose notes this thread holds, and those that
    // ended at an `until` inside this span, whose notes lie after it.
    const note = this.sessions
      .getEntries()
      .find((entry) => foldEntryDetails(entry)?.handle === fold.handle);
    const covered = new Set<string>();
    for (const record of note ? foldRecords(this.sessions.getBranch(note.id)) : []) {
      const first = path.findIndex((entry) => entry.id === record.firstEntryId);
      if (first < 0) continue;
      const last = path.findIndex((entry) => entry.id === record.lastEntryId);
      for (const entry of path.slice(first, last < 0 ? undefined : last + 1)) {
        covered.add(entry.id);
      }
    }

    return {
      handle: fold.handle,
      kind: 'fold',
      label: fold.until ? `turns ${fold.since}-${fold.until}` : `turns from ${fold.since} on`,
      messages: path
        .slice(start)
        .filter((entry) => !covered.has(entry.id))
        .flatMap(sessionEntryToContextMessages),
      context: () => {
        // As the agent saw the span when it folded: the edits of earlier folds are written
        // after their notes, which can be after this span.
        const entries = this.sessions.getEntries();
        const seen = buildSessionProjection(entries, note?.id ?? fold.lastEntryId).entries;
        const end = seen.findIndex((entry) => entry.sourceEntry.id === fold.lastEntryId);
        const messages =
          end < 0
            ? buildSessionProjection(entries, fold.lastEntryId).messages
            : seen.slice(0, end + 1).flatMap((entry) => entry.messages);
        // A span folded through /fold ends with the request to fold; a question appended
        // after it would read as that request, and the thread would try to fold again.
        while (isFoldRequest(messages.at(-1))) messages.pop();
        return messages;
      },
      record: fold
    };
  }

  /**
   * The fork, then the folds it made of its own work. The fork's thread holds what the child
   * saw at the end, so a span folded inside it is matched under its own handle only.
   */
  private async forkThreads(fork: ForkDetails): Promise<Thread[]> {
    const archive = this.memory.get(fork.handle) ?? (await readArchive(fork.archive));
    if (!archive) return [];

    const folds = archive.folds ?? [];
    const prefix = () => [
      ...buildSessionProjection(this.sessions.getEntries(), fork.parentEntryId).messages,
      ...(archive.siblings ?? [])
    ];
    const visible = withoutFolded(archive.messages, folds);
    const threads: Thread[] = [
      {
        handle: fork.handle,
        kind: 'fork',
        label: `fork: ${fork.task.slice(0, 80)}`,
        messages: visible,
        context: () => [...prefix(), ...visible],
        record: fork
      }
    ];
    folds.forEach((fold, order) => {
      // What earlier folds already removed is archived under their handles.
      const earlier = new Set(folds.slice(0, order).flatMap((other) => other.omit));
      threads.push({
        handle: fold.record.handle,
        kind: 'fold',
        label: `${fold.record.until ? `turns ${fold.record.since}-${fold.record.until}` : `turns from ${fold.record.since} on`}, inside ${fork.handle}`,
        messages: archive.messages
          .slice(fold.first, fold.last + 1)
          .filter((_, index) => !earlier.has(fold.first + index)),
        // What the child saw when it folded: its earlier folds applied, this one not yet.
        context: () => [
          ...prefix(),
          ...withoutFolded(archive.messages.slice(0, fold.last + 1), folds.slice(0, order))
        ],
        record: fold.record
      });
    });
    return threads;
  }
}

function isFoldRequest(message: AgentMessage | undefined): boolean {
  return message?.role === 'custom' && message.customType === FOLD_REQUEST_TYPE;
}

async function readArchive(file: string | undefined): Promise<ForkArchive | undefined> {
  if (!file) return undefined;
  try {
    return JSON.parse(await readFile(file, 'utf8')) as ForkArchive;
  } catch {
    return undefined;
  }
}
