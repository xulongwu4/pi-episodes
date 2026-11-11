import { createHash } from 'node:crypto';
import { relative } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';
import type {
  ContextEditEntryDraft,
  ExtensionAPI,
  MessageRenderer,
  ProjectedSessionEntry,
  SessionEntry
} from '@earendil-works/pi-coding-agent';
import { Text } from '@earendil-works/pi-tui';
import { Type } from 'typebox';
import {
  appliedFolds,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_REQUEST_TYPE,
  FOLD_STARTED_TYPE,
  FOLD_TOOL,
  type FoldDetails,
  foldNoteDetails,
  ThreadIndex
} from './archive.js';
import { anchorGapTokens } from './cache.js';
import { type FoldRejection, recordEvent } from './events.js';
import { estimateTokens, readOnlyPaths, skillPaths, textOf, touchedPaths } from './messages.js';
import type { DiskChange } from './snapshots.js';
import { parseStamp, readStamp, stampLabel, unstampedText } from './stamps.js';
import type { EpisodeState } from './state.js';
import { episodeStats, statsLines } from './stats.js';
import { foldRenderers, formatTokens, refreshStatus } from './ui.js';

/** User text above this size is kept as a head plus a pointer into the archive. */
export const USER_TEXT_CAP = 4000;
const MAX_LISTED_CHANGES = 40;

export interface FoldPlan {
  handle: string;
  /**
   * Stamp of the last message before the span, if any, whether or not it is still in context:
   * the working tree at that stamp is what the span started from.
   */
  anchorStamp: number | undefined;
  /** Stamps inside the span before its first response, in order. */
  laterStamps: number[];
  firstEntryId: string;
  lastEntryId: string;
  /** Stamp of the first turn that stays after the span, when the span stops before the call. */
  keptFrom: number | undefined;
  /** Messages leaving the context. */
  omitted: AgentMessage[];
  /** Everything the span held, for file tracking. */
  spanMessages: AgentMessage[];
  /** Messages before the span, and those the request that called fold was sent with. */
  before: AgentMessage[];
  sent: AgentMessage[];
  keptUserTurns: string[];
  /** Earlier fold notes inside the span, which the new note stands for from now on. */
  covers: { handle: string; headline: string }[];
  edits: ContextEditEntryDraft[];
}

function entryMessage(entry: ProjectedSessionEntry): AgentMessage | undefined {
  return entry.sourceEntry.type === 'message' ? entry.sourceEntry.message : undefined;
}

function callsTool(entry: ProjectedSessionEntry, toolCallId: string): boolean {
  const message = entryMessage(entry);
  return (
    message?.role === 'assistant' &&
    message.content.some((block) => block.type === 'toolCall' && block.id === toolCallId)
  );
}

function isFoldNote(entry: ProjectedSessionEntry, applied: ReadonlySet<string>): boolean {
  if (entry.messages.length === 0) return false;
  const details = foldNoteDetails(entryMessage(entry));
  // A note whose edits never landed left its span in context; it is not a boundary.
  return details !== undefined && applied.has(details.handle);
}

const CAPPED =
  /\n\[\d+ more characters of this message are in ep-[0-9a-f]+; recall can search them\]$/;

function capUserContent(
  content: string | (TextContent | ImageContent)[],
  handle: string
): (TextContent | ImageContent)[] | undefined {
  const blocks: (TextContent | ImageContent)[] =
    typeof content === 'string' ? [{ type: 'text', text: content }] : content;
  const total = blocks.reduce(
    (sum, block) => sum + (block.type === 'text' ? block.text.length : 0),
    0
  );
  // A message an earlier fold capped is not capped again.
  if (
    total <= USER_TEXT_CAP ||
    blocks.some((block) => block.type === 'text' && CAPPED.test(block.text))
  ) {
    return undefined;
  }

  let budget = USER_TEXT_CAP;
  return blocks.map((block, index) => {
    // The trailing stamp block must survive untouched so the turn stays addressable.
    if (block.type !== 'text' || index === blocks.length - 1) return block;
    if (block.text.length <= budget) {
      budget -= block.text.length;
      return block;
    }
    const head = block.text.slice(0, budget);
    const cut = block.text.length - budget;
    budget = 0;
    return {
      type: 'text',
      text: `${head}\n[${cut} more characters of this message are in ${handle}; recall can search them]`
    };
  });
}

export interface FoldableTurn {
  stamp: number;
  label: string;
}

const PREVIEW_WIDTH = 72;

/**
 * Index of the latest compaction, or with `notes` of the latest applied fold note too. No fold
 * reaches back past a compaction; one that reaches back past a note merges it, which the nudge
 * asks for only when notes take up room.
 */
export function foldBoundary(entries: readonly ProjectedSessionEntry[], notes = true): number {
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  return entries.findLastIndex(
    (entry) => entry.sourceEntry.type === 'compaction' || (notes && isFoldNote(entry, applied))
  );
}

/**
 * Index of the entry a span starting at this message starts at. A span starting at a tool
 * result starts at the response that made the call, which can be behind the boundary: a call
 * made next to a fold call has its result after the note.
 */
function spanStart(
  entries: readonly ProjectedSessionEntry[],
  index: number,
  message: AgentMessage
): number {
  if (message.role !== 'toolResult') return index;
  const caller = entries.findLastIndex(
    (entry, at) => at < index && callsTool(entry, message.toolCallId)
  );
  return caller < 0 ? index : caller;
}

/**
 * Where the turn after the entry at `index` starts, and its stamp: the end of a span whose
 * last turn is that entry. A tool result made next to that entry's own, by the same response,
 * is not a later turn, so no call is left without its results. Undefined when no turn follows
 * before `limit`.
 */
function turnAfter(
  entries: readonly ProjectedSessionEntry[],
  index: number,
  limit: number
): { stamp: number; index: number } | undefined {
  for (let at = index + 1; at < limit; at++) {
    for (const message of entries[at]!.messages) {
      const stamp = readStamp(message);
      if (stamp === undefined || (message.role !== 'user' && message.role !== 'toolResult')) {
        continue;
      }
      const start = spanStart(entries, at, message);
      if (start > index) return { stamp, index: start };
    }
  }
  return undefined;
}

/**
 * The earliest stamp a fold can start from after the latest note, or with `notes` false after
 * the latest compaction, and the index of the entry its span starts at.
 */
export function foldStart(
  entries: readonly ProjectedSessionEntry[],
  notes = true
): { stamp: number; index: number } | undefined {
  return stampedStarts(entries, foldBoundary(entries, notes)).next().value;
}

/** The stamps a fold can start from after the entry at `boundary`, with where each span starts. */
function* stampedStarts(
  entries: readonly ProjectedSessionEntry[],
  boundary: number
): Generator<{ stamp: number; index: number }, undefined> {
  for (let index = boundary + 1; index < entries.length; index++) {
    for (const message of entries[index]!.messages) {
      const stamp = readStamp(message);
      if (stamp === undefined || (message.role !== 'user' && message.role !== 'toolResult')) {
        continue;
      }
      const start = spanStart(entries, index, message);
      if (start > boundary) yield { stamp, index: start };
    }
  }
  return undefined;
}

/**
 * Work that sits behind the latest note and that no fold removed: what a fold from a later
 * stamp left there, or what a fold with `until` kept. Its entries, without the notes and the
 * responses that called fold, and the stamp a fold of all of it starts from. A user message
 * counts only when a fold would cut it: a fold keeps the others whole. A message an extension
 * put in, such as a reminder, goes with the work it stands in and starts none: a fold from
 * the stamp before it would cover notes for the sake of a few lines. What stands between a
 * note and the next stamp, such as the reply that followed a fold, is left out: only a fold
 * that covers that note reaches it. `keptAt` is the index of the latest note of a fold that
 * stopped at `until`, where the agent chose to keep turns, or -1: a later fold that covered
 * that note did not undo the choice.
 */
export function unfoldedBehindNote(
  entries: readonly ProjectedSessionEntry[]
): { stamp: number; index: number; work: ProjectedSessionEntry[]; keptAt: number } | undefined {
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  const floor = foldBoundary(entries, false);
  const boundary = foldBoundary(entries);
  const starts = [...stampedStarts(entries, floor)];
  const calls = new Set<string>();
  for (const entry of entries.slice(floor + 1, boundary + 1)) {
    const message = entryMessage(entry);
    if (message?.role === 'toolResult' && isFoldNote(entry, applied)) calls.add(message.toolCallId);
  }
  const work: ProjectedSessionEntry[] = [];
  let start: { stamp: number; index: number } | undefined;
  // The first stamp a fold can start from without covering the latest note so far.
  let next = 0;
  for (let index = floor + 1; index < boundary; index++) {
    const entry = entries[index]!;
    if (isFoldNote(entry, applied)) {
      while (next < starts.length && starts[next]!.index <= index) next++;
      continue;
    }
    if (entry.messages.length === 0 || next >= starts.length || starts[next]!.index > index) {
      continue;
    }
    if (entry.sourceEntry.type === 'custom_message') {
      if (start) work.push(entry);
      continue;
    }
    const message = entryMessage(entry);
    const shown = entry.messages[0];
    const counts =
      message?.role === 'toolResult' ||
      (message?.role === 'assistant' &&
        !message.content.some((block) => block.type === 'toolCall' && calls.has(block.id))) ||
      (shown?.role === 'user' && capUserContent(shown.content, '') !== undefined);
    if (!counts) continue;
    if (!start) {
      // The latest span start that holds this first entry: an earlier one adds nothing. Of the
      // results of one response, which all start there, the first names it.
      const at = starts.findLast((candidate) => candidate.index <= index)?.index;
      start = starts.find((candidate) => candidate.index === at);
    }
    work.push(entry);
  }
  if (!start) return undefined;
  const keptAt = entries.findLastIndex((entry, index) => {
    const details = index > floor ? foldNoteDetails(entryMessage(entry)) : undefined;
    return details?.keptFrom !== undefined && applied.has(details.handle);
  });
  return { ...start, work, keptAt };
}

/**
 * What became of the work done after the entry at `after`: whether a fold removed some of it,
 * and what of it is still in context. Work is told from folding itself: a note and the
 * response that called its fold are neither, so a fold that only merges notes removes no work
 * and adds none, however long its note. A tool result counts as folded only when an applied
 * fold recorded its removal in the plan it writes before its edits: another extension can
 * remove one too, also inside the span of a fold. A session from before folds wrote that
 * plan shows none.
 */
export function workSince(
  entries: readonly ProjectedSessionEntry[],
  after: number
): { folded: boolean; left: ProjectedSessionEntry[] } {
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  const removed = new Set<string>();
  const calls = new Set<string>();
  for (const entry of entries) {
    const source = entry.sourceEntry;
    if (source.type === 'custom' && source.customType === FOLD_STARTED_TYPE) {
      const { handle, edits } = (source.data ?? {}) as { handle?: unknown; edits?: unknown };
      if (typeof handle !== 'string' || !applied.has(handle) || !Array.isArray(edits)) continue;
      for (const edit of edits as ContextEditEntryDraft[]) {
        if (edit.replacement === null) removed.add(edit.targetId);
      }
    }
    const message = entryMessage(entry);
    if (message?.role === 'toolResult' && foldNoteDetails(message)) calls.add(message.toolCallId);
  }
  let folded = false;
  const left: ProjectedSessionEntry[] = [];
  for (let index = after + 1; index < entries.length; index++) {
    const entry = entries[index]!;
    const message = entryMessage(entry);
    if (message?.role === 'toolResult' && foldNoteDetails(message)) continue;
    if (
      message?.role === 'assistant' &&
      message.content.some((block) => block.type === 'toolCall' && calls.has(block.id))
    ) {
      continue;
    }
    if (entry.messages.length > 0) left.push(entry);
    else if (message?.role === 'toolResult') folded ||= removed.has(entry.sourceEntry.id);
  }
  return { folded, left };
}

/**
 * Stamped turns a fold can start from, newest first: those after the latest compaction
 * summary, with a one-line preview to recognize them by.
 */
export function foldableTurns(entries: readonly ProjectedSessionEntry[]): FoldableTurn[] {
  const boundary = foldBoundary(entries, false);
  const turns: FoldableTurn[] = [];
  for (let index = boundary + 1; index < entries.length; index++) {
    for (const message of entries[index]!.messages) {
      const stamp = readStamp(message);
      if (stamp === undefined || (message.role !== 'user' && message.role !== 'toolResult')) {
        continue;
      }
      if (spanStart(entries, index, message) <= boundary) continue;
      const who = message.role === 'user' ? 'you' : message.toolName;
      const line =
        unstampedText(message.content)
          .split('\n')
          .find((text) => text.trim())
          ?.trim() ?? '';
      const preview = line.length > PREVIEW_WIDTH ? `${line.slice(0, PREVIEW_WIDTH - 3)}...` : line;
      turns.push({ stamp, label: `${stampLabel(stamp)}  ${who}: ${preview}` });
    }
  }
  return turns.reverse();
}

/**
 * What a fold does to one entry of its span: a user message keeps its first USER_TEXT_CAP
 * characters, a response, tool result or extension message is removed. System messages,
 * compaction checkpoints and `!` commands are not editable and stay.
 */
function entryEdit(
  entry: ProjectedSessionEntry,
  handle: string
): ContextEditEntryDraft | undefined {
  const source = entry.sourceEntry;
  const message = entryMessage(entry);
  const projected = entry.messages[0];
  if (!projected) return undefined;
  if (message?.role === 'user' && projected.role === 'user') {
    const capped = capUserContent(projected.content, handle);
    return (
      capped && { type: 'context_edit', targetId: source.id, replacement: { content: capped } }
    );
  }
  if (
    source.type === 'custom_message' ||
    message?.role === 'assistant' ||
    message?.role === 'toolResult'
  ) {
    return { type: 'context_edit', targetId: source.id, replacement: null };
  }
  return undefined;
}

type PendingDraft = { type: string; targetId?: string; replacement?: unknown };

/** Whether the latest edit of an entry, on the branch or among queued drafts, removes it. */
function removed(
  id: string,
  branch: readonly SessionEntry[],
  pending: readonly PendingDraft[]
): boolean {
  let gone = false;
  for (const entry of [...branch, ...pending]) {
    if (entry.type === 'context_edit' && entry.targetId === id) gone = entry.replacement === null;
  }
  return gone;
}

/**
 * The rest of the edits of folds cut short partway. An applied fold writes a start entry holding
 * its planned edits, then the edits in that order, then the applied marker; Pi writes them one
 * at a time, so a crash can stop the batch midway. For a start entry without a marker, each
 * planned edit whose target does not already carry that replacement, as its latest edit, is
 * written again, in the planned order, then the marker. That holds for a removal whatever was
 * done to its target since: a span removed in part must not keep a result whose call is gone.
 * The cut of a user message only saves tokens, so an edit made to that message after the fold
 * started, by anyone, stands and the cut is not written over it. A fold without a start entry
 * never took effect. One whose note another extension has removed since is completed like any
 * other: its record then says what a fold removed whole and lost its note afterwards says,
 * and no part of the span stays in context for a later fold to archive a second time. No fold
 * of this extension removes that note, because fold refuses to run while a start entry has
 * no marker.
 * `pending` holds the drafts earlier handlers of this boundary queued, which land first.
 */
export function completeCutShortFolds(
  branch: readonly SessionEntry[],
  pending: readonly PendingDraft[] = []
): (ContextEditEntryDraft | { type: 'custom'; customType: string; data: { handle: string } })[] {
  const applied = appliedFolds(branch);
  return branch.flatMap((start, index) => {
    if (start.type !== 'custom' || start.customType !== FOLD_STARTED_TYPE) return [];
    const { handle, edits } = (start.data ?? {}) as { handle?: unknown; edits?: unknown };
    if (typeof handle !== 'string' || applied.has(handle) || !Array.isArray(edits)) return [];
    const latest = new Map<string, string>();
    for (const entry of [...branch.slice(index + 1), ...pending]) {
      if (entry.type === 'context_edit' && entry.targetId !== undefined) {
        latest.set(entry.targetId, JSON.stringify(entry.replacement));
      }
    }
    const marker = { type: 'custom' as const, customType: FOLD_APPLIED_TYPE, data: { handle } };
    const rest = (edits as ContextEditEntryDraft[]).filter((edit) => {
      const now = latest.get(edit.targetId);
      if (now === JSON.stringify(edit.replacement)) return false;
      return edit.replacement === null || now === undefined;
    });
    return [...rest, marker];
  });
}

/** The handle of a fold that started and has no applied marker; the next turn end completes it. */
export function cutShortFold(branch: readonly SessionEntry[]): string | undefined {
  const applied = appliedFolds(branch);
  for (const entry of branch) {
    if (entry.type !== 'custom' || entry.customType !== FOLD_STARTED_TYPE) continue;
    const handle = (entry.data as { handle?: unknown } | undefined)?.handle;
    if (typeof handle === 'string' && !applied.has(handle)) return handle;
  }
  return undefined;
}

export interface FoldOptions {
  /** The newest stamp written in the session, to tell a made-up stamp from one that is gone. */
  latestStamp?: number;
  /** No span may start at or before this index; a fork folds only its own work. */
  floorIndex?: number;
  /** Handles of folds that took effect, when they are not recorded as session entries. */
  applied?: ReadonlySet<string>;
  /**
   * The call in the transcript whose response made this fold: the codemode call when a script
   * called fold. Defaults to the fold call itself.
   */
  callerId?: string;
  /** The last turn inside the span; later turns stay. Without it the span runs up to the call. */
  until?: number;
}

/** A fold the tool refused, with why, so rejections can be counted by cause. */
export class FoldError extends Error {
  readonly reason: FoldRejection;

  constructor(reason: FoldRejection, message: string) {
    super(message);
    this.reason = reason;
  }
}

/**
 * Works out which entries a fold removes. `since` names the first stamped message inside
 * the episode; when that is a tool result, the span starts at the response that made the
 * call, so the episode is folded from its turn boundary and no call loses its result. The
 * span ends before the response that called fold: that response and the fold result stay,
 * so the context still ends on a tool result. With `until` the span ends where the turn after
 * that one starts, and the turns from there to the call stay too.
 */
export function planFold(
  entries: readonly ProjectedSessionEntry[],
  foldCallId: string,
  since: number,
  options: FoldOptions = {}
): FoldPlan {
  const { latestStamp, floorIndex = -1, callerId = foldCallId } = options;
  const foldIndex = entries.findIndex((entry) => callsTool(entry, callerId));
  if (foldIndex < 0) {
    throw new FoldError(
      'not-in-context',
      'The response that called fold is not in the current context.'
    );
  }

  const foldResponse = entryMessage(entries[foldIndex]!);
  const firstFold =
    foldResponse?.role === 'assistant'
      ? foldResponse.content.find((block) => block.type === 'toolCall' && block.name === FOLD_TOOL)
      : undefined;
  if (firstFold?.type === 'toolCall' && firstFold.id !== foldCallId) {
    throw new FoldError('one-per-response', 'Only one fold per response; this one was ignored.');
  }

  // Nothing before the compaction summary can be folded: it stands for everything it
  // replaced. An earlier note can: the new note covers it, and its span stays archived.
  const applied = options.applied ?? appliedFolds(entries.map((entry) => entry.sourceEntry));
  const compaction = entries.findLastIndex(
    (entry, index) => index < foldIndex && entry.sourceEntry.type === 'compaction'
  );
  const boundary = Math.max(compaction, floorIndex);
  let earliest: number | undefined;
  for (let index = boundary + 1; index < foldIndex && earliest === undefined; index++) {
    earliest = entries[index]!.messages.filter(
      (message) => spanStart(entries, index, message) > boundary
    )
      .map(readStamp)
      .find((stamp) => stamp !== undefined);
  }
  const pick =
    earliest === undefined
      ? 'Nothing before this call can be folded yet.'
      : `The earliest stamp you can fold from is ${stampLabel(earliest)}.`;

  if (latestStamp !== undefined && since > latestStamp) {
    throw new FoldError(
      'no-such-stamp',
      `${stampLabel(since)} does not exist; the latest stamp is ${stampLabel(latestStamp)}. ${pick}`
    );
  }
  const sinceIndex = entries.findIndex((entry) =>
    entry.messages.some((message) => readStamp(message) === since)
  );
  if (sinceIndex < 0) {
    // A compaction summary can still mention it, but the turn itself is gone.
    throw new FoldError(
      'gone',
      `${stampLabel(since)} is no longer in your context: it was compacted or already folded. ${pick}`
    );
  }
  if (sinceIndex >= foldIndex) {
    throw new FoldError('not-before-call', `${stampLabel(since)} is not before this fold call.`);
  }

  let startIndex = sinceIndex;
  const sinceMessage = entryMessage(entries[sinceIndex]!);
  if (sinceMessage?.role === 'toolResult') {
    const caller = entries.findLastIndex(
      (entry, index) => index < sinceIndex && callsTool(entry, sinceMessage.toolCallId)
    );
    if (caller >= 0) startIndex = caller;
  }

  if (startIndex <= boundary) {
    const [reason, what]: [FoldRejection, string] =
      boundary === compaction
        ? ['behind-compaction', 'the compaction summary']
        : ['behind-fork-start', 'the start of this fork; a fork folds only its own work'];
    throw new FoldError(reason, `A fold cannot reach behind ${what}. ${pick}`);
  }

  let endIndex = foldIndex;
  let keptFrom: number | undefined;
  if (options.until !== undefined) {
    const until = options.until;
    if (until < since) {
      throw new FoldError(
        'bad-stamp',
        `until ${stampLabel(until)} is before since ${stampLabel(since)}.`
      );
    }
    const untilIndex = entries.findIndex((entry) =>
      entry.messages.some((message) => readStamp(message) === until)
    );
    if (untilIndex < 0) {
      throw latestStamp !== undefined && until > latestStamp
        ? new FoldError(
            'no-such-stamp',
            `${stampLabel(until)} does not exist; the latest stamp is ${stampLabel(latestStamp)}.`
          )
        : new FoldError(
            'gone',
            `${stampLabel(until)} is no longer in your context: it was compacted or already folded.`
          );
    }
    if (untilIndex >= foldIndex) {
      throw new FoldError('not-before-call', `${stampLabel(until)} is not before this fold call.`);
    }
    const next = turnAfter(entries, untilIndex, foldIndex);
    if (next) {
      endIndex = next.index;
      keptFrom = next.stamp;
    }
  }

  const span = entries.slice(startIndex, endIndex);
  const range =
    options.until === undefined
      ? `from ${stampLabel(since)}`
      : `from ${stampLabel(since)} to ${stampLabel(options.until)}`;
  if (
    !span.some((entry) => entry.messages.length > 0 && entryMessage(entry)?.role === 'assistant')
  ) {
    // The usual slip: since names the message being answered, to fold the request before it.
    const note = entries.findLastIndex(
      (entry, index) => index < startIndex && isFoldNote(entry, applied)
    );
    const before =
      options.until === undefined
        ? stampedStarts(entries, Math.max(boundary, note)).next().value
        : undefined;
    const pending =
      before !== undefined &&
      entries
        .slice(before.index, startIndex)
        .some((entry) => entry.messages.length > 0 && entryMessage(entry)?.role === 'assistant');
    throw new FoldError(
      'nothing-to-fold',
      `Nothing to fold ${range}: the span holds no responses.${pending ? ` To fold the work before it, use since "${stampLabel(before.stamp)}": that is where the work not yet folded starts.` : ''}`
    );
  }

  const first = span[0]!;
  const last = span.at(-1)!;
  // Derived from the call, not the span, so folding the same span again on another branch
  // gets its own handle.
  const handle = `ep-${createHash('sha256').update(foldCallId).digest('hex').slice(0, 8)}`;
  const edits: ContextEditEntryDraft[] = [];
  const omitted: AgentMessage[] = [];
  const keptUserTurns: string[] = [];
  const laterStamps: number[] = [];
  const covers: { handle: string; headline: string }[] = [];
  let reachedResponse = false;

  for (const entry of span) {
    if (entry.messages.length === 0) continue;
    const message = entryMessage(entry);
    const projected = entry.messages[0]!;
    if (message?.role === 'assistant') reachedResponse = true;
    if (!reachedResponse) {
      const stamp = readStamp(projected);
      if (stamp !== undefined) laterStamps.push(stamp);
    }

    const covered = isFoldNote(entry, applied) ? foldNoteDetails(message) : undefined;
    if (covered) covers.push({ handle: covered.handle, headline: covered.headline });
    const edit = entryEdit(entry, handle);
    if (edit) edits.push(edit);
    if (message?.role === 'user' && projected.role === 'user') {
      const stamp = readStamp(projected);
      if (stamp !== undefined) keptUserTurns.push(stampLabel(stamp));
    } else if (edit) {
      omitted.push(...entry.messages);
    }
  }

  // Pi writes these edits one entry at a time, so a crash can apply only some of them. Removing
  // the span from its end keeps every applied prefix a valid context: a call that lost its
  // result gets a placeholder from pi-ai, but a result whose call is gone reaches the provider.
  edits.reverse();

  let anchorStamp: number | undefined;
  for (let index = startIndex - 1; index >= 0 && anchorStamp === undefined; index--) {
    // An earlier fold that ended at an until can have removed the turn before this span.
    const before = entryMessage(entries[index]!);
    anchorStamp = before && readStamp(before);
  }

  return {
    handle,
    anchorStamp,
    laterStamps,
    firstEntryId: first.sourceEntry.id,
    lastEntryId: last.sourceEntry.id,
    keptFrom,
    omitted,
    spanMessages: span.flatMap((entry) => entry.messages),
    before: entries.slice(0, startIndex).flatMap((entry) => entry.messages),
    sent: entries.slice(0, foldIndex).flatMap((entry) => entry.messages),
    keptUserTurns,
    covers,
    edits
  };
}

const SECTION = /^\s*[-*]?\s*\**\s*(changed|learned|decided|rejected|handled|open)\b[^:]*:\**\s*/i;
const NOTHING_OPEN = /^(none|nothing|n\/a|-)\.?$/i;

/**
 * The note's Open section, as the template asks for it: from the line that starts with
 * "Open:" to the next section or blank line. Undefined when there is none or it says none.
 */
export function openItems(note: string): string | undefined {
  const lines = note.split('\n');
  const start = lines.findIndex((line) => SECTION.exec(line)?.[1]?.toLowerCase() === 'open');
  if (start < 0) return undefined;

  const items = [lines[start]!.replace(SECTION, '').trim()];
  for (const line of lines.slice(start + 1)) {
    if (!line.trim() || SECTION.test(line)) break;
    items.push(line.trim());
  }
  const text = items.filter(Boolean).join(' ').trim();
  return text && !NOTHING_OPEN.test(text) ? text : undefined;
}

export function headline(note: string): string {
  const line = note.split('\n').find((text) => text.trim()) ?? '';
  const clean = line.replace(/^[\s#*-]+/, '').trim();
  return clean.length > 120 ? `${clean.slice(0, 117)}...` : clean;
}

/** What changed on disk, from the working tree diff, or the touched files when there is none. */
export function changeLines(
  changes: readonly DiskChange[] | undefined,
  touched: readonly string[],
  cwd: string
): string[] {
  if (!changes) {
    return touched.length === 0
      ? []
      : [
          'Files used by file tools (no working tree diff is available, so shell side effects are not listed):',
          ...touched.map((path) => relative(cwd, path))
        ];
  }
  if (changes.length === 0) return ['Working tree unchanged.'];

  const lines = ['Changed on disk (working tree diff):'];
  for (const change of changes.slice(0, MAX_LISTED_CHANGES)) {
    lines.push(`${change.status} ${relative(cwd, change.path)}`);
  }
  if (changes.length > MAX_LISTED_CHANGES) {
    lines.push(`... and ${changes.length - MAX_LISTED_CHANGES} more`);
  }
  return lines;
}

export function renderNote(options: {
  handle: string;
  since: string;
  until: string | undefined;
  messages: number;
  tokens: number;
  note: string;
  changes: readonly DiskChange[] | undefined;
  touched: readonly string[];
  keptUserTurns: readonly string[];
  /** First turn kept after the span, when the fold stopped before the call. */
  keptFrom?: string | undefined;
  covers?: readonly { handle: string; headline: string }[];
  /** Skill files read inside the span. */
  skills?: readonly string[];
  cwd: string;
}): string {
  const lines = [
    `[fold ${options.handle}: ${options.messages} messages, ${options.until ? `${options.since}-${options.until}` : `from ${options.since} on`}, ~${formatTokens(options.tokens)} tokens, archived]`,
    '',
    options.note.trim(),
    '',
    ...changeLines(options.changes, options.touched, options.cwd)
  ];

  if (options.keptUserTurns.length > 0) {
    lines.push(
      '',
      `User messages from inside the span stay above with their stamps: ${options.keptUserTurns.join(', ')}.`
    );
  }

  if (options.keptFrom) {
    lines.push('', `The turns from ${options.keptFrom} on were not folded and stay above.`);
  }
  if (options.covers?.length) {
    lines.push(
      '',
      `This fold covers the notes of earlier folds; recall and ask still reach their spans by handle: ${options.covers.map((fold) => `${fold.handle} (${fold.headline})`).join('; ')}.`
    );
  }

  // A note says what was learned, not how to work; a skill's instructions would be lost unseen.
  if (options.skills?.length) {
    lines.push(
      '',
      `Skills read inside the span, whose instructions left your context with it: ${options.skills.join(', ')}. Read a skill again before relying on it.`
    );
  }
  lines.push(
    '',
    `To go back: recall("<exact words>") searches this span; ask("${options.handle}", "<question>") asks it.`
  );
  return lines.join('\n');
}

const FoldParams = Type.Object({
  since: Type.String({
    description:
      'Stamp of the first message inside the episode, such as "t13", usually the result of its first tool call. The fold starts at the response that made that call and runs up to this call, or to the end of the turn `until` names.'
  }),
  until: Type.Optional(
    Type.String({
      description:
        'Stamp of the last turn inside the span, such as "t20", when the turns after it must stay verbatim. Leave it out to fold up to this call. What stays after the span is sent uncached once.'
    })
  ),
  note: Type.String({
    description:
      'Replaces the span. First line: the question and how it ended. Then changed, learned (with conditions), decided and rejected (with why), handled user messages by stamp, open (with the next step). Quote exact paths, symbols, commands and error strings.'
  })
});

/** One dim line for a message that asks the agent to fold, from /fold or a nudge. */
export const renderFoldAsk: MessageRenderer = (message, _options, theme) => {
  const text = typeof message.content === 'string' ? message.content : '';
  const since = /since "(t\d+)"/.exec(text)?.[1];
  return new Text(
    theme.fg('dim', `episodes: asked the agent to fold${since ? ` from ${since}` : ''}`),
    0,
    0
  );
};
/** How to ask for a fold call. Under codemode "only" fold is not declared; scripts reach it. */
/** Whether tools such as fold are hidden from the model and called from codemode scripts. */
export function codemodeOnly(pi: ExtensionAPI): boolean {
  return pi.getSettings().codemode?.mode === 'only' && pi.getActiveTools().includes('codemode');
}

export function foldInstruction(pi: ExtensionAPI): string {
  return codemodeOnly(pi)
    ? 'make one codemode call whose script calls only fold and prints its result'
    : 'call only fold in this response';
}
export function registerFold(pi: ExtensionAPI, state: EpisodeState): void {
  pi.registerTool({
    name: FOLD_TOOL,
    label: 'Fold',
    description:
      'Fold a finished episode: remove the span starting at `since` from your context, archive it, and keep `note` in its place. User messages from the span stay. With `until` the span ends with that turn and later turns stay. Call it as the only tool in the response, once the next step no longer depends on the span beyond the note, and not right before your final reply. From a codemode script, call only fold in that script and print its result.',
    promptSnippet: 'Fold a finished episode into a note; the span is archived and stays reachable',
    parameters: FoldParams,
    executionMode: 'sequential',
    renderCall: foldRenderers.renderCall,
    renderResult: foldRenderers.renderResult,

    async execute(toolCallId, params, signal, _onUpdate, ctx) {
      // A /fold request is used up by this call, whether it succeeds or not.
      const manual = state.manualFold;
      state.manualFold = undefined;
      let plan: FoldPlan;
      let since: number | undefined;
      const note = params.note.trim();
      // A call from a codemode script is not in the transcript; the response that called the
      // script made it.
      const root = state.rootCall(toolCallId);
      const scriptCall = root === toolCallId ? undefined : root;
      const run = state.run;
      const entries = ctx.sessionManager.buildSessionProjection().entries;
      try {
        // The first fold of a response holds it, whether or not it succeeds, as for folds the
        // response calls itself. Reserved before anything is awaited, so a second fold from the
        // same response, in the same script or another one, sees it.
        const caller = entries.find((entry) => callsTool(entry, scriptCall ?? toolCallId));
        if (caller) {
          const reserved = state.foldResponses.get(caller.sourceEntry.id);
          if (reserved !== undefined && reserved !== toolCallId) {
            throw new FoldError(
              'one-per-response',
              'Only one fold per response; this one was ignored.'
            );
          }
          state.foldResponses.set(caller.sourceEntry.id, toolCallId);
        }
        // Part of that fold's span is still in context and its note does not count as one
        // yet; a fold planned over this state would archive the span twice.
        const unfinished = cutShortFold(ctx.sessionManager.getBranch());
        if (unfinished !== undefined) {
          throw new FoldError(
            'earlier-fold-unfinished',
            unfinished +
              ' was interrupted partway and is completed when this response ends. Fold again in your next response; the turns of its span are gone by then.'
          );
        }
        since = parseStamp(params.since);
        if (since === undefined) {
          throw new FoldError(
            'bad-stamp',
            `since must be a turn stamp such as "t12", got "${params.since}".`
          );
        }
        const until = params.until?.trim() ? parseStamp(params.until) : undefined;
        if (params.until?.trim() && until === undefined) {
          throw new FoldError(
            'bad-stamp',
            `until must be a turn stamp such as "t20", got "${params.until}".`
          );
        }
        if (!note) {
          throw new FoldError(
            'empty-note',
            'The note is empty. Write what the rest of the task needs from the span.'
          );
        }
        plan = planFold(entries, toolCallId, since, {
          latestStamp: state.nextStamp - 1,
          ...(until === undefined ? {} : { until }),
          callerId: scriptCall ?? toolCallId
        });
      } catch (error) {
        if (error instanceof FoldError) {
          recordEvent(pi, { kind: 'fold-rejected', toolCallId, reason: error.reason });
        }
        throw error;
      }
      const sinceLabel = stampLabel(since);
      const lastStamp = plan.spanMessages.map(readStamp).findLast((stamp) => stamp !== undefined);
      const untilLabel =
        lastStamp !== undefined && lastStamp !== since ? stampLabel(lastStamp) : undefined;
      const keptFrom = plan.keptFrom === undefined ? undefined : stampLabel(plan.keptFrom);
      const touched = touchedPaths(plan.spanMessages, state.cwd);
      const files = new Set(touched);

      let changes: DiskChange[] | undefined;
      let treeAfter: string | undefined;
      const snapshots = state.snapshots;
      const anchorPlaced = state.anchorPlaced;
      if (snapshots) {
        // A span that stops before the call ends with the tree at its last stamp: what the
        // turns kept after it changed is not the episode's.
        treeAfter =
          plan.keptFrom === undefined || lastStamp === undefined
            ? await snapshots.snapshot()
            : await snapshots.treeAt(lastStamp);
        // The span starts from the tree at its last stamp before the first response, such as
        // the user's prompt; the anchor before the span can be from an earlier run, and files
        // the user edited between runs are not the episode's changes.
        let from: string | undefined;
        for (const stamp of [...plan.laterStamps].reverse().concat(plan.anchorStamp ?? [])) {
          from ??= await snapshots.treeAt(stamp);
        }
        if (from && treeAfter) {
          changes = await snapshots.diff(from, treeAfter);
          for (const change of changes ?? []) files.add(change.path);
        }
      }

      const tokens = estimateTokens(plan.omitted);

      const details: FoldDetails = {
        kind: 'fold',
        handle: plan.handle,
        since: sinceLabel,
        ...(untilLabel ? { until: untilLabel } : {}),
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        messages: plan.omitted.length,
        tokens,
        headline: headline(note),
        keptUserTurns: plan.keptUserTurns,
        files: [...files]
      };
      if (plan.covers.length > 0) details.covers = plan.covers.map((fold) => fold.handle);
      if (keptFrom) details.keptFrom = keptFrom;
      if (treeAfter) details.treeAfter = treeAfter;
      const open = openItems(note);
      if (open) details.open = open;
      const readOnly = readOnlyPaths(
        plan.spanMessages,
        state.cwd,
        (changes ?? []).map((change) => change.path)
      );
      if (readOnly.length > 0) details.readOnly = readOnly;
      details.noteTokens = Math.ceil(note.length / 4);
      if (manual) details.manual = manual;
      const gapTokens = anchorGapTokens(
        plan.sent,
        plan.before,
        appliedFolds(ctx.sessionManager.getBranch())
      );
      details.anchor = {
        ...(anchorPlaced === undefined ? {} : { placed: anchorPlaced }),
        ...(gapTokens === undefined ? {} : { gapTokens })
      };

      const text = renderNote({
        handle: plan.handle,
        since: sinceLabel,
        until: untilLabel,
        messages: plan.omitted.length,
        tokens,
        note,
        changes,
        touched,
        keptUserTurns: plan.keptUserTurns,
        keptFrom,
        covers: plan.covers,
        skills: skillPaths(plan.spanMessages, state.cwd),
        cwd: state.cwd
      });
      // Snapshots take time; a run aborted meanwhile, or already settled, must not get edits.
      if (signal?.aborted || run !== state.run) throw new Error('The fold was cancelled.');
      // A fold from a script takes effect with the script's result, which then stands as the
      // note; the edits wait under the script's call, the one the turn reports.
      state.pendingFolds.set(scriptCall ?? toolCallId, {
        handle: plan.handle,
        messages: plan.omitted.length,
        tokens,
        edits: plan.edits,
        ...(scriptCall ? { script: { foldCallId: toolCallId, note: text, details } } : {})
      });

      // A /fold request ends the run here; an autonomous fold continues the task. Codemode
      // does not pass this on from a script, so a /fold done there does not end the run.
      const terminate = manual !== undefined;

      return { content: [{ type: 'text', text }], details, terminate };
    }
  });

  // Calls a codemode script makes are not in the transcript; remember which call made them so
  // a fold from a script finds the response that called the script.
  pi.on('tool_call', (event) => {
    if (event.parentToolCallId) state.parentCalls.set(event.toolCallId, event.parentToolCallId);
  });

  // A result hook can still fail the fold after it ran, and the script can catch that; or
  // change its text, which is then what the script printed.
  pi.on('tool_execution_end', (event) => {
    if (!event.parentToolCallId) return;
    const root = state.rootCall(event.toolCallId);
    const pending = state.pendingFolds.get(root);
    if (pending?.script?.foldCallId !== event.toolCallId) return;
    const content = (event.result as { content?: unknown } | undefined)?.content;
    const note = Array.isArray(content) ? textOf(content).trim() : '';
    // A fold whose text a hook emptied would leave the span with nothing in its place.
    if (event.isError || !note) {
      state.pendingFolds.delete(root);
      return;
    }
    pending.script.note = note;
  });
  // A fold from a script takes effect only if the script succeeds. Its result then carries the
  // fold's details, and the note too when the script did not print it, written once with the
  // result rather than edited in later.
  pi.on('tool_result', (event) => {
    if (event.parentToolCallId) return undefined;
    const script = state.pendingFolds.get(event.toolCallId)?.script;
    if (!script) return undefined;
    if (event.isError) {
      state.pendingFolds.delete(event.toolCallId);
      return undefined;
    }
    const own =
      typeof event.details === 'object' && event.details !== null && !Array.isArray(event.details)
        ? event.details
        : {};
    const details = { ...own, [FOLD_DETAILS_KEY]: script.details };
    if (textOf(event.content).includes(script.note)) return { details };
    const content = [...event.content, { type: 'text' as const, text: script.note }];
    return event.structuredContent === undefined
      ? { content, details }
      : { content, details, structuredContent: event.structuredContent };
  });

  // Context edits can only be appended at a turn boundary; by then the fold result exists,
  // so the call and its note stay paired and the next request already sees the fold.
  pi.on('turn_end', (event, ctx) => {
    state.foldResponses.clear();
    const branch = ctx.sessionManager.getBranch();
    const applied = event.toolResults
      .filter((result) => !result.isError)
      .flatMap((result) => {
        const pending = state.pendingFolds.get(result.toolCallId);
        // A later hook may have rewritten the script's result; without the note in it, the
        // span would leave with nothing in its place.
        const stands = pending?.script
          ? foldNoteDetails(result)?.handle === pending.handle &&
            textOf(result.content).includes(pending.script.note)
          : result.toolName === FOLD_TOOL;
        if (!pending || !stands) return [];
        state.pendingFolds.delete(result.toolCallId);
        // An earlier handler that removes the note leaves nothing to stand for the span.
        const note = branch.find(
          (entry) =>
            entry.type === 'message' &&
            entry.message.role === 'toolResult' &&
            entry.message.toolCallId === result.toolCallId
        );
        if (note && removed(note.id, branch, event.entries)) return [];
        return [pending];
      });
    const repairs = completeCutShortFolds(branch, event.entries);
    if (applied.length === 0 && repairs.length === 0) return undefined;
    // The cut of a user message only saves tokens, so an edit an earlier handler queued for
    // that message stands, as it does when a cut-short fold is completed. The note was written
    // before this boundary: if that edit removes the message, the note still lists its turn
    // among the ones that stay.
    const edited = new Set(
      (event.entries as readonly PendingDraft[]).flatMap((draft) =>
        draft.type === 'context_edit' && draft.targetId !== undefined ? [draft.targetId] : []
      )
    );
    const editsOf = (fold: { edits: readonly ContextEditEntryDraft[] }) =>
      fold.edits.filter((edit) => edit.replacement === null || !edited.has(edit.targetId));

    for (const fold of applied) {
      ctx.ui.notify(
        `Folded ${fold.messages} messages (~${formatTokens(fold.tokens)} tokens) into ${fold.handle}`,
        'info'
      );
    }
    return {
      entries: [
        ...event.entries,
        ...repairs,
        // The marker goes last. A fold cut short leaves part of its span removed, a valid
        // context, and no applied note; a marker written first would leave the rest of the span
        // stuck behind a note no fold can reach past.
        ...applied.flatMap((fold) => [
          {
            type: 'custom' as const,
            customType: FOLD_STARTED_TYPE,
            data: { handle: fold.handle, edits: editsOf(fold) }
          },
          ...editsOf(fold),
          { type: 'custom' as const, customType: FOLD_APPLIED_TYPE, data: { handle: fold.handle } }
        ])
      ]
    };
  });

  // The status counts records on the branch, so it refreshes once the edits are committed.
  pi.on('turn_end', (_event, ctx) => {
    refreshStatus(ctx, state);
  });

  // The edits from the last turn_end are committed after its handlers run; a fold that ended
  // the run would otherwise not show until the next response.
  pi.on('agent_settled', (_event, ctx) => {
    state.pendingFolds.clear();
    state.foldResponses.clear();
    state.parentCalls.clear();
    state.run++;
    state.manualFold = undefined;
    refreshStatus(ctx, state);
  });

  pi.registerCommand('fold', {
    description: 'Fold now: pick the turn the span starts at, and the agent writes the note',
    handler: async (args, ctx) => {
      let since = args.trim() ? parseStamp(args) : undefined;
      if (args.trim() && since === undefined) {
        ctx.ui.notify('Usage: /fold [tN]', 'error');
        return;
      }

      await ctx.waitForIdle();
      // Picking the start by hand is the post's baseline: the scope is the user's, so there is
      // no false fold. Without a dialog to pick in, the agent picks.
      if (since === undefined && ctx.hasUI) {
        const turns = foldableTurns(ctx.sessionManager.buildSessionProjection().entries);
        if (turns.length === 0) {
          ctx.ui.notify('Nothing can be folded yet: no turns since the last compaction.', 'info');
          return;
        }
        const choice = await ctx.ui.select(
          'Fold from which turn? The span runs from there to now; the agent writes the note.',
          turns.map((turn) => turn.label)
        );
        if (choice === undefined) return;
        since = turns.find((turn) => turn.label === choice)?.stamp;
        if (since === undefined) return;
      }
      state.manualFold = since === undefined ? 'request' : 'since';
      const how = foldInstruction(pi);
      const target =
        since === undefined ? 'Pick the span that is done' : `Use since "${stampLabel(since)}"`;
      pi.sendMessage(
        {
          customType: FOLD_REQUEST_TYPE,
          display: true,
          content: `The user asked you to fold now. ${target}, write the note, and ${how}.`
        },
        { triggerTurn: true }
      );
    }
  });

  pi.registerMessageRenderer(FOLD_REQUEST_TYPE, renderFoldAsk);

  pi.registerCommand('episodes', {
    description:
      'List the folded spans and forks archived on this branch; /episodes stats shows the numbers',
    getArgumentCompletions: (prefix) =>
      'stats'.startsWith(prefix.trim())
        ? [{ value: 'stats', label: 'stats', description: 'Measurements for this branch' }]
        : null,
    handler: async (args, ctx) => {
      if (args.trim() === 'stats') {
        const lines = statsLines(episodeStats(ctx.sessionManager.getBranch(), state.cwd));
        ctx.ui.notify(lines.join('\n'), 'info');
        return;
      }
      if (args.trim()) {
        ctx.ui.notify('Usage: /episodes [stats]', 'error');
        return;
      }
      const threads = await new ThreadIndex(ctx.sessionManager, state.forkArchives).onBranch();
      if (threads.length === 0) {
        ctx.ui.notify('Nothing archived on this branch yet.', 'info');
        return;
      }
      const lines = threads.map((thread) => {
        const size =
          thread.record.kind === 'fold'
            ? `${thread.record.messages} messages, ~${formatTokens(thread.record.tokens)} tokens`
            : `${thread.messages.length} messages`;
        return `${thread.handle}  ${thread.label} (${size})  ${thread.record.headline}`;
      });
      ctx.ui.notify(lines.join('\n'), 'info');
    }
  });
}
