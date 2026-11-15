import { createHash } from 'node:crypto';
import { relative } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { AssistantMessage, ImageContent, TextContent } from '@earendil-works/pi-ai';
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
  archiveDir,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_RECOVERY_STARTED_TYPE,
  FOLD_REQUEST_TYPE,
  FOLD_STARTED_TYPE,
  FOLD_TOOL,
  FOLD_UNFOLD_STARTED_TYPE,
  FOLD_UNFOLDED_TYPE,
  type FoldDetails,
  type FoldSpanDetails,
  foldNoteDetails,
  foldSpans,
  formatFoldRange,
  ThreadIndex,
  UNFOLD_TOOL
} from './archive.js';
import { anchorGapTokens } from './cache.js';
import { type FoldRejection, recordEvent } from './events.js';
import {
  type ArchivedImage,
  archivedImagesOf,
  type CarriedMediaItem,
  carriedMediaBlocks,
  FOLD_ABORTED_MESSAGE,
  holdsArchivedManifest,
  holdsCarriedMedia,
  matchArchivedRefBlocks,
  prepareCarriedMedia,
  withoutTrailingStamp
} from './images.js';
import {
  estimateTokens,
  headline,
  readOnlyPaths,
  skillPaths,
  textOf,
  touchedPaths
} from './messages.js';
import { completeReadBlocks } from './output.js';
import { planFoldTailPruning, type ReadPruneDependency } from './prune.js';

export * from './prune.js';

import type { DiskChange } from './snapshots.js';
import {
  isStampBlock,
  parseStamp,
  readStamp,
  stampLabel,
  unstampedText,
  withStamp
} from './stamps.js';
import type { EpisodeState } from './state.js';
import { episodeStats, statsLines } from './stats.js';
import {
  foldRenderers,
  formatTokens,
  refreshStatus,
  type UnfoldDetails,
  unfoldRenderers
} from './ui.js';

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
  /** Stamps of `read` turns inside the span still cited by live duplicate-read stubs before it. */
  archivedReadTurns: string[];
  /** The notes of earlier folds that the new note stands for from now on, oldest first. */
  covers: CoveredNote[];
  /** Covered-note media items (already-archived references and inline images) in block order. */
  media: CarriedMediaItem[];
  edits: ContextEditEntryDraft[];
  /** Ids of the entries where the edits leave the fold's line, one for each span that has one. */
  lines: string[];
  /** Ids of the lines of covered folds that the edits remove, inside the span or before it. */
  linesRemoved: string[];
}

/** The note of an earlier fold, as a later fold that covers it keeps it. */
export interface CoveredNote {
  handle: string;
  /** The stamps its span ran over, such as "t13-t20". */
  range: string;
  /** What the note said, as it stood in context when it was covered. */
  text: string;
}

const KEPT_TITLE = 'Notes of the earlier folds covered here, oldest first, as they were written:';
/**
 * The line over a kept note. It says how long the note is, so that the note is read back by
 * length and nothing it says, a quoted title or a line like this one, can pass for the start
 * of the next.
 */
const KEPT_HEAD = /\n\n\[note of (ep-[0-9a-f]{8}), ([^\]\n]*), (\d+) characters\]\n/y;

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

// Over the code units of the string: as UTF-8 a lone surrogate and U+FFFD would hash alike.
const sha256 = (text: string) =>
  createHash('sha256').update(Buffer.from(text, 'utf16le')).digest('hex');

/**
 * What a fold records of its result as it left it: how many blocks, their hash, and the
 * length and hash of the kept notes at the end of the last one. A later fold that covers the
 * note tells by it whether the result, and the kept notes in it, are still what the fold
 * wrote. `changed` leaves the hash of the blocks out, for a result whose note a hook had
 * already changed: it never was what the agent wrote.
 */
export function writtenMark(
  content: readonly (TextContent | ImageContent)[],
  carried?: string,
  changed = false
): NonNullable<FoldDetails['written']> {
  const blocks = content
    .map((block) =>
      block.type === 'text'
        ? `text ${block.text.length}\n${block.text}\n`
        : `image ${block.mimeType} ${block.data.length}\n${block.data}\n`
    )
    .join('');
  return {
    blocks: content.length,
    ...(changed ? {} : { sha256: sha256(blocks) }),
    ...(carried ? { kept: { chars: carried.length, sha256: sha256(carried) } } : {})
  };
}

/** The notes in text that `renderCovered` wrote: each under its line, as long as the line says. */
function keptUnder(text: string): CoveredNote[] | undefined {
  if (!text.startsWith(KEPT_TITLE)) return undefined;
  const notes: CoveredNote[] = [];
  let index = KEPT_TITLE.length;
  while (index < text.length) {
    KEPT_HEAD.lastIndex = index;
    const head = KEPT_HEAD.exec(text);
    if (!head) return undefined;
    const start = index + head[0].length;
    index = start + Number(head[3]);
    if (index > text.length) return undefined;
    notes.push({ handle: head[1]!, range: head[2]!, text: text.slice(start, index) });
  }
  return notes;
}

function intactWrittenPrefixLength(
  blocks: readonly (TextContent | ImageContent)[],
  written: FoldDetails['written']
): number | undefined {
  const count = written?.blocks;
  if (
    typeof count !== 'number' ||
    count < 0 ||
    blocks.length < count ||
    typeof written?.sha256 !== 'string'
  ) {
    return undefined;
  }
  return writtenMark(blocks.slice(0, count)).sha256 === written.sha256 ? count : undefined;
}

/**
 * A note's result as it stands in context, read by what the fold recorded of it in
 * `written`. The stamp is left out.
 *
 * `kept`: the notes kept under this one, taken from the end of the block the fold wrote them
 * in, only while the text there has the length and hash of what the fold wrote. `own`: all
 * the other text of the result. `added`: set while the blocks the fold left are, to the
 * character, what it left; it is then the text of the blocks added after them, a cue say, and
 * everything else in the result is known to be the fold's own writing, with the agent's note
 * in it as recorded.
 *
 * So nothing is taken for kept notes by its form, wherever it stands, and once anything was
 * changed, moved or cut, the changed text is in `own`, whole.
 */
function standing(
  content: readonly (TextContent | ImageContent)[],
  written: FoldDetails['written'],
  archived: readonly ArchivedImage[] = []
): { own: string; kept: CoveredNote[]; added?: string } {
  const rawBlocks = withoutTrailingStamp(content);
  const refMap = matchArchivedRefBlocks(
    rawBlocks,
    archived,
    intactWrittenPrefixLength(rawBlocks, written)
  );
  const blocks = refMap.size === 0 ? rawBlocks : rawBlocks.filter((_, index) => !refMap.has(index));
  const count = typeof written?.blocks === 'number' ? written.blocks : 0;
  const holder = blocks[count - 1];
  const at =
    holder?.type === 'text' && typeof written?.kept?.chars === 'number'
      ? holder.text.length - written.kept.chars
      : -1;
  const tail =
    holder?.type === 'text' && at >= 0 && (at === 0 || holder.text.slice(0, at).endsWith('\n\n'))
      ? holder.text.slice(at)
      : undefined;
  const kept =
    (tail !== undefined && sha256(tail) === written?.kept?.sha256 ? keptUnder(tail) : undefined) ??
    [];
  const texts = (from: number, to?: number) =>
    blocks
      .slice(from, to)
      .flatMap((block) =>
        block.type !== 'text'
          ? []
          : [block === holder && kept.length > 0 ? block.text.slice(0, at) : block.text]
      )
      .join('\n')
      .trim();
  const asWritten =
    count > 0 &&
    blocks.length >= count &&
    typeof written?.sha256 === 'string' &&
    writtenMark(blocks.slice(0, count)).sha256 === written.sha256 &&
    (written.kept === undefined) === (kept.length === 0);
  return { own: texts(0), kept, ...(asWritten ? { added: texts(count) } : {}) };
}

/**
 * The notes a fold of `span` covers, in the order they were written: the applied notes inside
 * it and the notes each of them kept. The fold keeps them under its own note, so covering a
 * note loses nothing it said. Left to write one note in place of many, one model wrote 400
 * to 800 tokens whatever the note covered: over three such merges in a session of 302 turns
 * the facts of its first note went, and it answered 2 of 12 questions from context where it
 * answered 12 with every note standing.
 *
 * What is kept is read from the note's result as it stands in context: a handler or another
 * extension may have changed it since, and the notes under it are there and nowhere else in
 * context. While the result is what the fold wrote (`standing`), the note's own text is the
 * one the agent wrote, without the lines the harness put around it or what a script printed
 * next to it, and with whatever was added to the result afterwards. Otherwise all of the
 * result is kept under the note's handle, which is also what a fold made before this was
 * recorded gets. So nothing that stands in a covered note's result is dropped unless the
 * fold itself put it there.
 */
export function coveredNotes(
  entries: readonly ProjectedSessionEntry[],
  span: readonly ProjectedSessionEntry[],
  applied: ReadonlySet<string> = appliedFolds(entries.map((entry) => entry.sourceEntry))
): CoveredNote[] {
  if (!span.some((entry) => isFoldNote(entry, applied))) return [];
  // Where a note stands on the branch is when it was written: a fold with `until` can cover a
  // note older than one that precedes it in the span.
  const written = new Map<string, number>();
  entries.forEach((entry, index) => {
    const fold = foldNoteDetails(entryMessage(entry));
    if (fold) written.set(fold.handle, index);
  });
  const covered = new Map<string, { note: CoveredNote; at: number }>();
  for (const entry of span) {
    const shown = entry.messages[0];
    const fold = isFoldNote(entry, applied) ? foldNoteDetails(entryMessage(entry)) : undefined;
    if (!fold || shown?.role !== 'toolResult') continue;
    const archived = archivedImagesOf(fold.archivedImages);
    const read = standing(shown.content, fold.written, archived);
    // A record that names more folds than its result has notes for is not to be trusted.
    const { own, kept, added } =
      read.kept.length === (fold.covers?.length ?? 0)
        ? read
        : standing(shown.content, undefined, archived);
    const at = written.get(fold.handle) ?? 0;
    const recorded =
      added === undefined || fold.note === undefined
        ? own
        : [fold.note.trim(), added].filter((part) => part).join('\n\n');
    for (const note of kept) {
      if (!covered.has(note.handle))
        covered.set(note.handle, { note, at: written.get(note.handle) ?? at });
    }
    covered.set(fold.handle, {
      note: {
        handle: fold.handle,
        range: formatFoldRange(fold, { onSuffix: false }),
        text: recorded
      },
      at
    });
  }
  return [...covered.values()].sort((a, b) => a.at - b.at).map(({ note }) => note);
}

/**
 * Covered-note media items in block order: inline images and already-archived image references
 * standing on the results of the notes in `span`.
 */
export function coveredMedia(
  span: readonly ProjectedSessionEntry[],
  applied: ReadonlySet<string>
): CarriedMediaItem[] {
  return span.flatMap((entry) => {
    const shown = entry.messages[0];
    if (shown?.role !== 'toolResult' || !isFoldNote(entry, applied)) return [];
    const fold = foldNoteDetails(entryMessage(entry));
    const blocks = withoutTrailingStamp(shown.content);
    const refMap = matchArchivedRefBlocks(
      blocks,
      archivedImagesOf(fold?.archivedImages),
      intactWrittenPrefixLength(blocks, fold?.written)
    );
    const items: CarriedMediaItem[] = [];
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i]!;
      if (block.type === 'image') {
        items.push({ kind: 'image', image: block });
        continue;
      }
      const ref = refMap.get(i);
      if (ref) items.push({ kind: 'ref', ref });
    }
    return items;
  });
}

/** Estimated tokens of the notes a fold keeps as written: what covering them does not free. */
export function coveredTokens(covers: readonly CoveredNote[]): number {
  return covers.reduce((sum, fold) => sum + Math.ceil(fold.text.length / 4), 0);
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
 * What a fold leaves in place of the first response of its span, in the agent's voice.
 * With the span removed whole, nothing in context showed that the agent had done the work
 * its note describes, and one model then told the user that it had never run the commands
 * and that its own note was made up. The line says why nothing is shown, because the fold
 * call can come right after it, and a line that only claimed the work read as the opening
 * of that same response, with no call behind it. MEASUREMENTS.md, section 15, has the
 * wordings and the other forms tried. A span without a tool call gets a line that claims
 * none.
 */
export function foldMarker(handle: string, turns: string, calls = true): string {
  return calls
    ? `[${handle}: earlier responses of mine stood here, with the tool calls I made (${turns}) and the outputs they returned. I read those outputs, and then I folded all of it out of this conversation: that is why no call and no output is shown here. The note in the result of the fold call below is my own record of what the outputs showed, written with them in view; recall with this handle reads them back.]`
    : `[${handle}: earlier responses of mine stood here (${turns}). I folded them out of this conversation: that is why they are not shown here. The note in the result of the fold call below is my own record of them, written with them in view; recall with this handle reads them back.]`;
}

/**
 * Where the lines of the folds in effect stand: the id of the entry to the handle of the fold.
 * Read from the plan a fold writes before its edits, not from what the entry shows: another
 * extension can edit a line afterwards, and a response can quote one. Where a later fold put
 * its line in place of an earlier one, the entry is the later fold's while that is in effect.
 */
export function foldLines(
  entries: readonly ProjectedSessionEntry[],
  applied: ReadonlySet<string>
): Map<string, string> {
  const lines = new Map<string, string>();
  for (const { sourceEntry: source } of entries) {
    if (source.type !== 'custom' || source.customType !== FOLD_STARTED_TYPE) continue;
    const { handle, lines: ids } = (source.data ?? {}) as { handle?: unknown; lines?: unknown };
    if (typeof handle !== 'string' || !applied.has(handle) || !Array.isArray(ids)) continue;
    for (const id of ids) if (typeof id === 'string') lines.set(id, handle);
  }
  return lines;
}

/**
 * Index of the latest compaction, or with `notes` of the latest applied fold note too. No fold
 * reaches back past a compaction; one that reaches back to a note covers it and keeps it under
 * its own, which the nudge asks for only when what stands around the notes takes up room.
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
 * Where a fold that covers the notes in context starts: at the earliest thing behind the
 * latest note that a fold removes or cuts, which is the earliest note unless work was left
 * before it. A fold from an earlier stamp removes no more, since user messages stay; and one
 * model, asked to fold from the first stamp of a long session, would not, where it did fold
 * from a stamp further on. The marker of a fold is not a thing to start at: the one before
 * the earliest note goes with the fold that covers that note, and one further back marks
 * turns the agent kept. Undefined when no note stands after the latest compaction.
 */
export function mergeStart(
  entries: readonly ProjectedSessionEntry[]
): { stamp: number; index: number } | undefined {
  const floor = foldBoundary(entries, false);
  const boundary = foldBoundary(entries);
  const starts = [...stampedStarts(entries, floor)];
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  const lines = foldLines(entries, applied);
  for (let index = floor + 1; index <= boundary; index++) {
    const shown = entries[index]!.messages[0];
    const folds =
      lines.get(entries[index]!.sourceEntry.id) === undefined &&
      (shown?.role === 'assistant' ||
        shown?.role === 'toolResult' ||
        (shown?.role === 'user' && capUserContent(shown.content, '') !== undefined));
    if (!folds) continue;
    // The latest span start that holds this entry; of the results of one response, the first.
    // No start holds what stands before the first stamp, such as the end of a turn that a
    // compaction kept without its beginning.
    const at = starts.findLast((candidate) => candidate.index <= index)?.index;
    if (at !== undefined) return starts.find((candidate) => candidate.index === at);
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
 * that covers that note reaches it. The marker of a fold is no work. `keptAt` is the index of the latest note of a fold that
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
  const lines = foldLines(entries, applied);
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
    if (lines.get(entry.sourceEntry.id) !== undefined) continue;
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
 * and adds none, however long its note. A tool result or a response counts as folded only
 * when an applied fold recorded its removal in the plan it writes before its edits, or left
 * its marker in place of it: another extension can remove one too, also inside the span of a
 * fold. A session from before folds wrote that plan shows none.
 */
export function workSince(
  entries: readonly ProjectedSessionEntry[],
  after: number
): { folded: boolean; left: ProjectedSessionEntry[] } {
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  const lines = foldLines(entries, applied);
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
    if (message?.role === 'toolResult') {
      const fold = foldNoteDetails(message);
      if (fold && applied.has(fold.handle)) calls.add(message.toolCallId);
    }
  }
  let folded = false;
  const left: ProjectedSessionEntry[] = [];
  for (let index = after + 1; index < entries.length; index++) {
    const entry = entries[index]!;
    const message = entryMessage(entry);
    const fold = message?.role === 'toolResult' ? foldNoteDetails(message) : undefined;
    if (fold && applied.has(fold.handle)) continue;
    if (
      message?.role === 'assistant' &&
      message.content.some((block) => block.type === 'toolCall' && calls.has(block.id))
    ) {
      continue;
    }
    if (lines.get(entry.sourceEntry.id) !== undefined) folded = true;
    else if (entry.messages.length > 0) left.push(entry);
    else if (message?.role === 'toolResult' || message?.role === 'assistant') {
      folded ||= removed.has(entry.sourceEntry.id);
    }
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
 * characters, a response, tool result or extension message is removed; `planFold` then
 * puts the marker in place of the first response. System messages, compaction checkpoints
 * and `!` commands are not editable and stay.
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

/**
 * Whether a script's output holds the note: as text, or inside JSON, which is how a script
 * prints a value it does not print as it is. Adding the note next to that copy would put it
 * in context twice, for as long as the note stands.
 */
function holdsNote(content: readonly (TextContent | ImageContent)[], note: string): boolean {
  const text = textOf(content);
  return text.includes(note) || text.includes(JSON.stringify(note).slice(1, -1));
}

/** How much text a script's result holds next to the note of its fold, in characters. */
export function besideNote(content: readonly (TextContent | ImageContent)[], note: string): number {
  const text = textOf(content);
  const printed = JSON.stringify(note).slice(1, -1);
  const held = text.includes(note) ? note.length : text.includes(printed) ? printed.length : 0;
  return text.length - held;
}

type PendingDraft = { type: string; targetId?: string; replacement?: unknown };

/**
 * What the latest edit of an entry, on the branch or among queued drafts, leaves of it in
 * context: null when it removes the entry, undefined when no edit touched it.
 */
function editedContent(
  id: string,
  branch: readonly SessionEntry[],
  pending: readonly PendingDraft[]
): readonly (TextContent | ImageContent)[] | null | undefined {
  let content: readonly (TextContent | ImageContent)[] | null | undefined;
  for (const entry of [...branch, ...pending]) {
    if (entry.type !== 'context_edit' || entry.targetId !== id) continue;
    const replacement = entry.replacement as { content?: unknown } | null | undefined;
    // Pi takes a string for content and makes one text block of it.
    content = !replacement
      ? null
      : typeof replacement.content === 'string'
        ? [{ type: 'text', text: replacement.content }]
        : Array.isArray(replacement.content)
          ? (replacement.content as (TextContent | ImageContent)[])
          : [];
  }
  return content;
}

function unfinishedBatches(branch: readonly SessionEntry[]): {
  index: number;
  kind: 'fold' | 'unfold';
  handle: string;
  edits: ContextEditEntryDraft[];
  lines: string[];
  readDeps: ReadPruneDependency[];
}[] {
  const open = new Map<
    string,
    {
      index: number;
      kind: 'fold' | 'unfold';
      handle: string;
      edits: ContextEditEntryDraft[];
      lines: string[];
      readDeps: ReadPruneDependency[];
    }
  >();
  branch.forEach((entry, index) => {
    if (entry.type !== 'custom') return;
    const data = (entry.data ?? {}) as {
      handle?: unknown;
      edits?: unknown;
      lines?: unknown;
      readDeps?: unknown;
    };
    if (typeof data.handle !== 'string') return;
    if (entry.customType === FOLD_STARTED_TYPE && Array.isArray(data.edits)) {
      open.set(`fold:${data.handle}`, {
        index,
        kind: 'fold',
        handle: data.handle,
        edits: data.edits as ContextEditEntryDraft[],
        lines: Array.isArray(data.lines) ? (data.lines as string[]) : [],
        readDeps: Array.isArray(data.readDeps) ? (data.readDeps as ReadPruneDependency[]) : []
      });
    } else if (entry.customType === FOLD_APPLIED_TYPE) {
      open.delete(`fold:${data.handle}`);
    } else if (entry.customType === FOLD_UNFOLD_STARTED_TYPE && Array.isArray(data.edits)) {
      open.set(`unfold:${data.handle}`, {
        index,
        kind: 'unfold',
        handle: data.handle,
        edits: data.edits as ContextEditEntryDraft[],
        lines: [],
        readDeps: []
      });
    } else if (entry.customType === FOLD_UNFOLDED_TYPE) {
      open.delete(`unfold:${data.handle}`);
    }
  });
  return [...open.values()].sort((a, b) => a.index - b.index);
}

/**
 * The rest of the edits of folds or unfolds cut short partway. Each batch writes a start entry
 * holding its planned edits, then the edits in that order, then the completion marker; Pi writes
 * them one at a time, so a crash can stop the batch midway. `pending` holds the drafts earlier
 * handlers of this boundary queued, which land first.
 */
export function completeCutShortFolds(
  branch: readonly SessionEntry[],
  pending: readonly PendingDraft[] = [],
  laterRemovals: ReadonlySet<string> = new Set()
): (
  | ContextEditEntryDraft
  | {
      type: 'custom';
      customType: string;
      data:
        | { handle: string }
        | { kind: 'fold' | 'unfold'; handle: string; edits: ContextEditEntryDraft[] };
    }
)[] {
  const branchById = new Map(branch.map((entry) => [entry.id, entry]));
  const effectiveReplacement = (targetId: string): unknown => {
    const source = branchById.get(targetId);
    const base = source ? defaultEntryContent(source) : undefined;
    let state: unknown = base === undefined ? undefined : { content: base };
    for (const entry of [...branch, ...pending]) {
      if (entry.type === 'context_edit' && entry.targetId === targetId) {
        state = entry.replacement;
      }
    }
    return state;
  };
  return unfinishedBatches(branch).flatMap(({ index, kind, handle, edits, lines, readDeps }) => {
    const depByTarget = new Map(readDeps.map((dep) => [dep.targetId, dep]));
    const latestRaw = new Map<string, unknown>();
    for (const entry of [...branch.slice(index + 1), ...pending]) {
      if (entry.type === 'context_edit' && entry.targetId !== undefined) {
        latestRaw.set(entry.targetId, entry.replacement);
      }
    }
    const marker = {
      type: 'custom' as const,
      customType: kind === 'fold' ? FOLD_APPLIED_TYPE : FOLD_UNFOLDED_TYPE,
      data: { handle }
    };
    const rest = edits.flatMap((edit): ContextEditEntryDraft[] => {
      const dep = kind === 'fold' ? depByTarget.get(edit.targetId) : undefined;
      if (dep) {
        const currentNewer = effectiveReplacement(dep.newerEntryId);
        const newerSource = branchById.get(dep.newerEntryId);
        const newerBlocks =
          currentNewer && typeof currentNewer === 'object'
            ? (currentNewer as { content?: unknown }).content
            : undefined;
        const newerUnstamped = Array.isArray(newerBlocks)
          ? newerBlocks.at(-1) && isStampBlock(newerBlocks.at(-1) as TextContent | ImageContent)
            ? (newerBlocks.slice(0, -1) as (TextContent | ImageContent)[])
            : (newerBlocks as (TextContent | ImageContent)[])
          : undefined;
        const newerIntact =
          !laterRemovals.has(dep.newerEntryId) &&
          !latestRaw.has(dep.newerEntryId) &&
          currentNewer !== null &&
          currentNewer !== undefined &&
          JSON.stringify(currentNewer) === JSON.stringify(dep.newerExpected) &&
          newerSource?.type === 'message' &&
          newerSource.message.role === 'toolResult' &&
          newerUnstamped !== undefined &&
          completeReadBlocks(newerUnstamped, newerSource.message.details) !== undefined;
        if (!latestRaw.has(edit.targetId)) {
          return newerIntact ? [edit] : [];
        }
        const written = latestRaw.get(edit.targetId);
        if (!newerIntact && JSON.stringify(written) === JSON.stringify(edit.replacement)) {
          return [
            {
              type: 'context_edit',
              targetId: edit.targetId,
              replacement: dep.restoreReplacement
            }
          ];
        }
        return [];
      }
      if (!latestRaw.has(edit.targetId)) return [edit];
      const newer = latestRaw.get(edit.targetId);
      if (JSON.stringify(newer) === JSON.stringify(edit.replacement)) return [];
      const source = branchById.get(edit.targetId);
      // A fold's line is written like a removal: what it replaces left with the span.
      if (kind === 'fold') {
        return edit.replacement === null || lines.includes(edit.targetId) ? [edit] : [];
      }
      const noteRecord = source?.type === 'message' ? foldNoteDetails(source.message) : undefined;
      if (noteRecord?.handle === handle) {
        const pendingNote = editedContent(edit.targetId, branch, pending);
        if (pendingNote !== null && pendingNote !== undefined && source?.type === 'message') {
          return [
            {
              type: 'context_edit',
              targetId: edit.targetId,
              replacement: { content: unfoldedNoteContent(source, pendingNote, handle, noteRecord) }
            }
          ];
        }
        return [];
      }
      return [];
    });
    if (rest.length === 0) return [marker];
    return [
      {
        type: 'custom' as const,
        customType: FOLD_RECOVERY_STARTED_TYPE,
        data: { kind, handle, edits: rest }
      },
      ...rest,
      marker
    ];
  });
}

/** The handle of a fold or unfold that started and has no completion marker; the next turn end completes it. */
export function cutShortFold(branch: readonly SessionEntry[]): string | undefined {
  return unfinishedBatches(branch)[0]?.handle;
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
  /** Where the lines of folds in effect stand, when no session entries record them (see foldLines). */
  lines?: ReadonlyMap<string, string>;
  /** Handles of the folds whose notes the whole call covers, when it folds more than this span. */
  covered?: ReadonlySet<string>;
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
      ? foldResponse.content.find(
          (block) =>
            block.type === 'toolCall' && (block.name === FOLD_TOOL || block.name === UNFOLD_TOOL)
        )
      : undefined;
  if (firstFold?.type === 'toolCall' && firstFold.id !== foldCallId) {
    throw new FoldError('one-per-response', 'Only one fold per response; this one was ignored.');
  }

  // Nothing before the compaction summary can be folded: it stands for everything it
  // replaced. An earlier note can: the new note covers it, and its span stays archived.
  const applied = options.applied ?? appliedFolds(entries.map((entry) => entry.sourceEntry));
  const lines = options.lines ?? foldLines(entries, applied);
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
    // The marker of another fold stands for turns after these, which that fold removed: the
    // span stops before it, as it would before those turns.
    const marker = entries.findIndex(
      (entry, index) =>
        index > untilIndex &&
        index < (next?.index ?? foldIndex) &&
        lines.get(entry.sourceEntry.id) !== undefined
    );
    if (marker >= 0) endIndex = marker;
    else if (next) endIndex = next.index;
    keptFrom = next?.stamp;
  }

  const span = entries.slice(startIndex, endIndex);
  const range =
    options.until === undefined
      ? `from ${stampLabel(since)}`
      : `from ${stampLabel(since)} to ${stampLabel(options.until)}`;
  // The line of an earlier fold stands for responses that fold removed: it is none itself.
  const responds = (entry: ProjectedSessionEntry) =>
    entry.messages.length > 0 &&
    entryMessage(entry)?.role === 'assistant' &&
    lines.get(entry.sourceEntry.id) === undefined;
  if (!span.some(responds)) {
    // The usual slip: since names the message being answered, to fold the request before it.
    const note = entries.findLastIndex(
      (entry, index) => index < startIndex && isFoldNote(entry, applied)
    );
    const before =
      options.until === undefined
        ? stampedStarts(entries, Math.max(boundary, note)).next().value
        : undefined;
    const pending = before !== undefined && entries.slice(before.index, startIndex).some(responds);
    throw new FoldError(
      'nothing-to-fold',
      `Nothing to fold ${range}: the span holds no responses.${pending ? ` To fold the work before it, use since "${stampLabel(before.stamp)}": that is where the work not yet folded starts.` : ''}`
    );
  }

  const first = span.find((entry) => entry.messages.length > 0)!;
  const last = span.findLast((entry) => entry.messages.length > 0)!;
  // Derived from the call, not the span, so folding the same span again on another branch
  // gets its own handle.
  const handle = `ep-${createHash('sha256').update(foldCallId).digest('hex').slice(0, 8)}`;
  const edits: ContextEditEntryDraft[] = [];
  const omitted: AgentMessage[] = [];
  const keptUserTurns: string[] = [];
  const laterStamps: number[] = [];
  let reachedResponse = false;
  const lined: string[] = [];
  const linesRemoved: string[] = [];
  const covers = coveredNotes(entries, span, applied);
  const covered = options.covered ?? new Set(covers.map((note) => note.handle));
  // Only a span that holds a tool call gets a line that speaks of calls and outputs.
  const calls = span.some((entry) =>
    entry.messages.some(
      (shown) =>
        shown.role === 'assistant' && shown.content.some((block) => block.type === 'toolCall')
    )
  );
  // The turns the marker names: from since to the last stamp in the span, as the note does.
  const lastStamp = span
    .flatMap((entry) => entry.messages.map(readStamp))
    .findLast((stamp) => stamp !== undefined);
  const turns =
    lastStamp === undefined || lastStamp === since
      ? stampLabel(since)
      : `${stampLabel(since)}-${stampLabel(lastStamp)}`;

  for (const entry of span) {
    if (entry.messages.length === 0) continue;
    const owner = lines.get(entry.sourceEntry.id);
    if (owner !== undefined) {
      // The line of an earlier fold goes with a fold that covers its note. Under a span that
      // ends before that note it stays: the fold it stands for is still in effect.
      if (covered.has(owner)) {
        edits.push({ type: 'context_edit', targetId: entry.sourceEntry.id, replacement: null });
        linesRemoved.push(entry.sourceEntry.id);
      }
      continue;
    }
    const message = entryMessage(entry);
    const projected = entry.messages[0]!;
    let edit = entryEdit(entry, handle);
    // The line goes where the first response stood that is sent again: pi leaves a response
    // that ended in an error or was aborted out of every request, and the line with it.
    if (
      message?.role === 'assistant' &&
      message.stopReason !== 'error' &&
      message.stopReason !== 'aborted' &&
      lined.length === 0 &&
      edit
    ) {
      edit = {
        ...edit,
        replacement: { content: [{ type: 'text', text: foldMarker(handle, turns, calls) }] }
      };
      lined.push(edit.targetId);
    }
    if (message?.role === 'assistant') reachedResponse = true;
    if (!reachedResponse) {
      const stamp = readStamp(projected);
      if (stamp !== undefined) laterStamps.push(stamp);
    }

    if (edit) edits.push(edit);
    if (message?.role === 'user' && projected.role === 'user') {
      const stamp = readStamp(projected);
      if (stamp !== undefined) keptUserTurns.push(stampLabel(stamp));
    } else if (edit) {
      omitted.push(...entry.messages);
    }
  }

  // Pi writes these edits one entry at a time, so a crash can apply only some of them.
  // Removing the span from its end keeps every intermediate prefix valid.
  edits.reverse();

  // A fold that starts at the note of the fold before it stands behind that fold's line, with
  // at most the user messages that fold kept between them. Its own line says the same for
  // both from now on, so the older one goes: a chain of such folds would otherwise leave a
  // row of them. A line with a response between it and this span stays where it is.
  for (let index = startIndex - 1; index > boundary; index--) {
    const entry = entries[index]!;
    if (entry.messages.length === 0) continue;
    const owner = lines.get(entry.sourceEntry.id);
    if (owner !== undefined && covered.has(owner)) {
      edits.push({ type: 'context_edit', targetId: entry.sourceEntry.id, replacement: null });
      linesRemoved.push(entry.sourceEntry.id);
    } else if (entry.messages[0]!.role !== 'user') break;
  }

  let anchorStamp: number | undefined;
  for (let index = startIndex - 1; index >= 0 && anchorStamp === undefined; index--) {
    // An earlier fold that ended at an until can have removed the turn before this span.
    const before = entryMessage(entries[index]!);
    anchorStamp = before && readStamp(before);
  }

  const beforeMessages = entries.slice(0, startIndex).flatMap((entry) => entry.messages);
  const omittedReadStamps = new Set<string>();
  for (const msg of omitted) {
    if (msg.role === 'toolResult' && msg.toolName === 'read') {
      const s = readStamp(msg);
      if (s !== undefined) omittedReadStamps.add(stampLabel(s));
    }
  }
  const archivedReadTurns: string[] = [];
  const seenArchived = new Set<string>();
  for (const msg of beforeMessages) {
    if (msg.role !== 'toolResult' || msg.toolName !== 'read') continue;
    const hit = /^\[duplicate read of .+; identical to (t\d+)\b/.exec(textOf(msg.content));
    const ref = hit?.[1];
    if (ref && omittedReadStamps.has(ref) && !seenArchived.has(ref)) {
      seenArchived.add(ref);
      archivedReadTurns.push(ref);
    }
  }
  const media = coveredMedia(span, applied);
  return {
    handle,
    anchorStamp,
    laterStamps,
    firstEntryId: first.sourceEntry.id,
    lastEntryId: last.sourceEntry.id,
    keptFrom,
    omitted,
    spanMessages: span.flatMap((entry) => entry.messages),
    before: beforeMessages,
    sent: entries.slice(0, foldIndex).flatMap((entry) => entry.messages),
    keptUserTurns,
    archivedReadTurns,
    covers,
    media,
    edits,
    lines: lined,
    linesRemoved
  };
}

export interface NormalizedFoldSpan {
  since: number;
  until?: number;
  note: string;
}

export interface FoldBatchSpanPlan {
  since: number;
  until: number | undefined;
  keptFrom: number | undefined;
  note: string;
  startIndex: number;
  endIndex: number;
  firstEntryId: string;
  lastEntryId: string;
  anchorStamp: number | undefined;
  laterStamps: number[];
  lastStamp: number | undefined;
  spanMessages: AgentMessage[];
  omitted: AgentMessage[];
}

export interface FoldBatchPlan {
  handle: string;
  spans: FoldBatchSpanPlan[];
  firstEntryId: string;
  lastEntryId: string;
  /** First stamp retained unfolded after the earliest span (interior gap or trailing tail). */
  keptFrom: string | undefined;
  /** Human-readable notice naming retained interior gaps and/or trailing turns. */
  keptNotice: string | undefined;
  note: string;
  headline: string;
  open: string | undefined;
  omitted: AgentMessage[];
  spanMessages: AgentMessage[];
  tokens: number;
  before: AgentMessage[];
  sent: AgentMessage[];
  keptUserTurns: string[];
  archivedReadTurns: string[];
  covers: CoveredNote[];
  media: CarriedMediaItem[];
  edits: ContextEditEntryDraft[];
  /** Ids of the entries where the edits leave the fold's line, one for each span that has one. */
  lines: string[];
  /** Estimated tokens the fold's lines add, less those of the covered folds' lines it removes. */
  lineTokens: number;
}

function parseSingleFoldSpan(raw: Record<string, unknown>): NormalizedFoldSpan {
  const rawSince = typeof raw.since === 'string' ? raw.since : '';
  const since = parseStamp(rawSince);
  if (since === undefined) {
    throw new FoldError(
      'bad-stamp',
      `since must be a turn stamp such as "t12", got "${String(raw.since ?? '')}".`
    );
  }
  if (raw.until !== undefined && raw.until !== null && typeof raw.until !== 'string') {
    throw new FoldError(
      'bad-stamp',
      `until must be a turn stamp such as "t20", got "${String(raw.until)}".`
    );
  }
  const untilStr = typeof raw.until === 'string' ? raw.until.trim() : '';
  const until = untilStr ? parseStamp(untilStr) : undefined;
  if (untilStr && until === undefined) {
    throw new FoldError(
      'bad-stamp',
      `until must be a turn stamp such as "t20", got "${raw.until as string}".`
    );
  }
  const note = typeof raw.note === 'string' ? raw.note.trim() : '';
  if (!note) {
    throw new FoldError(
      'empty-note',
      'The note is empty. Write what the rest of the task needs from the span.'
    );
  }
  return {
    since,
    ...(until === undefined ? {} : { until }),
    note
  };
}

/** Validates single-span `{ since, until?, note }` or batch `{ spans: [...] }` arguments. */
export function normalizeFoldSpans(params: Record<string, unknown>): NormalizedFoldSpan[] {
  const hasSpans = params.spans !== undefined && params.spans !== null;
  const isPresent = (value: unknown): boolean =>
    value !== undefined && value !== null && (typeof value !== 'string' || value.trim() !== '');
  const hasSingle = isPresent(params.since) || isPresent(params.until) || isPresent(params.note);
  if (hasSpans && hasSingle) {
    throw new FoldError('bad-spans', 'Pass either spans or top-level since/until/note, not both.');
  }
  if (!hasSpans) {
    return [parseSingleFoldSpan(params)];
  }
  if (!Array.isArray(params.spans) || params.spans.length === 0) {
    throw new FoldError(
      'bad-spans',
      'spans must be a non-empty array of { since, until?, note } items.'
    );
  }
  return params.spans.map((item) => {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
      throw new FoldError(
        'bad-spans',
        'Each item in spans must be an object with { since, until?, note }.'
      );
    }
    return parseSingleFoldSpan(item as Record<string, unknown>);
  });
}

export function mergeSpanDiskChanges(
  spanChangesList: readonly (readonly DiskChange[] | undefined)[]
): DiskChange[] | undefined {
  if (spanChangesList.length === 0 || spanChangesList.some((list) => list === undefined)) {
    return undefined;
  }
  const byPath = new Map<string, string>();
  for (const list of spanChangesList) {
    for (const change of list!) {
      const prev = byPath.get(change.path);
      if (prev === undefined) {
        byPath.set(change.path, change.status);
      } else if (prev === 'A') {
        if (change.status === 'D') byPath.delete(change.path);
        else byPath.set(change.path, 'A');
      } else if (change.status === 'D') {
        byPath.set(change.path, 'D');
      } else if (prev === 'D' && change.status === 'A') {
        byPath.set(change.path, 'M');
      } else if (change.status === 'T') {
        byPath.set(change.path, 'T');
      } else {
        byPath.set(change.path, 'M');
      }
    }
  }
  return [...byPath.entries()].map(([path, status]) => ({ status, path }));
}

/**
 * Resolves one or more non-overlapping spans against the pre-call projection, sorts them
 * chronologically, rejects overlaps after turn-boundary expansion, and composes one crash-safe
 * edit sequence (span removals and cuts, newest first).
 */
export function planFoldBatch(
  entries: readonly ProjectedSessionEntry[],
  foldCallId: string,
  spans: readonly NormalizedFoldSpan[],
  options: Omit<FoldOptions, 'until'> = {}
): FoldBatchPlan {
  if (spans.length === 0) {
    throw new FoldError(
      'bad-spans',
      'spans must be a non-empty array of { since, until?, note } items.'
    );
  }
  const entryIndexById = new Map(entries.map((entry, idx) => [entry.sourceEntry.id, idx]));

  const planned = (item: NormalizedFoldSpan, covered?: ReadonlySet<string>) =>
    planFold(entries, foldCallId, item.since, {
      ...options,
      ...(item.until === undefined ? {} : { until: item.until }),
      ...(covered ? { covered } : {})
    });
  // The line of a covered fold can stand in or before another span than the one that holds
  // its note, so each span is planned knowing what all of them cover.
  const coveredByAll =
    spans.length > 1
      ? new Set(spans.flatMap((item) => planned(item).covers.map((note) => note.handle)))
      : undefined;
  const resolved = spans.map((item) => {
    const plan = planned(item, coveredByAll);
    const startIndex = entryIndexById.get(plan.firstEntryId) ?? -1;
    const lastIndex = entryIndexById.get(plan.lastEntryId) ?? -1;
    return {
      item,
      plan,
      startIndex,
      endIndex: lastIndex + 1
    };
  });

  resolved.sort((a, b) => a.startIndex - b.startIndex || a.endIndex - b.endIndex);

  for (let i = 0; i < resolved.length - 1; i++) {
    const spanItem = resolved[i]!.item;
    if (spanItem.until === undefined) {
      throw new FoldError(
        'bad-spans',
        `Span from ${stampLabel(spanItem.since)} must specify "until"; only the final span in a batch can omit "until".`
      );
    }
  }

  for (let i = 1; i < resolved.length; i++) {
    const prev = resolved[i - 1]!;
    const curr = resolved[i]!;
    if (prev.endIndex > curr.startIndex) {
      const desc = (s: NormalizedFoldSpan) =>
        s.until !== undefined
          ? `${stampLabel(s.since)}-${stampLabel(s.until)}`
          : `from ${stampLabel(s.since)} on`;
      throw new FoldError(
        'overlapping-spans',
        `Spans ${desc(prev.item)} and ${desc(curr.item)} overlap; batch spans must be non-overlapping.`
      );
    }
  }

  const selectedIndices = new Set<number>();
  for (const item of resolved) {
    for (let idx = item.startIndex; idx < item.endIndex; idx++) selectedIndices.add(idx);
  }

  const selectedSpanEdits: ContextEditEntryDraft[] = [];
  for (let i = resolved.length - 1; i >= 0; i--) {
    const spanIds = new Set(
      entries
        .slice(resolved[i]!.startIndex, resolved[i]!.endIndex)
        .map((entry) => entry.sourceEntry.id)
    );
    for (const edit of resolved[i]!.plan.edits) {
      // An older marker that a span removes with it stands before the span; one that another
      // span of this call holds is that span's to edit.
      const index = entryIndexById.get(edit.targetId) ?? -1;
      if (spanIds.has(edit.targetId) || !selectedIndices.has(index)) selectedSpanEdits.push(edit);
    }
  }

  // What the fold's lines add to the context: those it writes, less the lines of covered
  // folds it removes.
  const lines = resolved.flatMap((entry) => entry.plan.lines);
  const linesRemoved = new Set(resolved.flatMap((entry) => entry.plan.linesRemoved));
  let lineTokens = 0;
  for (const edit of selectedSpanEdits) {
    const index = entryIndexById.get(edit.targetId) ?? -1;
    const shown = entries[index]?.messages[0];
    if (!shown) continue;
    if (edit.replacement === null) {
      if (linesRemoved.has(edit.targetId)) lineTokens -= estimateTokens([shown]);
    } else if (lines.includes(edit.targetId)) {
      lineTokens += estimateTokens([
        { ...shown, content: edit.replacement.content } as AgentMessage
      ]);
    }
  }

  const interiorGaps: { firstStamp: string; lastStamp: string }[] = [];
  for (let i = 0; i < resolved.length - 1; i++) {
    const gapMessages = entries
      .slice(resolved[i]!.endIndex, resolved[i + 1]!.startIndex)
      .flatMap((entry) => entry.messages);
    const gapStamps = gapMessages
      .map(readStamp)
      .filter((stamp): stamp is number => stamp !== undefined);
    if (gapStamps.length > 0) {
      interiorGaps.push({
        firstStamp: stampLabel(gapStamps[0]!),
        lastStamp: stampLabel(gapStamps.at(-1)!)
      });
    }
  }

  const lastResolved = resolved.at(-1)!;
  const trailingKeptFrom =
    lastResolved.plan.keptFrom === undefined ? undefined : stampLabel(lastResolved.plan.keptFrom);
  const firstKeptFrom = interiorGaps[0]?.firstStamp ?? trailingKeptFrom;

  let keptNotice: string | undefined;
  if (interiorGaps.length === 0) {
    keptNotice = trailingKeptFrom
      ? `The turns from ${trailingKeptFrom} on were not folded and stay above.`
      : undefined;
  } else {
    const parts = [
      ...interiorGaps.map((gap) =>
        gap.firstStamp === gap.lastStamp ? gap.firstStamp : `${gap.firstStamp}-${gap.lastStamp}`
      ),
      ...(trailingKeptFrom ? [`from ${trailingKeptFrom} on`] : [])
    ];
    const list =
      parts.length === 1 ? parts[0]! : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)!}`;
    keptNotice = `The turns ${list} were not folded and stay above.`;
  }

  const omitted = resolved.flatMap((entry) => entry.plan.omitted);
  const spanMessages = resolved.flatMap((entry) => entry.plan.spanMessages);
  const keptUserTurns = resolved.flatMap((entry) => entry.plan.keptUserTurns);
  const selectedEntries = entries.filter((_, idx) => selectedIndices.has(idx));
  const applied = options.applied ?? appliedFolds(entries.map((entry) => entry.sourceEntry));
  const covers = coveredNotes(entries, selectedEntries, applied);
  const media = coveredMedia(selectedEntries, applied);

  const omittedReadStamps = new Set<string>();
  for (const msg of omitted) {
    if (msg.role === 'toolResult' && msg.toolName === 'read') {
      const stamp = readStamp(msg);
      if (stamp !== undefined) omittedReadStamps.add(stampLabel(stamp));
    }
  }
  const archivedReadTurns: string[] = [];
  const seenArchived = new Set<string>();
  for (let idx = 0; idx < lastResolved.startIndex; idx++) {
    if (selectedIndices.has(idx)) continue;
    for (const msg of entries[idx]!.messages) {
      if (msg.role !== 'toolResult' || msg.toolName !== 'read') continue;
      const hit = /^\[duplicate read of .+; identical to (t\d+)\b/.exec(textOf(msg.content));
      const ref = hit?.[1];
      if (ref && omittedReadStamps.has(ref) && !seenArchived.has(ref)) {
        seenArchived.add(ref);
        archivedReadTurns.push(ref);
      }
    }
  }

  const batchSpans: FoldBatchSpanPlan[] = resolved.map((entry) => {
    const lastStamp = entry.plan.spanMessages
      .map(readStamp)
      .findLast((stamp) => stamp !== undefined);
    return {
      since: entry.item.since,
      until: entry.item.until,
      keptFrom: entry.plan.keptFrom,
      note: entry.item.note,
      startIndex: entry.startIndex,
      endIndex: entry.endIndex,
      firstEntryId: entry.plan.firstEntryId,
      lastEntryId: entry.plan.lastEntryId,
      anchorStamp: entry.plan.anchorStamp,
      laterStamps: entry.plan.laterStamps,
      lastStamp,
      spanMessages: entry.plan.spanMessages,
      omitted: entry.plan.omitted
    };
  });

  const combinedNote =
    batchSpans.length === 1
      ? batchSpans[0]!.note
      : batchSpans
          .map((span) => {
            const sinceLabel = stampLabel(span.since);
            const runsToCall = span.keptFrom === undefined;
            const endLabel = runsToCall
              ? span.lastStamp !== undefined && span.lastStamp !== span.since
                ? stampLabel(span.lastStamp)
                : undefined
              : stampLabel(span.lastStamp ?? span.until ?? span.since);
            const singleSpanRange = {
              since: sinceLabel,
              ...(endLabel ? { until: endLabel } : {})
            };
            const header = formatFoldRange(
              { ...singleSpanRange, spans: [singleSpanRange] },
              { turnsPrefix: true }
            );
            return `[${header}]\n${span.note}`;
          })
          .join('\n\n');

  const combinedHeadlineRaw =
    batchSpans.length === 1
      ? headline(batchSpans[0]!.note)
      : batchSpans
          .map((span) => headline(span.note))
          .filter(Boolean)
          .join(' · ');
  const combinedHeadline =
    combinedHeadlineRaw.length > 120
      ? `${combinedHeadlineRaw.slice(0, 117)}...`
      : combinedHeadlineRaw;

  const combinedOpenList = batchSpans.flatMap((span) => {
    const open = openItems(span.note);
    return open ? [open] : [];
  });
  const combinedOpen = combinedOpenList.length > 0 ? combinedOpenList.join('; ') : undefined;

  const tokens = estimateTokens(omitted);
  return {
    handle: resolved[0]!.plan.handle,
    spans: batchSpans,
    firstEntryId: resolved[0]!.plan.firstEntryId,
    lastEntryId: lastResolved.plan.lastEntryId,
    keptFrom: firstKeptFrom,
    keptNotice,
    note: combinedNote,
    headline: combinedHeadline,
    open: combinedOpen,
    omitted,
    spanMessages,
    tokens,
    before: resolved[0]!.plan.before,
    sent: resolved[0]!.plan.sent,
    keptUserTurns,
    archivedReadTurns,
    covers,
    media,
    edits: selectedSpanEdits.sort(
      (a, b) => (entryIndexById.get(b.targetId) ?? -1) - (entryIndexById.get(a.targetId) ?? -1)
    ),
    lines,
    lineTokens
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

export { headline } from './messages.js';

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
  spans?: readonly { since: string; until?: string | undefined }[] | undefined;
  messages: number;
  tokens: number;
  note: string;
  changes: readonly DiskChange[] | undefined;
  touched: readonly string[];
  keptUserTurns: readonly string[];
  /** First turn kept after the span, when the fold stopped before the call. */
  keptFrom?: string | undefined;
  /** Explicit notice for retained interior gaps and/or trailing turns in a batch fold. */
  keptNotice?: string | undefined;
  covers?: readonly CoveredNote[];
  /** Skill files read inside the span. */
  skills?: readonly string[];
  /** Read turns inside this span still referenced by live duplicate-read stubs above it. */
  archivedReadTurns?: readonly string[];
  cwd: string;
}): string {
  const range = formatFoldRange({
    since: options.since,
    until: options.until,
    spans: options.spans
  });
  const lines = [
    `[fold ${options.handle}: ${options.messages} messages, ${range}, ~${formatTokens(options.tokens)} tokens, archived]`,
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

  if (options.keptNotice) {
    lines.push('', options.keptNotice);
  } else if (options.keptFrom) {
    lines.push('', `The turns from ${options.keptFrom} on were not folded and stay above.`);
  }
  if (options.covers?.length) {
    lines.push(
      '',
      `This fold covers ${options.covers.length === 1 ? 'the note of an earlier fold, which follows' : `the notes of ${options.covers.length} earlier folds, which follow`} this note as written; recall and ask still reach ${options.covers.length === 1 ? 'its span' : 'their spans'} by handle.`
    );
  }

  if (options.archivedReadTurns?.length) {
    const calls = options.archivedReadTurns
      .map((turn) => `recall({ handle: "${options.handle}", turn: "${turn}" })`)
      .join(', ');
    lines.push(
      '',
      `Earlier duplicate-read stub(s) above point into this span: read them back with ${calls}.`
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

/** The notes a fold covers, to stand under its own; `standing` reads them back. */
export function renderCovered(covers: readonly CoveredNote[]): string | undefined {
  if (covers.length === 0) return undefined;
  return [
    KEPT_TITLE,
    ...covers.flatMap((fold) => [
      '',
      `[note of ${fold.handle}, ${fold.range}, ${fold.text.length} characters]`,
      fold.text
    ])
  ].join('\n');
}

export interface UnfoldPlan {
  handle: string;
  record: FoldDetails;
  noteEntryId: string;
  edits: ContextEditEntryDraft[];
}

function defaultEntryContent(
  entry: SessionEntry
): (TextContent | ImageContent)[] | AssistantMessage['content'] | string | undefined {
  if (entry.type === 'custom_message') return entry.content ?? [];
  if (entry.type !== 'message') return undefined;
  const msg = entry.message;
  if (msg.role === 'user' || msg.role === 'assistant' || msg.role === 'toolResult') {
    return msg.content ?? [];
  }
  return undefined;
}

function stripKeptSuffix(text: string, keptChars: number | undefined): string {
  if (keptChars !== undefined && keptChars > 0 && text.length >= keptChars + 2) {
    const start = text.length - keptChars;
    if (text.slice(start - 2, start) === '\n\n' && text.slice(start).startsWith(KEPT_TITLE)) {
      return text.slice(0, start - 2);
    }
  }
  const idx = text.indexOf(`\n\n${KEPT_TITLE}`);
  return idx >= 0 ? text.slice(0, idx) : text;
}

function unfoldedNoteContent(
  noteEntry: SessionEntry & { type: 'message' },
  current: readonly (TextContent | ImageContent)[],
  handle: string,
  details: FoldDetails
): (TextContent | ImageContent)[] {
  const stamp = readStamp(noteEntry.message);
  const marker = `[unfolded ${handle}]`;
  if (noteEntry.message.role === 'toolResult' && noteEntry.message.toolName === FOLD_TOOL) {
    return stamp === undefined ? [{ type: 'text', text: marker }] : withStamp(marker, stamp);
  }
  // Script result: keep what the script emitted on its own (including script images), drop
  // covered-note images appended by fold, and replace the rendered or JSON-escaped note.
  const last = current.at(-1);
  const rawBody = last && isStampBlock(last) ? current.slice(0, -1) : [...current];
  const carriedCount = details.carriedImages ?? 0;
  const archivedRefs = archivedImagesOf(details.archivedImages);
  let body = rawBody;
  if (carriedCount > 0 || archivedRefs.length > 0) {
    const protectedPrefix = intactWrittenPrefixLength(rawBody, details.written);
    const dropIndices = new Set<number>(
      matchArchivedRefBlocks(rawBody, archivedRefs, protectedPrefix).keys()
    );
    if (carriedCount > 0) {
      let droppedImages = 0;
      if (protectedPrefix !== undefined) {
        for (let i = protectedPrefix; i < rawBody.length && droppedImages < carriedCount; i++) {
          if (rawBody[i]?.type === 'image') {
            dropIndices.add(i);
            droppedImages++;
          }
        }
      } else if (typeof details.written?.sha256 === 'string') {
        // Recorded written.sha256 failed: inline image ownership is uncertain, so preserve all
        // (preferring potential duplicate images after unfold over losing script-owned images).
      } else if (details.written?.blocks !== undefined) {
        for (
          let i = details.written.blocks;
          i < rawBody.length && droppedImages < carriedCount;
          i++
        ) {
          if (rawBody[i]?.type === 'image') {
            dropIndices.add(i);
            droppedImages++;
          }
        }
      } else {
        for (let i = rawBody.length - 1; i >= 0 && droppedImages < carriedCount; i--) {
          if (rawBody[i]?.type === 'image') {
            dropIndices.add(i);
            droppedImages++;
          }
        }
      }
    }
    if (dropIndices.size > 0) {
      body = rawBody.filter((_, idx) => !dropIndices.has(idx));
    }
  }

  const candidates = [
    details.renderedNote,
    details.renderedNote ? JSON.stringify(details.renderedNote).slice(1, -1) : undefined,
    details.note?.trim(),
    details.note ? JSON.stringify(details.note.trim()).slice(1, -1) : undefined
  ].filter((item): item is string => Boolean(item && item.length > 0));

  const next: (TextContent | ImageContent)[] = [];
  let marked = false;
  for (const block of body) {
    if (block.type !== 'text') {
      next.push(block);
      continue;
    }
    if (block.text.startsWith(KEPT_TITLE)) {
      if (!marked) {
        next.push({ type: 'text', text: marker });
        marked = true;
      }
      continue;
    }
    const cleaned = stripKeptSuffix(block.text, details.written?.kept?.chars);
    if (cleaned.startsWith(`[fold ${handle}:`)) {
      if (!marked) {
        next.push({ type: 'text', text: marker });
        marked = true;
      }
      continue;
    }
    const hit = candidates.find((candidate) => cleaned.includes(candidate));
    if (hit !== undefined) {
      next.push({ type: 'text', text: cleaned.replace(hit, marker) });
      marked = true;
      continue;
    }
    if (cleaned !== block.text) {
      if (cleaned.length > 0) next.push({ type: 'text', text: cleaned });
      if (!marked) {
        next.push({ type: 'text', text: marker });
        marked = true;
      }
      continue;
    }
    next.push(block);
  }
  if (!marked) next.push({ type: 'text', text: marker });
  return stamp === undefined ? next : withStamp(next, stamp);
}

function editSignature(targetId: string, replacement: unknown): string {
  return `${targetId}\0${JSON.stringify(replacement)}`;
}

export function attributeEpisodesEdits(
  branch: readonly SessionEntry[]
): Map<string, { kind: 'fold' | 'unfold'; handle: string }> {
  const owners = new Map<string, { kind: 'fold' | 'unfold'; handle: string }>();
  const claimForward = (
    fromIndex: number,
    kind: 'fold' | 'unfold',
    handle: string,
    sigs: ReadonlySet<string>
  ) => {
    for (let j = fromIndex + 1; j < branch.length; j++) {
      const step = branch[j]!;
      if (step.type !== 'context_edit') break;
      if (!sigs.has(editSignature(step.targetId, step.replacement))) break;
      owners.set(step.id, { kind, handle });
    }
  };
  for (let i = 0; i < branch.length; i++) {
    const entry = branch[i]!;
    if (entry.type !== 'custom') continue;
    const data = (entry.data ?? {}) as {
      kind?: unknown;
      handle?: unknown;
      edits?: unknown;
      readDeps?: unknown;
    };
    if (typeof data.handle !== 'string') continue;
    if (
      (entry.customType === FOLD_STARTED_TYPE || entry.customType === FOLD_UNFOLD_STARTED_TYPE) &&
      Array.isArray(data.edits)
    ) {
      const kind: 'fold' | 'unfold' = entry.customType === FOLD_STARTED_TYPE ? 'fold' : 'unfold';
      const sigs = new Set<string>(
        (data.edits as ContextEditEntryDraft[]).map((edit) =>
          editSignature(edit.targetId, edit.replacement)
        )
      );
      if (kind === 'fold' && Array.isArray(data.readDeps)) {
        for (const dep of data.readDeps as ReadPruneDependency[]) {
          sigs.add(editSignature(dep.targetId, dep.restoreReplacement));
        }
      }
      claimForward(i, kind, data.handle, sigs);
    } else if (
      entry.customType === FOLD_RECOVERY_STARTED_TYPE &&
      (data.kind === 'fold' || data.kind === 'unfold') &&
      Array.isArray(data.edits)
    ) {
      const sigs = new Set<string>(
        (data.edits as ContextEditEntryDraft[]).map((edit) =>
          editSignature(edit.targetId, edit.replacement)
        )
      );
      claimForward(i, data.kind, data.handle, sigs);
    }
  }
  return owners;
}

/**
 * Works out the edits that reverse an applied fold in place up to the latest compaction.
 * Any fold still covering or overlapping `handle` must be unfolded first; any earlier fold
 * that `handle` covered stays in effect, and newer external edits stand.
 */
export function planUnfold(
  branch: readonly SessionEntry[],
  projection: readonly ProjectedSessionEntry[],
  handle: string,
  options: { callerId?: string } = {}
): UnfoldPlan {
  const trimmed = handle.trim();
  if (trimmed.startsWith('fk-')) {
    throw new Error(
      `${trimmed} is a fork, not an inline fold; fork transcripts stay archived and can be searched with recall or queried with ask.`
    );
  }
  if (trimmed.startsWith('cp-')) {
    throw new Error(
      `${trimmed} is a compaction archive, not a fold; compacted turns remain accessible via recall.`
    );
  }

  const branchIndexById = new Map<string, number>();
  const noteEntryIdByHandle = new Map<string, string>();
  branch.forEach((entry, idx) => {
    branchIndexById.set(entry.id, idx);
    const fold = entry.type === 'message' ? foldNoteDetails(entry.message) : undefined;
    if (fold) noteEntryIdByHandle.set(fold.handle, entry.id);
  });

  const applied = appliedFolds(branch);
  const noteEntryId = noteEntryIdByHandle.get(trimmed);
  const noteIndex = noteEntryId === undefined ? -1 : (branchIndexById.get(noteEntryId) ?? -1);
  if (noteIndex < 0 || !applied.has(trimmed)) {
    throw new Error(`No active fold ${trimmed} on this branch.`);
  }
  const noteEntry = branch[noteIndex] as SessionEntry & { type: 'message' };
  const record = foldNoteDetails(noteEntry.message)!;

  const resolveAllIntervals = (target: FoldDetails): [number, number][] | undefined => {
    const out: [number, number][] = [];
    for (const span of foldSpans(target)) {
      const first = branchIndexById.get(span.firstEntryId) ?? -1;
      if (first < 0) return undefined;
      const rawLast = branchIndexById.get(span.lastEntryId) ?? -1;
      out.push([first, rawLast >= first ? rawLast : first]);
    }
    return out;
  };
  const resolveKnownIntervals = (target: FoldDetails): [number, number][] => {
    const out: [number, number][] = [];
    for (const span of foldSpans(target)) {
      const first = branchIndexById.get(span.firstEntryId) ?? -1;
      if (first < 0) continue;
      const rawLast = branchIndexById.get(span.lastEntryId) ?? -1;
      out.push([first, rawLast >= first ? rawLast : first]);
    }
    return out;
  };

  const compactionIndex = branch.findLastIndex((entry) => entry.type === 'compaction');
  if (compactionIndex >= 0) {
    const compaction = branch[compactionIndex] as SessionEntry & {
      type: 'compaction';
      firstKeptEntryId: string;
    };
    const firstKeptIndex = branchIndexById.get(compaction.firstKeptEntryId) ?? -1;
    const isBeforeKept = (idx: number) =>
      idx <= compactionIndex && (firstKeptIndex < 0 || idx < firstKeptIndex);
    const anyCompactedSpan = foldSpans(record).some((span) => {
      const first = branchIndexById.get(span.firstEntryId) ?? -1;
      const last = branchIndexById.get(span.lastEntryId) ?? -1;
      if (first >= 0) return isBeforeKept(first);
      return isBeforeKept(noteIndex) || (last >= 0 && isBeforeKept(last));
    });
    if (isBeforeKept(noteIndex) || anyCompactedSpan) {
      throw new Error(
        `${trimmed} lies before the latest compaction: its turns were compacted by pi and remain accessible via recall and ask.`
      );
    }
  }
  const ownIntervals = resolveAllIntervals(record);
  if (!ownIntervals || ownIntervals.length === 0) {
    throw new Error(`Cannot find the start of ${trimmed} on this branch.`);
  }

  for (let index = noteIndex + 1; index < branch.length; index++) {
    const later = branch[index]!;
    const laterFold = later.type === 'message' ? foldNoteDetails(later.message) : undefined;
    if (!laterFold || !applied.has(laterFold.handle)) continue;
    if (laterFold.covers?.includes(trimmed)) {
      throw new Error(
        `${trimmed} is covered by ${laterFold.handle}; unfold ${laterFold.handle} first.`
      );
    }
    const laterIntervals = resolveKnownIntervals(laterFold);
    const overlaps = ownIntervals.some(([aFirst, aLast]) =>
      laterIntervals.some(([bFirst, bLast]) => aFirst <= bLast && bFirst <= aLast)
    );
    if (overlaps) {
      throw new Error(
        `${trimmed} overlaps active fold ${laterFold.handle}; unfold ${laterFold.handle} first.`
      );
    }
  }

  if (
    options.callerId !== undefined &&
    !projection.some((entry) => callsTool(entry, options.callerId!))
  ) {
    throw new Error('The response that called unfold is not in the current context.');
  }

  const remaining = new Set(applied);
  remaining.delete(trimmed);
  const editOwners = attributeEpisodesEdits(branch);

  // Chronological replay of what stands on `target` when folds no longer in `remaining` are
  // removed, while active folds in `remaining`, unfolded note stubs, and external edits stand.
  const desiredWithoutOwn = (
    target: SessionEntry
  ): ContextEditEntryDraft['replacement'] | undefined => {
    const baseContent = defaultEntryContent(target);
    let state: ContextEditEntryDraft['replacement'] | undefined =
      baseContent === undefined ? undefined : { content: baseContent };
    for (const entry of branch) {
      if (entry.type !== 'context_edit' || entry.targetId !== target.id) continue;
      const owner = editOwners.get(entry.id);
      if (!owner) {
        state = entry.replacement;
        continue;
      }
      if (owner.kind === 'fold') {
        if (remaining.has(owner.handle)) state = entry.replacement;
        continue;
      }
      // Unfold batch: keep its `[unfolded ...]` stub on its own note (while still unfolded);
      // skip span/tail restorations so active folds and external edits win.
      const isUnfoldedStub =
        !remaining.has(owner.handle) && target.id === noteEntryIdByHandle.get(owner.handle);
      if (isUnfoldedStub) {
        state = entry.replacement;
      }
    }
    return state;
  };

  const ownEditedTargetIds = new Set<string>();
  for (const entry of branch) {
    if (entry.type !== 'context_edit') continue;
    const owner = editOwners.get(entry.id);
    if (owner?.kind === 'fold' && owner.handle === trimmed) {
      ownEditedTargetIds.add(entry.targetId);
    }
  }
  const spanSlice = ownIntervals.flatMap(([first, last]) => branch.slice(first, last + 1));
  const spanIds = new Set(spanSlice.map((entry) => entry.id));
  // What the fold edited outside its span: an older marker right before it, which it removed,
  // and entries after it that it pruned.
  const postSpanOwned = branch.filter(
    (entry) =>
      !spanIds.has(entry.id) && ownEditedTargetIds.has(entry.id) && entry.id !== noteEntry.id
  );
  const byId = new Map(projection.map((entry) => [entry.sourceEntry.id, entry]));
  const restoreEdits: ContextEditEntryDraft[] = [];
  const tailRestoreEdits: ContextEditEntryDraft[] = [];

  // Restore span entries in forward order so each assistant turn returns before its toolResults,
  // plus any post-span entries that this fold pruned.
  for (const entry of [...spanSlice, ...postSpanOwned]) {
    const editable =
      entry.type === 'custom_message' ||
      (entry.type === 'message' &&
        (entry.message.role === 'user' ||
          entry.message.role === 'assistant' ||
          entry.message.role === 'toolResult'));
    if (!editable) continue;
    const desired = desiredWithoutOwn(entry);
    if (desired === undefined) continue;
    const projected = byId.get(entry.id);
    const firstMsg = projected?.messages[0];
    const current =
      projected === undefined
        ? undefined
        : firstMsg === undefined
          ? null
          : 'content' in firstMsg
            ? { content: firstMsg.content }
            : undefined;
    if (JSON.stringify(current) !== JSON.stringify(desired)) {
      (spanIds.has(entry.id) ? restoreEdits : tailRestoreEdits).push({
        type: 'context_edit',
        targetId: entry.id,
        replacement: desired
      });
    }
  }

  const noteProjectedEntry = byId.get(noteEntry.id);
  const noteProjected = noteProjectedEntry?.messages[0];
  const currentNoteBlocks =
    noteProjected?.role === 'toolResult'
      ? noteProjected.content
      : (noteEntry.message as { content: (TextContent | ImageContent)[] }).content;
  const noteReplacement =
    noteProjectedEntry !== undefined && noteProjectedEntry.messages.length === 0
      ? null
      : { content: unfoldedNoteContent(noteEntry, currentNoteBlocks, trimmed, record) };
  const noteEdits: ContextEditEntryDraft[] =
    noteReplacement !== null
      ? [
          {
            type: 'context_edit',
            targetId: noteEntry.id,
            replacement: noteReplacement
          }
        ]
      : [];

  return {
    handle: trimmed,
    record,
    noteEntryId: noteEntry.id,
    edits: [...restoreEdits, ...tailRestoreEdits, ...noteEdits]
  };
}

const UnfoldParams = Type.Object({
  handle: Type.String({
    description:
      'Handle of the applied fold to restore in place, such as "ep-1a2b3c4d". Must lie after the latest compaction and not be covered by a later active fold.'
  })
});

const FoldSpanItemParams = Type.Object({
  since: Type.String({
    description:
      'Stamp of the first message inside this span, such as "t13", usually the result of its first tool call.'
  }),
  until: Type.Optional(
    Type.String({
      description:
        'Stamp of the last turn inside this span, such as "t20". Required on all spans before the final one; omit on the final span to fold up to this call.'
    })
  ),
  note: Type.String({
    description:
      'Note replacing this span: question and outcome on the first line, then changed, learned, decided/rejected, handled user stamps, and open.'
  })
});

const FoldParams = Type.Object({
  since: Type.Optional(
    Type.String({
      description:
        'Stamp of the first message inside the episode, such as "t13", usually the result of its first tool call. The fold starts at the response that made that call and runs up to this call, or to the end of the turn `until` names. Mutually exclusive with `spans`.'
    })
  ),
  until: Type.Optional(
    Type.String({
      description:
        'Stamp of the last turn inside the span, such as "t20", when the turns after it must stay in context. Leave it out to fold up to this call. What stays after the span is sent uncached once (with older identical reads and aged failed inputs trimmed).'
    })
  ),
  note: Type.Optional(
    Type.String({
      description:
        'Replaces the span. First line: the question and how it ended. Then changed, learned (with conditions), decided and rejected (with why), handled user messages by stamp, open (with the next step). Quote exact paths, symbols, commands and error strings.'
    })
  ),
  spans: Type.Optional(
    Type.Array(FoldSpanItemParams, {
      minItems: 1,
      description:
        'Non-overlapping spans to fold in one batch, such as [{ since: "t10", until: "t25", note: "..." }, { since: "t45", note: "..." }]. Retires all listed spans in one turn with a single prefix rewrite from the earliest `since`, keeping turns between them in context. Mutually exclusive with top-level `since`/`until`/`note`.'
    })
  )
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
      'Fold one or more finished episodes: remove the span starting at `since` (or multiple non-overlapping `spans: [{ since, until?, note }, ...]`) from your context, archive them, and keep the note in their place. User messages from folded spans stay. With `until` a span ends with that turn and later turns stay. Call it as the only tool in the response, once the next step no longer depends on the folded spans beyond the note, and not right before your final reply. From a codemode script, call only fold in that script and print its result.',
    promptSnippet: 'Fold a finished episode into a note; the span is archived and stays reachable',
    parameters: FoldParams,
    executionMode: 'sequential',
    renderCall: foldRenderers.renderCall,
    renderResult: foldRenderers.renderResult,

    async execute(toolCallId, params, signal, _onUpdate, ctx) {
      // A /fold request is used up by this call, whether it succeeds or not.
      const manual = state.manualFold;
      state.manualFold = undefined;
      // A call from a codemode script is not in the transcript; the response that called the
      // script made it.
      const root = state.rootCall(toolCallId);
      const scriptCall = root === toolCallId ? undefined : root;
      const run = state.run;
      if (signal?.aborted || run !== state.run) throw new Error(FOLD_ABORTED_MESSAGE);
      const entries = ctx.sessionManager.buildSessionProjection().entries;
      const foldIndex = entries.findIndex((entry) => callsTool(entry, scriptCall ?? toolCallId));
      const caller = foldIndex >= 0 ? entries[foldIndex] : undefined;
      let batch: FoldBatchPlan;
      try {
        // The first fold of a response holds it, whether or not it succeeds, as for folds the
        // response calls itself. Reserved before anything is awaited, so a second fold from the
        // same response, in the same script or another one, sees it.
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
        const normalizedSpans = normalizeFoldSpans(params as Record<string, unknown>);
        batch = planFoldBatch(entries, toolCallId, normalizedSpans, {
          latestStamp: state.nextStamp - 1,
          callerId: scriptCall ?? toolCallId
        });
      } catch (error) {
        if (error instanceof FoldError) {
          recordEvent(pi, { kind: 'fold-rejected', toolCallId, reason: error.reason });
        }
        throw error;
      }

      const spanDetails: FoldSpanDetails[] = batch.spans.map((span) => {
        const sinceLabel = stampLabel(span.since);
        const runsToCall = span.keptFrom === undefined;
        const untilLabel = runsToCall
          ? span.lastStamp !== undefined && span.lastStamp !== span.since
            ? stampLabel(span.lastStamp)
            : undefined
          : batch.spans.length === 1
            ? span.lastStamp !== undefined && span.lastStamp !== span.since
              ? stampLabel(span.lastStamp)
              : undefined
            : stampLabel(span.lastStamp ?? span.until ?? span.since);
        return {
          since: sinceLabel,
          ...(untilLabel ? { until: untilLabel } : {}),
          firstEntryId: span.firstEntryId,
          lastEntryId: span.lastEntryId,
          note: span.note
        };
      });
      const firstSpanDetail = spanDetails[0]!;
      const lastSpanDetail = spanDetails.at(-1)!;
      const sinceLabel = firstSpanDetail.since;
      const untilLabel = batch.spans.length === 1 ? firstSpanDetail.until : lastSpanDetail.until;
      const keptFrom = batch.keptFrom;
      const touched = touchedPaths(batch.spanMessages, state.cwd);
      const files = new Set(touched);
      const perSpanFiles = batch.spans.map(
        (span) => new Set(touchedPaths(span.spanMessages, state.cwd))
      );
      const changedPathsAnySpan = new Set<string>();

      let changes: DiskChange[] | undefined;
      let treeAfter: string | undefined;
      const snapshots = state.snapshots;
      const anchorPlaced = state.anchorPlaced;
      if (snapshots) {
        let currentTreePromise: Promise<string | undefined> | undefined;
        const currentTree = () => {
          currentTreePromise ??= snapshots.snapshot();
          return currentTreePromise;
        };
        const perSpanChanges: (DiskChange[] | undefined)[] = [];
        for (let i = 0; i < batch.spans.length; i++) {
          const span = batch.spans[i]!;
          const runsToCall = span.keptFrom === undefined;
          const spanTreeAfter =
            runsToCall || span.lastStamp === undefined
              ? await currentTree()
              : await snapshots.treeAt(span.lastStamp);
          if (spanTreeAfter) {
            treeAfter = spanTreeAfter;
            spanDetails[i]!.treeAfter = spanTreeAfter;
          }
          let from: string | undefined;
          for (const stamp of [...span.laterStamps].reverse().concat(span.anchorStamp ?? [])) {
            from ??= await snapshots.treeAt(stamp);
          }
          if (from && spanTreeAfter) {
            const spanDiff = await snapshots.diff(from, spanTreeAfter);
            perSpanChanges.push(spanDiff);
            for (const change of spanDiff ?? []) {
              files.add(change.path);
              perSpanFiles[i]!.add(change.path);
              changedPathsAnySpan.add(change.path);
            }
          } else {
            perSpanChanges.push(undefined);
          }
        }
        changes = mergeSpanDiskChanges(perSpanChanges);
      }
      for (let i = 0; i < spanDetails.length; i++) {
        if (perSpanFiles[i]!.size > 0) spanDetails[i]!.files = [...perSpanFiles[i]!];
      }

      const tokens = batch.tokens;
      const details: FoldDetails = {
        kind: 'fold',
        handle: batch.handle,
        since: sinceLabel,
        ...(untilLabel ? { until: untilLabel } : {}),
        ...(spanDetails.length > 1 ? { spans: spanDetails } : {}),
        firstEntryId: batch.firstEntryId,
        lastEntryId: batch.lastEntryId,
        messages: batch.omitted.length,
        tokens,
        headline: batch.headline,
        keptUserTurns: batch.keptUserTurns,
        files: [...files]
      };
      if (batch.covers.length > 0) details.covers = batch.covers.map((fold) => fold.handle);
      details.note = batch.note;
      if (keptFrom) details.keptFrom = keptFrom;
      if (treeAfter) details.treeAfter = treeAfter;
      if (batch.open) details.open = batch.open;
      const readOnly = readOnlyPaths(
        batch.spanMessages,
        state.cwd,
        changedPathsAnySpan.size > 0
          ? [...changedPathsAnySpan]
          : (changes ?? []).map((change) => change.path)
      );
      if (readOnly.length > 0) details.readOnly = readOnly;
      details.noteTokens = Math.ceil(batch.note.length / 4);
      // The notes it covers stay in context under it: the agent did not write them again, and
      // the fold did not free them.
      if (batch.covers.length > 0) details.keptTokens = coveredTokens(batch.covers);
      if (batch.lineTokens !== 0) details.lineTokens = batch.lineTokens;
      if (manual) details.manual = manual;
      const gapTokens = anchorGapTokens(
        batch.sent,
        batch.before,
        appliedFolds(ctx.sessionManager.getBranch())
      );
      details.anchor = {
        ...(anchorPlaced === undefined ? {} : { placed: anchorPlaced }),
        ...(gapTokens === undefined ? {} : { gapTokens })
      };

      const text = renderNote({
        handle: batch.handle,
        since: sinceLabel,
        until: untilLabel,
        ...(spanDetails.length > 1 ? { spans: spanDetails } : {}),
        messages: batch.omitted.length,
        tokens,
        note: batch.note,
        changes,
        touched,
        keptUserTurns: batch.keptUserTurns,
        archivedReadTurns: batch.archivedReadTurns,
        keptFrom,
        keptNotice: batch.keptNotice,
        covers: batch.covers,
        skills: skillPaths(batch.spanMessages, state.cwd),
        cwd: state.cwd
      });
      const carried = renderCovered(batch.covers);
      const media = await prepareCarriedMedia(
        batch.handle,
        batch.media,
        archiveDir(ctx.sessionManager),
        signal
      );
      // Snapshots and image archiving take time; a run aborted meanwhile, or already settled, must not get edits.
      if (signal?.aborted || run !== state.run) throw new Error(FOLD_ABORTED_MESSAGE);
      const inlineImages = media.flatMap((item) => (item.kind === 'image' ? [item.image] : []));
      const archivedRefs = media.flatMap((item) => (item.kind === 'ref' ? [{ ...item.ref }] : []));
      // A fold from a script takes effect with the script's result, which then stands as the
      // note; the edits wait under the script's call, the one the turn reports.
      state.pendingFolds.set(scriptCall ?? toolCallId, {
        handle: batch.handle,
        messages: batch.omitted.length,
        tokens,
        edits: batch.edits,
        lines: batch.lines,
        lastEntryId: batch.lastEntryId,
        pruneAfterEntryId: batch.spans[0]!.lastEntryId,
        ...(caller ? { callerEntryId: caller.sourceEntry.id } : {}),
        ...(carried ? { carried } : {}),
        ...(media.length > 0 ? { media } : {}),
        ...(scriptCall ? { script: { foldCallId: toolCallId, note: text, details } } : {})
      });

      // A /fold request ends the run here; an autonomous fold continues the task. Codemode
      // does not pass this on from a script, so a /fold done there does not end the run.
      const terminate = manual !== undefined;

      // A script gets its own note only. The notes it covers are added to the script's result:
      // printed by the script they would be there twice, or cut where codemode cuts long output.
      const shown = carried && !scriptCall ? `${text}\n\n${carried}` : text;
      const written = [{ type: 'text' as const, text: shown }];
      // A script's result is recorded when it is there, with what the script printed.
      details.renderedNote = text;
      if (inlineImages.length > 0) details.carriedImages = inlineImages.length;
      if (archivedRefs.length > 0) details.archivedImages = archivedRefs;
      if (scriptCall) return { content: written, details, terminate };
      details.written = writtenMark(written, carried);
      // Covered-note media (archived references and inline fallbacks) follow what the fold wrote.
      return { content: [...written, ...carriedMediaBlocks(media)], details, terminate };
    }
  });

  pi.registerTool({
    name: UNFOLD_TOOL,
    label: 'Unfold',
    description:
      'Restore a folded episode in place: bring back the turns `handle` removed after the latest compaction and retire its fold note. A fold covered by a later fold can be unfolded once the covering fold is unfolded first.',
    promptSnippet: 'Restore a post-compaction folded episode in place',
    parameters: UnfoldParams,
    executionMode: 'sequential',
    renderCall: unfoldRenderers.renderCall,
    renderResult: unfoldRenderers.renderResult,

    async execute(toolCallId, params, signal, _onUpdate, ctx) {
      const root = state.rootCall(toolCallId);
      const scriptCall = root === toolCallId ? undefined : root;
      const run = state.run;
      const projection = ctx.sessionManager.buildSessionProjection().entries;
      const branch = ctx.sessionManager.getBranch();
      const caller = projection.find((entry) => callsTool(entry, scriptCall ?? toolCallId));
      if (caller) {
        const reserved = state.foldResponses.get(caller.sourceEntry.id);
        if (reserved !== undefined && reserved !== toolCallId) {
          throw new Error('Only one fold or unfold per response; this one was ignored.');
        }
        state.foldResponses.set(caller.sourceEntry.id, toolCallId);
      }
      const unfinished = cutShortFold(branch);
      if (unfinished !== undefined) {
        throw new Error(
          `${unfinished} was interrupted partway and is completed when this response ends. Unfold again in your next response.`
        );
      }
      const response = caller ? entryMessage(caller) : undefined;
      const firstAction =
        response?.role === 'assistant'
          ? response.content.find(
              (block) =>
                block.type === 'toolCall' &&
                (block.name === FOLD_TOOL || block.name === UNFOLD_TOOL)
            )
          : undefined;
      if (!scriptCall && firstAction?.type === 'toolCall' && firstAction.id !== toolCallId) {
        throw new Error('Only one fold or unfold per response; this one was ignored.');
      }

      const plan = planUnfold(branch, projection, params.handle, {
        callerId: scriptCall ?? toolCallId
      });
      if (signal?.aborted || run !== state.run) throw new Error('The unfold was cancelled.');

      state.pendingUnfolds.set(scriptCall ?? toolCallId, {
        handle: plan.handle,
        record: plan.record,
        messages: plan.record.messages,
        tokens: plan.record.tokens,
        edits: plan.edits,
        noteEntryId: plan.noteEntryId
      });

      const range = formatFoldRange(plan.record);
      const details: UnfoldDetails = {
        kind: 'unfold',
        handle: plan.handle,
        since: plan.record.since,
        ...(plan.record.until ? { until: plan.record.until } : {}),
        ...(plan.record.spans && plan.record.spans.length > 1
          ? {
              spans: plan.record.spans.map((span) => ({
                since: span.since,
                ...(span.until ? { until: span.until } : {})
              }))
            }
          : {}),
        messages: plan.record.messages,
        tokens: plan.record.tokens
      };
      return {
        content: [
          {
            type: 'text',
            text: `[unfolded ${plan.handle}: restored ${plan.record.messages} messages, ${range}, ~${formatTokens(plan.record.tokens)} tokens]`
          }
        ],
        details
      };
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
    // What the agent wrote is then no longer what the result says.
    if (note !== pending.script.note.trim()) pending.script.changed = true;
    pending.script.note = note;
  });

  // A result hook can fail a nested unfold call after it ran, and the script can catch that.
  pi.on('tool_execution_end', (event) => {
    if (!event.parentToolCallId || event.toolName !== UNFOLD_TOOL || !event.isError) return;
    state.pendingUnfolds.delete(state.rootCall(event.toolCallId));
  });
  // A fold from a script takes effect only if the script succeeds. Its result then carries the
  // fold's details, the note too when the script did not print it, and the notes the fold
  // covers, written once with the result rather than edited in later.
  pi.on('tool_result', (event) => {
    if (event.parentToolCallId) return undefined;
    const pending = state.pendingFolds.get(event.toolCallId);
    if (event.isError) {
      state.pendingUnfolds.delete(event.toolCallId);
    }
    const script = pending?.script;
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
    const added = [
      ...(holdsNote(event.content, script.note) ? [] : [script.note]),
      ...(pending.carried ? [pending.carried] : [])
    ];
    const written = [...event.content, ...added.map((text) => ({ type: 'text' as const, text }))];
    // The result as the fold leaves it, for the fold that covers this note to know it by;
    // not when a hook changed the note on its way to the script, which that fold must keep
    // as the result has it.
    script.details.renderedNote = script.note;
    script.details.written = writtenMark(written, pending.carried, script.changed);
    const mediaItems = pending.media ?? [];
    const inlineCount = mediaItems.filter((item) => item.kind === 'image').length;
    const archivedRefs = mediaItems.flatMap((item) =>
      item.kind === 'ref' ? [{ ...item.ref }] : []
    );
    if (inlineCount > 0) script.details.carriedImages = inlineCount;
    if (archivedRefs.length > 0) script.details.archivedImages = archivedRefs;
    const mediaBlocks = carriedMediaBlocks(mediaItems);
    if (added.length === 0 && mediaBlocks.length === 0) return { details };
    // Covered-note media (archived references and inline fallbacks) follow what the fold wrote.
    const content = [...written, ...mediaBlocks];
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
        const note = branch.find(
          (entry) =>
            entry.type === 'message' &&
            entry.message.role === 'toolResult' &&
            entry.message.toolCallId === result.toolCallId
        );
        // What will stand in context for the result: a later hook may have rewritten it, and
        // an earlier handler of this boundary may have queued an edit of it. Without the note
        // in a script's result, the span would leave with nothing in its place; without the
        // notes the fold covers, or the images that stood in their results, those would leave
        // with it; and a result that is removed leaves nothing to stand for the span.
        const edited = note && editedContent(note.id, branch, event.entries);
        const content = edited ?? result.content;
        const persistedDetails = foldNoteDetails(note?.type === 'message' ? note.message : result);
        const mediaItems = pending?.media ?? [];
        const unstamped = withoutTrailingStamp(content);
        const protectedPrefix = intactWrittenPrefixLength(unstamped, persistedDetails?.written);
        const hasRefs = mediaItems.some((item) => item.kind === 'ref');
        const stands =
          edited !== null &&
          (pending?.script
            ? persistedDetails?.handle === pending.handle && holdsNote(content, pending.script.note)
            : result.toolName === FOLD_TOOL &&
              (!hasRefs || persistedDetails?.handle === pending?.handle)) &&
          (pending?.carried === undefined || textOf(content).includes(pending.carried)) &&
          holdsArchivedManifest(persistedDetails?.archivedImages, mediaItems) &&
          holdsCarriedMedia(
            content,
            mediaItems,
            protectedPrefix,
            archiveDir(ctx.sessionManager),
            pending?.handle
          );
        if (!pending || !stands) return [];
        state.pendingFolds.delete(result.toolCallId);
        return [pending];
      });
    const unfolded = event.toolResults
      .filter((result) => !result.isError)
      .flatMap((result) => {
        const pending = state.pendingUnfolds.get(result.toolCallId);
        if (!pending) return [];
        state.pendingUnfolds.delete(result.toolCallId);
        return [pending];
      });
    const appliedSpanRemovals = new Set(
      applied.flatMap((fold) =>
        fold.edits.flatMap((edit) => (edit.replacement === null ? [edit.targetId] : []))
      )
    );
    const repairs = completeCutShortFolds(branch, event.entries, appliedSpanRemovals);
    const repairDrafts = repairs.filter(
      (item): item is ContextEditEntryDraft => item.type === 'context_edit'
    );
    if (applied.length === 0 && unfolded.length === 0 && repairs.length === 0) return undefined;

    const branchById = new Map(branch.map((entry) => [entry.id, entry]));
    const branchIndexById = new Map(branch.map((entry, index) => [entry.id, index]));
    const editOwners =
      applied.length > 0
        ? attributeEpisodesEdits(branch)
        : new Map<string, { kind: 'fold' | 'unfold'; handle: string }>();

    // The cut of a user message only saves tokens, so an edit an earlier handler queued for
    // that message stands, as it does when a cut-short fold is completed. The note was written
    // before this boundary: if that edit removes the message, the note still lists its turn
    // among the ones that stay.
    const edited = new Set(
      (event.entries as readonly PendingDraft[]).flatMap((draft) =>
        draft.type === 'context_edit' && draft.targetId !== undefined ? [draft.targetId] : []
      )
    );
    const standingLines =
      applied.length > 0
        ? foldLines(ctx.sessionManager.buildSessionProjection().entries, appliedFolds(branch))
        : new Map<string, string>();
    const editsOf = (fold: {
      edits: readonly ContextEditEntryDraft[];
      lines?: readonly string[];
      lastEntryId?: string;
      pruneAfterEntryId?: string;
      callerEntryId?: string;
    }) => {
      const beforeEntryId = fold.callerEntryId ?? event.messageEntryId;
      const pruneAfter = fold.pruneAfterEntryId ?? fold.lastEntryId;
      const ownRemovalDrafts = fold.edits.filter((edit) => edit.replacement === null);
      const tailPrune =
        pruneAfter !== undefined && beforeEntryId !== undefined
          ? planFoldTailPruning({
              branch,
              pending: event.entries as readonly PendingDraft[],
              repairDrafts: [...repairDrafts, ...ownRemovalDrafts],
              editOwners,
              afterEntryId: pruneAfter,
              beforeEntryId,
              cwd: state.cwd,
              lines: new Set([...standingLines.keys(), ...(fold.lines ?? [])])
            })
          : {
              assistantReplacements: new Map<string, AssistantMessage['content']>(),
              readEdits: [] as ContextEditEntryDraft[],
              readDeps: [] as ReadPruneDependency[]
            };

      const foldAssistantMap = new Map<string, ContextEditEntryDraft>();
      const spanEdits: ContextEditEntryDraft[] = [];
      for (const edit of fold.edits) {
        if (edit.replacement === null) {
          spanEdits.push(edit);
          continue;
        }
        const source = branchById.get(edit.targetId);
        if (source?.type === 'message' && source.message.role === 'assistant') {
          foldAssistantMap.set(edit.targetId, edit);
          continue;
        }
        if (!edited.has(edit.targetId)) spanEdits.push(edit);
      }

      const sortedAssistantIds = [
        ...new Set([...foldAssistantMap.keys(), ...tailPrune.assistantReplacements.keys()])
      ].sort((a, b) => (branchIndexById.get(b) ?? -1) - (branchIndexById.get(a) ?? -1));

      const assistantEdits: ContextEditEntryDraft[] = [];
      for (const targetId of sortedAssistantIds) {
        // An edit an earlier handler queued for a response stands, as for a user message. The
        // fold's line does not give way: the response it replaces leaves with its span, as a
        // removal would have it, and left standing it would hold calls without results.
        if (edited.has(targetId) && !fold.lines?.includes(targetId)) continue;
        const composed = tailPrune.assistantReplacements.get(targetId);
        if (composed) {
          assistantEdits.push({
            type: 'context_edit',
            targetId,
            replacement: { content: composed }
          });
          continue;
        }
        const existing = foldAssistantMap.get(targetId);
        if (existing) assistantEdits.push(existing);
      }

      // Order the edits of responses (failed-argument trims) and the span removals and cuts
      // newest to oldest by branch index, so no crash prefix leaves a tool result without its call.
      const orderedEdits = [...assistantEdits, ...spanEdits].sort(
        (a, b) => (branchIndexById.get(b.targetId) ?? -1) - (branchIndexById.get(a.targetId) ?? -1)
      );

      const readEdits = tailPrune.readEdits.filter((edit) => !edited.has(edit.targetId));
      const keptReadIds = new Set(readEdits.map((edit) => edit.targetId));
      const readDeps = tailPrune.readDeps.filter((dep) => keptReadIds.has(dep.targetId));
      return {
        edits: [...orderedEdits, ...readEdits],
        readDeps
      };
    };
    const unfoldEditsOf = (unfold: {
      handle: string;
      record: FoldDetails;
      edits: readonly ContextEditEntryDraft[];
      noteEntryId: string;
    }) => {
      const base: ContextEditEntryDraft[] = [];
      for (const edit of unfold.edits) {
        if (edit.targetId === unfold.noteEntryId) {
          if (!edited.has(unfold.noteEntryId)) {
            base.push(edit);
            continue;
          }
          const pendingNote = editedContent(unfold.noteEntryId, branch, event.entries);
          const noteEntry = branchById.get(unfold.noteEntryId);
          if (pendingNote !== null && pendingNote !== undefined && noteEntry?.type === 'message') {
            base.push({
              type: 'context_edit',
              targetId: unfold.noteEntryId,
              replacement: {
                content: unfoldedNoteContent(noteEntry, pendingNote, unfold.handle, unfold.record)
              }
            });
          }
          continue;
        }
        if (!edited.has(edit.targetId)) base.push(edit);
      }
      return base;
    };

    for (const fold of applied) {
      ctx.ui.notify(
        `Folded ${fold.messages} messages (~${formatTokens(fold.tokens)} tokens) into ${fold.handle}`,
        'info'
      );
    }
    for (const unfold of unfolded) {
      ctx.ui.notify(
        `Unfolded ${unfold.handle} (${unfold.messages} messages, ~${formatTokens(unfold.tokens)} tokens)`,
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
        ...applied.flatMap((fold) => {
          const { edits, readDeps } = editsOf(fold);
          const lines = fold.lines ?? [];
          return [
            {
              type: 'custom' as const,
              customType: FOLD_STARTED_TYPE,
              data: {
                handle: fold.handle,
                edits,
                ...(lines.length > 0 ? { lines } : {}),
                ...(readDeps.length > 0 ? { readDeps } : {})
              }
            },
            ...edits,
            {
              type: 'custom' as const,
              customType: FOLD_APPLIED_TYPE,
              data: { handle: fold.handle }
            }
          ];
        }),
        ...unfolded.flatMap((unfold) => {
          const edits = unfoldEditsOf(unfold);
          return [
            {
              type: 'custom' as const,
              customType: FOLD_UNFOLD_STARTED_TYPE,
              data: { handle: unfold.handle, edits }
            },
            ...edits,
            {
              type: 'custom' as const,
              customType: FOLD_UNFOLDED_TYPE,
              data: { handle: unfold.handle }
            }
          ];
        })
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
    state.pendingUnfolds.clear();
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
