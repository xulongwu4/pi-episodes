import type { AgentMessage, AgentTool, AgentToolResult } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent, Tool } from '@earendil-works/pi-ai';
import type { ProjectedSessionEntry, SessionEntry } from '@earendil-works/pi-coding-agent';
import {
  type FoldDetails,
  type FoldSpanDetails,
  foldNoteDetails,
  formatFoldRange,
  type NestedFold,
  nestedFoldSpans,
  UNFOLD_TOOL
} from './archive.js';
import {
  coveredTokens,
  normalizeFoldSpans,
  planFoldBatch,
  renderCovered,
  renderNote,
  writtenMark
} from './fold.js';
import { carriedMediaBlocks, FOLD_ABORTED_MESSAGE, prepareCarriedMedia } from './images.js';
import { skillPaths, touchedPaths } from './messages.js';
import { readStamp, stampLabel, withStamp } from './stamps.js';
import type { EpisodeState } from './state.js';
import { formatTokens, type UnfoldDetails } from './ui.js';

/**
 * Folding inside a fork. The child is a plain agent, not a pi session, so there are no
 * session entries to edit: it gets synthetic entries for `planFold` and hides folded spans
 * per request through its `transformContext`. Its tool results are stamped from the
 * session's own counter, so a stamp means one message across parent and children.
 */
export class ForkFolds {
  readonly folds: NestedFold[] = [];
  private readonly forkHandle: string;
  private readonly state: EpisodeState;
  private readonly parentEntries: readonly ProjectedSessionEntry[];
  /** Messages before the child's own: the parent's view plus closed sibling fork calls. */
  private readonly prefixLength: number;
  private readonly siblings: number;
  private readonly archiveDir: string | undefined;
  private readonly omitted = new Set<AgentMessage>();
  private readonly replaced = new Map<AgentMessage, (TextContent | ImageContent)[]>();
  private readonly applied = new Set<string>();

  constructor(options: {
    forkHandle: string;
    state: EpisodeState;
    parentEntries: readonly ProjectedSessionEntry[];
    prefixLength: number;
    siblings: number;
    archiveDir?: string | undefined;
  }) {
    this.forkHandle = options.forkHandle;
    this.state = options.state;
    this.parentEntries = options.parentEntries;
    this.prefixLength = options.prefixLength;
    this.siblings = options.siblings;
    this.archiveDir = options.archiveDir;
  }

  /** Active folds inside this fork: excludes any that the child unfolded. */
  get activeFolds(): NestedFold[] {
    return this.folds.filter((fold) => !fold.unfolded);
  }

  /** Handles of active folds inside this fork, for prompt-cache anchor placement. */
  activeHandles(): ReadonlySet<string> {
    return this.applied;
  }

  /** What the child sees: everything but the spans it folded. */
  view(messages: readonly AgentMessage[]): AgentMessage[] {
    const out: AgentMessage[] = [];
    for (const message of messages) {
      if (this.omitted.has(message)) continue;
      const noteReplacement = this.replaced.get(message);
      out.push(
        noteReplacement && message.role === 'toolResult'
          ? { ...message, content: noteReplacement }
          : message
      );
    }
    return out;
  }

  /** Stamps a child tool result in place, before the loop sends it. */
  stamp(message: AgentMessage): void {
    if (message.role !== 'toolResult') return;
    message.content = withStamp(message.content, this.state.nextStamp++);
  }

  tool(declaration: Tool, messages: () => readonly AgentMessage[]): AgentTool {
    return {
      ...declaration,
      label: declaration.name,
      executionMode: 'sequential',
      execute: async (toolCallId, args, signal) =>
        declaration.name === UNFOLD_TOOL
          ? this.unfold(args, messages())
          : this.fold(toolCallId, args, messages(), signal)
    };
  }

  private entries(messages: readonly AgentMessage[]): ProjectedSessionEntry[] {
    const own = messages.slice(this.prefixLength - this.siblings).map((message, index) => {
      const sibling = index < this.siblings;
      const ownIndex = index - this.siblings;
      const id = sibling ? `${this.forkHandle}/s${index}` : `${this.forkHandle}/${ownIndex}`;
      const noteReplacement = this.replaced.get(message);
      const projected: AgentMessage =
        noteReplacement && message.role === 'toolResult'
          ? { ...message, content: noteReplacement }
          : message;
      const sourceEntry = {
        type: 'message',
        id,
        parentId: null,
        timestamp: new Date(message.timestamp).toISOString(),
        message
      } as SessionEntry;
      return { sourceEntry, messages: this.omitted.has(message) ? [] : [projected] };
    });
    return [...this.parentEntries, ...own];
  }

  private async fold(
    toolCallId: string,
    args: unknown,
    messages: readonly AgentMessage[],
    signal?: AbortSignal
  ): Promise<AgentToolResult<FoldDetails>> {
    if (signal?.aborted) throw new Error(FOLD_ABORTED_MESSAGE);
    const normalizedSpans = normalizeFoldSpans((args ?? {}) as Record<string, unknown>);

    const entries = this.entries(messages);
    // The task is the first entry after the parent's view and the siblings; the fork's own
    // work starts after it.
    const batch = planFoldBatch(entries, toolCallId, normalizedSpans, {
      latestStamp: this.state.nextStamp - 1,
      floorIndex: this.parentEntries.length + this.siblings,
      applied: this.applied
    });

    // Archive covered-note media once for the combined batch before mutating any ForkFolds
    // state so errors or cancellation leave omitted/applied/folds untouched.
    const prepared = await prepareCarriedMedia(batch.handle, batch.media, this.archiveDir, signal);
    if (signal?.aborted) throw new Error(FOLD_ABORTED_MESSAGE);

    const ownMessages = messages.slice(this.prefixLength);
    const index = (id: string) => Number(id.slice(this.forkHandle.length + 1));
    const omit: number[] = [];
    for (const edit of batch.edits) {
      // Children have no user messages, so a fold only omits span entries.
      if (edit.replacement !== null) continue;
      const ownIndex = index(edit.targetId);
      const message = ownMessages[ownIndex];
      if (!message) continue;
      this.omitted.add(message);
      omit.push(ownIndex);
    }
    const firstIndex = index(batch.firstEntryId);
    const lastIndex = index(batch.lastEntryId);
    const childSpans = batch.spans.map((span) => ({
      first: index(span.firstEntryId),
      last: index(span.lastEntryId)
    }));
    this.applied.add(batch.handle);

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
      const spanFiles = touchedPaths(span.spanMessages, this.state.cwd);
      return {
        since: sinceLabel,
        ...(untilLabel ? { until: untilLabel } : {}),
        firstEntryId: span.firstEntryId,
        lastEntryId: span.lastEntryId,
        note: span.note,
        ...(spanFiles.length > 0 ? { files: spanFiles } : {})
      };
    });
    const firstSpanDetail = spanDetails[0]!;
    const lastSpanDetail = spanDetails.at(-1)!;
    const sinceLabel = firstSpanDetail.since;
    const until = batch.spans.length === 1 ? firstSpanDetail.until : lastSpanDetail.until;
    const touched = touchedPaths(batch.spanMessages, this.state.cwd);
    const tokens = batch.tokens;
    const record: FoldDetails = {
      kind: 'fold',
      handle: batch.handle,
      since: sinceLabel,
      ...(until ? { until } : {}),
      ...(spanDetails.length > 1 ? { spans: spanDetails } : {}),
      firstEntryId: batch.firstEntryId,
      lastEntryId: batch.lastEntryId,
      messages: batch.omitted.length,
      tokens,
      headline: batch.headline,
      keptUserTurns: batch.keptUserTurns,
      files: touched
    };
    if (batch.covers.length > 0) record.covers = batch.covers.map((fold) => fold.handle);
    record.note = batch.note;
    record.noteTokens = Math.ceil(batch.note.length / 4);
    if (batch.covers.length > 0) record.keptTokens = coveredTokens(batch.covers);
    const carried = renderCovered(batch.covers);
    const keptFrom = batch.keptFrom;
    if (keptFrom) record.keptFrom = keptFrom;
    if (batch.open) record.open = batch.open;
    this.folds.push({
      record,
      first: firstIndex,
      last: lastIndex,
      ...(childSpans.length > 1 ? { spans: childSpans } : {}),
      omit,
      createdAt: ownMessages.length
    });

    const content = [
      {
        type: 'text' as const,
        text: `${renderNote({
          handle: batch.handle,
          since: sinceLabel,
          until,
          ...(spanDetails.length > 1 ? { spans: spanDetails } : {}),
          messages: batch.omitted.length,
          tokens,
          note: batch.note,
          // The fork's own result carries the working tree diff for all of its work.
          changes: undefined,
          touched,
          keptUserTurns: batch.keptUserTurns,
          archivedReadTurns: batch.archivedReadTurns,
          keptFrom,
          keptNotice: batch.keptNotice,
          covers: batch.covers,
          skills: skillPaths(batch.spanMessages, this.state.cwd),
          cwd: this.state.cwd
        })}${carried ? `\n\n${carried}` : ''}`
      }
    ];
    record.written = writtenMark(content, carried);
    const mediaBlocks = carriedMediaBlocks(prepared);
    const archivedManifest = prepared.flatMap((item) =>
      item.kind === 'ref' ? [{ ...item.ref }] : []
    );
    const carriedCount = prepared.filter((item) => item.kind === 'image').length;
    if (carriedCount > 0) record.carriedImages = carriedCount;
    if (archivedManifest.length > 0) record.archivedImages = archivedManifest;
    return {
      content: [...content, ...mediaBlocks],
      details: structuredClone(record)
    };
  }

  private async unfold(
    args: unknown,
    messages: readonly AgentMessage[]
  ): Promise<AgentToolResult<UnfoldDetails>> {
    const raw = (args as { handle?: unknown } | undefined)?.handle;
    const handle = typeof raw === 'string' ? raw.trim() : '';
    const nestedIndex = this.folds.findIndex(
      (fold) => !fold.unfolded && fold.record.handle === handle
    );
    const nested = nestedIndex >= 0 ? this.folds[nestedIndex] : undefined;
    if (!nested || !this.applied.has(handle)) {
      throw new Error(`No active fold ${handle} inside this fork.`);
    }
    for (let i = nestedIndex + 1; i < this.folds.length; i++) {
      const other = this.folds[i]!;
      if (other.unfolded) continue;
      if (other.record.covers?.includes(handle)) {
        throw new Error(
          `${handle} is covered by ${other.record.handle}; unfold ${other.record.handle} first.`
        );
      }
      const ownSpans = nestedFoldSpans(nested);
      const otherSpans = nestedFoldSpans(other);
      const overlaps = ownSpans.some((a) =>
        otherSpans.some((b) => a.first <= b.last && b.first <= a.last)
      );
      if (overlaps) {
        throw new Error(
          `${handle} overlaps active fold ${other.record.handle}; unfold ${other.record.handle} first.`
        );
      }
    }

    const ownMessages = messages.slice(this.prefixLength);
    nested.unfolded = true;
    nested.unfoldedAt = ownMessages.length;
    this.applied.delete(handle);
    const stillOmitted = new Set(
      this.folds.filter((fold) => !fold.unfolded).flatMap((fold) => fold.omit)
    );
    this.omitted.clear();
    for (const idx of stillOmitted) {
      const msg = ownMessages[idx];
      if (msg) this.omitted.add(msg);
    }

    const noteMsg = ownMessages.find((msg) => foldNoteDetails(msg)?.handle === handle);
    if (noteMsg?.role === 'toolResult') {
      const stamp = readStamp(noteMsg);
      const marker = `[unfolded ${handle}]`;
      this.replaced.set(
        noteMsg,
        stamp === undefined ? [{ type: 'text', text: marker }] : withStamp(marker, stamp)
      );
    }
    const record = nested.record;
    const range = formatFoldRange(record);
    return {
      content: [
        {
          type: 'text',
          text: `[unfolded ${handle}: restored ${record.messages} messages, ${range}, ~${formatTokens(record.tokens)} tokens]`
        }
      ],
      details: {
        kind: 'unfold',
        handle,
        since: record.since,
        ...(record.until ? { until: record.until } : {}),
        ...(record.spans && record.spans.length > 1
          ? {
              spans: record.spans.map((span) => ({
                since: span.since,
                ...(span.until ? { until: span.until } : {})
              }))
            }
          : {}),
        messages: record.messages,
        tokens: record.tokens
      }
    };
  }
}
