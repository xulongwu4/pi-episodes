import type { AgentMessage, AgentTool, AgentToolResult } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent, Tool } from '@earendil-works/pi-ai';
import type { ProjectedSessionEntry, SessionEntry } from '@earendil-works/pi-coding-agent';
import { type FoldDetails, foldNoteDetails, type NestedFold, UNFOLD_TOOL } from './archive.js';
import {
  coveredTokens,
  headline,
  openItems,
  planFold,
  renderCovered,
  renderNote,
  retireAssistantMessage,
  retiredResponsesCallIds,
  writtenMark
} from './fold.js';
import { estimateTokens, skillPaths, touchedPaths } from './messages.js';
import { parseStamp, readStamp, stampLabel, withStamp } from './stamps.js';
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
  private readonly omitted = new Set<AgentMessage>();
  private readonly replaced = new Map<AgentMessage, (TextContent | ImageContent)[]>();
  private readonly applied = new Set<string>();
  /** Child assistant message indices whose replay signatures were retired by an earlier fold/unfold. */
  private readonly retiredIndices = new Set<number>();

  constructor(options: {
    forkHandle: string;
    state: EpisodeState;
    parentEntries: readonly ProjectedSessionEntry[];
    prefixLength: number;
    siblings: number;
  }) {
    this.forkHandle = options.forkHandle;
    this.state = options.state;
    this.parentEntries = options.parentEntries;
    this.prefixLength = options.prefixLength;
    this.siblings = options.siblings;
  }

  /** Active folds inside this fork: excludes any that the child unfolded. */
  get activeFolds(): NestedFold[] {
    return this.folds.filter((fold) => !fold.unfolded);
  }

  /** Handles of active folds inside this fork, for prompt-cache anchor placement. */
  activeHandles(): ReadonlySet<string> {
    return this.applied;
  }

  /** Whether any fold or unfold has mutated this child's history. */
  get hasMutated(): boolean {
    return this.retiredIndices.size > 0 || this.folds.length > 0;
  }

  /** What the child sees: everything but the spans it folded, with stale replay signatures retired. */
  view(messages: readonly AgentMessage[]): AgentMessage[] {
    const out: AgentMessage[] = [];
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i]!;
      if (this.omitted.has(message)) continue;
      const ownIndex = i - this.prefixLength;
      const noteReplacement = this.replaced.get(message);
      let current: AgentMessage =
        noteReplacement && message.role === 'toolResult'
          ? { ...message, content: noteReplacement }
          : message;
      if (ownIndex >= 0 && this.retiredIndices.has(ownIndex) && current.role === 'assistant') {
        current = retireAssistantMessage(current);
      }
      out.push(current);
    }
    return out;
  }

  /** Responses `call_id` values from retired child assistant turns (excluding turns added after the latest edit). */
  retiredCallIds(messages: readonly AgentMessage[]): Set<string> {
    const ids = new Set<string>();
    for (const ownIndex of this.retiredIndices) {
      const message = messages[this.prefixLength + ownIndex];
      if (!message || this.omitted.has(message) || message.role !== 'assistant') continue;
      for (const id of retiredResponsesCallIds([retireAssistantMessage(message)])) ids.add(id);
    }
    return ids;
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
      execute: async (toolCallId, args) =>
        declaration.name === UNFOLD_TOOL
          ? this.unfold(args, messages())
          : this.fold(toolCallId, args, messages())
    };
  }

  private entries(messages: readonly AgentMessage[]): ProjectedSessionEntry[] {
    const own = messages.slice(this.prefixLength - this.siblings).map((message, index) => {
      const sibling = index < this.siblings;
      const ownIndex = index - this.siblings;
      const id = sibling ? `${this.forkHandle}/s${index}` : `${this.forkHandle}/${ownIndex}`;
      const noteReplacement = this.replaced.get(message);
      let projected: AgentMessage =
        noteReplacement && message.role === 'toolResult'
          ? { ...message, content: noteReplacement }
          : message;
      if (!sibling && this.retiredIndices.has(ownIndex) && projected.role === 'assistant') {
        projected = retireAssistantMessage(projected);
      }
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

  private collectRetiredFrom(
    ownMessages: readonly AgentMessage[],
    startOwnIndex: number
  ): number[] {
    const retired: number[] = [];
    for (let idx = Math.max(0, startOwnIndex); idx < ownMessages.length; idx++) {
      const message = ownMessages[idx];
      if (!message || this.omitted.has(message) || message.role !== 'assistant') continue;
      this.retiredIndices.add(idx);
      retired.push(idx);
    }
    return retired;
  }

  private async fold(
    toolCallId: string,
    args: unknown,
    messages: readonly AgentMessage[]
  ): Promise<AgentToolResult<FoldDetails>> {
    const params = args as { since?: unknown; until?: unknown; note?: unknown };
    const since = typeof params.since === 'string' ? parseStamp(params.since) : undefined;
    if (since === undefined) {
      throw new Error(`since must be a turn stamp such as "t12", got "${String(params.since)}".`);
    }
    const bounded = typeof params.until === 'string' && params.until.trim() !== '';
    const last = bounded ? parseStamp(params.until as string) : undefined;
    if (bounded && last === undefined) {
      throw new Error(`until must be a turn stamp such as "t20", got "${String(params.until)}".`);
    }
    const note = typeof params.note === 'string' ? params.note.trim() : '';
    if (!note) {
      throw new Error('The note is empty. Write what the rest of the task needs from the span.');
    }

    const entries = this.entries(messages);
    // The task is the first entry after the parent's view and the siblings; the fork's own
    // work starts after it.
    const plan = planFold(entries, toolCallId, since, {
      latestStamp: this.state.nextStamp - 1,
      floorIndex: this.parentEntries.length + this.siblings,
      applied: this.applied,
      ...(last === undefined ? {} : { until: last })
    });

    const ownMessages = messages.slice(this.prefixLength);
    const index = (id: string) => Number(id.slice(this.forkHandle.length + 1));
    const omit: number[] = [];
    for (const edit of plan.edits) {
      // Children have no user messages, so a fold only omits span entries (and retires assistant
      // signatures through `mutatedFrom` below).
      if (edit.replacement !== null) continue;
      const ownIndex = index(edit.targetId);
      const message = ownMessages[ownIndex];
      if (!message) continue;
      this.omitted.add(message);
      omit.push(ownIndex);
    }
    const firstIndex = index(plan.firstEntryId);
    const retired = this.collectRetiredFrom(ownMessages, firstIndex);
    this.applied.add(plan.handle);
    const sinceLabel = stampLabel(since);
    const lastStamp = plan.spanMessages.map(readStamp).findLast((stamp) => stamp !== undefined);
    const until =
      lastStamp !== undefined && lastStamp !== since ? stampLabel(lastStamp) : undefined;
    const touched = touchedPaths(plan.spanMessages, this.state.cwd);
    const tokens = estimateTokens(plan.omitted);
    const record: FoldDetails = {
      kind: 'fold',
      handle: plan.handle,
      since: sinceLabel,
      ...(until ? { until } : {}),
      firstEntryId: plan.firstEntryId,
      lastEntryId: plan.lastEntryId,
      messages: plan.omitted.length,
      tokens,
      headline: headline(note),
      keptUserTurns: plan.keptUserTurns,
      files: touched
    };
    if (plan.covers.length > 0) record.covers = plan.covers.map((fold) => fold.handle);
    record.note = note;
    record.noteTokens = Math.ceil(note.length / 4);
    if (plan.covers.length > 0) record.keptTokens = coveredTokens(plan.covers);
    const carried = renderCovered(plan.covers);
    const keptFrom = plan.keptFrom === undefined ? undefined : stampLabel(plan.keptFrom);
    if (keptFrom) record.keptFrom = keptFrom;
    const open = openItems(note);
    if (open) record.open = open;
    this.folds.push({
      record,
      first: firstIndex,
      last: index(plan.lastEntryId),
      omit,
      createdAt: ownMessages.length,
      retired
    });

    const content = [
      {
        type: 'text' as const,
        text: `${renderNote({
          handle: plan.handle,
          since: sinceLabel,
          until,
          messages: plan.omitted.length,
          tokens,
          note,
          // The fork's own result carries the working tree diff for all of its work.
          changes: undefined,
          touched,
          keptUserTurns: plan.keptUserTurns,
          keptFrom,
          covers: plan.covers,
          skills: skillPaths(plan.spanMessages, this.state.cwd),
          cwd: this.state.cwd
        })}${carried ? `\n\n${carried}` : ''}`
      }
    ];
    record.written = writtenMark(content, carried);
    return { content, details: record };
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
      if (other.first <= nested.last && other.last >= nested.first) {
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
    nested.unfoldRetired = this.collectRetiredFrom(ownMessages, nested.first);
    const record = nested.record;
    const range = record.until ? `${record.since}-${record.until}` : `from ${record.since} on`;
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
        messages: record.messages,
        tokens: record.tokens
      }
    };
  }
}
