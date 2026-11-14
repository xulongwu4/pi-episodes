import { relative } from 'node:path';
import type { AssistantMessage, ImageContent, TextContent, ToolCall } from '@earendil-works/pi-ai';
import {
  buildSessionProjection,
  type ContextEditEntryDraft,
  type SessionEntry
} from '@earendil-works/pi-coding-agent';
import { FOLD_TOOL, FORK_TOOL, UNFOLD_TOOL } from './archive.js';
import { absolutePath } from './messages.js';
import { type ReadPayloadCoverage, verifyCompleteRead } from './output.js';
import { retireAssistantContent } from './retire.js';
import { isStampBlock, readStamp, stampLabel, withStamp } from './stamps.js';

/** Surviving assistant turns after a failed call required before trimming its large arguments. */
export const ERROR_PRUNE_AFTER_TURNS = 3;
/** Minimum string argument length in characters before a failed call's argument is trimmed. */
export const ERROR_ARG_MIN_CHARS = 512;
/** Minimum character savings required before replacing an older identical `read` result. */
export const MIN_DUPLICATE_READ_SAVINGS = 64;

/** Results and tool calls belonging to episodes itself: bounded or standing as notes. */
export const UNCAPPED_TOOLS: ReadonlySet<string> = new Set([
  FOLD_TOOL,
  UNFOLD_TOOL,
  FORK_TOOL,
  'recall',
  'ask'
]);
const PRESERVED_ARG_KEYS = new Set(['path', 'file', 'filePath']);

export interface FoldTailPruneOptions {
  branch: readonly SessionEntry[];
  pending?: readonly { type: string; targetId?: string; replacement?: unknown }[];
  /** Cut-short recovery edits emitted just ahead of this fold in the same `turn_end` batch. */
  repairDrafts?: readonly ContextEditEntryDraft[];
  editOwners: ReadonlyMap<string, { kind: 'fold' | 'unfold'; handle: string }>;
  afterEntryId: string;
  beforeEntryId: string;
  cwd: string;
}

export interface ReadPruneDependency {
  targetId: string;
  newerEntryId: string;
  newerExpected: NonNullable<ContextEditEntryDraft['replacement']>;
  restoreReplacement: NonNullable<ContextEditEntryDraft['replacement']>;
}

export interface FoldTailPrunePlan {
  assistantReplacements: Map<string, AssistantMessage['content']>;
  readEdits: ContextEditEntryDraft[];
  readDeps: ReadPruneDependency[];
}

function stripTrailingStamp(
  content: readonly (TextContent | ImageContent)[]
): (TextContent | ImageContent)[] {
  const last = content.at(-1);
  return last && isStampBlock(last) ? content.slice(0, -1) : [...content];
}

function sameTextBlocks(a: readonly TextContent[], b: readonly TextContent[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i]!.text !== b[i]!.text) return false;
  }
  return true;
}

function totalChars(blocks: readonly (TextContent | ImageContent)[]): number {
  let chars = 0;
  for (const block of blocks) {
    if (block.type === 'text') chars += block.text.length;
  }
  return chars;
}

function payloadCovers(later: ReadPayloadCoverage, earlier: ReadPayloadCoverage): boolean {
  if (later.uncapped) return true;
  if (earlier.uncapped) return false;
  return later.block === earlier.block && later.head >= earlier.head && later.tail >= earlier.tail;
}

function readSignature(
  args: Record<string, unknown> | undefined,
  cwd: string
): { key: string; resolvedPath: string } | undefined {
  if (!args || typeof args.path !== 'string') return undefined;
  const stripped = args.path.trim().replace(/^@/, '');
  if (!stripped) return undefined;
  const resolvedPath = absolutePath(cwd, stripped);
  const rest: Record<string, unknown> = {};
  for (const key of Object.keys(args).sort()) {
    const value = args[key];
    if (key === 'path' || value === undefined) continue;
    if (key === 'offset' && value === 1) continue;
    rest[key] = value;
  }
  return { key: `${resolvedPath}\0${JSON.stringify(rest)}`, resolvedPath };
}

function displayPath(cwd: string, resolvedPath: string): string {
  const rel = relative(cwd, resolvedPath);
  if (rel && !rel.startsWith('..') && !rel.startsWith('/')) return rel;
  return resolvedPath;
}

function trimFailedArgValue(
  value: unknown,
  toolName: string,
  key?: string
): { value: unknown; changed: boolean } {
  if (typeof value === 'string') {
    if (key && PRESERVED_ARG_KEYS.has(key)) return { value, changed: false };
    if (
      value.length < ERROR_ARG_MIN_CHARS ||
      /^\[\d+ characters omitted from failed /.test(value)
    ) {
      return { value, changed: false };
    }
    return {
      value: `[${value.length} characters omitted from failed ${toolName} call]`,
      changed: true
    };
  }
  if (Array.isArray(value)) {
    let changed = false;
    const next = value.map((item) => {
      const res = trimFailedArgValue(item, toolName);
      if (res.changed) changed = true;
      return res.value;
    });
    return changed ? { value: next, changed: true } : { value, changed: false };
  }
  if (value && typeof value === 'object') {
    let changed = false;
    const next: Record<string, unknown> = {};
    for (const [prop, child] of Object.entries(value as Record<string, unknown>)) {
      const res = trimFailedArgValue(child, toolName, prop);
      if (res.changed) changed = true;
      next[prop] = res.value;
    }
    return changed ? { value: next, changed: true } : { value, changed: false };
  }
  return { value, changed: false };
}

/**
 * Plans deterministic cleanup on the surviving tail after a folded span (`afterEntryId`, exclusive)
 * and before the assistant turn that invoked `fold` (`beforeEntryId`, exclusive).
 *
 * Pre-fold prefixes are never inspected or edited.
 */
export function planFoldTailPruning(options: FoldTailPruneOptions): FoldTailPrunePlan {
  const {
    branch,
    pending = [],
    repairDrafts = [],
    editOwners,
    afterEntryId,
    beforeEntryId,
    cwd
  } = options;
  const assistantReplacements = new Map<string, AssistantMessage['content']>();
  const readEdits: ContextEditEntryDraft[] = [];
  const readDeps: ReadPruneDependency[] = [];

  const afterIndex = branch.findIndex((entry) => entry.id === afterEntryId);
  const beforeIndex = branch.findIndex((entry) => entry.id === beforeEntryId);
  if (afterIndex < 0 || beforeIndex < 0 || afterIndex + 1 >= beforeIndex) {
    return { assistantReplacements, readEdits, readDeps };
  }

  const pendingEditedIds = new Set<string>();
  const synthEdits: SessionEntry[] = [];
  let seq = 0;
  let lastParentId: string | null = branch.at(-1)?.id ?? null;
  for (const draft of pending) {
    if (draft.type === 'context_edit' && typeof draft.targetId === 'string') {
      pendingEditedIds.add(draft.targetId);
      const id = `__pending_${seq++}`;
      synthEdits.push({
        type: 'context_edit',
        id,
        parentId: lastParentId,
        timestamp: new Date(0).toISOString(),
        targetId: draft.targetId,
        replacement: draft.replacement as ContextEditEntryDraft['replacement']
      });
      lastParentId = id;
    }
  }
  for (const draft of repairDrafts) {
    const id = `__repair_${seq++}`;
    synthEdits.push({
      type: 'context_edit',
      id,
      parentId: lastParentId,
      timestamp: new Date(0).toISOString(),
      targetId: draft.targetId,
      replacement: draft.replacement
    });
    lastParentId = id;
  }

  const externallyEditedIds = new Set<string>(pendingEditedIds);
  for (const entry of branch) {
    if (entry.type === 'context_edit' && !editOwners.has(entry.id)) {
      externallyEditedIds.add(entry.targetId);
    }
  }

  const effectiveEntries: SessionEntry[] =
    synthEdits.length > 0 ? [...branch, ...synthEdits] : [...branch];
  const projection = buildSessionProjection(effectiveEntries);
  const projectedById = new Map(projection.entries.map((entry) => [entry.sourceEntry.id, entry]));

  interface TailAssistant {
    entryId: string;
    order: number;
    projected: AssistantMessage;
    externallyEdited: boolean;
  }
  interface TailToolResult {
    entryId: string;
    order: number;
    projected: Extract<
      (typeof projection.entries)[number]['messages'][number],
      { role: 'toolResult' }
    >;
    sourceDetails: unknown;
    externallyEdited: boolean;
  }

  const survivingAssistants: TailAssistant[] = [];
  const tailResults: TailToolResult[] = [];
  const callsById = new Map<
    string,
    { assistantId: string; assistantExternallyEdited: boolean; call: ToolCall }
  >();
  const ambiguousCallIds = new Set<string>();
  const resultsByCallId = new Map<string, TailToolResult>();
  const ambiguousResultIds = new Set<string>();

  for (let idx = afterIndex + 1; idx < beforeIndex; idx++) {
    const source = branch[idx]!;
    if (source.type !== 'message') continue;
    const projEntry = projectedById.get(source.id);
    if (!projEntry || projEntry.messages.length === 0) continue;
    const externallyEdited = externallyEditedIds.has(source.id);
    const msg = projEntry.messages[0]!;
    if (source.message.role === 'assistant' && msg.role === 'assistant') {
      survivingAssistants.push({
        entryId: source.id,
        order: idx,
        projected: msg,
        externallyEdited
      });
      for (const block of msg.content) {
        if (block.type !== 'toolCall') continue;
        if (callsById.has(block.id) || ambiguousCallIds.has(block.id)) {
          callsById.delete(block.id);
          ambiguousCallIds.add(block.id);
        } else {
          callsById.set(block.id, {
            assistantId: source.id,
            assistantExternallyEdited: externallyEdited,
            call: block
          });
        }
      }
    } else if (source.message.role === 'toolResult' && msg.role === 'toolResult') {
      const item: TailToolResult = {
        entryId: source.id,
        order: idx,
        projected: msg,
        sourceDetails: source.message.details,
        externallyEdited
      };
      if (resultsByCallId.has(msg.toolCallId) || ambiguousResultIds.has(msg.toolCallId)) {
        resultsByCallId.delete(msg.toolCallId);
        ambiguousResultIds.add(msg.toolCallId);
      } else {
        resultsByCallId.set(msg.toolCallId, item);
      }
      tailResults.push(item);
    }
  }

  // 1. Duplicate read pruning across the surviving tail.
  interface ReadCandidate {
    entryId: string;
    order: number;
    resolvedPath: string;
    stamp: number | undefined;
    projectedContent: (TextContent | ImageContent)[];
    visibleChars: number;
    coverage: ReadPayloadCoverage;
    complete: TextContent[];
  }
  interface RawReadItem {
    item: TailToolResult;
    resolvedPath: string;
    stamp: number | undefined;
    unstamped: (TextContent | ImageContent)[];
  }
  const rawBuckets = new Map<string, RawReadItem[]>();
  for (const item of tailResults) {
    if (
      item.externallyEdited ||
      item.projected.isError ||
      item.projected.toolName !== 'read' ||
      ambiguousCallIds.has(item.projected.toolCallId) ||
      ambiguousResultIds.has(item.projected.toolCallId)
    ) {
      continue;
    }
    const pairedCall = callsById.get(item.projected.toolCallId);
    if (!pairedCall || pairedCall.assistantExternallyEdited || pairedCall.call.name !== 'read') {
      continue;
    }
    const sig = readSignature(
      pairedCall.call.arguments as Record<string, unknown> | undefined,
      cwd
    );
    if (!sig) continue;
    const stamp = readStamp(item.projected);
    const unstamped = stripTrailingStamp(item.projected.content);
    if (
      unstamped.length === 1 &&
      unstamped[0]!.type === 'text' &&
      unstamped[0]!.text.startsWith('[duplicate read of ')
    ) {
      continue;
    }
    let bucket = rawBuckets.get(sig.key);
    if (!bucket) {
      bucket = [];
      rawBuckets.set(sig.key, bucket);
    }
    bucket.push({ item, resolvedPath: sig.resolvedPath, stamp, unstamped });
  }

  const groups = new Map<string, ReadCandidate[]>();
  for (const [key, bucket] of rawBuckets) {
    if (bucket.length < 2) continue;
    const list: ReadCandidate[] = [];
    for (const { item, resolvedPath, stamp, unstamped } of bucket) {
      const verified = verifyCompleteRead(unstamped, item.sourceDetails);
      if (!verified) continue;
      list.push({
        entryId: item.entryId,
        order: item.order,
        resolvedPath,
        stamp,
        projectedContent: [...item.projected.content],
        visibleChars: totalChars(unstamped),
        coverage: verified.coverage,
        complete: verified.blocks
      });
    }
    if (list.length >= 2) groups.set(key, list);
  }

  const pendingReadEdits: {
    order: number;
    edit: ContextEditEntryDraft;
    dep: ReadPruneDependency;
  }[] = [];
  for (const candidates of groups.values()) {
    if (candidates.length < 2) continue;
    for (let i = 0; i < candidates.length - 1; i++) {
      const earlier = candidates[i]!;
      let newestMatch: ReadCandidate | undefined;
      for (let j = candidates.length - 1; j > i; j--) {
        const later = candidates[j]!;
        if (
          payloadCovers(later.coverage, earlier.coverage) &&
          sameTextBlocks(earlier.complete, later.complete)
        ) {
          newestMatch = later;
          break;
        }
      }
      if (!newestMatch) continue;
      const shownPath = displayPath(cwd, earlier.resolvedPath);
      const fullChars = totalChars(earlier.complete);
      const laterRef =
        newestMatch.stamp !== undefined ? stampLabel(newestMatch.stamp) : 'a later read';
      const redirectHint =
        newestMatch.stamp !== undefined
          ? ` (if ${laterRef} is later folded, check that fold note for its handle and call recall with turn "${laterRef}")`
          : '';
      const stub = `[duplicate read of ${shownPath}, ${fullChars} chars; identical to ${laterRef}${redirectHint}]`;
      if (earlier.visibleChars - stub.length < MIN_DUPLICATE_READ_SAVINGS) continue;
      const content =
        earlier.stamp !== undefined
          ? withStamp(stub, earlier.stamp)
          : [{ type: 'text' as const, text: stub }];
      pendingReadEdits.push({
        order: earlier.order,
        edit: { type: 'context_edit', targetId: earlier.entryId, replacement: { content } },
        dep: {
          targetId: earlier.entryId,
          newerEntryId: newestMatch.entryId,
          newerExpected: { content: newestMatch.projectedContent },
          restoreReplacement: { content: earlier.projectedContent }
        }
      });
    }
  }
  pendingReadEdits.sort((a, b) => a.order - b.order);
  for (const { edit, dep } of pendingReadEdits) {
    readEdits.push(edit);
    readDeps.push(dep);
  }

  // 2. Aged failed/aborted tool-call argument pruning.
  for (let i = 0; i < survivingAssistants.length; i++) {
    const laterTurns = survivingAssistants.length - 1 - i;
    if (laterTurns < ERROR_PRUNE_AFTER_TURNS) continue;
    const item = survivingAssistants[i]!;
    if (item.externallyEdited) continue;
    let modified = false;
    const nextBlocks = item.projected.content.map((block) => {
      if (
        block.type !== 'toolCall' ||
        UNCAPPED_TOOLS.has(block.name) ||
        ambiguousCallIds.has(block.id) ||
        ambiguousResultIds.has(block.id)
      ) {
        return block;
      }
      const paired = resultsByCallId.get(block.id);
      if (!paired || paired.externallyEdited || !paired.projected.isError) return block;
      const trimmed = trimFailedArgValue(block.arguments, block.name);
      if (!trimmed.changed) return block;
      modified = true;
      return {
        ...block,
        arguments: trimmed.value as ToolCall['arguments']
      };
    });
    if (!modified) continue;
    const retired = retireAssistantContent(nextBlocks) ?? nextBlocks;
    assistantReplacements.set(item.entryId, retired);
  }

  return { assistantReplacements, readEdits, readDeps };
}
