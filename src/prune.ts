import { relative } from 'node:path';
import type { AssistantMessage, ImageContent, TextContent, ToolCall } from '@earendil-works/pi-ai';
import {
  buildSessionProjection,
  type ContextEditEntryDraft,
  type SessionEntry
} from '@earendil-works/pi-coding-agent';
import { FOLD_TOOL, FORK_TOOL, UNFOLD_TOOL } from './archive.js';
import { absolutePath } from './messages.js';
import { completeReadBlocks } from './output.js';
import { retireAssistantContent } from './retire.js';
import { isStampBlock, readStamp, stampLabel, withStamp } from './stamps.js';

/** Surviving assistant turns after a failed call required before trimming its large arguments. */
export const ERROR_PRUNE_AFTER_TURNS = 3;
/** Minimum string argument length in characters before a failed call's argument is trimmed. */
export const ERROR_ARG_MIN_CHARS = 512;
/** Minimum character savings required before replacing an older identical `read` result. */
export const MIN_DUPLICATE_READ_SAVINGS = 64;

const OWN_TOOLS = new Set([FOLD_TOOL, UNFOLD_TOOL, FORK_TOOL, 'recall', 'ask']);
const PRESERVED_ARG_KEYS = new Set(['path', 'file', 'filePath']);
const ALREADY_OMITTED_ARG = /^\[\d+ characters omitted from failed .+ call\]$/;

export interface FoldTailPruneOptions {
  branch: readonly SessionEntry[];
  pending?: readonly { type: string; targetId?: string; replacement?: unknown }[];
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

function readSignature(
  args: Record<string, unknown> | undefined,
  cwd: string
): { key: string; resolvedPath: string } | undefined {
  if (!args || typeof args.path !== 'string' || !args.path.trim()) return undefined;
  const resolvedPath = absolutePath(cwd, args.path);
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
  if (!rel) return '.';
  return rel.startsWith('..') ? resolvedPath : rel;
}

function trimFailedArgValue(
  value: unknown,
  toolName: string,
  key?: string
): { value: unknown; changed: boolean } {
  if (key !== undefined && PRESERVED_ARG_KEYS.has(key)) {
    return { value, changed: false };
  }
  if (typeof value === 'string') {
    if (value.length >= ERROR_ARG_MIN_CHARS && !ALREADY_OMITTED_ARG.test(value)) {
      return {
        value: `[${value.length} characters omitted from failed ${toolName} call]`,
        changed: true
      };
    }
    return { value, changed: false };
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
  if (value !== null && typeof value === 'object') {
    let changed = false;
    const next: Record<string, unknown> = {};
    for (const [entryKey, entryValue] of Object.entries(value as Record<string, unknown>)) {
      const res = trimFailedArgValue(entryValue, toolName, entryKey);
      if (res.changed) changed = true;
      next[entryKey] = res.value;
    }
    return changed ? { value: next, changed: true } : { value, changed: false };
  }
  return { value, changed: false };
}

/**
 * Plans deterministic context edits on the surviving tail strictly after a folded span
 * (`afterEntryId`) and strictly before the assistant response that called `fold` (`beforeEntryId`):
 * 1. Stubs older `read` results when a newer `read` in the surviving tail has identical file/range
 *    arguments and identical verified complete content, keeping the older stamp and pairing.
 * 2. Trims large string arguments on tool calls paired with `isError` results once at least
 *    `ERROR_PRUNE_AFTER_TURNS` later surviving assistant turns follow them in the tail,
 *    composing into a single replacement with `retireAssistantContent`.
 */
export function planFoldTailPruning(options: FoldTailPruneOptions): FoldTailPrunePlan {
  const assistantReplacements = new Map<string, AssistantMessage['content']>();
  const readEdits: ContextEditEntryDraft[] = [];
  const readDeps: ReadPruneDependency[] = [];

  const { branch, pending = [], editOwners, afterEntryId, beforeEntryId, cwd } = options;
  const afterIndex = branch.findIndex((entry) => entry.id === afterEntryId);
  const beforeIndex = branch.findIndex((entry) => entry.id === beforeEntryId);
  if (afterIndex < 0 || beforeIndex <= afterIndex + 1) {
    return { assistantReplacements, readEdits, readDeps };
  }

  const externallyEdited = new Set<string>();
  const pendingNull = new Set<string>();
  for (const entry of branch) {
    if (entry.type === 'context_edit' && !editOwners.has(entry.id)) {
      externallyEdited.add(entry.targetId);
    }
  }
  for (const draft of pending) {
    if (draft.type === 'context_edit' && draft.targetId !== undefined) {
      externallyEdited.add(draft.targetId);
      if (draft.replacement === null) pendingNull.add(draft.targetId);
      else pendingNull.delete(draft.targetId);
    }
  }

  const projection = buildSessionProjection([...branch]);
  const byId = new Map(projection.entries.map((entry) => [entry.sourceEntry.id, entry]));
  const tailSlice = branch.slice(afterIndex + 1, beforeIndex);

  interface TailAssistant {
    entryId: string;
    projected: AssistantMessage;
    externallyEdited: boolean;
  }
  interface TailToolResult {
    entryId: string;
    sourceDetails: unknown;
    projected: Extract<SessionEntry, { type: 'message' }>['message'] & { role: 'toolResult' };
    externallyEdited: boolean;
    order: number;
  }

  const survivingAssistants: TailAssistant[] = [];
  const callsById = new Map<string, { call: ToolCall; assistantExternallyEdited: boolean }>();
  const resultsByCallId = new Map<string, TailToolResult>();
  const tailResults: TailToolResult[] = [];

  for (let i = 0; i < tailSlice.length; i++) {
    const entry = tailSlice[i]!;
    if (entry.type !== 'message' || pendingNull.has(entry.id)) continue;
    const projectedMsg = byId.get(entry.id)?.messages[0];
    if (!projectedMsg) continue;
    const isExternal = externallyEdited.has(entry.id);
    if (entry.message.role === 'assistant' && projectedMsg.role === 'assistant') {
      survivingAssistants.push({
        entryId: entry.id,
        projected: projectedMsg,
        externallyEdited: isExternal
      });
      for (const block of projectedMsg.content) {
        if (block.type === 'toolCall') {
          callsById.set(block.id, { call: block, assistantExternallyEdited: isExternal });
        }
      }
    } else if (entry.message.role === 'toolResult' && projectedMsg.role === 'toolResult') {
      const item: TailToolResult = {
        entryId: entry.id,
        sourceDetails: entry.message.details,
        projected: projectedMsg,
        externallyEdited: isExternal,
        order: i
      };
      resultsByCallId.set(projectedMsg.toolCallId, item);
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
    unstamped: (TextContent | ImageContent)[];
    complete: TextContent[];
  }
  const groups = new Map<string, ReadCandidate[]>();
  for (const item of tailResults) {
    if (item.externallyEdited || item.projected.isError || item.projected.toolName !== 'read') {
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
    const complete = completeReadBlocks(unstamped, item.sourceDetails);
    if (!complete) continue;
    let list = groups.get(sig.key);
    if (!list) {
      list = [];
      groups.set(sig.key, list);
    }
    list.push({
      entryId: item.entryId,
      order: item.order,
      resolvedPath: sig.resolvedPath,
      stamp,
      projectedContent: [...item.projected.content],
      unstamped,
      complete
    });
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
        if (sameTextBlocks(earlier.complete, later.complete)) {
          newestMatch = later;
          break;
        }
      }
      if (!newestMatch) continue;
      const shownPath = displayPath(cwd, earlier.resolvedPath);
      const fullChars = totalChars(earlier.complete);
      const projectedChars = totalChars(earlier.unstamped);
      const laterRef =
        newestMatch.stamp !== undefined ? stampLabel(newestMatch.stamp) : 'a later read';
      const stub = `[duplicate read of ${shownPath}, ${fullChars} chars; identical to ${laterRef}]`;
      if (projectedChars - stub.length < MIN_DUPLICATE_READ_SAVINGS) continue;
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
      if (block.type !== 'toolCall' || OWN_TOOLS.has(block.name)) return block;
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
