import { basename, resolve } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';

const READ_TOOLS = new Set(['read']);
/**
 * Tools that change a file: pi's own, and the anchor tools of pi-hashline-edit-pro. grep is not
 * one, nor a file tool at all: its path is where it searched.
 */
const WRITE_TOOLS = new Set([
  'edit',
  'write',
  'replace',
  'replace_match',
  'insert',
  'copy',
  'move',
  'undo_last_change'
]);
export const FILE_TOOLS = new Set([...READ_TOOLS, ...WRITE_TOOLS]);

/**
 * Details key of a tool result for the files its call changed without naming them in its
 * arguments, and for a script's result the files its calls changed that way.
 */
export const FILES_KEY = 'episodesFiles';

export interface FileTouch {
  name: string;
  path: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * The file a call of a file tool touched. pi's tools name it in their arguments. An anchor tool
 * takes line anchors and names the file only in its structured result, which the session does
 * not keep: `fromResult` then says the path is found later only if it is recorded. A move
 * between two files names the one it moved to. A result that says it failed or changed nothing
 * names no file.
 */
export function fileTouch(
  name: string,
  args: Record<string, unknown> | undefined,
  structured: unknown
): { path: string; fromResult: boolean } | undefined {
  if (!FILE_TOOLS.has(name)) return undefined;
  if (typeof args?.path === 'string' && args.path) return { path: args.path, fromResult: false };
  const path =
    WRITE_TOOLS.has(name) &&
    isRecord(structured) &&
    structured.ok !== false &&
    structured.classification !== 'noop'
      ? structured.path
      : undefined;
  return typeof path === 'string' && path ? { path, fromResult: true } : undefined;
}

/** `details` with these files recorded, each once; undefined when the details cannot hold them. */
export function withFiles(
  details: unknown,
  files: readonly FileTouch[]
): Record<string, unknown> | undefined {
  if (details !== undefined && !isRecord(details)) return undefined;
  const once = files.filter(
    (file, index) =>
      files.findIndex((other) => other.name === file.name && other.path === file.path) === index
  );
  return { ...details, [FILES_KEY]: once };
}

function recordedFiles(details: unknown): FileTouch[] {
  const files = isRecord(details) ? details[FILES_KEY] : undefined;
  if (!Array.isArray(files)) return [];
  return files.filter(
    (file): file is FileTouch =>
      isRecord(file) && typeof file.name === 'string' && typeof file.path === 'string'
  );
}

function blocksText(content: string | readonly (TextContent | ImageContent)[]): string {
  if (typeof content === 'string') {
    return content;
  }

  return content.map((block) => (block.type === 'text' ? block.text : '[image]')).join('\n');
}

/**
 * Everything a message holds as searchable text: tool calls with their arguments, tool
 * outputs and, where the provider returned it, reasoning.
 */
export function messageText(message: AgentMessage): string {
  switch (message.role) {
    case 'user':
    case 'toolResult':
    case 'custom':
      return blocksText(message.content);
    case 'assistant':
      return message.content
        .map((block) => {
          if (block.type === 'text') return block.text;
          if (block.type === 'thinking')
            return block.redacted ? '' : `(thinking) ${block.thinking}`;
          return `${block.name}(${JSON.stringify(block.arguments)})`;
        })
        .filter(Boolean)
        .join('\n');
    case 'bashExecution':
      return `$ ${message.command}\n${message.output}`;
    case 'branchSummary':
    case 'compactionSummary':
      return message.summary;
    default:
      return '';
  }
}

/** Rough token count, only used to tell the user and the model how much a fold saved. */
export function estimateTokens(messages: readonly AgentMessage[]): number {
  let chars = 0;
  for (const message of messages) {
    chars += messageText(message).length;
  }
  return Math.ceil(chars / 4);
}

/**
 * Absolute paths that file tools read or wrote in these messages, including calls a codemode
 * script made. Pi records those on the script's result with bounded arguments, so a call whose
 * arguments were dropped for size is missed. A path that was not in the arguments is the one
 * recorded with the result.
 */
export function touchedPaths(
  messages: readonly AgentMessage[],
  cwd: string,
  tools: ReadonlySet<string> = FILE_TOOLS
): string[] {
  const paths = new Set<string>();
  const add = (name: string, args: Record<string, unknown> | undefined) => {
    const path = args?.path;
    if (tools.has(name) && typeof path === 'string' && path) paths.add(resolve(cwd, path));
  };
  for (const message of messages) {
    if (message.role === 'toolResult') {
      for (const call of message.nestedCalls?.calls ?? []) add(call.name, call.arguments);
      for (const file of recordedFiles(message.details)) add(file.name, { path: file.path });
      continue;
    }
    if (message.role !== 'assistant') continue;
    for (const block of message.content) {
      if (block.type === 'toolCall') add(block.name, block.arguments);
    }
  }
  return [...paths];
}

/** Skill files these messages read. Pi has no skill tool: loading a skill is reading its SKILL.md. */
export function skillPaths(messages: readonly AgentMessage[], cwd: string): string[] {
  return touchedPaths(messages, cwd, READ_TOOLS).filter((path) => basename(path) === 'SKILL.md');
}

/**
 * Files these messages only read: not edited or written by a file tool, and not in the
 * working tree diff when there is one. A cue on such a file points at what was learned, not
 * at what was changed.
 */
export function readOnlyPaths(
  messages: readonly AgentMessage[],
  cwd: string,
  changed: readonly string[]
): string[] {
  const written = new Set([...touchedPaths(messages, cwd, WRITE_TOOLS), ...changed]);
  return touchedPaths(messages, cwd, READ_TOOLS).filter((path) => !written.has(path));
}

export function textOf(content: readonly (TextContent | ImageContent)[]): string {
  return content
    .filter((block): block is TextContent => block.type === 'text')
    .map((block) => block.text)
    .join('\n');
}
