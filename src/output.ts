import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';

/** Key in a tool result's `details` that says where the text it was cut from is kept. */
export const CAPPED_KEY = 'episodesOutput';
/** The status block that opens a codemode result, as codemode's renderer looks for it. */
const SCRIPT_HEADER = /^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n$/;

export interface CappedOutput {
  /** File holding the text the preview was cut from. */
  path: string;
  /** Length of that text. */
  chars: number;
  /** Index of the content block that holds the preview. */
  block: number;
  /** Length of the preview when it was made; another extension can cut it further. */
  preview?: number;
}

type Block = TextContent | ImageContent;

export function cappedOutput(details: unknown): CappedOutput | undefined {
  if (typeof details !== 'object' || details === null) return undefined;
  const capped = (details as Record<string, unknown>)[CAPPED_KEY] as
    | Partial<CappedOutput>
    | undefined;
  return typeof capped?.path === 'string' &&
    typeof capped.chars === 'number' &&
    typeof capped.block === 'number'
    ? (capped as CappedOutput)
    : undefined;
}

/**
 * Where another extension kept the text of a result it cut on arrival, as tool-result-offload
 * records it: the file, and the hash of the text to check the file against.
 */
function offloadedOutput(details: unknown): { path: string; sha256: string } | undefined {
  if (typeof details !== 'object' || details === null) return undefined;
  const offload = (details as Record<string, unknown>).offload as
    | { path?: unknown; sha256?: unknown }
    | undefined;
  return typeof offload?.path === 'string' && typeof offload.sha256 === 'string'
    ? { path: offload.path, sha256: offload.sha256 }
    : undefined;
}

/** How a preview made here names its file; a preview is found again by it. */
function named(path: string): string {
  return `All of this output is in ${path};`;
}

/** One cut of a result: the file with the text, and how to find the preview made from it. */
interface Cut {
  path: string;
  /** The hash the file's text must have, when the record gives one. */
  sha256?: string;
  /** Where the preview was when it was made. */
  block?: number;
  /**
   * The line a preview has in place of the middle, with the blank lines around it. It names
   * the file in the cutter's own words, and its first group is how many characters it says
   * were left out.
   */
  marker: RegExp;
}

function literal(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Cut at a line break when one is in the nearer half, so the preview does not end mid-line.
function head(text: string, chars: number): string {
  const end = text.lastIndexOf('\n', chars);
  return text.slice(0, end > chars / 2 ? end : chars).toWellFormed();
}

function tail(text: string, chars: number): string {
  const start = text.indexOf('\n', text.length - chars);
  return text
    .slice(start >= 0 && start < text.length - chars / 2 ? start + 1 : text.length - chars)
    .toWellFormed();
}

/**
 * Cuts a tool result of `limit` characters or more down to about `keep`: the start and the
 * end, two to one, around a line naming the file that holds all of it. The file is written
 * before the result is changed, so a preview never points at text that was not kept.
 * Undefined when the result is under the limit, or has an image between its text blocks:
 * one preview could not keep that order.
 */
export async function capOutput(
  content: readonly Block[],
  dir: string,
  limit: number,
  keep: number
): Promise<{ content: Block[]; capped: CappedOutput } | undefined> {
  const first = content[0];
  // That block stays where the renderer looks for it; it is not part of the output.
  const header = first?.type === 'text' && SCRIPT_HEADER.test(first.text) ? 1 : 0;
  const texts = content.flatMap((block, index) =>
    index >= header && block.type === 'text' ? [index] : []
  );
  const from = texts[0];
  const to = texts.at(-1);
  if (from === undefined || to === undefined || to - from + 1 !== texts.length) return undefined;
  const text = content
    .slice(from, to + 1)
    .flatMap((block) => (block.type === 'text' ? [block.text] : []))
    .join('\n')
    .toWellFormed();
  // A preview at least half the size of the original would save little and can outgrow it.
  if (text.length < Math.max(limit, 2 * keep)) return undefined;

  const hash = createHash('sha256').update(text).digest('hex').slice(0, 16);
  const path = join(dir, `output-${hash}.txt`);
  await mkdir(dir, { recursive: true });
  // Parallel results with the same text must not share a temporary file.
  const temporary = `${path}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, text);
    await rename(temporary, path);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }

  const start = head(text, Math.floor((keep * 2) / 3));
  const end = tail(text, Math.floor(keep / 3));
  const omitted = text.length - start.length - end.length;
  const preview = `${start}\n\n[episodes] ${omitted} of ${text.length} characters left out here. ${named(path)} grep it or read a slice, not the whole file. recall searches it once it is folded.\n\n${end}`;
  return {
    content: [...content.slice(0, from), { type: 'text', text: preview }, ...content.slice(to + 1)],
    capped: { path, chars: text.length, block: from, preview: preview.length }
  };
}

// A cut that fell inside a surrogate pair left U+FFFD for the half it kept.
function sameCut(shown: string, part: string): boolean {
  return shown === part || shown === part.toWellFormed();
}

/**
 * The text of a block with the preview of `full` in it replaced by `full`. A preview is the
 * start and the end of the text around the marker, as many characters as the marker says were
 * not left out. Only that much is replaced: a block can hold more than the preview, once
 * another cut that took in the preview and a block after it has been undone. Undefined when
 * the block holds no such preview, or the text fits in more than one way.
 */
function putBack(text: string, marker: RegExp, full: string): string | undefined {
  const fits: string[] = [];
  for (const match of text.matchAll(marker)) {
    const before = match.index;
    const after = before + match[0].length;
    const kept = full.length - Number(match[1]);
    for (let start = Math.min(kept, before); start >= 0; start--) {
      const end = kept - start;
      if (end > text.length - after) break;
      // The first and the last character of a preview are those of the text: most places
      // are ruled out by them.
      if (start > 0 && text[before - start] !== full[0]) continue;
      if (end > 0 && text[after + end - 1] !== full.at(-1)) continue;
      if (!sameCut(text.slice(before - start, before), full.slice(0, start))) continue;
      if (!sameCut(text.slice(after, after + end), full.slice(full.length - end))) continue;
      fits.push(text.slice(0, before - start) + full + text.slice(after + end));
    }
  }
  return fits.length === 1 ? fits[0] : undefined;
}

async function cutText(cut: Cut): Promise<string | undefined> {
  try {
    const text = await readFile(cut.path, 'utf8');
    const intact =
      cut.sha256 === undefined || createHash('sha256').update(text).digest('hex') === cut.sha256;
    return intact ? text : undefined;
  } catch {
    return undefined;
  }
}

/**
 * The content with the preview of a cut replaced by the text it was cut from. The path alone
 * does not tell the preview: a later handler can add a block that mentions the file, so it is
 * the block where the start and the end of the text stand around the cutter's own line. Where
 * the preview was made is tried first; several such blocks otherwise are not told apart, and
 * none is taken. Undefined when no block is taken, or the file is gone or not the text.
 */
async function restored(content: readonly Block[], cut: Cut): Promise<Block[] | undefined> {
  const marked = content.flatMap((block, index) =>
    block.type === 'text' && block.text.search(cut.marker) >= 0 ? [{ index, text: block.text }] : []
  );
  if (marked.length === 0) return undefined;
  const full = await cutText(cut);
  if (full === undefined) return undefined;
  const fits = marked.flatMap(({ index, text }) => {
    const whole = putBack(text, cut.marker, full);
    return whole === undefined ? [] : [{ index, text: whole }];
  });
  const fit =
    fits.find(({ index }) => index === cut.block) ?? (fits.length === 1 ? fits[0] : undefined);
  if (!fit) return undefined;
  return content.map((block, index) =>
    index === fit.index ? { type: 'text', text: fit.text } : block
  );
}

/**
 * The message with the text its preview was cut from put back, for searching and reading
 * back an archived thread, whether this extension cut it or tool-result-offload did. When
 * both did, one cut the other's preview: the preview in the message is put back first, and
 * the text that brings holds the preview of the other, with whatever a handler added between
 * the two cuts, which stays. Unchanged when it was not cut, its file is gone, or the file is
 * not the text the other extension recorded.
 */
export async function withFullOutput(message: AgentMessage): Promise<AgentMessage> {
  if (message.role !== 'toolResult') return message;
  const capped = cappedOutput(message.details);
  const offloaded = offloadedOutput(message.details);
  let cuts: Cut[] = [
    ...(capped
      ? [
          {
            path: capped.path,
            block: capped.block,
            marker: new RegExp(
              `\\n\\n\\[episodes\\] (\\d+) of \\d+ characters left out here\\. ${literal(named(capped.path))}[^\\n]*\\n\\n`,
              'g'
            )
          }
        ]
      : []),
    ...(offloaded
      ? [
          {
            ...offloaded,
            marker: new RegExp(
              `\\n\\n\\[tool-result-offload: (\\d+) of \\d+ chars omitted\\.[^\\n]*${literal(offloaded.path)}[^\\n]*\\n\\n`,
              'g'
            )
          }
        ]
      : [])
  ];
  let content: Block[] = message.content;
  for (let left = cuts.length; left > 0; left--) {
    let done: Cut | undefined;
    for (const cut of cuts) {
      const next = await restored(content, cut);
      if (!next) continue;
      content = next;
      done = cut;
      break;
    }
    if (!done) break;
    cuts = cuts.filter((cut) => cut !== done);
  }
  return content === message.content ? message : { ...message, content };
}
