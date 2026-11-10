import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';

/** Key in a tool result's `details` that says where the text it was cut from is kept. */
export const CAPPED_KEY = 'episodesOutput';
/** A first text block up to this long is taken for a tool's status header and left alone. */
const HEADER_CHARS = 200;

export interface CappedOutput {
  /** File holding the text the preview was cut from. */
  path: string;
  /** Length of that text. */
  chars: number;
  /** Index of the content block that holds the preview. */
  block: number;
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
 * Undefined when the result is under the limit.
 */
export async function capOutput(
  content: readonly Block[],
  dir: string,
  limit: number,
  keep: number
): Promise<{ content: Block[]; capped: CappedOutput } | undefined> {
  const first = content[0];
  // codemode's result opens with a short status block that its renderer looks for there.
  const header =
    first?.type === 'text' && first.text.length <= HEADER_CHARS && content.length > 1 ? 1 : 0;
  const body = content.slice(header);
  const text = body
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
  await writeFile(temporary, text);
  await rename(temporary, path);

  const start = head(text, Math.floor((keep * 2) / 3));
  const end = tail(text, Math.floor(keep / 3));
  const omitted = text.length - start.length - end.length;
  const preview = `${start}\n\n[episodes] ${omitted} of ${text.length} characters left out here. All of this output is in ${path}; grep it or read a slice, not the whole file. recall searches it once it is folded.\n\n${end}`;
  return {
    content: [
      ...content.slice(0, header),
      { type: 'text', text: preview },
      ...body.filter((block) => block.type !== 'text')
    ],
    capped: { path, chars: text.length, block: header }
  };
}

/**
 * The message with the text its preview was cut from put back, for searching and reading
 * back an archived thread. Unchanged when it was not cut or its file is gone.
 */
export async function withFullOutput(message: AgentMessage): Promise<AgentMessage> {
  if (message.role !== 'toolResult') return message;
  const capped = cappedOutput(message.details);
  if (!capped || message.content[capped.block]?.type !== 'text') return message;
  let text: string;
  try {
    text = await readFile(capped.path, 'utf8');
  } catch {
    return message;
  }
  return {
    ...message,
    content: message.content.map((block, index) =>
      index === capped.block ? { type: 'text', text } : block
    )
  };
}
