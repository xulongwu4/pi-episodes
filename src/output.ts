import { createHash, randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';

/** Key in a tool result's `details` that says where the text it was cut from is kept. */
export const CAPPED_KEY = 'episodesOutput';
/** The status block that opens a codemode result, as codemode's renderer looks for it. */
const SCRIPT_HEADER = /^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n$/;
/** The line codemode puts before each thing a script printed, when it printed several. */
const SCRIPT_PART = /^==> text (\d+)\/(\d+) <==$/gm;
/** How a preview made here opens; no other text of this extension does. */
export const PREVIEW_OPENS = '[episodes] Only part of this output is shown: ';
/** How codemode opens the output of a script that it cut to its own budget. */
const CODEMODE_CUT =
  /^Warning: truncated output \(original token count: (\d+)\)\nTotal output lines: (\d+)\n\n/;
/** What codemode puts between the start and the end that it kept. */
const CODEMODE_GAP = /…\d+ tokens truncated…/g;
/** The largest file of codemode's that is read to stand for a script's output. */
export const CODEMODE_FULL_BYTES = 16 * 1024 * 1024;
/**
 * The least a part of a script's output keeps when it is cut by itself. With parts too many
 * for that, the output is cut as one text.
 */
const MIN_PART_CHARS = 600;

/** A stretch of a text, from an index up to another. */
type Range = [from: number, to: number];

export interface CappedOutput {
  /** File holding the text the preview was cut from. */
  path: string;
  /** Length of that text. */
  chars: number;
  /** Index of the content block that holds the preview. */
  block: number;
  /** Length of the preview when it was made; another extension can cut it further. */
  preview?: number;
  /**
   * Length of the text as it arrived, when that was less than `chars`: codemode had cut the
   * output to its own budget first, and the text was taken from the file it kept all of it in.
   */
  arrived?: number;
  /** Length of the start of the text that the preview shows. */
  head?: number;
  /**
   * The preview opens with a line that says it is a part (see partLine). Previews made before
   * that line was written have none, and open with the start of the text.
   */
  top?: true;
  /**
   * The file a `read` returned this text of, as the call named it. The preview then points at
   * that file and not at `path`: the copy kept there is what the read returned, which is less
   * than the file when the read was a slice of it or stopped at its size limit.
   */
  source?: string;
  /** The tool had cut its output itself before it arrived, and the preview says so. */
  stopped?: true;
  /** Lengths of the text blocks the text was joined from, when there were several. */
  parts?: number[];
  /**
   * The stretches of the text that the preview of a script's output shows. That preview says
   * first that it is cut, and can leave out more than one stretch; it is found again by
   * building it anew from the text and these.
   */
  shown?: Range[];
}

type Block = TextContent | ImageContent;

export function cappedOutput(details: unknown): CappedOutput | undefined {
  if (typeof details !== 'object' || details === null) return undefined;
  const capped = (details as Record<string, unknown>)[CAPPED_KEY] as
    | Partial<CappedOutput>
    | undefined;
  return typeof capped?.path === 'string' &&
    typeof capped.chars === 'number' &&
    typeof capped.block === 'number' &&
    ['undefined', 'number'].includes(typeof capped.preview) &&
    ['undefined', 'number'].includes(typeof capped.arrived) &&
    ['undefined', 'number'].includes(typeof capped.head) &&
    (capped.top === undefined || capped.top === true) &&
    ['undefined', 'string'].includes(typeof capped.source) &&
    (capped.stopped === undefined || capped.stopped === true) &&
    (capped.parts === undefined ||
      (Array.isArray(capped.parts) && capped.parts.every((part) => typeof part === 'number'))) &&
    (capped.shown === undefined ||
      (Array.isArray(capped.shown) &&
        capped.shown.every(
          (range) =>
            Array.isArray(range) &&
            range.length === 2 &&
            range.every((index) => Number.isInteger(index))
        )))
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

/**
 * How the line in place of the middle says where the rest is; a preview is found again by it.
 * For a file that was read that is the file, and for other output the copy kept of it.
 */
function named(path: string, source?: string): string {
  return source === undefined
    ? `All of this output is in ${path};`
    : `They are in ${source}, the file that was read;`;
}

/** What a preview says about where the rest of its text is (see CappedOutput). */
interface Rest {
  path: string;
  source?: string | undefined;
  stopped?: true | undefined;
}

/** What the opening line of a recorded preview was written from; none when it has no such line. */
function opening(cut: Rest & { top?: true | undefined }): Rest | undefined {
  return cut.top && { path: cut.path, source: cut.source, stopped: cut.stopped };
}

/** One cut of a result: the file with the text, and how to find the preview made from it. */
interface Cut {
  path: string;
  /** The hash the file's text must have, when the record gives one. */
  sha256?: string;
  /** Where the preview was when it was made. */
  block?: number;
  /** How much of the start of the text the preview shows, when the record says. */
  head?: number | undefined;
  /** The preview opens with the line that says it is a part, when the record says. */
  top?: true | undefined;
  /** The file that was read, when the preview names that and not `path`. */
  source?: string | undefined;
  /** The preview says that the tool had cut the output first, when the record says. */
  stopped?: true | undefined;
  /** The lengths of the blocks the text was joined from, when the record says. */
  parts?: readonly number[] | undefined;
  /**
   * The line a preview has in place of the middle, with the blank lines around it. It names
   * the file in the cutter's own words, and its first group is how many characters it says
   * were left out. None for the preview of a script's output, which is found by `shown`.
   */
  marker?: RegExp;
  /** What the preview of a script's output shows of the text (see CappedOutput.shown). */
  shown?: readonly Range[] | undefined;
}

/**
 * The line that opens the preview of a text cut as one. The line in place of the middle is
 * easy to read past: an agent that found no field in the start and the end of a report
 * answered that the report had none. So the preview says first that it is a part, and what
 * that means for an answer. A preview is found again by this line as it is written here: when
 * the words change, the line of before has to stay readable for the sessions that have it.
 *
 * Where it sends the agent for the rest depends on what the text is. The copy kept of a tool's
 * output holds all of that output. Of a file that was read it holds what the read returned,
 * which pi stops at a size limit: an agent sent to the copy of such a read took it for the
 * file and counted 281 of its 400 records, in 36 turns running. So a read is sent to the file
 * itself, and a text that its tool had cut first is said to be less than the whole.
 */
function partLine(kept: number, chars: number, rest: Rest): string {
  const part = `${PREVIEW_OPENS}${kept} of ${chars} characters, its start and its end, with the middle left out at the place marked below. If what you need is not in the part shown, do not answer without it, and do not conclude that it is not there: `;
  if (rest.source === undefined) {
    // The copy holds what arrived. That is all of the output, unless the tool left some out.
    return rest.stopped
      ? `${part}grep ${rest.path}, which holds all that arrived, or read a slice of that file. The tool had cut its output itself before it arrived, and what it left out is not in that file.\n\n`
      : `${part}grep ${rest.path}, which holds all of it, or read a slice of that file.\n\n`;
  }
  const cut = rest.stopped
    ? ' The read tool stopped at its size limit before the end of that file, so this output is not all of the file either.'
    : '';
  return `${part}grep ${rest.source}, the file that was read, or read a slice of it.${cut}\n\n`;
}

/**
 * The path a `read` was called with, when the preview can name it for the file that was read.
 * It is named as the call gave it and no other way, and the agent goes to it with whatever
 * tool it has. So a path is not named when pi's `read`, the `read` of pi-hashline-edit-pro
 * and a shell are known to take it for different files, or when it does not fit in a line:
 * - a blank at either end (pi keeps it as part of the name);
 * - an @ in front (pi drops it, the other read looks for a file of that name);
 * - one of the spaces pi turns into a plain one (U+00A0 and the like);
 * - `file:` in front, whatever follows: pi resolves a URL, the others take it for a relative
 *   path;
 * - a `..` step: pi drops it with the step before it, and so does the other read in a
 *   relative path, where a shell follows a link;
 * - a `/` or `/.` at the end, which pi drops and a shell does not take for a file;
 * - a `~` in front other than `~` alone or `~/`, which only a shell expands;
 * - a line break or other control character; more than 1,024 characters.
 * The copy is named for those. Not seen: pi finds a file whose name is spelled another way
 * (composed or decomposed accents, a curly apostrophe) when the name as given is not there.
 */
export function readSource(path: unknown): string | undefined {
  return typeof path === 'string' &&
    path.length > 0 &&
    path.length <= 1024 &&
    path === path.trim() &&
    !path.startsWith('@') &&
    !/[\p{Cc}\u2028\u2029\u00A0\u2000-\u200A\u202F\u205F\u3000]/u.test(path) &&
    !/^file:/i.test(path) &&
    !/(^|[\\/])\.\.([\\/]|$)/.test(path) &&
    !/[\\/]\.?$/.test(path) &&
    !/^~(?!\/|$)/.test(path)
    ? path
    : undefined;
}

/** True when the details of a result say that its tool cut the output before returning it. */
function cutByTool(details: unknown): boolean {
  const truncation = (details as { truncation?: unknown } | null | undefined)?.truncation;
  return (
    typeof truncation === 'object' &&
    truncation !== null &&
    (truncation as { truncated?: unknown }).truncated === true
  );
}

function literal(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Cut at a line break when one is in the nearer half, so the preview does not end mid-line.
function headEnd(text: string, chars: number): number {
  const end = text.lastIndexOf('\n', chars);
  return end > chars / 2 ? end : chars;
}

function tailStart(text: string, chars: number): number {
  const start = text.indexOf('\n', text.length - chars);
  return start >= 0 && start < text.length - chars / 2 ? start + 1 : text.length - chars;
}

function head(text: string, chars: number): string {
  return text.slice(0, headEnd(text, chars)).toWellFormed();
}

function tail(text: string, chars: number): string {
  return text.slice(tailStart(text, chars)).toWellFormed();
}

/** `keep` shared out among texts of these lengths: a short one gets all it needs, and what it leaves goes to the longer ones. */
function shares(lengths: readonly number[], keep: number): number[] {
  const out = lengths.map(() => 0);
  const order = lengths.map((_, index) => index).sort((a, b) => lengths[a]! - lengths[b]!);
  let left = keep;
  order.forEach((index, position) => {
    const share = Math.min(lengths[index]!, Math.floor(left / (order.length - position)));
    out[index] = share;
    left -= share;
  });
  return out;
}

/**
 * What the preview of a script's output shows. A script prints several things, and the one the
 * agent needs is as likely a short one in the middle as the start or the end of them all: so
 * each thing printed is cut by itself, to its share of `keep`, start and end two to one, and
 * one within its share stays whole.
 */
function scriptRanges(text: string, keep: number): Range[] {
  // Codemode numbers the things 1/n to n/n. A line of the output itself that reads like one
  // of those is out of step with them, and starts nothing; one that happens to be in step
  // cannot be told apart.
  const starts: number[] = [];
  let count: string | undefined;
  for (const match of text.matchAll(SCRIPT_PART)) {
    if (Number(match[1]) !== starts.length + 1 || (count ?? match[2]) !== match[2]) continue;
    count = match[2];
    starts.push(match.index);
  }
  // What stands before the first thing, such as codemode's own warning that it cut the output.
  if (starts[0] !== 0) starts.unshift(0);
  const parts = starts.map((start, index): Range => [start, starts[index + 1] ?? text.length]);
  const each = shares(
    parts.map(([from, to]) => to - from),
    keep
  );
  const tooMany = parts.some(
    ([from, to], index) => to - from > each[index]! && each[index]! < MIN_PART_CHARS
  );
  const pieces: Range[] = tooMany ? [[0, text.length]] : parts;
  const budgets = tooMany ? [keep] : each;
  const ranges: Range[] = [];
  const add = (from: number, to: number) => {
    const last = ranges.at(-1);
    if (last && last[1] === from) last[1] = to;
    else if (to > from) ranges.push([from, to]);
  };
  pieces.forEach(([from, to], index) => {
    const part = text.slice(from, to);
    const budget = budgets[index]!;
    const start = headEnd(part, Math.floor((budget * 2) / 3));
    const end = tailStart(part, Math.floor(budget / 3));
    if (part.length <= budget || start >= end) add(from, to);
    else {
      add(from, from + start);
      add(from + end, to);
    }
  });
  return ranges;
}

/**
 * The preview of a script's output that shows these stretches of it: a line first that says
 * it is cut and where all of it is, then the stretches, with a line wherever something is left
 * out: between them, and before the first or after the last when a small `keep` left nothing
 * of a start or an end. Stretches on record that are not those the preview was made of give a
 * text that is not the preview, and so is found nowhere.
 */
function scriptPreview(text: string, path: string, shown: readonly Range[]): string {
  const kept = shown.reduce((sum, [from, to]) => sum + to - from, 0);
  const pieces: string[] = [];
  let gaps = 0;
  let at = 0;
  for (const [from, to] of [...shown, [text.length, text.length] as const]) {
    if (from > at) {
      pieces.push(`\n\n[episodes] ${from - at} characters left out here.\n\n`);
      gaps++;
    }
    pieces.push(text.slice(from, to).toWellFormed());
    at = to;
  }
  const places = gaps === 1 ? 'the place' : `the ${gaps} places`;
  const body = pieces.join('');
  return `${PREVIEW_OPENS}${kept} of ${text.length} characters, with the rest left out at ${places} marked below. If what you need is not in the part shown, do not answer without it: print less of it (filter in the script), or grep or read a slice of ${path}, which holds all of it. recall searches it once it is folded.\n\n${body}`;
}

/**
 * The result of a script as codemode would have given it had it not cut the output to its own
 * budget (10,000 tokens unless the script asks for another): codemode then keeps the start and
 * the end around a line with the count, and writes all of it to a temporary file that the
 * result's details name. Cut again here, such a result had a preview whose file held only what
 * codemode passed on, with the middle in neither, so the text is taken from codemode's file
 * instead. Only when the result is exactly what codemode writes and the file is the text it was
 * cut from: the counts in its first line, and the start and the end shown. Undefined otherwise,
 * also for a file over `limit` bytes, and the result is then cut as it arrived. `arrived` is
 * the length of the text that did arrive.
 */
export async function scriptOutput(
  content: readonly Block[],
  details: unknown,
  limit = CODEMODE_FULL_BYTES,
  least = 0
): Promise<{ content: Block[]; arrived: number } | undefined> {
  const path =
    typeof details === 'object' && details !== null
      ? (details as Record<string, unknown>).fullOutputPath
      : undefined;
  const [header, output] = content;
  if (typeof path !== 'string' || content.length !== 2) return undefined;
  if (header?.type !== 'text' || !SCRIPT_HEADER.test(header.text) || output?.type !== 'text') {
    return undefined;
  }
  const opening = CODEMODE_CUT.exec(output.text);
  const trailer = `\n\n[Full output: ${path} (read with offset/limit)]`;
  if (!opening || !output.text.endsWith(trailer)) return undefined;
  // No file is read for a result that is left as it arrived.
  if (output.text.length < least) return undefined;
  let full: string;
  try {
    const found = await stat(path);
    if (found.size > limit) return undefined;
    full = await readFile(path, 'utf8');
  } catch {
    return undefined;
  }
  if (Math.ceil(full.length / 4) !== Number(opening[1])) return undefined;
  if (full.split('\n').length !== Number(opening[2])) return undefined;
  const shown = output.text.slice(opening[0].length, output.text.length - trailer.length);
  // The output can hold a line that reads as codemode's: the gap is where the start and the
  // end of the file stand around it. Both are measured once: many such lines cost no more.
  const starts = shared(shown, full, false);
  const ends = shared(shown, full, true);
  for (const gap of shown.matchAll(CODEMODE_GAP)) {
    if (gap.index > starts || shown.length - gap.index - gap[0].length > ends) continue;
    return { content: [header, { type: 'text', text: full }], arrived: output.text.length };
  }
  return undefined;
}

/**
 * How many characters what is shown and the file have in common, from the start or from the
 * end. The file is UTF-8, so it holds U+FFFD where the script printed half a character.
 */
function shared(shown: string, full: string, fromEnd: boolean): number {
  const most = Math.min(shown.length, full.length);
  let count = 0;
  for (; count < most; count++) {
    const at = fromEnd ? shown.length - 1 - count : count;
    const there = full.charCodeAt(fromEnd ? full.length - 1 - count : count);
    if (shown.charCodeAt(at) !== there && !(there === 0xfffd && halved(shown, at))) break;
  }
  return count;
}

/** Whether the text has half a character there, with no other half next to it. */
function halved(text: string, at: number): boolean {
  const high = (unit: number) => unit >= 0xd800 && unit <= 0xdbff;
  const low = (unit: number) => unit >= 0xdc00 && unit <= 0xdfff;
  const unit = text.charCodeAt(at);
  if (high(unit)) return !low(text.charCodeAt(at + 1));
  return low(unit) && !high(text.charCodeAt(at - 1));
}

/**
 * Cuts a tool result of `limit` characters or more down to about `keep`: the start and the
 * end, two to one, around a line naming the file that holds all of it, under a line that says
 * it is a part (see partLine). A script's output is cut thing by thing that it printed, under
 * a line of its own that says so (see scriptRanges). The file
 * is written before the result is changed, so a preview never points at text that was not
 * kept. Undefined when the result is under the limit, or has an image between its text
 * blocks: one preview could not keep that order.
 */
export async function capOutput(
  content: readonly Block[],
  dir: string,
  limit: number,
  keep: number,
  details?: unknown,
  source?: string
): Promise<{ content: Block[]; capped: CappedOutput } | undefined> {
  // A preview at least half the size of the original would save little and can outgrow it.
  const least = Math.max(limit, 2 * keep);
  const rest = { source, stopped: cutByTool(details) };
  // What codemode cut to its own budget first is cut from all of it (see scriptOutput), when
  // what arrived is over the limit and that preview is shorter than what arrived: a result is
  // never made longer here, and one that a script asked to be small is left as it is.
  const whole =
    details === undefined
      ? undefined
      : await scriptOutput(content, details, CODEMODE_FULL_BYTES, least);
  if (whole) {
    // All that the script printed is there: no file was read, and nothing is cut any more.
    const all = { source: undefined, stopped: false };
    const cut = await cutOutput(whole.content, dir, 0, keep, whole.arrived, all);
    if (cut) return { content: cut.content, capped: { ...cut.capped, arrived: whole.arrived } };
  }
  return cutOutput(content, dir, least, keep, Number.POSITIVE_INFINITY, rest);
}

/** Cuts a text of `least` characters or more to a preview shorter than it and than `under`. */
async function cutOutput(
  content: readonly Block[],
  dir: string,
  least: number,
  keep: number,
  under: number,
  rest: { source: string | undefined; stopped: boolean }
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
  const joined = content
    .slice(from, to + 1)
    .flatMap((block) => (block.type === 'text' ? [block.text] : []));
  const text = joined.join('\n').toWellFormed();
  if (text.length < least) return undefined;

  const hash = createHash('sha256').update(text).digest('hex').slice(0, 16);
  const path = join(dir, `output-${hash}.txt`);
  const shown = header === 1 ? scriptRanges(text, keep) : undefined;
  const script = shown && scriptPreview(text, path, shown);
  const start = head(text, Math.floor((keep * 2) / 3));
  const end = tail(text, Math.floor(keep / 3));
  const omitted = text.length - start.length - end.length;
  const said: Rest = { path, source: rest.source, stopped: rest.stopped || undefined };
  const where =
    rest.source === undefined
      ? 'grep it or read a slice, not the whole file. recall searches it once it is folded.'
      : 'grep it or read a slice of it, not the whole file again. recall searches this output once it is folded.';
  const preview = `${partLine(start.length + end.length, text.length, said)}${start}\n\n[episodes] ${omitted} of ${text.length} characters left out here. ${named(path, rest.source)} ${where}\n\n${end}`;
  // The line that says it is cut is long next to a small `keep`.
  if ((script ?? preview).length >= Math.min(text.length, under)) return undefined;
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

  if (shown && script !== undefined) {
    return {
      content: [
        ...content.slice(0, from),
        { type: 'text', text: script },
        ...content.slice(to + 1)
      ],
      capped: {
        path,
        chars: text.length,
        block: from,
        preview: script.length,
        shown,
        ...(joined.length > 1 ? { parts: joined.map((part) => part.length) } : {})
      }
    };
  }
  return {
    content: [...content.slice(0, from), { type: 'text', text: preview }, ...content.slice(to + 1)],
    capped: {
      path,
      chars: text.length,
      block: from,
      preview: preview.length,
      head: start.length,
      top: true,
      ...(rest.source === undefined ? {} : { source: rest.source }),
      ...(rest.stopped ? { stopped: true as const } : {}),
      ...(joined.length > 1 ? { parts: joined.map((part) => part.length) } : {})
    }
  };
}

// A cut that fell inside a surrogate pair left U+FFFD for the half it kept.
function sameCut(shown: string, part: string): boolean {
  return shown === part || shown === part.toWellFormed();
}

/**
 * The text of a block with the preview of `full` in it replaced by `full`. A preview is the
 * start and the end of the text around the marker, as many characters as the marker says were
 * not left out. With the length of the start on record, only that much is replaced: a block
 * can hold more than the preview, once another cut that took in the preview and a block next
 * to it has been undone. Without it the preview has to be all of the block. A preview that
 * opens with the line that says it is a part (`top` is then what the line was written from) is taken only
 * with that line before its start, as it was written, and the line goes with it. Where a preview
 * stands in a longer block is not searched for: text whose start or end repeats fits in more
 * than one place, and the wrong one would take a handler's line for part of the output.
 * Undefined when the text does not fit.
 */
function putBackSpan(
  text: string,
  marker: RegExp,
  full: string,
  head?: number,
  top?: Rest
): { whole: string; head: number; tail: number } | undefined {
  for (const match of text.matchAll(marker)) {
    const before = match.index;
    const after = before + match[0].length;
    const start = head ?? before;
    const end = full.length - Number(match[1]) - start;
    if (head === undefined && end !== text.length - after) continue;
    if (start < 0 || start > before || end < 0 || end > text.length - after) continue;
    if (!sameCut(text.slice(before - start, before), full.slice(0, start))) continue;
    if (!sameCut(text.slice(after, after + end), full.slice(full.length - end))) continue;
    const line = top === undefined ? '' : partLine(start + end, full.length, top);
    const opens = before - start - line.length;
    if (opens < 0 || text.slice(opens, before - start) !== line) continue;
    return {
      whole: text.slice(0, opens) + full + text.slice(after + end),
      head: start,
      tail: end
    };
  }
  return undefined;
}

function putBack(
  text: string,
  marker: RegExp,
  full: string,
  head?: number,
  top?: Rest
): string | undefined {
  return putBackSpan(text, marker, full, head, top)?.whole;
}

/**
 * The blocks a text was joined from, by their recorded lengths. A preview that another
 * extension made before the join is then all of its block again, as that extension left it.
 * The text as one block when the lengths are not those of this text.
 */
function unjoined(text: string, parts: readonly number[] | undefined): string[] {
  if (!parts) return [text];
  const blocks: string[] = [];
  let at = 0;
  for (const length of parts) {
    if (blocks.length > 0 && text[at++] !== '\n') return [text];
    blocks.push(text.slice(at, at + length));
    at += length;
  }
  return at === text.length ? blocks : [text];
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
 * `whole` says that the preview was all of its block.
 */
async function restored(
  content: readonly Block[],
  cut: Cut
): Promise<{ content: Block[]; whole: boolean } | undefined> {
  const { marker, shown } = cut;
  const marked = content.flatMap((block, index) =>
    block.type === 'text' &&
    (marker ? block.text.search(marker) >= 0 : block.text.includes(cut.path))
      ? [{ index, text: block.text }]
      : []
  );
  if (marked.length === 0) return undefined;
  const full = await cutText(cut);
  if (full === undefined) return undefined;
  // The preview of a script's output is known whole from its record: it is looked for as it
  // was written, and taken only where it stands once.
  const written = shown && scriptPreview(full, cut.path, shown);
  const fits = marked.flatMap(({ index, text }) => {
    if (written === undefined) {
      const whole = marker && putBack(text, marker, full, cut.head, opening(cut));
      return whole === undefined ? [] : [{ index, text: whole }];
    }
    const at = text.indexOf(written);
    return at < 0 || at !== text.lastIndexOf(written)
      ? []
      : [{ index, text: text.slice(0, at) + full + text.slice(at + written.length) }];
  });
  const fit =
    fits.find(({ index }) => index === cut.block) ?? (fits.length === 1 ? fits[0] : undefined);
  if (!fit) return undefined;
  // Only a preview that was all of its block is the text as it was joined.
  const whole = fit.text.length === full.length;
  const blocks = whole ? unjoined(full, cut.parts) : [fit.text];
  return {
    content: content.flatMap((block, index): Block[] =>
      index === fit.index ? blocks.map((text) => ({ type: 'text', text })) : [block]
    ),
    whole
  };
}

/**
 * The message with the text its preview was cut from put back, for searching and reading
 * back an archived thread, whether this extension cut it or tool-result-offload did. When
 * both did, the later cut took in the preview of the earlier one, and is undone first: its
 * preview is all of a block, and the earlier preview can be in sight inside it, where putting
 * that back first would leave the block no longer the preview of the later cut. What a
 * handler added between the two cuts stays: in blocks of its own when this extension made
 * the later cut, and in one block with the output when the other did, which keeps no record
 * of the join. Unchanged when it was not cut, its file is gone, or the file is not the text
 * the other extension recorded.
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
            head: capped.head,
            top: capped.top,
            source: capped.source,
            stopped: capped.stopped,
            parts: capped.parts,
            ...(capped.shown
              ? { shown: capped.shown }
              : {
                  marker: new RegExp(
                    `\\n\\n\\[episodes\\] (\\d+) of \\d+ characters left out here\\. ${literal(named(capped.path, capped.source))}[^\\n]*\\n\\n`,
                    'g'
                  )
                })
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
    const found: { cut: Cut; content: Block[]; whole: boolean }[] = [];
    for (const cut of cuts) {
      const next = await restored(content, cut);
      if (next) found.push({ cut, ...next });
    }
    const next = found.find(({ whole }) => whole) ?? found[0];
    if (!next) break;
    content = next.content;
    cuts = cuts.filter((cut) => cut !== next.cut);
  }
  return content === message.content ? message : { ...message, content };
}

const UNTRACKED_CUT_MARKER =
  /\n\n\[(?:episodes\] \d+ of \d+ characters left out here|tool-result-offload: \d+ of \d+ chars omitted)\./;

/**
 * Verifies and returns the complete uncut text blocks of a `read` tool result after any trailing
 * `[tN]` stamp block has been removed. Returns `undefined` when the result holds an image, was
 * cut by an external offloader, carries an untracked cut marker, or had its archived
 * `episodesOutput` file removed or altered on disk.
 */
export type ReadPayloadCoverage =
  | { uncapped: true }
  | { uncapped: false; block: number; head: number; tail: number };

export interface VerifiedRead {
  blocks: TextContent[];
  coverage: ReadPayloadCoverage;
}

export function verifyCompleteRead(
  content: readonly Block[],
  details: unknown
): VerifiedRead | undefined {
  if (content.some((block) => block.type !== 'text')) return undefined;
  const textBlocks = content as readonly TextContent[];
  if (offloadedOutput(details)) return undefined;
  const capped = cappedOutput(details);
  if (!capped) {
    if (textBlocks.some((block) => UNTRACKED_CUT_MARKER.test(block.text))) return undefined;
    return {
      blocks: textBlocks.map((block) => ({ type: 'text', text: block.text })),
      coverage: { uncapped: true }
    };
  }
  let full: string;
  try {
    full = readFileSync(capped.path, 'utf8');
  } catch {
    return undefined;
  }
  if (full.length !== capped.chars) return undefined;
  const nameMatch = /(?:^|[/\\])output-([0-9a-f]{16})\.txt$/.exec(capped.path);
  if (!nameMatch) return undefined;
  const digest = createHash('sha256').update(full).digest('hex');
  if (!digest.startsWith(nameMatch[1]!)) return undefined;
  const marker = new RegExp(
    '\\n\\n\\[episodes\\] (\\d+) of \\d+ characters left out here\\. ' +
      literal(named(capped.path, capped.source)) +
      '[^\\n]*\\n\\n',
    'g'
  );
  const marked = textBlocks.flatMap((block, index) =>
    block.text.search(marker) >= 0 ? [{ index, text: block.text }] : []
  );
  if (marked.length === 0) return undefined;
  const fits = marked.flatMap(({ index, text }) => {
    const span = putBackSpan(text, marker, full, capped.head, opening(capped));
    return span === undefined
      ? []
      : [{ index, text: span.whole, head: span.head, tail: span.tail }];
  });
  const fit =
    fits.find(({ index }) => index === capped.block) ?? (fits.length === 1 ? fits[0] : undefined);
  if (!fit || fit.text.length !== full.length) return undefined;
  const rebuilt = textBlocks.flatMap((block, index): TextContent[] =>
    index === fit.index
      ? unjoined(full, capped.parts).map((text) => ({ type: 'text', text }))
      : [{ type: 'text', text: block.text }]
  );
  if (rebuilt.some((block) => UNTRACKED_CUT_MARKER.test(block.text))) return undefined;
  return {
    blocks: rebuilt,
    coverage: { uncapped: false, block: fit.index, head: fit.head, tail: fit.tail }
  };
}

export function completeReadBlocks(
  content: readonly Block[],
  details: unknown
): TextContent[] | undefined {
  return verifyCompleteRead(content, details)?.blocks;
}
