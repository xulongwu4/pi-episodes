import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { archiveDir, archiveDirs, callingFork, type Thread, ThreadIndex } from './archive.js';
import { archivedImagesOf, readArchivedImage, resolveArchivedImagePath } from './images.js';
import { messageText } from './messages.js';
import { withFullOutput } from './output.js';
import { excerpt, RecallIndex, type RecallMatch, type SearchableThread } from './search.js';
import { parseStamp, readStamp, stampLabel } from './stamps.js';
import type { EpisodeState } from './state.js';
import { type RecallDetails, recallRenderers } from './ui.js';

const DEFAULT_LIMIT = 30;
const MAX_LIMIT = 200;
/** How much of a turn read back by its stamp goes into context; the rest goes to a file. */
const READ_BACK_CHARS = 8000;
/** The same for a range of turns: about what three turns would get one by one. */
const READ_BACK_RANGE_CHARS = 24_000;

export function createMatcher(
  query: string,
  regex: boolean
): (line: string) => { index: number; length: number } | undefined {
  if (regex) {
    const pattern = new RegExp(query, 'i');
    return (line) => {
      const match = pattern.exec(line);
      return match ? { index: match.index, length: match[0].length } : undefined;
    };
  }
  const needle = query.toLowerCase();
  return (line) => {
    const index = line.toLowerCase().indexOf(needle);
    return index < 0 ? undefined : { index, length: needle.length };
  };
}

/**
 * Greps every line a thread held, for quoted and regex queries. Lines are attributed to the
 * newest stamp seen so far, which is the turn the model would cite to find its way back.
 */
export function grepThreads(
  threads: readonly SearchableThread[],
  match: (line: string) => { index: number; length: number } | undefined
): RecallMatch[] {
  const matches: RecallMatch[] = [];
  for (const thread of threads) {
    for (const segment of thread.segments ?? [thread.messages]) {
      let turn: string | undefined;
      for (const message of segment) {
        const stamp = readStamp(message);
        for (const line of messageText(message).split('\n')) {
          const hit = match(line);
          if (!hit) continue;
          matches.push({
            handle: thread.handle,
            turn: stamp !== undefined ? stampLabel(stamp) : turn,
            role: message.role,
            line: excerpt(line, hit.index, hit.length)
          });
        }
        if (stamp !== undefined) turn = stampLabel(stamp);
      }
    }
  }
  return matches;
}

export function formatMatch(match: RecallMatch): string {
  return `${match.handle} ${match.turn ?? '-'} ${match.role}: ${match.line}`;
}

/**
 * The messages a turn stands for in recall's matches: the one carrying the stamp and the
 * responses after it, up to the next stamp. With `last`, every turn from `stamp` to that one.
 * Empty when the thread does not hold both.
 */
export function turnMessages(
  thread: SearchableThread,
  stamp: number,
  last = stamp
): AgentMessage[] {
  const segments = thread.segments ?? [thread.messages];
  let startSeg = -1;
  let startIdx = -1;
  let endSeg = -1;
  let endIdx = -1;
  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s]!;
    for (let i = 0; i < seg.length; i++) {
      const found = readStamp(seg[i]!);
      if (found === stamp && startSeg < 0) {
        startSeg = s;
        startIdx = i;
      }
      if (found === last && endSeg < 0) {
        endSeg = s;
        endIdx = i;
      }
    }
  }
  if (startSeg < 0 || endSeg < startSeg || (endSeg === startSeg && endIdx < startIdx)) {
    return [];
  }
  const finalSeg = segments[endSeg]!;
  const nextStamped = finalSeg.findIndex(
    (message, index) => index > endIdx && readStamp(message) !== undefined
  );
  const endCut = nextStamped < 0 ? finalSeg.length : nextStamped;
  if (startSeg === endSeg) return [...finalSeg.slice(startIdx, endCut)];
  return [
    ...segments[startSeg]!.slice(startIdx),
    ...segments.slice(startSeg + 1, endSeg).flatMap((seg) => [...seg]),
    ...finalSeg.slice(0, endCut)
  ];
}

/** Turns as they were, each message under a line saying whose it is, of which turn and how long. */
export function formatTurn(
  handle: string,
  turn: string,
  messages: readonly AgentMessage[],
  segmentStarts?: ReadonlySet<AgentMessage>
) {
  let label = turn;
  return messages
    .map((message) => {
      const stamp = readStamp(message);
      if (stamp !== undefined) label = stampLabel(stamp);
      else if (segmentStarts?.has(message)) label = '-';
      const text = messageText(message);
      const tool = message.role === 'toolResult' ? ` (${message.toolName})` : '';
      return `${handle} ${label} ${message.role}${tool}, ${text.length} chars\n${text}`;
    })
    .join('\n\n');
}

/** What does not fit the result goes to a file the agent can grep or read in part. */
async function spill(dir: string | undefined, name: string, text: string): Promise<string> {
  const target = dir ?? join(tmpdir(), 'pi-episodes');
  await mkdir(target, { recursive: true });
  const file = join(target, name);
  await writeFile(file, `${text}\n`);
  return file;
}

export interface Staleness {
  handle: string;
  /** Absolute paths the thread touched that changed since it was archived. */
  files: string[];
  line: string;
}

function spanCoversStamp(
  spans: readonly { since: string; until?: string | undefined }[],
  idx: number,
  stamp: number
): boolean {
  const span = spans[idx]!;
  const start = parseStamp(span.since);
  if (start === undefined) return false;
  const explicitEnd = span.until ? parseStamp(span.until) : undefined;
  const nextStart = idx + 1 < spans.length ? parseStamp(spans[idx + 1]!.since) : undefined;
  const end = explicitEnd ?? (nextStart !== undefined ? nextStart - 1 : Number.POSITIVE_INFINITY);
  return stamp >= start && stamp <= end;
}

/** Which files an archived thread touched have changed since it was archived. */
export async function staleness(
  state: EpisodeState,
  threads: readonly Thread[],
  relevantStampsByHandle?: ReadonlyMap<string, ReadonlySet<number>>
): Promise<Staleness[]> {
  const snapshots = state.snapshots;
  const records = threads.flatMap((thread) => {
    const rec = thread.record;
    if (rec.files.length === 0) return [];
    const hasSpanTree =
      rec.kind === 'fold' && Boolean(rec.spans?.some((span) => Boolean(span.treeAfter)));
    return rec.treeAfter || hasSpanTree ? [rec] : [];
  });
  if (!snapshots || records.length === 0) return [];

  const now = await snapshots.snapshot();
  if (!now) return [];

  const notes: Staleness[] = [];
  for (const record of records) {
    const filesByTree = new Map<string, string[]>();
    const addFileForTree = (tree: string, file: string) => {
      const group = filesByTree.get(tree) ?? [];
      if (!group.includes(file)) group.push(file);
      filesByTree.set(tree, group);
    };

    if (record.kind === 'fold' && record.spans && record.spans.length > 1) {
      const anySpanMetadata = record.spans.some(
        (span) => span.files !== undefined || span.treeAfter !== undefined
      );
      if (!anySpanMetadata && record.treeAfter) {
        filesByTree.set(record.treeAfter, [...record.files]);
      } else {
        const filterStamps = relevantStampsByHandle?.get(record.handle);
        const matchedSpans = filterStamps
          ? record.spans.filter((_, idx) =>
              [...filterStamps].some((stamp) => spanCoversStamp(record.spans!, idx, stamp))
            )
          : record.spans;
        const targetSpans = matchedSpans.length > 0 ? matchedSpans : record.spans;
        for (const span of targetSpans) {
          if (!span.treeAfter || !span.files?.length) continue;
          for (const file of span.files) addFileForTree(span.treeAfter, file);
        }
      }
    } else if (record.treeAfter) {
      filesByTree.set(record.treeAfter, [...record.files]);
    }
    const changedPaths = new Set<string>();
    for (const [tree, groupFiles] of filesByTree) {
      const changed = await snapshots.diff(tree, now, groupFiles);
      for (const change of changed ?? []) changedPaths.add(change.path);
    }
    const ordered = record.files.filter((file) => changedPaths.has(file));
    if (ordered.length > 0) {
      notes.push({
        handle: record.handle,
        files: ordered,
        line: `${record.handle} predates changes to: ${ordered.map((path) => relative(state.cwd, path)).join(', ')}. Conclusions about those files may no longer hold.`
      });
    }
  }
  return notes;
}

const RecallParams = Type.Object({
  query: Type.Optional(
    Type.String({
      description:
        'Words to find, such as an error code, symbol, path or a few words of what happened. Wrap it in double quotes to match that exact text instead. Not needed with turn or image.'
    })
  ),
  handle: Type.Optional(
    Type.String({ description: 'Only search this archived thread, such as "ep-1a2b3c4d".' })
  ),
  turn: Type.Optional(
    Type.String({
      description:
        'With handle, instead of searching: return what that turn held, verbatim. Use the turn a match named, such as "t14", or a range of turns such as "t14-t16".'
    })
  ),
  image: Type.Optional(
    Type.Integer({
      minimum: 1,
      description:
        'With handle, instead of searching: return that offloaded image (#1, #2, ...) from a covered fold note.'
    })
  ),
  regex: Type.Optional(
    Type.Boolean({
      description: 'Treat query as a case-insensitive regular expression, matched per line.'
    })
  ),
  limit: Type.Optional(
    Type.Integer({
      minimum: 1,
      maximum: MAX_LIMIT,
      description: `Matches to return, default ${DEFAULT_LIMIT}, at most ${MAX_LIMIT}. The rest go to a file.`
    })
  )
});

const RecallOutput = Type.Union([
  Type.String(),
  Type.Object({
    type: Type.Literal('image'),
    data: Type.String(),
    mimeType: Type.String(),
    note: Type.String()
  })
]);

export function registerRecall(pi: ExtensionAPI, state: EpisodeState): void {
  pi.registerTool<typeof RecallParams, RecallDetails>({
    name: 'recall',
    label: 'Recall',
    description:
      'Search the archived threads (folded spans and finished forks): tool outputs, your own messages and reasoning, commands and their arguments. Ranked by relevance: messages holding every word come first, in any order, and rare words such as error codes weigh more than common ones. Returns the best line of each match with the thread handle and turn it came from; matches past the limit go to a file. A "quoted" query or regex matches exact text per line instead. Cheap, no model call. Query for the specific thing you need. When the line of a match is not enough, pass its handle and turn without a query to get that turn verbatim; pass handle and image to read back an offloaded [ep-... image #k] from a covered note.',
    promptSnippet: 'Search archived threads (folded spans, forks) by keyword or exact text',
    parameters: RecallParams,
    outputSchema: RecallOutput,
    renderCall: recallRenderers.renderCall,
    renderResult: recallRenderers.renderResult,

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const query = params.query ?? '';
      const asked = params.turn?.trim();
      const askedImage = params.image;
      if (asked && askedImage !== undefined) {
        throw new Error('recall takes either turn or image, not both.');
      }
      if (askedImage !== undefined && query.trim()) {
        throw new Error('recall takes either query or image, not both.');
      }
      if (!asked && askedImage === undefined && !query.trim()) {
        throw new Error(
          'recall needs a query, or a handle with turn or image to read that item back.'
        );
      }
      if (asked && !params.handle) {
        throw new Error('turn needs the handle of the thread it is in.');
      }
      if (askedImage !== undefined && !params.handle) {
        throw new Error('image needs the handle of the thread it is in.');
      }
      const index = new ThreadIndex(
        ctx.sessionManager,
        state.forkArchives,
        callingFork(ctx.sessionManager)
      );
      if (askedImage !== undefined) {
        const thread = await index.find(params.handle!);
        if (!thread) throw new Error(`No archived thread ${params.handle}.`);
        const archived =
          thread.record.kind === 'fold'
            ? archivedImagesOf(thread.record.archivedImages).find(
                (item) => item.handle === thread.handle && item.index === askedImage
              )
            : undefined;
        if (!archived) {
          throw new Error(
            `No archived image #${askedImage} in ${thread.handle} (${thread.label}).`
          );
        }
        const dirs = await archiveDirs(ctx.sessionManager);
        if (dirs.length === 0) {
          throw new Error(
            `No session archive directory available for ${thread.handle} image #${askedImage}.`
          );
        }
        const loaded = await readArchivedImage(archived, dirs);
        const stale = await staleness(state, [thread]);
        const details: RecallDetails = {
          query,
          threads: 1,
          shown: 1,
          total: 1,
          hits: [thread.handle],
          image: {
            index: archived.index,
            mimeType: archived.mimeType,
            bytes: archived.bytes,
            ...(archived.width !== undefined && archived.height !== undefined
              ? { width: archived.width, height: archived.height }
              : {})
          }
        };
        if (stale.length > 0) {
          details.stale = stale.map(({ handle, files }) => ({ handle, files }));
        }
        const dims =
          archived.width !== undefined && archived.height !== undefined
            ? `, ${archived.width}x${archived.height}`
            : '';
        const livePath = resolveArchivedImagePath(archived, dirs) ?? archived.path;
        const header = `[${archived.handle} image #${archived.index}: ${archived.mimeType}${dims}, ${archived.bytes} bytes, sha256 ${archived.sha256} — ${livePath}]`;
        const note = [...stale.map((item) => item.line), header].join('\n');
        return {
          content: [{ type: 'text', text: note }, loaded],
          structuredContent: {
            type: 'image',
            data: loaded.data,
            mimeType: loaded.mimeType,
            note
          },
          details
        };
      }

      let threads: Thread[];
      if (params.handle) {
        const thread = await index.find(params.handle);
        if (!thread) throw new Error(`No archived thread ${params.handle}.`);
        threads = [thread];
      } else {
        threads = await index.onBranch();
      }
      // What a tool result was cut from when it arrived is part of the record.
      threads = await Promise.all(
        threads.map(async (thread) => {
          if (thread.segments) {
            const segments = await Promise.all(
              thread.segments.map((seg) => Promise.all(seg.map(withFullOutput)))
            );
            return { ...thread, segments, messages: segments.flat() };
          }
          return {
            ...thread,
            messages: await Promise.all(thread.messages.map(withFullOutput))
          };
        })
      );

      if (threads.length === 0) {
        // Read alone, "nothing is archived" was taken to mean that nothing can be known. It says where to
        // look, and not why the archive is empty: another extension's edits and a lost archive file
        // leave it empty too. Nor does it promise a file for a cut result: only some cuts leave one.
        const emptyText =
          'No archived thread is on this branch, so recall has nothing to search. Look in your context instead; for a tool result that arrived cut, follow any pointer its preview gives to the rest.';
        return {
          content: [{ type: 'text', text: emptyText }],
          structuredContent: emptyText,
          details: { query, threads: 0, shown: 0, total: 0 }
        };
      }

      if (asked) {
        const thread = threads[0]!;
        const [first, last, ...more] = asked.split('-').map(parseStamp);
        const ranged = asked.includes('-');
        const valid =
          first !== undefined && more.length === 0 && (!ranged || (last ?? -1) >= first);
        const messages = valid ? turnMessages(thread, first, last) : [];
        if (messages.length === 0) {
          throw new Error(`No turn ${asked} in ${thread.handle} (${thread.label}).`);
        }
        const from = stampLabel(first!);
        const turn = last === undefined || last === first ? from : `${from}-${stampLabel(last)}`;
        const segmentStarts =
          thread.segments && thread.segments.length > 1
            ? new Set(thread.segments.slice(1).flatMap((seg) => (seg[0] ? [seg[0]] : [])))
            : undefined;
        const text = formatTurn(thread.handle, from, messages, segmentStarts);
        const cap = turn === from ? READ_BACK_CHARS : READ_BACK_RANGE_CHARS;
        const details: RecallDetails = {
          query,
          threads: 1,
          shown: messages.length,
          total: messages.length,
          hits: [thread.handle],
          turn,
          chars: text.length
        };
        const askedStamps = new Set<number>();
        for (let s = first!; s <= (last ?? first!); s++) askedStamps.add(s);
        const stale = await staleness(state, [thread], new Map([[thread.handle, askedStamps]]));
        if (stale.length > 0) {
          details.stale = stale.map(({ handle, files }) => ({ handle, files }));
        }
        const lines = stale.map((note) => note.line);
        if (text.length > cap) {
          const file = await spill(
            archiveDir(ctx.sessionManager),
            `${thread.handle}-${turn}.txt`,
            text
          );
          lines.push(
            text.slice(0, cap).toWellFormed(),
            `... ${text.length - cap} more characters. All of ${turn} is in ${file}; grep it or read a slice rather than the whole file.`
          );
          details.file = file;
        } else {
          lines.push(text);
        }
        const out = lines.join('\n');
        return { content: [{ type: 'text', text: out }], structuredContent: out, details };
      }
      const quoted = /^"(.+)"$/s.exec(query.trim())?.[1];
      let all: RecallMatch[];
      let partial = false;
      if (params.regex || quoted !== undefined) {
        let matcher: ReturnType<typeof createMatcher>;
        try {
          matcher = createMatcher(params.regex ? query : quoted!, params.regex === true);
        } catch (error) {
          throw new Error(`Invalid regular expression: ${(error as Error).message}`);
        }
        all = grepThreads(threads, matcher);
      } else {
        // Built on the first recall of the session and only ever added to.
        state.recallIndex ??= new RecallIndex();
        state.recallIndex.add(threads);
        ({ matches: all, partial } = state.recallIndex.search(
          query,
          new Set(threads.map((thread) => thread.handle))
        ));
      }

      const limit = params.limit ?? DEFAULT_LIMIT;
      const matches = all.slice(0, limit);
      const total = all.length;
      const details: RecallDetails = {
        query,
        threads: threads.length,
        shown: matches.length,
        total
      };
      if (total === 0) {
        const none = `No match for "${query}" in ${threads.length} archived thread(s): ${threads.map((thread) => thread.handle).join(', ')}.`;
        return {
          content: [{ type: 'text', text: none }],
          structuredContent: none,
          details
        };
      }

      const hitThreads = threads.filter((thread) =>
        matches.some((found) => found.handle === thread.handle)
      );
      details.hits = hitThreads.map((thread) => thread.handle);
      const relevantStampsByHandle = new Map<string, ReadonlySet<number>>();
      for (const thread of hitThreads) {
        const threadMatches = matches.filter((found) => found.handle === thread.handle);
        const stamps: number[] = [];
        let allStamped = threadMatches.length > 0;
        for (const found of threadMatches) {
          const parsed = found.turn !== undefined ? parseStamp(found.turn) : undefined;
          if (parsed === undefined) {
            allStamped = false;
            break;
          }
          stamps.push(parsed);
        }
        if (allStamped) relevantStampsByHandle.set(thread.handle, new Set(stamps));
      }
      const stale = await staleness(state, hitThreads, relevantStampsByHandle);
      if (stale.length > 0) {
        details.stale = stale.map(({ handle, files }) => ({ handle, files }));
      }
      // Staleness goes first: a cue only works if it is seen before the matches are acted on.
      const lines = [
        ...stale.map((note) => note.line),
        ...(partial ? ['No message holds every word of the query; these hold some of them.'] : []),
        ...matches.map(formatMatch)
      ];
      if (total > matches.length) {
        const file = await spill(
          archiveDir(ctx.sessionManager),
          `recall-${Date.now()}-${query.replace(/[^a-z0-9]+/gi, '-').slice(0, 40)}.txt`,
          all.map(formatMatch).join('\n')
        );
        lines.push(
          `... ${total - matches.length} more matches. All ${total} are in ${file}; grep it or read a slice rather than the whole file. A narrower query is usually better.`
        );
        details.file = file;
      }

      const out = lines.join('\n');
      return { content: [{ type: 'text', text: out }], structuredContent: out, details };
    }
  });
}
