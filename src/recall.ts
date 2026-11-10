import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { archiveDir, type Thread, ThreadIndex } from './archive.js';
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
    let turn: string | undefined;
    for (const message of thread.messages) {
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
  const start = thread.messages.findIndex((message) => readStamp(message) === stamp);
  const final = thread.messages.findIndex((message) => readStamp(message) === last);
  if (start < 0 || final < start) return [];
  const end = thread.messages.findIndex(
    (message, index) => index > final && readStamp(message) !== undefined
  );
  return thread.messages.slice(start, end < 0 ? undefined : end);
}

/** Turns as they were, each message under a line saying whose it is, of which turn and how long. */
export function formatTurn(handle: string, turn: string, messages: readonly AgentMessage[]) {
  let label = turn;
  return messages
    .map((message) => {
      const stamp = readStamp(message);
      if (stamp !== undefined) label = stampLabel(stamp);
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

/** Which files an archived thread touched have changed since it was archived. */
export async function staleness(
  state: EpisodeState,
  threads: readonly Thread[]
): Promise<Staleness[]> {
  const snapshots = state.snapshots;
  const records = threads.flatMap((thread) =>
    thread.record.treeAfter && thread.record.files.length > 0 ? [thread.record] : []
  );
  if (!snapshots || records.length === 0) return [];

  const now = await snapshots.snapshot();
  if (!now) return [];

  const notes: Staleness[] = [];
  for (const record of records) {
    const changed = await snapshots.diff(record.treeAfter!, now, record.files);
    if (changed && changed.length > 0) {
      notes.push({
        handle: record.handle,
        files: changed.map((change) => change.path),
        line: `${record.handle} predates changes to: ${changed.map((change) => relative(state.cwd, change.path)).join(', ')}. Conclusions about those files may no longer hold.`
      });
    }
  }
  return notes;
}

const RecallParams = Type.Object({
  query: Type.Optional(
    Type.String({
      description:
        'Words to find, such as an error code, symbol, path or a few words of what happened. Wrap it in double quotes to match that exact text instead. Not needed with turn.'
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

export function registerRecall(pi: ExtensionAPI, state: EpisodeState): void {
  pi.registerTool<typeof RecallParams, RecallDetails>({
    name: 'recall',
    label: 'Recall',
    description:
      'Search the archived threads (folded spans and finished forks): tool outputs, your own messages and reasoning, commands and their arguments. Ranked by relevance: messages holding every word come first, in any order, and rare words such as error codes weigh more than common ones. Returns the best line of each match with the thread handle and turn it came from; matches past the limit go to a file. A "quoted" query or regex matches exact text per line instead. Cheap, no model call. Query for the specific thing you need. When the line of a match is not enough, pass its handle and turn without a query to get that turn verbatim; it is not a way to read a whole span back.',
    promptSnippet: 'Search archived threads (folded spans, forks) by keyword or exact text',
    parameters: RecallParams,
    renderCall: recallRenderers.renderCall,
    renderResult: recallRenderers.renderResult,

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const query = params.query ?? '';
      if (!params.turn && !query.trim()) {
        throw new Error('recall needs a query, or a handle and a turn to read that turn back.');
      }
      if (params.turn && !params.handle) {
        throw new Error('turn needs the handle of the thread it is in.');
      }
      const index = new ThreadIndex(ctx.sessionManager, state.forkArchives);
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
        threads.map(async (thread) => ({
          ...thread,
          messages: await Promise.all(thread.messages.map(withFullOutput))
        }))
      );

      if (threads.length === 0) {
        return {
          content: [{ type: 'text', text: 'Nothing is archived on this branch yet.' }],
          details: { query, threads: 0, shown: 0, total: 0 }
        };
      }

      if (params.turn) {
        const thread = threads[0]!;
        const [first, last, ...more] = params.turn.split('-').map(parseStamp);
        const ranged = params.turn.includes('-');
        const valid =
          first !== undefined && more.length === 0 && (!ranged || (last ?? -1) >= first);
        const messages = valid ? turnMessages(thread, first, last) : [];
        if (messages.length === 0) {
          throw new Error(`No turn ${params.turn} in ${thread.handle} (${thread.label}).`);
        }
        const from = stampLabel(first!);
        const turn = last === undefined || last === first ? from : `${from}-${stampLabel(last)}`;
        const text = formatTurn(thread.handle, from, messages);
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
        const stale = await staleness(state, [thread]);
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
        return { content: [{ type: 'text', text: lines.join('\n') }], details };
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
        return {
          content: [
            {
              type: 'text',
              text: `No match for "${query}" in ${threads.length} archived thread(s): ${threads.map((thread) => thread.handle).join(', ')}.`
            }
          ],
          details
        };
      }

      const hitThreads = threads.filter((thread) =>
        matches.some((found) => found.handle === thread.handle)
      );
      details.hits = hitThreads.map((thread) => thread.handle);
      const stale = await staleness(state, hitThreads);
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

      return { content: [{ type: 'text', text: lines.join('\n') }], details };
    }
  });
}
