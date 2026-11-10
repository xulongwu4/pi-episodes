import { relative, resolve } from 'node:path';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { appliedFolds, FORK_TOOL, type ThreadRecord, threadRecords } from './archive.js';
import { registerAsk } from './ask.js';
import { cacheAnchor, placeAnchor } from './cache.js';
import { registerCompactionNotice } from './compaction.js';
import { registerStampDisplay } from './display.js';
import { recordEvent, recordMeta } from './events.js';
import { codemodeOnly, registerFold } from './fold.js';
import { registerFork } from './fork.js';
import { ForkRegistry, registerForksView } from './forks-view.js';
import { FILE_TOOLS } from './messages.js';
import { registerNudge } from './nudge.js';
import { CODEMODE_CALLS, EPISODES_SECTION } from './prompt.js';
import { registerRecall } from './recall.js';
import { registerReminder } from './reminder.js';
import { episodesSettings } from './settings.js';
import { unstampedText, withStamp } from './stamps.js';
import { EpisodeState } from './state.js';
import { refreshStatus } from './ui.js';

const STALE_STAMP = /\s*\[t\d+\]\s*$/;
const MAX_SURFACED_THREADS = 2;

/** Threads to point at from a tool result that touched one of their files. */
interface Cue {
  absolute: string;
  records: ThreadRecord[];
  matched: number;
}

export default function episodes(pi: ExtensionAPI): void {
  const state = new EpisodeState();
  const forks = new ForkRegistry();

  pi.on('session_start', (_event, ctx) => {
    state.reset(ctx);
    recordMeta(pi, ctx);
    refreshStatus(ctx, state);
  });

  pi.on('session_tree', (_event, ctx) => {
    refreshStatus(ctx, state);
  });

  pi.on('session_shutdown', async () => {
    await state.dispose();
  });

  pi.on('before_agent_start', (event) => {
    event.systemPromptOptions.sections.episodes = codemodeOnly(pi)
      ? `${EPISODES_SECTION}\n\n${CODEMODE_CALLS}`
      : EPISODES_SECTION;
  });

  // Text copied back from an earlier message (for example by /tree) carries its old stamp
  // inline; the new message gets its own. Only an exact copy is touched, so a user who types
  // a stamp at the end of a question keeps it.
  pi.on('input', (event, ctx) => {
    const text = event.text.replace(STALE_STAMP, '');
    if (text === event.text) return { action: 'continue' };
    const copied = ctx.sessionManager
      .getEntries()
      .some(
        (entry) =>
          entry.type === 'message' &&
          entry.message.role === 'user' &&
          unstampedText(entry.message.content) === text
      );
    if (!copied) return { action: 'continue' };
    return event.images
      ? { action: 'transform', text, images: event.images }
      : { action: 'transform', text };
  });

  // Stamps are written when the message is persisted, never later, so they never change
  // bytes that are already cached.
  pi.on('message_end', (event, ctx) => {
    const message = event.message;
    // The graph moves with each response, not only at the end of a turn.
    if (message.role === 'assistant') refreshStatus(ctx, state, message);
    if (message.role !== 'user' && message.role !== 'toolResult') return undefined;

    const stamp = state.nextStamp++;
    // Results of a tool batch are saved after every call in it finished, so the tree is final
    // for this stamp. Not awaited: tool calls wait for it below.
    state.snapshots?.recordAt(stamp);
    return { message: { ...message, content: withStamp(message.content, stamp) } };
  });

  // The anchor comes from the model's view, not the raw branch: after a compaction the
  // newest stamped user message may only exist in the summary.
  pi.on('before_provider_request', (event, ctx) => {
    const anchor = episodesSettings(pi.getSettings()).cacheAnchor
      ? cacheAnchor(
          ctx.sessionManager.buildSessionProjection().messages,
          appliedFolds(ctx.sessionManager.getBranch())
        )
      : undefined;
    const placed = placeAnchor(event.payload, anchor);
    state.anchorPlaced = placed.placed;
    return placed.payload;
  });

  // A snapshot still being written must not see the changes of the next tool call.
  pi.on('tool_call', async () => {
    await state.snapshots?.settled();
  });

  // The cue for a folded thread is attached to the tool result that touches one of its
  // files again, so it is written once with that result instead of edited in later. A call a
  // codemode script made has a result the model may never see, so its cue waits for the
  // script's own result.
  const scriptCues = new Map<string, Cue[]>();
  // A script that never returned, for example when the run was aborted, leaves its cues.
  pi.on('agent_settled', () => scriptCues.clear());
  pi.on('session_start', () => scriptCues.clear());
  pi.on('tool_result', (event, ctx) => {
    // A fork's child calls tools under the fork's call too; its reads are not the parent's.
    const cues =
      event.parentToolCallId || event.toolName === FORK_TOOL
        ? []
        : (scriptCues.get(event.toolCallId) ?? []);
    scriptCues.delete(event.toolCallId);
    const path = event.input.path;
    if (FILE_TOOLS.has(event.toolName) && typeof path === 'string' && path) {
      const cue = cueFor(resolve(state.cwd, path), threadRecords(ctx.sessionManager.getBranch()));
      if (cue && event.parentToolCallId) {
        const root = state.rootCall(event.toolCallId);
        scriptCues.set(root, [...(scriptCues.get(root) ?? []), cue]);
        return undefined;
      }
      if (cue) cues.push(cue);
    }

    const lines = cues.flatMap((cue) => surface(cue, event.toolCallId));
    if (lines.length === 0) return undefined;
    const content = [...event.content, { type: 'text' as const, text: lines.join('\n') }];
    // Replacing content alone would drop structured output for programmatic callers.
    return event.structuredContent === undefined
      ? { content }
      : { content, structuredContent: event.structuredContent };
  });

  function cueFor(absolute: string, threads: readonly ThreadRecord[]): Cue | undefined {
    const matched = threads.filter(
      (record) =>
        // Compactions record paths as the tools received them, possibly relative.
        record.files.some((file) => resolve(state.cwd, file) === absolute) &&
        !state.surfaced.has(`${record.handle}:${absolute}`)
    );
    // A cue that points at many threads primes none of them; the newest ones are the likely
    // ones, and the rest stay one recall away.
    const records = matched.slice(-MAX_SURFACED_THREADS);
    return records.length === 0 ? undefined : { absolute, records, matched: matched.length };
  }

  /** The cue's lines, marking its threads surfaced; none for threads a sibling cue already showed. */
  function surface(cue: Cue, toolCallId: string): string[] {
    const records = cue.records.filter(
      (record) => !state.surfaced.has(`${record.handle}:${cue.absolute}`)
    );
    if (records.length === 0) return [];
    const shown = relative(state.cwd, cue.absolute);
    recordEvent(pi, {
      kind: 'cue',
      toolCallId,
      file: shown,
      threads: records.map((record) => ({
        handle: record.handle,
        readOnly: record.kind !== 'compaction' && (record.readOnly?.includes(cue.absolute) ?? false)
      })),
      matched: cue.matched
    });
    return records.map((record) => {
      state.surfaced.add(`${record.handle}:${cue.absolute}`);
      const kind = { fold: 'folded span', fork: 'fork', compaction: 'compacted turns' }[
        record.kind
      ];
      const back =
        record.kind === 'compaction'
          ? `recall("<exact words>", handle "${record.handle}")`
          : `recall, or ask("${record.handle}", "<question>")`;
      return `[episodes] ${shown} was also touched in ${kind} ${record.handle} (${record.headline}). If what happened there matters now: ${back}.`;
    });
  }

  registerFold(pi, state);
  registerRecall(pi, state);
  registerAsk(pi, state);
  registerFork(pi, state, forks);
  registerForksView(pi, forks);
  registerReminder(pi);
  registerNudge(pi);
  registerStampDisplay(pi);
  registerCompactionNotice(pi);
}
