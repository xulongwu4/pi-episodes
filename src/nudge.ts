import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { AssistantMessage, ImageContent } from '@earendil-works/pi-ai';
import {
  type ExtensionAPI,
  type ExtensionContext,
  estimateTokens as estimateMessageTokens,
  type ProjectedSessionEntry,
  type SessionEntry
} from '@earendil-works/pi-coding-agent';
import { appliedFolds, foldNoteDetails, RUN_END_FOLD_TYPE } from './archive.js';
import {
  codemodeOnly,
  coveredNotes,
  coveredTokens,
  foldBoundary,
  foldInstruction,
  foldLines,
  foldStart,
  mergeStart,
  renderFoldAsk,
  runEndRequest,
  turnOf,
  USER_TEXT_CAP,
  unfoldedBehindNote,
  workSince
} from './fold.js';
import { estimateTokens } from './messages.js';
import { askState, withoutStaleAsks } from './requests.js';
import { episodesSettings } from './settings.js';
import { readStamp, stampLabel } from './stamps.js';

export const NUDGE_TYPE = 'episodes-nudge';

/** What a call gets that the agent makes after the fold it was asked for at the end of a run. */
export const AFTER_RUN_END_FOLD =
  'Not run: the fold you were asked for is made, the request before it is answered, and the next request has not arrived. Make no further call.';

/**
 * Whether the fold asked for at the end of a run is made and nothing new has arrived since.
 * Read from the branch, like the fold's own tie to its request: a fold the agent made by
 * itself or was told to make with /fold does not count, nor one on another branch, nor one
 * that did not take effect, which the agent may try again.
 */
export function afterRunEndFold(branch: readonly SessionEntry[]): boolean {
  const applied = appliedFolds(branch);
  for (let index = branch.length - 1; index >= 0; index--) {
    const entry = branch[index]!;
    if (turnOf(entry) === 'new') return false;
    const fold = entry.type === 'message' ? foldNoteDetails(entry.message) : undefined;
    if (fold?.manual === 'run-end' && applied.has(fold.handle)) return true;
  }
  return false;
}

function* messagesAfter(
  entries: readonly ProjectedSessionEntry[],
  index: number
): Generator<AgentMessage> {
  for (let at = index + 1; at < entries.length; at++) yield* entries[at]!.messages;
}

/**
 * Removals of the requests for a fold that got no answer, for a boundary to write: from then
 * on the session shows them to no one, which covers what `withoutStaleAsks` cannot reach, a
 * compaction's summary among it. When a run has ended, a request that is still open is past
 * answering too.
 */
function staleAskEdits(entries: readonly ProjectedSessionEntry[], ended: boolean) {
  const edits: { type: 'context_edit'; targetId: string; replacement: null }[] = [];
  for (const [index, entry] of entries.entries()) {
    const source = entry.sourceEntry;
    if (
      entry.messages.length === 0 ||
      source.type !== 'custom_message' ||
      source.customType !== RUN_END_FOLD_TYPE
    ) {
      continue;
    }
    const state = askState(messagesAfter(entries, index));
    if (state === 'stale' || (ended && state === 'open')) {
      edits.push({ type: 'context_edit', targetId: source.id, replacement: null });
    }
  }
  return edits;
}

/**
 * A message of the extension that asked for a fold: the nudge, or the request at a run's end
 * once it was answered. One that was not is no decision of the agent about that work, so the
 * nudge at the next prompt is still due.
 */
function asksForFold(
  entry: ProjectedSessionEntry,
  index: number,
  entries: readonly ProjectedSessionEntry[]
): boolean {
  const source = entry.sourceEntry;
  if (source.type !== 'custom_message') return false;
  if (source.customType === NUDGE_TYPE) return true;
  return (
    source.customType === RUN_END_FOLD_TYPE &&
    askState(messagesAfter(entries, index)) === 'answered'
  );
}

/** At most a quarter of the window, so small windows are nudged well before pi compacts. */
export function nudgeThreshold(tokens: number, contextWindow: number | undefined): number {
  return contextWindow ? Math.min(tokens, Math.floor(contextWindow / 4)) : tokens;
}

/**
 * Tokens of work a fold from the earliest stamp could remove that built up since the latest
 * nudge, read from the entries rather than kept in memory, so a resumed session or another
 * branch is measured the same way. With `notes` the earliest stamp after the latest note,
 * without them the earliest after the compaction, for a fold that merges the notes. The system
 * prompt, tool declarations, results a fold cannot reach, such as those of calls made next to
 * a fold call, and what a fold keeps, such as the start of each user message, are not counted.
 * Nor are notes and the responses that called their folds: a fold adds no work, and counted
 * in, a long note brought the next nudge by itself, and the fold that answered it another.
 * That leaves out what a script did next to its fold, whose result is the note: it counts as
 * room the notes take, in `tokensBehindNote`, and leaves with the next merge. What the agent
 * wrote after a fold is work, and only a fold that takes in the note before it removes it.
 */
export function nudgeGrowth(entries: readonly ProjectedSessionEntry[], notes = true): number {
  const start = foldStart(entries, notes);
  if (!start) return 0;
  const nudged = entries.findLastIndex(asksForFold);
  return foldableTokens(workSince(entries, Math.max(start.index, nudged + 1) - 1).left);
}

/**
 * Tokens that a fold from `start`, by default where a fold that covers the notes starts, could
 * free and a fold from the earliest stamp after the latest note could not: the calls that
 * wrote the notes, work a fold that started later left behind them, and what stands between
 * a note and the next stamp, such as the reply that followed its fold, also after the latest
 * note. Not the notes themselves: the fold keeps them as written under its own.
 */
export function tokensBehindNote(
  entries: readonly ProjectedSessionEntry[],
  start = mergeStart(entries)
): number {
  if (!start || start.index > foldBoundary(entries)) return 0;
  const span = entries.slice(start.index, foldStart(entries)?.index);
  return Math.max(0, foldableTokens(span) - coveredTokens(coveredNotes(entries, span)));
}

/**
 * Whether the nudge about work left behind the latest note is due. The agent last decided
 * about that work at the latest nudge or, inside a run, at a later fold that stopped at
 * `until`: its own choice to keep those turns for its next steps, which the next prompt asks
 * about. With no such point the nudge is due at once. After one, it is due once a fold has
 * removed work done since and left this work behind again, or once a quarter of the
 * threshold of work has built up since. A fold that only merges notes removes no work and
 * its note is none, so it does not bring the nudge back: an agent that answered each nudge
 * that way would be asked without end.
 */
function dueBehindNote(
  entries: readonly ProjectedSessionEntry[],
  left: { keptAt: number },
  inRun: boolean,
  quarter: number
): boolean {
  const nudged = entries.findLastIndex(asksForFold);
  const decided = Math.max(nudged, inRun ? left.keptAt : -1);
  if (decided < 0) return true;
  const { folded, left: since } = workSince(entries, decided);
  return folded || foldableTokens(since) >= quarter;
}

/**
 * Says whose stamp a fold of the notes starts at, when it is a note's. Asked to fold from a
 * stamp far back, one model answered more than once that the stamp was behind a compaction
 * and folded its last request instead; the note is something it can find in its context.
 */
function noteAt(entries: readonly ProjectedSessionEntry[], stamp: number): string {
  const applied = appliedFolds(entries.map((entry) => entry.sourceEntry));
  for (const entry of entries) {
    const message = entry.messages[0];
    if (!message || readStamp(message) !== stamp) continue;
    const fold = foldNoteDetails(message);
    return fold && applied.has(fold.handle)
      ? `, the stamp at the end of your note ${fold.handle}`
      : '';
  }
  return '';
}

function foldableTokens(entries: readonly ProjectedSessionEntry[]): number {
  let tokens = 0;
  for (const entry of entries) {
    const source = entry.sourceEntry;
    const role = source.type === 'message' ? source.message.role : undefined;
    const size = estimateTokens(entry.messages);
    if (role === 'user') tokens += Math.max(0, size - USER_TEXT_CAP / 4);
    else if (source.type === 'custom_message' || role === 'assistant' || role === 'toolResult') {
      tokens += size;
    }
  }
  return tokens;
}

/**
 * Where a fold asked for at the end of a run stops, to leave the last request in view: the
 * latest stamp before the last user message, and where that message stands. Undefined when
 * no response stands before it from the start on, which leaves nothing to fold, when the
 * request is too large to keep (over a quarter of the threshold), or when keeping it would
 * leave more in context than the fold removes.
 */
export function keptTail(
  entries: readonly ProjectedSessionEntry[],
  start: { stamp: number; index: number },
  threshold: number
): { until: number; index: number } | undefined {
  const index = entries.findLastIndex(
    (entry) =>
      entry.messages.length > 0 &&
      entry.sourceEntry.type === 'message' &&
      entry.sourceEntry.message.role === 'user'
  );
  // With none shown the slice below would count from the end.
  if (index < 0) return undefined;
  const span = entries.slice(start.index, index);
  // The line of a fold is no response: a fold of a span that holds nothing else is refused.
  const lines = foldLines(entries, appliedFolds(entries.map((entry) => entry.sourceEntry)));
  const responds = span.some(
    (entry) =>
      !lines.has(entry.sourceEntry.id) &&
      entry.messages.some((message) => message.role === 'assistant')
  );
  if (!responds) return undefined;
  const tail = foldableTokens(entries.slice(index));
  const removed = foldableTokens(span) - coveredTokens(coveredNotes(entries, span));
  if (tail > threshold / 4 || tail > removed) return undefined;
  let until = start.stamp;
  for (const entry of span) {
    for (const message of entry.messages) {
      const stamp = readStamp(message);
      if (stamp === undefined || (message.role !== 'user' && message.role !== 'toolResult')) {
        continue;
      }
      if (stamp > until) until = stamp;
    }
  }
  return { until, index };
}

/** The room pi keeps free before it compacts, `compaction.reserveTokens`, read loosely. */
function reserveTokens(pi: ExtensionAPI, model: ExtensionContext['model']): number {
  type Reserve = { reserveTokens?: unknown };
  const compaction = (
    pi.getSettings() as { compaction?: Reserve & { modelOverrides?: Record<string, Reserve> } }
  ).compaction;
  const override = model && compaction?.modelOverrides?.[`${model.provider}/${model.id}`];
  const value = override?.reserveTokens ?? compaction?.reserveTokens;
  return typeof value === 'number' && value >= 0 ? value : 16_384;
}

/** A response pi follows with another request on its own: one that called tools. */
function continues(message: AssistantMessage): boolean {
  return (
    message.stopReason === 'toolUse' ||
    (message.stopReason === 'length' && message.content.some((block) => block.type === 'toolCall'))
  );
}

/**
 * The prompt leaves folding to the model, and in long sessions it rarely folds on its own.
 * Once `foldNudgeTokens` built up since the last fold, compaction or nudge, the next user prompt
 * gets one short message asking for a fold, and so does the next request of a run that grew by
 * twice as much on its own. Once less than `foldNudgeTokens` of room is left before pi compacts,
 * a quarter of it is enough. It names the earliest stamp after the latest note: a fold from a
 * later stamp leaves the work before it in context. A fold that reaches back past notes merges
 * them, and what a later fold left behind them, but rewrites the cache from its start. Two
 * things make that pay. Work that sits behind the latest note unfolded, left there by folds
 * from later stamps or kept by `until`, is sent again with every request: once it reaches
 * `foldNudgeTokens`, or a quarter of that close to compaction, the nudge names the stamp it
 * starts at; `dueBehindNote` says when it comes again.
 * A model that folds only its last request would otherwise not be asked about the work
 * before it until pi compacts; and asked then to fold from the first stamp of a long session,
 * one model did not, where it did fold from the stamp its unfolded work started at. Notes
 * take up room as well: each stays with the call that wrote it, whose argument is the note
 * again, and a session that folds on every nudge can stay far from compaction, where alone
 * they were merged at first. In one of 302 turns, 32 notes were 29k and their calls 20k of
 * the 124k tokens in context at the end. A fold that covers notes keeps them as written
 * (`coveredNotes` says why) and frees the rest. So once a quarter of the threshold that such
 * a fold would free sits up to the first stamp after the latest note, the nudge for new work
 * names `mergeStart`, the earliest note as a rule.
 * It is appended, never edited in, so the cached prefix is untouched.
 *
 * A fold made inside a request cost answers: asked to fold with a new request in front of it,
 * one model then worked from the archive, answered an earlier request or stopped, and kept to
 * it for tens of turns (MEASUREMENTS.md, section 16). So the fold is asked for when a run has
 * ended, by a message of its own that continues the run once: the request is answered, the
 * next one is not there yet, and the prefix is still cached. The extension names the span.
 * It starts where the nudge would, or at work a fold kept behind the latest note, and with
 * `until` it stops before the last request, whose calls and outputs stay in view: with
 * nothing but notes in view the same model looked for its tools again and took a worse one.
 * The next such fold takes those turns in, and with them the note before, which it keeps as
 * written; once notes that it would leave take up room, it starts at the earliest of them.
 * The nudge at the next prompt remains for where this is off, and for a request that got no
 * answer, which is then not sent again.
 */
export function registerNudge(pi: ExtensionAPI): void {
  /** What a fold asked for now would start from, or undefined when none is due. */
  const decide = (
    entries: readonly ProjectedSessionEntry[],
    ctx: ExtensionContext,
    /** At a prompt of the user, between two requests of a run, or when a run has ended. */
    at: 'prompt' | 'run' | 'end',
    /** Tokens about to be added that the context usage does not hold yet. */
    incoming = 0
  ) => {
    const settings = episodesSettings(pi.getSettings());
    if (!settings.foldNudge || !pi.getActiveTools().includes('fold')) return undefined;
    // The window of the model that answers, which a routing model does not report itself.
    const usage = ctx.getContextUsage();
    const threshold = nudgeThreshold(settings.foldNudgeTokens, usage?.contextWindow);
    const pressed =
      usage?.tokens != null &&
      usage.tokens + incoming + threshold >= usage.contextWindow - reserveTokens(pi, ctx.model);
    const left = unfoldedBehindNote(entries);
    const leftTokens = left ? foldableTokens(left.work) : 0;
    const stranded = leftTokens >= (pressed ? threshold / 4 : threshold) ? left : undefined;
    const behind = tokensBehindNote(entries, stranded);
    const merge = stranded !== undefined || behind >= threshold / 4;
    const growth = nudgeGrowth(entries, !merge);
    const inRun = at === 'run';
    const due = stranded
      ? dueBehindNote(entries, stranded, inRun, threshold / 4)
      : growth >= (pressed ? threshold / 4 : (inRun ? 2 : 1) * threshold);
    if (!due) return undefined;
    // A fold at the end of a run takes in what the one before it kept behind its note. Once
    // notes that no such fold took in take up room, it starts where a merge does, as a nudge
    // would: with these requests answered, no nudge comes to ask for that.
    const earliest =
      stranded ??
      (merge ? mergeStart(entries) : ((at === 'end' ? left : undefined) ?? foldStart(entries)));
    if (!earliest) return undefined;
    return { threshold, pressed, stranded, leftTokens, behind, merge, growth, earliest };
  };
  const k = (tokens: number) => Math.round(tokens / 1000);

  const nudge = (
    entries: readonly ProjectedSessionEntry[],
    ctx: ExtensionContext,
    inRun: boolean,
    incoming = 0
  ): string | undefined => {
    const asked = decide(entries, ctx, inRun ? 'run' : 'prompt', incoming);
    if (!asked) return undefined;
    const { pressed, stranded, leftTokens, behind, merge, growth, earliest } = asked;
    const when = inRun ? 'the task' : 'this request';
    const since = stampLabel(earliest.stamp);
    const lead = stranded
      ? `About ${k(leftTokens)}k tokens of work that later folds did not cover are still in your context behind their notes, from ${since} on, and are sent again with every request. `
      : merge
        ? `About ${k(growth)}k tokens of work built up that no fold has removed, since the last compaction or message like this one. `
        : `About ${k(growth)}k tokens of work built up since the last fold, compaction or message like this one. `;
    const start = stranded
      ? `with since "${since}"`
      : merge
        ? `with since "${since}"${noteAt(entries, earliest.stamp)}, early enough to take in all your notes,`
        : `with since "${since}", the earliest stamp you can fold from,`;
    const covering = merge
      ? `a note covering everything from there that your notes do not say already: the fold keeps each note of ${stranded ? 'the folds made since' : 'earlier folds there'} under the new one as you wrote it, and removes the calls that wrote them and what they left behind (about ${k(behind)}k tokens${pressed ? '; pi would compact soon otherwise' : stranded ? '' : ', sent again with every request'})`
      : 'a note covering everything from there';
    return (
      lead +
      `If the work since ${since} is finished and ${when} does not need its raw output, fold it first: ${foldInstruction(pi)}, ${start} and ${covering}; then continue. ` +
      'Do not fold from a later stamp instead: the work before it would stay in your context.'
    );
  };

  /** The request for a fold once a run has ended, with the span the extension picked. */
  const askAtEnd = (
    entries: readonly ProjectedSessionEntry[],
    ctx: ExtensionContext
  ): string | undefined => {
    const asked = decide(entries, ctx, 'end');
    if (!asked) return undefined;
    const { earliest, threshold } = asked;
    const tail = keptTail(entries, earliest, threshold);
    const span = entries.slice(earliest.index, tail?.index);
    const notes = coveredNotes(entries, span);
    const frees = Math.max(0, foldableTokens(span) - coveredTokens(notes));
    const since = `since "${stampLabel(earliest.stamp)}"${noteAt(entries, earliest.stamp)}`;
    const range = tail
      ? `with ${since} and until "${stampLabel(tail.until)}" (the turns after it stay in view for what comes next)`
      : `with ${since}`;
    const covering =
      `a note covering everything ${tail ? 'between them' : 'from there'}` +
      (notes.length > 0
        ? ' that your notes do not say already: the fold keeps each note of earlier folds there under the new one as you wrote it'
        : '');
    const ending = codemodeOnly(pi)
      ? 'When the fold is made, reply with the one word "Folded." and nothing else; the next request comes after that.'
      : 'Do nothing else in this response: the fold ends it, and the next request comes after it.';
    return `The request above is answered. About ${k(frees)}k tokens of finished work are still in your context and would be sent again with every request. Fold that work now, before the next request arrives: ${foldInstruction(pi)}, ${range}, and ${covering}. ${ending}`;
  };

  pi.on('before_agent_start', (event, ctx) => {
    // Extension messages, such as the /fold request, are not the user coming back to the task.
    const messageType = (event as { messageType?: string }).messageType;
    if (messageType !== undefined && messageType !== 'user') return undefined;
    // The prompt is not in the session yet: it adds to the context, but not to what can be folded.
    const { prompt = '', images = [] } = event as { prompt?: string; images?: ImageContent[] };
    const content = nudge(
      ctx.sessionManager.buildSessionProjection().entries,
      ctx,
      false,
      estimateMessageTokens({
        role: 'user',
        content: [{ type: 'text', text: prompt }, ...images],
        timestamp: Date.now()
      })
    );
    return content ? { message: { customType: NUDGE_TYPE, content, display: true } } : undefined;
  });

  // A long run gets no user prompt to nudge on. The preview already holds this turn's folds.
  pi.on('turn_end', (event, ctx) => {
    const entries = event.context.contextEntries;
    const gone = staleAskEdits(entries, false);
    // Not in the responses to a request for a fold at the end of a run: the message would
    // stand between that request and the fold that answers it, or behind that fold, and be
    // taken for something new there.
    const content =
      event.message.role === 'assistant' &&
      continues(event.message) &&
      runEndRequest(entries, entries.length) < 0
        ? nudge(entries, ctx, true)
        : undefined;
    if (gone.length === 0 && !content) return undefined;
    return {
      entries: [
        ...event.entries,
        ...gone,
        ...(content
          ? [{ type: 'custom_message' as const, customType: NUDGE_TYPE, content, display: true }]
          : [])
      ]
    };
  });

  // The request counts as a nudge, so when the reply to it ends the run nothing is due again:
  // growth is counted from it, whether the fold was made or not.
  pi.on('agent_before_settle', (event, ctx) => {
    const entries = event.context.contextEntries;
    const gone = staleAskEdits(entries, true);
    const drafts = [...event.entries, ...gone];
    const unasked = gone.length > 0 ? { entries: drafts } : undefined;
    const setting = episodesSettings(pi.getSettings()).foldAfterRun;
    if (setting === false || (setting !== 'always' && ctx.mode !== 'tui')) return unasked;
    // Only after an answer: a message another extension continues the run with, which is then
    // the last entry, would stand in front of the fold as a new request did. Pi has delivered
    // what was queued before this boundary.
    const last = entries.findLast((entry) => entry.messages.length > 0)?.sourceEntry;
    if (
      last?.type !== 'message' ||
      last.message.role !== 'assistant' ||
      last.message.stopReason !== 'stop'
    ) {
      return unasked;
    }
    const content = askAtEnd(entries, ctx);
    if (!content) return unasked;
    return {
      entries: [
        ...drafts,
        { type: 'custom_message' as const, customType: RUN_END_FOLD_TYPE, content, display: true }
      ],
      continue: true
    };
  });

  // A request that got no answer is not sent again. The boundaries above take it out of the
  // session; this covers the requests made before the next of them.
  pi.on('context', (event) => {
    const messages = withoutStaleAsks(event.messages);
    return messages.length === event.messages.length ? undefined : { messages };
  });

  // A direct fold ends the run it was asked for at the end of. A fold from a script cannot,
  // and nothing is asked of the agent then: in one of twenty such folds of a session it folded
  // the request it had been left, read it back and went on to work on a request that had not
  // come (MEASUREMENTS.md, section 16). So a call made once that fold has taken effect, and
  // before anything new, is not run; a call next to the fold in its response comes before
  // the fold is applied.
  //
  // Pi ends the run when every call of a response was not run. A call that pi refused before
  // it came by here, for arguments that do not validate, keeps the run going for another
  // response; nothing is run in it either. The run is not stopped then: a stop would also
  // take the answer from a message that arrived in the meantime.
  pi.on('tool_call', (_event, ctx) =>
    afterRunEndFold(ctx.sessionManager.getBranch())
      ? { block: true, reason: AFTER_RUN_END_FOLD, terminate: true }
      : undefined
  );

  pi.registerMessageRenderer(NUDGE_TYPE, renderFoldAsk);
  pi.registerMessageRenderer(RUN_END_FOLD_TYPE, renderFoldAsk);
}
