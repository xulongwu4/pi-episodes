import { resolve } from 'node:path';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { SessionEntry } from '@earendil-works/pi-coding-agent';
import {
  appliedFolds,
  FOLD_APPLIED_TYPE,
  FOLD_TOOL,
  FORK_TOOL,
  foldNoteDetails,
  isForkDetails
} from './archive.js';
import { COMPACTION_NOTICE_TYPE } from './compaction.js';
import { type EpisodeEvent, EVENT_TYPE } from './events.js';
import { NUDGE_TYPE } from './nudge.js';
import { cappedOutput } from './output.js';
import { EPISODES_SECTION } from './prompt.js';
import { REMINDER_TYPE } from './reminder.js';
import { isStampBlock } from './stamps.js';
import type { AskDetails, RecallDetails, StaleThread } from './ui.js';

/** The prompt's threshold: folding less than this saves less than the note and cache write cost. */
const SMALL_SPAN_TOKENS = 2000;
/** Responses after a cue or staleness line in which acting on it still counts as following it. */
const FOLLOW_WINDOW = 3;

/**
 * What a cached input token costs next to an uncached one when the session recorded no
 * prices: 0.1 at Anthropic and for OpenAI's current models.
 */
const ASSUMED_CACHED_RATE = 0.1;

/** What cached input and cache writes cost next to uncached input, and where that is from. */
export interface CacheRates {
  cached: number;
  /** 1 where the provider does not bill cache writes apart from input. */
  write: number;
  /** An output token next to an uncached input one; 0 when no costs say, and notes then count as free. */
  output: number;
  /**
   * `costs`: the prices the session's requests were recorded with. `uncached`: no request
   * read anything from cache, so cached input costs what uncached does. `assumed`: neither.
   */
  basis: 'costs' | 'uncached' | 'assumed';
  /** Dollars per uncached input token, when the session recorded costs. */
  inputPrice?: number;
}

const ASSUMED_RATES: CacheRates = {
  cached: ASSUMED_CACHED_RATE,
  write: 1,
  output: 0,
  basis: 'assumed'
};

export interface FoldCost {
  handle: string;
  /** Estimated tokens the fold keeps out of every later request: its span minus its note. */
  net: number;
  /** Input not read from cache on the first request after it; absent before that request. */
  uncachedAfter?: number;
  /** Estimated tokens of its note, which the model wrote as output. */
  note?: number;
  /** The part of it written to the cache, which some providers bill above the input price. */
  writeAfter?: number;
  /** Requests it has been in effect for, up to a compaction. */
  requests: number;
}

/**
 * What a fold cost, in tokens at the input price: the model wrote its note as output, and the
 * next request sent its miss at the full or the cache write rate instead of the cached one. A
 * miss that would have happened without the fold, such as a cache that expired while the user
 * was away, is counted too.
 */
function spent(fold: FoldCost, rates: CacheRates): number {
  const write = fold.writeAfter ?? 0;
  const input = (fold.uncachedAfter ?? 0) - write;
  const resent = input * (1 - rates.cached) + write * (rates.write - rates.cached);
  return Math.max(0, resent) + (fold.note ?? 0) * rates.output;
}

/**
 * Requests a fold must stay in effect for before it has saved more than it cost: every
 * request it is in effect for saves its net tokens at the cached rate. Infinity for a note as
 * long as its span; undefined before the first request.
 */
export function breakEven(fold: FoldCost, rates: CacheRates = ASSUMED_RATES): number | undefined {
  if (fold.uncachedAfter === undefined) return undefined;
  if (fold.net <= 0) return Number.POSITIVE_INFINITY;
  return Math.ceil(spent(fold, rates) / (fold.net * rates.cached));
}

/** The part of a request's usage the rates are read from. */
interface PricedUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cost?: { input: number; output: number; cacheRead: number; cacheWrite: number };
}
/**
 * The rates the session's own requests show. Costs come from pi's price table for the model
 * that answered; with several models the rates are the token-weighted mix.
 */
export function cacheRates(usages: readonly PricedUsage[]): CacheRates {
  const total = { input: 0, output: 0, read: 0, write: 0 };
  const cost = { input: 0, output: 0, read: 0, write: 0 };
  for (const usage of usages) {
    total.input += usage.input;
    total.output += usage.output;
    total.read += usage.cacheRead;
    total.write += usage.cacheWrite;
    cost.input += usage.cost?.input ?? 0;
    cost.output += usage.cost?.output ?? 0;
    cost.read += usage.cost?.cacheRead ?? 0;
    cost.write += usage.cost?.cacheWrite ?? 0;
  }
  const inputPrice = total.input > 0 && cost.input > 0 ? cost.input / total.input : undefined;
  const priced = inputPrice === undefined ? {} : { inputPrice };
  const output =
    inputPrice !== undefined && cost.output > 0 ? cost.output / total.output / inputPrice : 0;
  if (usages.length > 1 && total.read === 0) {
    return { cached: 1, write: 1, output, basis: 'uncached', ...priced };
  }
  if (inputPrice === undefined || cost.read <= 0) return { ...ASSUMED_RATES, output, ...priced };
  return {
    cached: cost.read / total.read / inputPrice,
    write: cost.write > 0 ? cost.write / total.write / inputPrice : 1,
    output,
    basis: 'costs',
    inputPrice
  };
}
export interface RequestPoint {
  /** Tokens the request was sent with: input plus cache read and write. */
  tokens: number;
  /** The first request after a fold or a compaction, where the context should drop. */
  after?: 'fold' | 'compaction';
}

export interface EpisodeStats {
  /** Main-thread requests on the branch, oldest first. */
  requests: RequestPoint[];
  folds: {
    count: number;
    /** By /fold, with the user picking the span start, or the agent picking it. */
    pickedByUser: number;
    requestedByUser: number;
    /** Estimated tokens the folds took out of context. */
    tokens: number;
    spanTokens: number[];
    small: number;
    /** Folds that stopped at `until` and left later turns in the context. */
    bounded: number;
    noteTokens: number;
    /** Folds sharing their response with other tool calls, which the prompt asks against. */
    notAlone: number;
    /** Folds followed directly by the final reply, which the prompt asks against. */
    beforeFinal: number;
    /** Estimated tokens not sent: each fold's span minus its note, on every later request. */
    saved: number;
    rejected: number;
    /** Rejections whose next fold call succeeded. */
    retriedOk: number;
    /**
     * Input not read from cache (uncached input plus cache write) on the first request after
     * each fold: what the fold cost in re-sent prefix. Providers that do not report cache
     * writes bill the whole miss as input, so the sum compares across providers.
     */
    uncachedAfter: number[];
    /** Each fold in the order applied, to tell which have paid for themselves. */
    each: FoldCost[];
    /** What the break-even of each fold is worked out with. */
    rates: CacheRates;
    /** Folds whose request reported on the anchor, and those where it was placed. */
    anchorKnown: number;
    anchored: number;
    /** Only where the provider takes breakpoints; elsewhere there is no anchor to measure. */
    gapTokens: number[];
  };
  /** `readBack`: calls that returned a turn verbatim instead of searching. */
  recall: { count: number; empty: number; compacted: number; readBack: number };
  ask: { count: number };
  stale: { shown: number; reread: number };
  cues: { shown: number; followed: number; readOnly: number; capped: number };
  forks: {
    count: number;
    turns: number;
    folds: number;
    failed: number;
    /** Refused before starting, such as for a context too full to fork from. */
    refused: number;
    cost: number;
  };
  /** Tool results cut on arrival, and the estimated tokens that kept out of context. */
  outputs: { capped: number; tokens: number };
  compactions: number;
  reminders: number;
  nudges: number;
  /** Estimated tokens the extension added to the transcript, and its prompt section. */
  overhead: { stamps: number; cues: number; notices: number; prompt: number };
}

function tokensOf(chars: number): number {
  return Math.ceil(chars / 4);
}

function message(entry: SessionEntry | undefined): AgentMessage | undefined {
  return entry?.type === 'message' ? entry.message : undefined;
}

function toolCalls(entry: SessionEntry | undefined): { id: string; name: string; path?: string }[] {
  const found = message(entry);
  if (found?.role !== 'assistant') return [];
  return found.content.flatMap((block) =>
    block.type === 'toolCall'
      ? [
          {
            id: block.id,
            name: block.name,
            ...(typeof block.arguments.path === 'string' ? { path: block.arguments.path } : {})
          }
        ]
      : []
  );
}

/** Tool calls made by the next `count` responses after `index`. */
function callsAfter(
  entries: readonly SessionEntry[],
  index: number,
  count: number
): { id: string; name: string; path?: string }[] {
  const calls: { id: string; name: string; path?: string }[] = [];
  let responses = 0;
  for (const entry of entries.slice(index + 1)) {
    if (message(entry)?.role !== 'assistant') continue;
    calls.push(...toolCalls(entry));
    if (++responses >= count) break;
  }
  return calls;
}

function textChars(found: AgentMessage, keep: (text: string) => boolean): number {
  if (found.role !== 'user' && found.role !== 'toolResult' && found.role !== 'custom') return 0;
  if (typeof found.content === 'string') return keep(found.content) ? found.content.length : 0;
  return found.content.reduce(
    (sum, block) => sum + (block.type === 'text' && keep(block.text) ? block.text.length : 0),
    0
  );
}

/**
 * Numbers for one branch of a session, read from its entries alone: the tool results carry
 * the fold, fork, recall and ask records, and `episodes-event` entries the rest. Nothing is
 * kept in memory, so a resumed or older session reads the same way, and a script outside pi
 * can call this on a parsed session file.
 */
export function episodeStats(entries: readonly SessionEntry[], cwd: string): EpisodeStats {
  const stats: EpisodeStats = {
    requests: [],
    folds: {
      count: 0,
      pickedByUser: 0,
      requestedByUser: 0,
      tokens: 0,
      spanTokens: [],
      small: 0,
      bounded: 0,
      noteTokens: 0,
      notAlone: 0,
      beforeFinal: 0,
      saved: 0,
      rejected: 0,
      retriedOk: 0,
      uncachedAfter: [],
      each: [],
      rates: ASSUMED_RATES,
      anchorKnown: 0,
      anchored: 0,
      gapTokens: []
    },
    recall: { count: 0, empty: 0, compacted: 0, readBack: 0 },
    ask: { count: 0 },
    stale: { shown: 0, reread: 0 },
    cues: { shown: 0, followed: 0, readOnly: 0, capped: 0 },
    forks: { count: 0, turns: 0, folds: 0, failed: 0, refused: 0, cost: 0 },
    outputs: { capped: 0, tokens: 0 },
    compactions: 0,
    reminders: 0,
    nudges: 0,
    overhead: { stamps: 0, cues: 0, notices: 0, prompt: tokensOf(EPISODES_SECTION.length) }
  };

  const applied = appliedFolds(entries);
  const foldNet = new Map<string, number>();
  const foldNote = new Map<string, number>();
  /** Net tokens of folds in effect, saved again on every request until a compaction. */
  let active = 0;
  let mark: RequestPoint['after'];
  /** Folds still saving on each request, and the one whose first request has not come yet. */
  let inEffect: FoldCost[] = [];
  let awaiting: FoldCost | undefined;
  const usages: PricedUsage[] = [];

  /** A recall that matched the thread, or an ask of it, within the next few responses. */
  const followed = (index: number, handle: string) => {
    const ids = new Set(callsAfter(entries, index, FOLLOW_WINDOW).map((call) => call.id));
    return entries.some((entry) => {
      const found = message(entry);
      if (found?.role !== 'toolResult' || found.isError || !ids.has(found.toolCallId)) return false;
      if (found.toolName === 'ask') {
        return (found.details as AskDetails | undefined)?.handle === handle;
      }
      if (found.toolName === 'recall') {
        return (found.details as RecallDetails | undefined)?.hits?.includes(handle) ?? false;
      }
      return false;
    });
  };

  const countStale = (stale: readonly StaleThread[] | undefined, index: number) => {
    if (!stale || stale.length === 0) return;
    stats.stale.shown += stale.length;
    const reads = callsAfter(entries, index, FOLLOW_WINDOW)
      .filter((call) => call.name === 'read' && call.path)
      .map((call) => resolve(cwd, call.path!));
    for (const thread of stale) {
      if (thread.files.some((file) => reads.includes(file))) stats.stale.reread++;
    }
  };

  entries.forEach((entry, index) => {
    if (entry.type === 'compaction') {
      stats.compactions++;
      active = 0;
      inEffect = [];
      mark = 'compaction';
      return;
    }
    if (entry.type === 'custom' && entry.customType === FOLD_APPLIED_TYPE) {
      const handle = (entry.data as { handle?: string } | undefined)?.handle;
      active += (handle && foldNet.get(handle)) || 0;
      if (handle) {
        awaiting = {
          handle,
          net: foldNet.get(handle) ?? 0,
          note: foldNote.get(handle) ?? 0,
          requests: 0
        };
        stats.folds.each.push(awaiting);
        inEffect.push(awaiting);
      }
      mark = 'fold';
      return;
    }
    if (entry.type === 'custom' && entry.customType === EVENT_TYPE) {
      const event = entry.data as EpisodeEvent;
      if (event.kind === 'fold-rejected') {
        stats.folds.rejected++;
        // Only one fold per response, so a retry is a fold in a later response; results of
        // this one, the rejected call's own or a script's that folded first, do not count.
        let laterResponse = false;
        const next = entries.slice(index + 1).find((later) => {
          const found = message(later);
          // A later rejection is the next attempt, and it failed.
          if (later.type === 'custom' && later.customType === EVENT_TYPE) {
            return (later.data as EpisodeEvent).kind === 'fold-rejected';
          }
          if (found?.role === 'assistant') laterResponse = true;
          if (!laterResponse || found?.role !== 'toolResult') return false;
          if (found.toolName === FOLD_TOOL) return true;
          const fold = foldNoteDetails(found);
          return fold !== undefined && applied.has(fold.handle);
        });
        const result = message(next);
        if (result?.role === 'toolResult' && !result.isError) stats.folds.retriedOk++;
      } else if (event.kind === 'cue') {
        stats.cues.shown += event.threads.length;
        stats.cues.readOnly += event.threads.filter((thread) => thread.readOnly).length;
        if (event.matched > event.threads.length) stats.cues.capped++;
        for (const thread of event.threads) {
          if (followed(index, thread.handle)) stats.cues.followed++;
        }
      }
      return;
    }
    if (entry.type === 'custom_message') {
      if (entry.customType === REMINDER_TYPE) stats.reminders++;
      if (entry.customType === NUDGE_TYPE) stats.nudges++;
      if (
        entry.customType === REMINDER_TYPE ||
        entry.customType === NUDGE_TYPE ||
        entry.customType === COMPACTION_NOTICE_TYPE
      ) {
        const content = entry.content;
        stats.overhead.notices += tokensOf(
          typeof content === 'string'
            ? content.length
            : content.reduce(
                (sum, block) => sum + (block.type === 'text' ? block.text.length : 0),
                0
              )
        );
      }
      return;
    }

    const found = message(entry);
    if (!found) return;

    if (found.role === 'assistant') {
      const usage = found.usage;
      const tokens = usage.input + usage.cacheRead + usage.cacheWrite;
      if (tokens === 0) return;
      usages.push(usage);
      stats.requests.push(mark ? { tokens, after: mark } : { tokens });
      if (mark === 'fold') {
        stats.folds.uncachedAfter.push(usage.input + usage.cacheWrite);
        if (awaiting) {
          awaiting.uncachedAfter = usage.input + usage.cacheWrite;
          awaiting.writeAfter = usage.cacheWrite;
        }
      }
      awaiting = undefined;
      for (const fold of inEffect) fold.requests++;
      mark = undefined;
      stats.folds.saved += active;
      return;
    }

    if (found.role === 'user' || found.role === 'toolResult') {
      stats.overhead.stamps += tokensOf(
        textChars(found, (text) => isStampBlock({ type: 'text', text }))
      );
    }
    if (found.role !== 'toolResult') return;
    stats.overhead.cues += tokensOf(textChars(found, (text) => text.startsWith('[episodes] ')));
    const capped = cappedOutput(found.details);
    const preview = capped && found.content[capped.block];
    if (capped && preview?.type === 'text') {
      stats.outputs.capped++;
      stats.outputs.tokens += tokensOf(Math.max(0, capped.chars - preview.text.length));
    }

    const details = found.details;
    const fold = foldNoteDetails(found);
    if (fold && applied.has(fold.handle)) {
      const folds = stats.folds;
      folds.count++;
      if (fold.manual === 'since') folds.pickedByUser++;
      if (fold.manual === 'request') folds.requestedByUser++;
      folds.tokens += fold.tokens;
      folds.spanTokens.push(fold.tokens);
      if (fold.tokens < SMALL_SPAN_TOKENS) folds.small++;
      if (fold.keptFrom) folds.bounded++;
      folds.noteTokens += fold.noteTokens ?? 0;
      foldNet.set(fold.handle, Math.max(0, fold.tokens - (fold.noteTokens ?? 0)));
      foldNote.set(fold.handle, fold.noteTokens ?? 0);
      if (fold.anchor?.placed !== undefined) {
        folds.anchorKnown++;
        if (fold.anchor.placed) folds.anchored++;
        if (fold.anchor.gapTokens !== undefined) folds.gapTokens.push(fold.anchor.gapTokens);
      }

      const caller = entries
        .slice(0, index)
        .findLast((earlier) => toolCalls(earlier).some((call) => call.id === found.toolCallId));
      if (toolCalls(caller).length > 1) folds.notAlone++;
      // A /fold ends the run on purpose; the reply after it answers a new prompt.
      const next = message(
        entries.slice(index + 1).find((later) => message(later)?.role === 'assistant')
      );
      if (
        !fold.manual &&
        next?.role === 'assistant' &&
        next.stopReason === 'stop' &&
        !next.content.some((block) => block.type === 'toolCall')
      ) {
        folds.beforeFinal++;
      }
      return;
    }
    // A fork that failed or was stopped still returns its record.
    if (found.toolName === FORK_TOOL) {
      const forks = stats.forks;
      if (!isForkDetails(details)) {
        if (found.isError) forks.refused++;
        return;
      }
      forks.count++;
      forks.turns += details.turns;
      forks.folds += details.folds ?? 0;
      if (details.status && details.status !== 'done') forks.failed++;
      forks.cost += details.usage?.cost.total ?? 0;
      return;
    }
    if (found.isError) return;
    if (found.toolName === 'recall') {
      const recall = details as RecallDetails | undefined;
      stats.recall.count++;
      if (recall?.total === 0) stats.recall.empty++;
      if (recall?.turn) stats.recall.readBack++;
      if (recall?.hits?.some((handle) => handle.startsWith('cp-'))) stats.recall.compacted++;
      countStale(recall?.stale, index);
    } else if (found.toolName === 'ask') {
      stats.ask.count++;
      countStale((details as AskDetails | undefined)?.stale, index);
    }
  });

  stats.folds.rates = cacheRates(usages);
  return stats;
}

const BARS = '▁▂▃▄▅▆▇█';

/**
 * One block per request, scaled to the largest shown so the drops stay visible however large
 * the window is. `paint` colors a bar, such as the ones right after a fold.
 */
export function sparkline(
  points: readonly RequestPoint[],
  width: number,
  paint: (bar: string, point: RequestPoint) => string = (bar) => bar
): string {
  const shown = points.slice(-width);
  const max = Math.max(...shown.map((point) => point.tokens));
  return shown
    .map((point) => {
      const level = Math.max(0, Math.ceil((point.tokens / max) * BARS.length) - 1);
      return paint(BARS[level]!, point);
    })
    .join('');
}

function k(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  return tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens);
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]!
    : Math.round((sorted[middle - 1]! + sorted[middle]!) / 2);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
function average(values: readonly number[]): number {
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

/** Only the parts that happened, with counts in parentheses when there are any. */
function parts(...items: [number, string][]): string {
  return items
    .filter(([count]) => count > 0)
    .map(([count, label]) => `${count} ${label}`)
    .join(', ');
}

/** The /episodes stats view: a few plain lines, each left out when it has nothing to say. */
export function statsLines(stats: EpisodeStats): string[] {
  const lines: string[] = [];
  if (stats.requests.length > 1) {
    lines.push(`context per request  ${sparkline(stats.requests, 48)}`);
  }

  const folds = stats.folds;
  if (folds.count > 0 || folds.rejected > 0) {
    const detail = parts(
      [folds.pickedByUser, 'span picked by you'],
      [folds.requestedByUser, 'asked by you'],
      [folds.small, 'under 2k'],
      [folds.bounded, 'ending at until'],
      [folds.notAlone, 'not alone in their response'],
      [folds.beforeFinal, 'right before a final reply']
    );
    lines.push(
      `folds ${folds.count}${folds.count > 0 ? `, median ~${k(median(folds.spanTokens))}, notes ~${k(folds.noteTokens)}` : ''}${detail ? ` (${detail})` : ''}`
    );
    if (folds.rejected > 0) {
      lines.push(`fold errors ${folds.rejected}, next fold succeeded after ${folds.retriedOk}`);
    }
  }
  if (folds.count > 0 || stats.compactions > 0) {
    lines.push(
      `context ~${k(folds.tokens)} folded, ~${k(folds.saved)} fewer tokens sent since, compactions ${stats.compactions}`
    );
  }
  const rates = folds.rates;
  const needed = folds.each.flatMap((fold) => breakEven(fold, rates) ?? []);
  if (needed.length > 0) {
    const finite = needed.filter(Number.isFinite);
    const paid = folds.each.filter((fold) => fold.requests >= (breakEven(fold, rates) ?? Infinity));
    const basis =
      rates.basis === 'costs'
        ? `cached input at ${Math.round(rates.cached * 100)}% of the input price, from recorded costs`
        : rates.basis === 'uncached'
          ? 'nothing was read from cache, so a fold costs only its note'
          : `cached input counted at ${rates.cached * 100}%`;
    const price = rates.inputPrice;
    const dollars =
      price === undefined
        ? ''
        : `; ~$${(sum(folds.each.map((fold) => spent(fold, rates))) * price).toFixed(3)} spent on notes and re-sent input, up to ~$${(sum(folds.each.map((fold) => fold.net * fold.requests)) * rates.cached * price).toFixed(3)} saved since`;
    lines.push(
      `fold cost ${finite.length > 0 ? `break-even after median ${median(finite)} request(s), ` : ''}${paid.length}/${needed.length} folds past it${dollars} (${basis})`
    );
  }
  if (stats.outputs.capped > 0) {
    lines.push(
      `outputs capped ${stats.outputs.capped}, ~${k(stats.outputs.tokens)} kept out of context`
    );
  }
  if (folds.uncachedAfter.length > 0 || folds.anchorKnown > 0) {
    const cache = [
      folds.uncachedAfter.length > 0
        ? `missed after fold avg ~${k(average(folds.uncachedAfter))}`
        : '',
      folds.anchorKnown > 0 ? `anchor placed ${folds.anchored}/${folds.anchorKnown}` : '',
      folds.gapTokens.length > 0 ? `anchor to span start avg ~${k(average(folds.gapTokens))}` : ''
    ].filter(Boolean);
    lines.push(`cache ${cache.join(', ')}`);
  }

  const back = [
    stats.recall.count > 0
      ? `recall ${stats.recall.count}${stats.recall.empty || stats.recall.compacted || stats.recall.readBack ? ` (${parts([stats.recall.empty, 'empty'], [stats.recall.compacted, 'in compacted turns'], [stats.recall.readBack, 'read back'])})` : ''}`
      : '',
    stats.ask.count > 0 ? `ask ${stats.ask.count}` : '',
    stats.stale.shown > 0 ? `stale ${stats.stale.shown} (re-read ${stats.stale.reread})` : ''
  ].filter(Boolean);
  if (back.length > 0) lines.push(back.join(', '));
  const cues = stats.cues;
  if (cues.shown > 0) {
    const detail = parts([cues.readOnly, 'read-only'], [cues.capped, 'capped']);
    lines.push(`cues ${cues.shown}, followed ${cues.followed}${detail ? ` (${detail})` : ''}`);
  }

  const forks = stats.forks;
  if (forks.count > 0 || forks.refused > 0) {
    const detail = parts(
      [forks.folds, 'folded inside'],
      [forks.failed, 'stopped or failed'],
      [forks.refused, 'refused']
    );
    lines.push(
      `forks ${forks.count}, ${forks.turns} turns${forks.cost > 0 ? `, $${forks.cost.toFixed(2)}` : ''}${detail ? ` (${detail})` : ''}`
    );
  }
  if (stats.reminders > 0) lines.push(`open-item reminders ${stats.reminders}`);
  if (stats.nudges > 0) lines.push(`fold nudges ${stats.nudges}`);

  const overhead = stats.overhead;
  lines.push(
    `added by the extension ~${k(overhead.stamps + overhead.cues + overhead.notices)} (stamps ~${k(overhead.stamps)}, cues ~${k(overhead.cues)}, reminders and notices ~${k(overhead.notices)}), prompt ~${k(overhead.prompt)} per request`
  );
  return lines;
}
