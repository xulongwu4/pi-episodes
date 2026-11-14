import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { SessionEntry } from '@earendil-works/pi-coding-agent';
import {
  everAppliedFolds,
  FOLD_APPLIED_TYPE,
  FOLD_TOOL,
  FOLD_UNFOLDED_TYPE,
  FORK_TOOL,
  foldNoteDetails,
  isForkDetails
} from './archive.js';
import { COMPACTION_NOTICE_TYPE } from './compaction.js';
import { type EpisodeEvent, EVENT_TYPE } from './events.js';
import { absolutePath } from './messages.js';
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
  // Where cached input is free, as where the note is as long as the span, nothing is saved.
  const saved = fold.net * rates.cached;
  if (saved <= 0) return Number.POSITIVE_INFINITY;
  // Rates worked out from costs carry rounding error, which must not add a request.
  return Math.ceil((spent(fold, rates) / saved) * (1 - 1e-9));
}

/** The part of a request's usage the rates are read from. */
interface PricedUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cost?: { input: number; output: number; cacheRead: number; cacheWrite: number };
  /** The model that answered: prices are per model. */
  model?: string;
}

/** The mean of the values that are known, each weighted; undefined when none is. */
function weighted(values: readonly { value: number | undefined; weight: number }[]) {
  const known = values.filter((entry) => entry.value !== undefined && entry.weight > 0);
  const weight = sum(known.map((entry) => entry.weight));
  return weight > 0 ? sum(known.map((entry) => entry.value! * entry.weight)) / weight : undefined;
}

/**
 * The rates the session's own requests show. Costs come from pi's price table for the model
 * that answered. Each model's rates are worked out from its own requests, since one model's
 * cached tokens say nothing next to another's input price, and requests without a recorded
 * cost are left out, since their tokens would count as free. With several models the input
 * price is the mix by the tokens each was sent, and each rate the mix by those tokens at the
 * model's input price, so that rate times input price is the mean price of such a token over
 * the models that show one. A cache read recorded at no cost is a rate of zero, not an
 * unknown one.
 */
export function cacheRates(usages: readonly PricedUsage[]): CacheRates {
  const models = new Map<string, Record<'input' | 'output' | 'read' | 'write', number[]>>();
  for (const usage of usages) {
    const cost = usage.cost;
    if (!cost || cost.input + cost.output + cost.cacheRead + cost.cacheWrite <= 0) continue;
    const key = usage.model ?? '';
    const model = models.get(key) ?? {
      input: [0, 0],
      output: [0, 0],
      read: [0, 0],
      write: [0, 0]
    };
    models.set(key, model);
    const add = (part: number[], tokens: number, dollars: number) => {
      part[0]! += tokens;
      part[1]! += dollars;
    };
    add(model.input, usage.input, cost.input);
    add(model.output, usage.output, cost.output);
    add(model.read, usage.cacheRead, cost.cacheRead);
    add(model.write, usage.cacheWrite, cost.cacheWrite);
  }
  const rates = [...models.values()].flatMap((model) => {
    const [input, inputCost] = model.input as [number, number];
    if (input <= 0 || inputCost <= 0) return [];
    const price = inputCost / input;
    const rate = ([tokens, dollars]: number[]) =>
      tokens! > 0 ? dollars! / tokens! / price : undefined;
    return [
      {
        price,
        cached: rate(model.read),
        // No cost for the tokens written: the provider bills them as input.
        write: model.write[1]! > 0 ? rate(model.write) : undefined,
        output: rate(model.output),
        weight: input + model.read[0]! + model.write[0]!
      }
    ];
  });
  const inputPrice = weighted(rates.map((model) => ({ value: model.price, weight: model.weight })));
  // A rate is a share of the input price, so a model counts by its tokens at its price: the
  // mixed rate times the input price of those models is then the mean of what they charge.
  // A model that shows no such rate, as one that read nothing from cache, is left out of that
  // mean and not counted at zero, so the rate is scaled to the input price of all models.
  const mix = (pick: (model: (typeof rates)[number]) => number | undefined) => {
    const known = rates.filter((model) => pick(model) !== undefined);
    const rate = weighted(
      known.map((model) => ({ value: pick(model), weight: model.weight * model.price }))
    );
    const price = weighted(known.map((model) => ({ value: model.price, weight: model.weight })));
    return rate === undefined || price === undefined || !inputPrice
      ? rate
      : rate * (price / inputPrice);
  };
  const priced = inputPrice === undefined ? {} : { inputPrice };
  const output = mix((model) => model.output) ?? 0;
  const cached = mix((model) => model.cached);
  if (cached !== undefined && inputPrice !== undefined) {
    return {
      cached,
      write: mix((model) => model.write) ?? 1,
      output,
      basis: 'costs',
      inputPrice
    };
  }
  if (usages.length > 1 && usages.every((usage) => usage.cacheRead === 0)) {
    return { cached: 1, write: 1, output, basis: 'uncached', ...priced };
  }
  return ASSUMED_RATES;
}
export interface RequestPoint {
  /** Tokens the request was sent with: input plus cache read and write. */
  tokens: number;
  /** The first request after a fold or a compaction, where the context should drop. */
  after?: 'fold' | 'compaction';
}

/** A request with what is needed to say why its input was not read from cache. */
export interface SentRequest extends RequestPoint {
  /** The part of the tokens not read from cache: input plus cache write. */
  uncached: number;
  /** When the response was saved, in ms. */
  at?: number;
}

/** Providers cache in blocks, so a little of every request misses; less than this is no miss. */
const MISS_FLOOR_TOKENS = 1024;
/** Without a request for about this long, a provider drops the prefix it cached. */
const PAUSE_MS = 5 * 60_000;

/** Input that was not read from cache, by what explains it. */
export interface UncachedSplit {
  /** New content: what the context grew by since the request before. */
  fresh: number;
  /** The first request after a fold or a compaction: the prefix it changed, sent again. */
  afterFold: number;
  afterCompaction: number;
  /** A prefix sent before and not read back, with no fold or compaction to explain it. */
  missed: number;
  missedRequests: number;
  /** The part of `missed` that came after a pause long enough for the cache to expire. */
  afterPause: number;
  afterPauseRequests: number;
}

/**
 * Says why input was not read from cache. A request cannot read what it adds, so growth
 * since the request before is new content; more than that is a prefix the provider had and
 * did not return, which a fold, a compaction or a long pause explains, or nothing does.
 */
export function uncachedSplit(requests: readonly SentRequest[]): UncachedSplit {
  const split: UncachedSplit = {
    fresh: 0,
    afterFold: 0,
    afterCompaction: 0,
    missed: 0,
    missedRequests: 0,
    afterPause: 0,
    afterPauseRequests: 0
  };
  let previous: SentRequest | undefined;
  for (const request of requests) {
    if (request.after === 'fold') split.afterFold += request.uncached;
    else if (request.after === 'compaction') split.afterCompaction += request.uncached;
    else {
      const grown = previous ? Math.max(0, request.tokens - previous.tokens) : request.tokens;
      const beyond = request.uncached - grown;
      if (beyond < MISS_FLOOR_TOKENS) split.fresh += request.uncached;
      else {
        split.fresh += grown;
        split.missed += beyond;
        split.missedRequests++;
        if (previous?.at !== undefined && request.at !== undefined) {
          if (request.at - previous.at >= PAUSE_MS) {
            split.afterPause += beyond;
            split.afterPauseRequests++;
          }
        }
      }
    }
    previous = request;
  }
  return split;
}

export interface EpisodeStats {
  /** Main-thread requests on the branch, oldest first. */
  requests: SentRequest[];
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
  /** Input not read from cache over the branch, by what explains it. */
  uncached: UncachedSplit;
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
    uncached: uncachedSplit([]),
    compactions: 0,
    reminders: 0,
    nudges: 0,
    overhead: { stamps: 0, cues: 0, notices: 0, prompt: tokensOf(EPISODES_SECTION.length) }
  };

  const applied = everAppliedFolds(entries);
  const foldNet = new Map<string, number>();
  const foldNote = new Map<string, number>();
  /** Net tokens of folds in effect, saved again on every request until a compaction. */
  let active = 0;
  let mark: RequestPoint['after'];
  /** Folds still saving on each request, and the one whose first request has not come yet. */
  let inEffect: FoldCost[] = [];
  let awaiting: FoldCost[] = [];
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
      .map((call) => absolutePath(cwd, call.path!));
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
        const fold = {
          handle,
          net: foldNet.get(handle) ?? 0,
          note: foldNote.get(handle) ?? 0,
          requests: 0
        };
        awaiting.push(fold);
        stats.folds.each.push(fold);
        inEffect.push(fold);
      }
      mark = 'fold';
      return;
    }
    if (entry.type === 'custom' && entry.customType === FOLD_UNFOLDED_TYPE) {
      const handle = (entry.data as { handle?: string } | undefined)?.handle;
      if (handle) {
        active = Math.max(0, active - (foldNet.get(handle) ?? 0));
        inEffect = inEffect.filter((fold) => fold.handle !== handle);
        awaiting = awaiting.filter((fold) => fold.handle !== handle);
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
      usages.push({ ...usage, model: `${found.provider}/${found.model}` });
      // The message carries the time its request began; the entry, the time it was written.
      const at = Number.isFinite(found.timestamp) ? found.timestamp : Date.parse(entry.timestamp);
      stats.requests.push({
        tokens,
        uncached: usage.input + usage.cacheWrite,
        ...(mark ? { after: mark } : {}),
        ...(Number.isNaN(at) ? {} : { at })
      });
      if (mark === 'fold') {
        stats.folds.uncachedAfter.push(usage.input + usage.cacheWrite);
        // Folds applied at one turn end share the request after them; its miss is counted
        // once, for the last of them.
        awaiting.forEach((fold, order) => {
          const last = order === awaiting.length - 1;
          fold.uncachedAfter = last ? usage.input + usage.cacheWrite : 0;
          fold.writeAfter = last ? usage.cacheWrite : 0;
        });
      }
      awaiting = [];
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
    const block = capped && found.content[capped.block];
    // The length recorded at the cut: what is there now can be shorter by another cut, or be
    // another block. Results cut before the length was recorded have only the block.
    const preview = capped?.preview ?? (block?.type === 'text' ? block.text.length : undefined);
    if (capped && preview !== undefined) {
      stats.outputs.capped++;
      stats.outputs.tokens += tokensOf(Math.max(0, capped.chars - preview));
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
      foldNet.set(
        fold.handle,
        Math.max(0, fold.tokens - (fold.noteTokens ?? 0) - (fold.keptTokens ?? 0))
      );
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
  stats.uncached = uncachedSplit(stats.requests);
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
    const middle = median(needed);
    const paid = folds.each.filter((fold) => fold.requests >= (breakEven(fold, rates) ?? Infinity));
    // A small share is not none: where cached input is free, no fold breaks even.
    const share =
      rates.cached > 0 && rates.cached < 0.005 ? 'under 1' : Math.round(rates.cached * 100);
    const basis =
      rates.basis === 'costs'
        ? `cached input at ${share}% of the input price, from recorded costs`
        : rates.basis === 'uncached'
          ? 'nothing was read from cache, so a fold costs only its note'
          : `cached input counted at ${rates.cached * 100}%`;
    const price = rates.inputPrice;
    const dollars =
      price === undefined
        ? ''
        : `; ~$${(sum(folds.each.map((fold) => spent(fold, rates))) * price).toFixed(3)} spent on notes and re-sent input, up to ~$${(sum(folds.each.map((fold) => fold.net * fold.requests)) * rates.cached * price).toFixed(3)} saved since`;
    lines.push(
      `fold cost ${Number.isFinite(middle) ? `break-even after median ${middle} request(s), ${paid.length}/${needed.length} folds past it` : `never breaks even for half the folds or more, ${paid.length}/${needed.length} folds past break-even`}${dollars} (${basis})`
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

  const split = stats.uncached;
  const uncached = split.fresh + split.afterFold + split.afterCompaction + split.missed;
  if (uncached > 0) {
    const pauses = split.afterPauseRequests;
    const causes = [
      `~${k(split.fresh)} new content`,
      split.afterFold > 0 ? `~${k(split.afterFold)} after folds` : '',
      split.afterCompaction > 0 ? `~${k(split.afterCompaction)} after compactions` : '',
      split.missed > 0
        ? `~${k(split.missed)} sent before and missed in ${split.missedRequests} request(s), ${pauses > 0 ? `${pauses} of them (~${k(split.afterPause)}) after a pause of 5 min or more` : 'none after a pause of 5 min or more'}`
        : ''
    ].filter(Boolean);
    lines.push(`uncached input ~${k(uncached)}: ${causes.join(', ')}`);
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
