import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  breakEven,
  cacheRates,
  episodeStats,
  sparkline,
  statsLines,
  uncachedSplit
} from '../src/stats.js';

test('the graph scales to the largest request shown and keeps the newest', () => {
  const points = [100, 200, 400, 800, 100].map((tokens) => ({ tokens }));
  assert.equal(sparkline(points, 5), '▁▂▄█▁');
  assert.equal(sparkline(points, 2), '█▁');
  const marked = sparkline(
    [{ tokens: 10 }, { tokens: 5, after: 'fold' as const }],
    2,
    (bar, point) => (point.after ? `<${bar}>` : bar)
  );
  assert.equal(marked, '█<▄>');
});

test('an empty session says only what the extension costs', () => {
  const lines = statsLines(episodeStats([], '/repo'));
  assert.equal(lines.length, 1);
  assert.match(lines[0]!, /^added by the extension ~0 .*prompt ~\d+(k)? per request$/);
});

test('a fold breaks even once its savings at the cached rate cover the prefix it re-sent', () => {
  // 9,000 tokens re-sent at nine tenths over the cached rate; 10,000 saved at one tenth each.
  assert.equal(breakEven({ handle: 'ep-1', net: 10_000, uncachedAfter: 9000, requests: 0 }), 9);
  assert.equal(breakEven({ handle: 'ep-1', net: 500, uncachedAfter: 9000, requests: 0 }), 162);
  // A note as long as its span never pays; before the next request nothing is known.
  assert.equal(breakEven({ handle: 'ep-1', net: 0, uncachedAfter: 9000, requests: 3 }), Infinity);
  assert.equal(breakEven({ handle: 'ep-1', net: 10_000, requests: 0 }), undefined);
});

test('the cache rates are those of the costs the session recorded, or say what they assume', () => {
  const priced = (input: number, cacheRead: number, cacheWrite: number) => ({
    input,
    output: 100,
    cacheRead,
    cacheWrite,
    cost: {
      input: input * 2e-6,
      output: 100 * 1e-5,
      cacheRead: cacheRead * 5e-7,
      cacheWrite: cacheWrite * 2.5e-6
    }
  });
  const rates = cacheRates([priced(1000, 0, 4000), priced(500, 4000, 500)]);
  assert.equal(rates.basis, 'costs');
  assert.ok(Math.abs(rates.cached - 0.25) < 1e-9);
  assert.ok(Math.abs(rates.write - 1.25) < 1e-9);
  assert.ok(Math.abs(rates.output - 5) < 1e-9);
  // A note of 500 output tokens at five times the input price, 6,000 tokens of input above
  // the cached rate, 3,000 written at 1.25 against 0.25; 10,000 saved at 0.25 on each request.
  const fold = {
    handle: 'ep-1',
    net: 10_000,
    note: 500,
    uncachedAfter: 9000,
    writeAfter: 3000,
    requests: 5
  };
  assert.equal(breakEven(fold, rates), 4);

  // Without recorded costs the usual discount is assumed.
  const unpriced = { input: 1000, output: 100, cacheRead: 500, cacheWrite: 0 };
  assert.deepEqual(cacheRates([unpriced, unpriced]), {
    cached: 0.1,
    write: 1,
    output: 0,
    basis: 'assumed'
  });
  // A provider that caches nothing bills a fold's span in full on every request: the fold
  // costs only its note.
  const uncached = cacheRates([priced(1000, 0, 0), priced(2000, 0, 0)]);
  assert.equal(uncached.basis, 'uncached');
  assert.equal(breakEven(fold, uncached), 1);

  const stats = episodeStats([], '/repo');
  stats.folds.each.push(fold);
  stats.folds.rates = rates;
  assert.ok(
    statsLines(stats).includes(
      'fold cost break-even after median 4 request(s), 1/1 folds past it; ~$0.020 spent on notes and re-sent input, up to ~$0.025 saved since (cached input at 25% of the input price, from recorded costs)'
    )
  );
});

test('uncached input is split by what explains it', () => {
  const minute = 60_000;
  const split = uncachedSplit([
    // Nothing was cached yet: all of it is new.
    { tokens: 10_000, uncached: 10_000, at: 0 },
    // Grew by 2,000 and missed a block more, as every request does.
    { tokens: 12_000, uncached: 2150, at: minute },
    // Grew by 1,000 and missed 6,000 of what was sent before, with nothing to explain it.
    { tokens: 13_000, uncached: 7000, at: 2 * minute },
    // The whole prefix again, after a pause the cache does not outlive.
    { tokens: 13_500, uncached: 13_500, at: 9 * minute },
    { tokens: 6000, uncached: 4000, after: 'fold', at: 10 * minute },
    { tokens: 3000, uncached: 3000, after: 'compaction', at: 11 * minute }
  ]);
  assert.deepEqual(split, {
    fresh: 10_000 + 2150 + 1000 + 500,
    afterFold: 4000,
    afterCompaction: 3000,
    missed: 6000 + 13_000,
    missedRequests: 2,
    afterPause: 13_000,
    afterPauseRequests: 1
  });

  const lines = statsLines({ ...episodeStats([], '/'), uncached: split });
  assert.ok(
    lines.includes(
      'uncached input ~40k: ~14k new content, ~4k after folds, ~3k after compactions, ~19k sent before and missed in 2 request(s), 1 of them (~13k) after a pause of 5 min or more'
    )
  );
});
