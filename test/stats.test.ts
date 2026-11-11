import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FOLD_APPLIED_TYPE } from '../src/archive.js';
import {
  breakEven,
  cacheRates,
  episodeStats,
  sparkline,
  statsLines,
  uncachedSplit
} from '../src/stats.js';
import { assistant, call, result, session, user } from './helpers.js';

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

test('rates keep each model apart and leave out requests without a cost', () => {
  const sent = (model: string, price: number, input: number, cacheRead: number, read = 0.1) => ({
    input,
    output: 0,
    cacheRead,
    cacheWrite: 0,
    model,
    cost: { input: input * price, output: 0, cacheRead: cacheRead * price * read, cacheWrite: 0 }
  });
  const near = (actual: number, expected: number) =>
    assert.ok(Math.abs(actual - expected) < 1e-9 * expected, `${actual}`);
  // Both models bill cached input at a tenth; one mostly read from cache, the other did not.
  near(cacheRates([sent('a', 1e-6, 100, 100_000), sent('b', 1e-4, 100_000, 100)]).cached, 0.1);
  // Models with different prices and discounts, sent as much: rate times input price is the
  // mean price of a cached token, $25.05 per million here.
  const mixed = cacheRates([sent('a', 1e-6, 1000, 1000), sent('b', 1e-4, 1000, 1000, 0.5)]);
  near(mixed.inputPrice!, 50.5e-6);
  near(mixed.cached * mixed.inputPrice!, 25.05e-6);
  // A model that read nothing from cache does not say what its cached input costs: the
  // price of a cached token is that of the model that did, not a share of the dearer one's.
  const partly = cacheRates([sent('a', 1e-6, 1000, 1000), sent('b', 1e-4, 2000, 0)]);
  near(partly.inputPrice!, 50.5e-6);
  near(partly.cached * partly.inputPrice!, 0.1e-6);
  // Tokens of a request with no recorded cost are not input that cost nothing.
  const free = {
    ...sent('a', 0, 9000, 0),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
  };
  near(cacheRates([sent('a', 1e-6, 1000, 4000), free]).cached, 0.1);

  // Cache reads recorded at no cost are free: a fold saves nothing on later requests.
  const gratis = cacheRates([sent('a', 1e-6, 1000, 4000, 0)]);
  assert.equal(gratis.basis, 'costs');
  assert.equal(gratis.cached, 0);
  const fold = { handle: 'ep-1', net: 10_000, uncachedAfter: 9000, requests: 5 };
  assert.equal(breakEven(fold, gratis), Infinity);
  assert.equal(breakEven({ ...fold, uncachedAfter: 0 }, gratis), Infinity);

  // One request that read nothing from cache does not say what cached input costs.
  assert.deepEqual(cacheRates([sent('a', 1e-6, 1000, 0)]), {
    cached: 0.1,
    write: 1,
    output: 0,
    basis: 'assumed'
  });
});

test('a rate off by rounding does not add a request, and folds that never pay are counted', () => {
  // 750 of re-sent input and a note of 2,500 against 3,250 saved on each request.
  const rates = { cached: 0.25, write: 1, output: 5.000000000000001, basis: 'costs' as const };
  const even = { handle: 'ep-1', net: 13_000, note: 500, uncachedAfter: 1000, requests: 1 };
  assert.equal(breakEven(even, rates), 1);

  const never = { handle: 'ep-2', net: 0, uncachedAfter: 1000, requests: 9 };
  const stats = episodeStats([], '/repo');
  stats.folds.rates = rates;
  stats.folds.each.push(even, never, { ...never, handle: 'ep-3' });
  assert.ok(
    statsLines(stats).includes(
      'fold cost never breaks even for half the folds or more, 1/3 folds past break-even (cached input at 25% of the input price, from recorded costs)'
    )
  );
});

test('folds applied together share the request after them, and a compaction ends their savings', () => {
  const { manager } = session([user('go', 1), assistant(call('a', 'bash'))]);
  const sent = (input: number, cacheRead: number, timestamp: number) => {
    manager.appendMessage({
      ...assistant('ok'),
      usage: {
        input,
        output: 10,
        cacheRead,
        cacheWrite: 0,
        totalTokens: input + cacheRead + 10,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
      },
      timestamp
    });
  };
  const minute = 60_000;
  sent(20_000, 0, 0);
  for (const [id, handle] of [
    ['f1', 'ep-1'],
    ['f2', 'ep-2']
  ] as const) {
    manager.appendMessage({
      ...result(id, 'fold', 'note', 2),
      details: {
        kind: 'fold',
        handle,
        since: 't2',
        firstEntryId: 'a',
        lastEntryId: 'b',
        messages: 2,
        tokens: 5000,
        headline: 'note',
        keptUserTurns: [],
        files: [],
        noteTokens: 1000
      }
    } as Parameters<typeof manager.appendMessage>[0]);
  }
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-1' });
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-2' });
  sent(7000, 5000, minute);
  // Seven minutes after the request before it began, though both were written just now.
  sent(11_000, 2000, 8 * minute);
  let stats = episodeStats(manager.getBranch(), '/repo');
  assert.deepEqual(
    stats.folds.each.map((fold) => [fold.handle, fold.uncachedAfter, fold.requests]),
    [
      ['ep-1', 0, 2],
      ['ep-2', 7000, 2]
    ]
  );
  assert.equal(stats.folds.saved, 2 * 2 * 4000);
  assert.deepEqual(
    [stats.uncached.afterFold, stats.uncached.missed, stats.uncached.afterPause],
    [7000, 10_000, 10_000]
  );

  manager.appendCompaction('summary', null, 13_000);
  sent(3000, 0, 9 * minute);
  sent(1000, 3000, 10 * minute);
  stats = episodeStats(manager.getBranch(), '/repo');
  assert.equal(stats.folds.saved, 2 * 2 * 4000);
  assert.deepEqual(
    stats.folds.each.map((fold) => fold.requests),
    [2, 2]
  );
});

test('a capped output counts what the cap left out, whatever the block holds now', () => {
  const kept = (preview?: number) => {
    const record = { path: '/s/output-1.txt', chars: 40_000, block: 0 };
    const cut = {
      ...result('a', 'bash', 'x'.repeat(6000), 2),
      details: { episodesOutput: preview === undefined ? record : { ...record, preview } }
    };
    const { manager } = session([user('go', 1), assistant(call('a', 'bash')), cut]);
    return episodeStats(manager.getBranch(), '/repo').outputs;
  };
  // Another extension cut the preview of 14,000 characters to 6,000: that is not the cap's.
  assert.deepEqual(kept(14_000), { capped: 1, tokens: 6500 });
  // A result cut before the length was recorded has only the block to go by.
  assert.deepEqual(kept(), { capped: 1, tokens: 8500 });
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
