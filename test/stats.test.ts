import assert from 'node:assert/strict';
import { test } from 'node:test';
import { breakEven, episodeStats, sparkline, statsLines } from '../src/stats.js';

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
