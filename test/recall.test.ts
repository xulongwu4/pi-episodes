import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createMatcher, formatTurn, grepThreads, turnMessages } from '../src/recall.js';
import { assistant, call, result, user } from './helpers.js';

const thread = {
  handle: 'ep-1',
  messages: [
    assistant(call('c2', 'bash', { command: 'pnpm build' })),
    result('c2', 'bash', 'ok\nsrc/client/api.ts(12,5): error TS2322', 3),
    user('do not edit generated files', 4),
    assistant('The TS2322 errors come from the stale generated client.')
  ]
};

test('finds lines in tool output, user text and the agent own messages', () => {
  const matches = grepThreads([thread], createMatcher('ts2322', false));

  assert.equal(matches.length, 2);
  assert.deepEqual(
    matches.map((match) => [match.handle, match.turn, match.role]),
    [
      ['ep-1', 't3', 'toolResult'],
      ['ep-1', 't4', 'assistant']
    ]
  );
  assert.equal(matches[0]!.line, 'src/client/api.ts(12,5): error TS2322');
});

test('searches tool call arguments and supports regular expressions', () => {
  const matches = grepThreads([thread], createMatcher('pnpm\\s+build', true));
  assert.equal(matches.length, 1);
  assert.match(matches[0]!.line, /bash\(\{"command":"pnpm build"\}\)/);
});

test('returns every match so the caller can cap and spill the rest', () => {
  const matches = grepThreads([thread, { ...thread, handle: 'ep-2' }], createMatcher('e', false));
  assert.ok(matches.length > 3);
  assert.deepEqual(new Set(matches.map((match) => match.handle)), new Set(['ep-1', 'ep-2']));
});

test('a turn read back is its stamped message and the responses up to the next stamp', () => {
  // The turn a match names: the assistant line above was attributed to t4.
  const turn = turnMessages(thread, 4);
  assert.deepEqual(
    turn.map((message) => message.role),
    ['user', 'assistant']
  );
  assert.deepEqual(
    turnMessages(thread, 3).map((message) => message.role),
    ['toolResult']
  );
  assert.deepEqual(turnMessages(thread, 9), []);

  assert.equal(
    formatTurn('ep-1', 't3', turnMessages(thread, 3)),
    'ep-1 t3 toolResult (bash), 45 chars\nok\nsrc/client/api.ts(12,5): error TS2322\n[t3]'
  );
});

test('a range read back runs from its first turn to the end of its last', () => {
  const range = turnMessages(thread, 3, 4);
  assert.deepEqual(
    range.map((message) => message.role),
    ['toolResult', 'user', 'assistant']
  );
  // Both ends must be in the thread, in order.
  assert.deepEqual(turnMessages(thread, 3, 9), []);
  assert.deepEqual(turnMessages(thread, 4, 3), []);

  const text = formatTurn('ep-1', 't3', range);
  assert.match(text, /^ep-1 t3 toolResult \(bash\), 45 chars\n/);
  assert.match(
    text,
    /\n\nep-1 t4 user, \d+ chars\ndo not edit generated files\n\[t4\]\n\nep-1 t4 assistant, /
  );
});
