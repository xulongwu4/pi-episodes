import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { Message } from '@earendil-works/pi-ai';
import { convertToLlm, type SessionManager } from '@earendil-works/pi-coding-agent';
import {
  FOLD_APPLIED_TYPE,
  FOLD_STARTED_TYPE,
  type ForkArchive,
  foldRecords,
  ThreadIndex
} from '../src/archive.js';
import {
  completeCutShortFolds,
  coveredTokens,
  cutShortFold,
  foldableTurns,
  headline,
  keptMark,
  mergeStart,
  openItems,
  planFold,
  renderCovered,
  renderNote,
  USER_TEXT_CAP,
  unfoldedBehindNote,
  workSince
} from '../src/fold.js';
import { skillPaths, touchedPaths } from '../src/messages.js';
import { nudgeGrowth, tokensBehindNote } from '../src/nudge.js';
import { readStamp } from '../src/stamps.js';
import { assistant, call, result, session, user } from './helpers.js';

function apply(manager: SessionManager, edits: ReturnType<typeof planFold>['edits']): void {
  for (const edit of edits) manager.appendContextEdit(edit.targetId, edit.replacement);
}

function roles(messages: readonly AgentMessage[]): string[] {
  return messages.filter((message) => message.role !== 'system').map((message) => message.role);
}

/** Every tool call must be followed by its result, or the provider rejects the request. */
function assertPaired(messages: readonly AgentMessage[]): void {
  const llm = convertToLlm([...messages]);
  const results = new Set(
    llm.flatMap((message) => (message.role === 'toolResult' ? [message.toolCallId] : []))
  );
  const calls = new Set(
    llm.flatMap((message) =>
      message.role === 'assistant'
        ? message.content.flatMap((block) => (block.type === 'toolCall' ? [block.id] : []))
        : []
    )
  );
  assert.deepEqual([...results].sort(), [...calls].sort());
}

const buildLoop = [
  user('add the claim endpoint', 1),
  assistant('reading', call('c1', 'read', { path: 'src/claim.ts' })),
  result('c1', 'read', 'export function claim() {}', 2),
  assistant(call('c2', 'bash', { command: 'pnpm build' })),
  result('c2', 'bash', 'src/client/api.ts(12,5): error TS2322', 3),
  user('do not edit generated files', 4),
  assistant(call('c3', 'bash', { command: 'pnpm generate && pnpm build' })),
  result('c3', 'bash', 'build passed', 5),
  assistant('The build passes.', call('f1', 'fold', { since: 't2' }))
];

test('folds from the response behind since and keeps user messages from the span', () => {
  const { manager, ids } = session(buildLoop);
  // t3 is the first build output; the span starts at the response that ran the build.
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 3);

  assert.match(plan.handle, /^ep-[0-9a-f]{8}$/);
  assert.equal(plan.anchorStamp, 2);
  assert.equal(plan.firstEntryId, ids[3]);
  assert.equal(plan.lastEntryId, ids[7]);
  assert.deepEqual(plan.keptUserTurns, ['t4']);
  assert.equal(plan.omitted.length, 4);
  assert.deepEqual(
    plan.edits.map((edit) => [edit.targetId, edit.replacement]),
    [ids[7], ids[6], ids[4], ids[3]].map((id) => [id, null])
  );

  apply(manager, plan.edits);
  manager.appendMessage(result('f1', 'fold', 'note', 6));
  const view = manager.buildSessionProjection().messages;

  assert.deepEqual(roles(view), [
    'user',
    'assistant',
    'toolResult',
    'user',
    'assistant',
    'toolResult'
  ]);
  assert.equal(readStamp(view.filter((message) => message.role === 'user')[1]!), 4);
  assertPaired(view);
});

test('every prefix of the edits of a fold leaves each tool result next to its call', () => {
  const { manager, ids } = session(buildLoop);
  const { edits } = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  assert.ok(edits.length > 2);
  // Pi writes the edits one at a time; a crash after any of them must leave a valid context.
  for (let applied = 0; applied <= edits.length; applied++) {
    const { manager: partial, ids: same } = session(buildLoop);
    const prefix = edits.slice(0, applied);
    apply(
      partial,
      prefix.map((edit) => ({ ...edit, targetId: same[ids.indexOf(edit.targetId)]! }))
    );
    partial.appendMessage(result('f1', 'fold', 'note', 6));
    const llm = convertToLlm([...partial.buildSessionProjection().messages]);
    const called = new Set<string>();
    for (const message of llm) {
      if (message.role === 'assistant') {
        for (const block of message.content) if (block.type === 'toolCall') called.add(block.id);
      }
      if (message.role === 'toolResult') assert.ok(called.has(message.toolCallId), `${applied}`);
    }
  }
});

test('the rest of a fold cut short after any of its edits completes it', () => {
  const build = (started: boolean) => {
    const made = session(buildLoop);
    const plan = planFold(made.manager.buildSessionProjection().entries, 'f1', 2);
    const note = Object.assign(result('f1', 'fold', 'note', 6), {
      details: {
        kind: 'fold',
        handle: plan.handle,
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId
      }
    });
    made.manager.appendMessage(note);
    if (started) {
      made.manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
    }
    return { ...made, plan };
  };
  const done = build(true);
  apply(done.manager, done.plan.edits);
  const folded = roles(done.manager.buildSessionProjection().messages);

  for (let applied = 0; applied <= done.plan.edits.length; applied++) {
    const { manager, plan } = build(true);
    apply(manager, plan.edits.slice(0, applied));
    const rest = completeCutShortFolds(manager.getBranch());
    // Only the edits that did not land, in their order.
    assert.deepEqual(
      rest.flatMap((draft) => (draft.type === 'context_edit' ? [draft.targetId] : [])),
      plan.edits.slice(applied).map((edit) => edit.targetId)
    );
    assert.deepEqual(rest.at(-1), {
      type: 'custom',
      customType: FOLD_APPLIED_TYPE,
      data: { handle: plan.handle }
    });
    for (const draft of rest) {
      if (draft.type === 'context_edit')
        manager.appendContextEdit(draft.targetId, draft.replacement);
      else manager.appendCustomEntry(draft.customType, draft.data);
    }
    assert.deepEqual(roles(manager.buildSessionProjection().messages), folded, `${applied}`);
    assert.deepEqual(completeCutShortFolds(manager.getBranch()), []);
  }
});

test('a cut-short fold is completed over edits that other extensions made to its span', () => {
  const { manager } = session(buildLoop);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const noteId = manager.appendMessage(
    Object.assign(result('f1', 'fold', 'note', 6), {
      details: {
        kind: 'fold',
        handle: plan.handle,
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId
      }
    })
  );
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
  apply(manager, plan.edits.slice(0, 2));
  // Another extension restores the first removed entry, shortens the next planned one, and
  // rewrites the note without hiding it.
  const restored = plan.edits[0]!.targetId;
  manager.appendContextEdit(restored, { content: [{ type: 'text', text: 'back' }] });
  manager.appendContextEdit(plan.edits[2]!.targetId, {
    content: [{ type: 'text', text: 'short' }]
  });
  manager.appendContextEdit(noteId, { content: [{ type: 'text', text: 'note' }] });

  const rest = completeCutShortFolds(manager.getBranch());
  assert.deepEqual(
    rest.flatMap((draft) => (draft.type === 'context_edit' ? [draft.targetId] : [])),
    [restored, ...plan.edits.slice(2).map((edit) => edit.targetId)]
  );
  for (const draft of rest) {
    if (draft.type === 'context_edit') manager.appendContextEdit(draft.targetId, draft.replacement);
    else manager.appendCustomEntry(draft.customType, draft.data);
  }
  assertPaired(manager.buildSessionProjection().messages);
  assert.equal(completeCutShortFolds(manager.getBranch()).length, 0);
});

test('a cut-short fold does not write its cut of a user message over a later edit of it', () => {
  const pasted = [...buildLoop];
  pasted[5] = user('x'.repeat(USER_TEXT_CAP * 2), 4);
  const build = () => {
    const { manager, ids } = session(pasted);
    const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
    manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
    const cut = plan.edits.findIndex((edit) => edit.targetId === ids[5]);
    assert.ok(cut > 0 && plan.edits[cut]!.replacement !== null);
    return { manager, plan, cut, paste: ids[5]! };
  };
  const targets = (rest: ReturnType<typeof completeCutShortFolds>) =>
    rest.flatMap((draft) => (draft.type === 'context_edit' ? [draft.targetId] : []));
  const redacted = { content: [{ type: 'text' as const, text: 'redacted' }] };

  // Left alone, the cut is completed like the removals.
  const plain = build();
  apply(plain.manager, plain.plan.edits.slice(0, 1));
  assert.ok(targets(completeCutShortFolds(plain.manager.getBranch())).includes(plain.paste));

  // Another extension rewrote or removed the message, on the branch or in a draft queued at
  // this boundary, whether the crash came before the cut or after it.
  for (const later of [redacted, null]) {
    for (const where of ['branch', 'queued'] as const) {
      for (const crash of ['before the cut', 'after the cut'] as const) {
        const { manager, plan, cut, paste } = build();
        const landed = crash === 'before the cut' ? 1 : cut + 1;
        apply(manager, plan.edits.slice(0, landed));
        const edit = { type: 'context_edit' as const, targetId: paste, replacement: later };
        if (where === 'branch') manager.appendContextEdit(paste, later);
        const rest = completeCutShortFolds(manager.getBranch(), where === 'queued' ? [edit] : []);
        assert.deepEqual(
          targets(rest),
          plan.edits
            .slice(landed)
            .map((edit) => edit.targetId)
            .filter((id) => id !== paste),
          [JSON.stringify(later), where, crash].join(', ')
        );
      }
    }
  }
});

test('a cut-short fold counts the edits earlier handlers queued at the same boundary', () => {
  const { manager } = session(buildLoop);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
  apply(manager, plan.edits.slice(0, 1));
  const restored = plan.edits[0]!.targetId;
  const pending = [
    {
      type: 'context_edit' as const,
      targetId: restored,
      replacement: { content: [{ type: 'text' as const, text: 'back' }] }
    }
  ];
  const rest = completeCutShortFolds(manager.getBranch(), pending);
  assert.equal(
    rest.flatMap((draft) => (draft.type === 'context_edit' ? [draft.targetId] : []))[0],
    restored
  );
});

test('a cut-short fold whose note an edit removed is completed like any other', () => {
  for (const where of ['before', 'after', 'queued'] as const) {
    const { manager } = session(buildLoop);
    const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
    const noteId = manager.appendMessage(
      Object.assign(result('f1', 'fold', 'note', 6), {
        details: {
          kind: 'fold',
          handle: plan.handle,
          firstEntryId: plan.firstEntryId,
          lastEntryId: plan.lastEntryId
        }
      })
    );
    if (where === 'before') manager.appendContextEdit(noteId, null);
    manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
    apply(manager, plan.edits.slice(0, 1));
    if (where === 'after') manager.appendContextEdit(noteId, null);
    const pending =
      where === 'queued' ? [{ type: 'context_edit', targetId: noteId, replacement: null }] : [];
    assert.equal(cutShortFold(manager.getBranch()), plan.handle, where);
    // The whole span leaves the context, so the record holds no part a later fold can take.
    const rest = completeCutShortFolds(manager.getBranch(), pending);
    assert.deepEqual(
      rest,
      [
        ...plan.edits.slice(1),
        { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: plan.handle } }
      ],
      where
    );
    apply(manager, plan.edits.slice(1));
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
    assert.deepEqual(
      foldRecords(manager.getBranch()).map((record) => record.handle),
      [plan.handle],
      where
    );
    assert.equal(cutShortFold(manager.getBranch()), undefined, where);
    assert.deepEqual(completeCutShortFolds(manager.getBranch(), pending), [], where);
  }
});

test('a fold whose batch never landed is not completed, whatever later edits touch its span', () => {
  // Without its start entry the fold never took effect; an edit by another extension to a
  // result in its span is not progress of the fold.
  const { manager, ids } = session(buildLoop);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  manager.appendMessage(
    Object.assign(result('f1', 'fold', 'note', 6), {
      details: {
        kind: 'fold',
        handle: plan.handle,
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId
      }
    })
  );
  manager.appendContextEdit(ids[4]!, { content: [{ type: 'text', text: 'shortened' }] });
  manager.appendContextEdit(ids[7]!, null);
  assert.deepEqual(completeCutShortFolds(manager.getBranch()), []);
});

test('a stamp on one result of a parallel batch folds the whole batch', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('a', 'read', { path: 'a.ts' }), call('b', 'read', { path: 'b.ts' })),
    result('a', 'read', 'a', 2),
    result('b', 'read', 'b', 3),
    assistant(call('c', 'bash')),
    result('c', 'bash', 'out', 4),
    assistant(call('f', 'fold'))
  ]);

  const plan = planFold(manager.buildSessionProjection().entries, 'f', 3);
  assert.equal(plan.anchorStamp, 1);
  assert.equal(plan.firstEntryId, ids[1]);
  assert.equal(plan.omitted.length, 5);

  apply(manager, plan.edits);
  manager.appendMessage(result('f', 'fold', 'note'));
  assertPaired(manager.buildSessionProjection().messages);
});

test('a fold with until removes the turns up to it and leaves the later ones', () => {
  const { manager, ids } = session(buildLoop);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 3 });

  assert.equal(plan.firstEntryId, ids[1]);
  assert.equal(plan.lastEntryId, ids[4]);
  assert.equal(plan.keptFrom, 4);
  assert.equal(plan.omitted.length, 4);
  assert.deepEqual(plan.keptUserTurns, []);

  apply(manager, plan.edits);
  manager.appendMessage(result('f1', 'fold', 'note', 6));
  const view = manager.buildSessionProjection().messages;
  assert.deepEqual(roles(view), [
    'user',
    'user',
    'assistant',
    'toolResult',
    'assistant',
    'toolResult'
  ]);
  assert.deepEqual(view.map(readStamp), [1, 4, undefined, 5, undefined, 6]);
  assertPaired(view);
});

test('a span starts from the turn before it, also when an earlier fold removed that turn', () => {
  const { manager } = session(buildLoop);
  const entries = () => manager.buildSessionProjection().entries;
  assert.equal(planFold(entries(), 'f1', 3, { until: 3 }).anchorStamp, 2);
  // The read leaves the context; the tree it left is still where the build began.
  apply(manager, planFold(entries(), 'f1', 2, { until: 2 }).edits);
  assert.equal(planFold(entries(), 'f1', 3, { until: 3 }).anchorStamp, 2);
});

test('until on one result of a parallel batch ends the span after the whole batch', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('a', 'read', { path: 'a.ts' }), call('b', 'read', { path: 'b.ts' })),
    result('a', 'read', 'a', 2),
    result('b', 'read', 'b', 3),
    assistant(call('c', 'bash')),
    result('c', 'bash', 'out', 4),
    assistant(call('f', 'fold'))
  ]);

  const plan = planFold(manager.buildSessionProjection().entries, 'f', 2, { until: 2 });
  assert.equal(plan.lastEntryId, ids[3]);
  assert.equal(plan.keptFrom, 4);

  apply(manager, plan.edits);
  manager.appendMessage(result('f', 'fold', 'note'));
  assertPaired(manager.buildSessionProjection().messages);
});

test('until on the last turn before the call is a fold up to the call', () => {
  const { manager, ids } = session(buildLoop);
  const entries = manager.buildSessionProjection().entries;
  const plan = planFold(entries, 'f1', 3, { until: 5 });
  assert.equal(plan.keptFrom, undefined);
  assert.equal(plan.lastEntryId, ids[7]);
  assert.deepEqual(plan.edits, planFold(entries, 'f1', 3).edits);
});

test('rejects an until that is before since, missing, or holds no responses', () => {
  const entries = session(buildLoop).manager.buildSessionProjection().entries;
  assert.throws(() => planFold(entries, 'f1', 3, { until: 2 }), /until t2 is before since t3/);
  assert.throws(
    () => planFold(entries, 'f1', 2, { until: 9, latestStamp: 5 }),
    /t9 does not exist; the latest stamp is t5/
  );
  assert.throws(
    () => planFold(entries, 'f1', 2, { until: 9, latestStamp: 12 }),
    /t9 is no longer in your context/
  );
  assert.throws(
    () => planFold(entries, 'f1', 4, { until: 4 }),
    /Nothing to fold from t4 to t4: the span holds no responses\.$/
  );
});

test('a fold from the message being answered names the stamp that folds the work before it', () => {
  const entries = (...messages: AgentMessage[]) =>
    session(messages).manager.buildSessionProjection().entries;
  const first = [
    user('first', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    assistant('done')
  ];
  assert.throws(
    () => planFold(entries(...first, user('second', 3), assistant(call('f1', 'fold'))), 'f1', 3),
    /Nothing to fold from t3: the span holds no responses\. To fold the work before it, use since "t1"/
  );
  // Nothing before it holds a response either.
  assert.throws(
    () => planFold(entries(user('first', 1), assistant(call('f1', 'fold'))), 'f1', 1),
    /Nothing to fold from t1: the span holds no responses\.$/
  );

  // After a note, the work not yet folded starts behind it; with none, there is no stamp to name.
  const noted = (...later: AgentMessage[]) => {
    const { manager } = session([
      ...first.slice(0, 3),
      assistant(call('f0', 'fold')),
      {
        ...result('f0', 'fold', 'first note', 3),
        details: { kind: 'fold', handle: 'ep-0', headline: 'first note' }
      } as AgentMessage
    ]);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-0' });
    for (const message of later) manager.appendMessage(message as Message);
    return manager.buildSessionProjection().entries;
  };
  assert.throws(
    () =>
      planFold(
        noted(
          user('second', 4),
          assistant(call('y', 'bash')),
          result('y', 'bash', 'out', 5),
          assistant('done'),
          user('third', 6),
          assistant(call('f1', 'fold'))
        ),
        'f1',
        6
      ),
    /the span holds no responses\. To fold the work before it, use since "t4"/
  );
  assert.throws(
    () => planFold(noted(user('second', 4), assistant(call('f1', 'fold'))), 'f1', 4),
    /Nothing to fold from t4: the span holds no responses\.$/
  );
});

test('work behind the latest note that no fold removed is found, with the stamp it starts at', () => {
  const { manager } = session([
    user('go', 1),
    assistant(call('a', 'bash')),
    result('a', 'bash', 'first', 2),
    assistant(call('b', 'bash')),
    result('b', 'bash', 'second', 3),
    assistant(call('f', 'fold'))
  ]);
  const entries = () => manager.buildSessionProjection().entries;
  assert.equal(unfoldedBehindNote(entries()), undefined);

  // A fold with until keeps the second command; its note lands after it.
  const plan = planFold(entries(), 'f', 2, { until: 2 });
  apply(manager, plan.edits);
  manager.appendMessage({
    ...result('f', 'fold', 'note', 4),
    details: { kind: 'fold', handle: plan.handle, headline: 'note' }
  } as Message);
  // A note that did not apply is no boundary.
  assert.equal(unfoldedBehindNote(entries()), undefined);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
  const left = unfoldedBehindNote(entries());
  assert.equal(left?.stamp, 3);
  // The command and its result, not the response that called fold or the note.
  assert.deepEqual(
    left?.work.map((entry) => entry.messages[0]?.role),
    ['assistant', 'toolResult']
  );

  // A fold from there covers the note and leaves nothing behind the new one.
  manager.appendMessage(assistant(call('g', 'fold')));
  const merged = planFold(entries(), 'g', 3);
  assert.deepEqual(
    merged.covers.map((fold) => fold.handle),
    [plan.handle]
  );
  apply(manager, merged.edits);
  manager.appendMessage({
    ...result('g', 'fold', 'both', 5),
    details: { kind: 'fold', handle: merged.handle, headline: 'both' }
  } as Message);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: merged.handle });
  assert.equal(unfoldedBehindNote(entries()), undefined);
});
test('a fold that merges the notes starts at the earliest thing a fold removes behind them', () => {
  const fold = (manager: SessionManager, id: string, since: number, stamp: number) => {
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(manager.buildSessionProjection().entries, id, since);
    apply(manager, plan.edits);
    manager.appendMessage({
      ...result(id, 'fold', 'note', stamp),
      details: { kind: 'fold', handle: plan.handle, headline: 'note' }
    } as Message);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
  };
  const start = (manager: SessionManager) =>
    mergeStart(manager.buildSessionProjection().entries)?.stamp;
  const command = (id: string, stamp: number) => [
    assistant(call(id, 'bash')),
    result(id, 'bash', 'out', stamp)
  ];

  const { manager } = session([user('first', 1), ...command('a', 2)]);
  assert.equal(start(manager), undefined, 'no note stands');
  fold(manager, 'f1', 1, 3);
  // The user message stays, so a fold from the note's own stamp removes as much as one from t1.
  assert.equal(start(manager), 3);
  manager.appendMessage(assistant('done'));
  manager.appendMessage(user('second', 4));
  for (const message of command('b', 5)) manager.appendMessage(message);
  fold(manager, 'f2', 4, 6);
  assert.equal(start(manager), 3, 'the earliest note, not the latest');

  // A command left before the earliest note is where it starts.
  const { manager: left } = session([user('first', 1), ...command('a', 2), ...command('b', 3)]);
  fold(left, 'f1', 3, 4);
  assert.equal(start(left), 2);

  // And so is a user message that a fold would cut and none did.
  const { manager: long } = session([user('x'.repeat(5000), 1), ...command('a', 2)]);
  fold(long, 'f1', 2, 3);
  assert.equal(start(long), 1);

  // What stands before the first stamp, as after a compaction that kept the end of a turn,
  // is out of reach of any fold and does not hide the notes after it.
  const { manager: tail } = session([assistant('kept'), user('first', 1), ...command('a', 2)]);
  fold(tail, 'f1', 1, 3);
  assert.equal(start(tail), 3);
});
test('growth counts the work after a note and what was written after its fold, and no note', () => {
  const { manager } = session([
    user('first', 1),
    assistant(call('a', 'bash')),
    result('a', 'bash', 'out', 2),
    assistant(call('f', 'fold', { note: 'n'.repeat(8000) }))
  ]);
  const entries = () => manager.buildSessionProjection().entries;
  const near = (tokens: number, expected: number) =>
    assert.ok(tokens >= expected && tokens < expected + 20, `${tokens} for ${expected}`);
  const plan = planFold(entries(), 'f', 1);
  apply(manager, plan.edits);
  manager.appendMessage({
    ...result('f', 'fold', 'n'.repeat(8000), 3),
    details: { kind: 'fold', handle: plan.handle, headline: 'note', note: 'n'.repeat(8000) }
  } as Message);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
  // The note and its call, 2,000 tokens each, are not work. A fold that covers the note
  // keeps it as written, so it frees the call only.
  assert.equal(nudgeGrowth(entries(), false), 0);
  near(tokensBehindNote(entries()), 2000);

  // The reply after the fold has no stamp of its own: only a fold from the note's reaches it.
  manager.appendMessage(assistant('r'.repeat(4000)));
  assert.equal(nudgeGrowth(entries()), 0);
  near(nudgeGrowth(entries(), false), 1000);
  near(tokensBehindNote(entries()), 3000);

  // The first stamp after the note is on a result; the span starts at the call that made it.
  manager.appendMessage(assistant(call('b', 'bash', { command: 'x'.repeat(4000) })));
  manager.appendMessage(result('b', 'bash', 'out', 4));
  near(nudgeGrowth(entries()), 1000);
  near(nudgeGrowth(entries(), false), 2000);
  near(tokensBehindNote(entries()), 3000);
});
test('work behind a note starts at the stamp a fold of it needs, whatever stands around it', () => {
  // A fold from `since`, applied, with its note at `stamp`.
  const fold = (manager: SessionManager, id: string, since: number, stamp: number) => {
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(manager.buildSessionProjection().entries, id, since);
    apply(manager, plan.edits);
    manager.appendMessage({
      ...result(id, 'fold', 'note', stamp),
      details: { kind: 'fold', handle: plan.handle, headline: 'note' }
    } as Message);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
  };
  // Two requests folded whole, the second with a reminder before its work, then a third
  // of which only the last command is folded.
  const left = (third: string, work: readonly AgentMessage[], last: number) => {
    const { manager } = session([
      user('first', 1),
      assistant(call('a', 'bash')),
      result('a', 'bash', 'out', 2)
    ]);
    fold(manager, 'f1', 2, 3);
    manager.appendMessage(assistant('done'));
    manager.appendMessage(user('second', 4));
    manager.appendCustomMessageEntry('episodes-open', 'Still open: the pin.', true);
    manager.appendMessage(assistant(call('b', 'bash')));
    manager.appendMessage(result('b', 'bash', 'out', 5));
    fold(manager, 'f2', 5, 6);
    manager.appendMessage(assistant('done'));
    manager.appendMessage(user(third, 7));
    for (const message of work) {
      if ('customType' in message) manager.appendCustomMessageEntry('episodes-open', 'x', true);
      else manager.appendMessage(message as Message);
    }
    manager.appendMessage(assistant(call('z', 'bash')));
    manager.appendMessage(result('z', 'bash', 'out', last));
    fold(manager, 'f3', last, last + 1);
    const found = unfoldedBehindNote(manager.buildSessionProjection().entries);
    return found && { stamp: found.stamp, work: found.work.map((entry) => entry.sourceEntry.type) };
  };
  const command = [assistant(call('c', 'bash')), result('c', 'bash', 'out', 8)];

  // The reminder behind the second note is no work: a fold from t4 would cover that note.
  assert.deepEqual(left('third', command, 9), { stamp: 8, work: ['message', 'message'] });
  // One that stands in the work goes with it; one before the work is not reached by its fold.
  const reminder = { customType: 'x' } as never;
  assert.deepEqual(left('third', [reminder, ...command, reminder], 9), {
    stamp: 8,
    work: ['message', 'message', 'custom_message']
  });
  // A user message counts when a fold would cut it, and not once one did.
  assert.deepEqual(left('x'.repeat(USER_TEXT_CAP + 1), command, 9), {
    stamp: 7,
    work: ['message', 'message', 'message']
  });
  const cut = `${'x'.repeat(USER_TEXT_CAP)}\n[9 more characters of this message are in ep-0a; recall can search them]`;
  assert.equal(left(cut, command, 9)?.stamp, 8);
  // The results of one response all start at it: the first of them names the start.
  const pair = [
    assistant(call('c', 'bash'), call('d', 'bash')),
    result('c', 'bash', 'out', 8),
    result('d', 'bash', 'out', 9)
  ];
  assert.equal(left('third', pair, 10)?.stamp, 8);
});

test('work since an entry is told from folding itself and from what other extensions remove', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('a', 'bash')),
    result('a', 'bash', 'first', 2),
    assistant(call('b', 'bash')),
    result('b', 'bash', 'second', 3),
    assistant(call('c', 'bash')),
    result('c', 'bash', 'third', 4)
  ]);
  const entries = () => manager.buildSessionProjection().entries;
  const since = (after: number) => {
    const { folded, left } = workSince(entries(), after);
    return { folded, left: left.map((entry) => entry.messages[0]?.role) };
  };
  const work = ['assistant', 'toolResult'];
  assert.deepEqual(since(0), { folded: false, left: [...work, ...work, ...work] });

  // Another extension removes the second command: gone, but no fold removed it, and the
  // fold below, whose span it lies in, does not either.
  manager.appendContextEdit(ids[4]!, null);
  manager.appendContextEdit(ids[3]!, null);
  assert.deepEqual(since(0), { folded: false, left: [...work, ...work] });

  // A fold with until removes the first command and keeps the rest.
  const fold = (
    id: string,
    since: number,
    until: number | undefined,
    stamp: number,
    land = true
  ) => {
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(entries(), id, since, until === undefined ? {} : { until });
    if (land) {
      manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
      apply(manager, plan.edits);
    }
    const details = {
      kind: 'fold',
      handle: plan.handle,
      headline: 'note',
      firstEntryId: plan.firstEntryId,
      lastEntryId: plan.lastEntryId,
      ...(plan.keptFrom === undefined ? {} : { keptFrom: `t${plan.keptFrom}` })
    };
    manager.appendMessage({
      ...result(id, 'fold', 'note '.repeat(400), stamp),
      details
    } as Message);
    return plan;
  };
  const kept = fold('f', 2, 2, 5).handle;
  // Its note counts for nothing before the fold applied, nor as a choice to keep turns.
  assert.equal(since(0).folded, false);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: kept });
  const keptAt = entries().findIndex((entry) => entry.sourceEntry.id === manager.getLeafId()) - 1;
  assert.equal(unfoldedBehindNote(entries())?.keptAt, keptAt);
  // The command it removed lay before entry 2, the one it kept after: nothing folded since.
  assert.deepEqual(since(0), { folded: true, left: work });
  assert.deepEqual(since(2), { folded: false, left: work });

  // A fold from the note's stamp covers the note and removes no work, however long its own.
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: fold('g', 5, undefined, 6).handle });
  assert.deepEqual(since(2), { folded: false, left: work });
  // The choice to keep turns outlives the note that recorded it.
  assert.equal(unfoldedBehindNote(entries())?.keptAt, keptAt);

  // A fold with until whose edits never landed is no such choice.
  manager.appendMessage(assistant(call('d', 'bash')));
  manager.appendMessage(result('d', 'bash', 'fourth', 7));
  assert.equal(fold('h', 6, 6, 8, false).keptFrom, 7);
  assert.equal(unfoldedBehindNote(entries())?.keptAt, keptAt);
});

function earlierFold(applied: boolean) {
  const note = {
    ...result('f0', 'fold', 'first note', 3),
    details: { kind: 'fold', handle: 'ep-0', headline: 'first note', since: 't2' }
  };
  const { manager } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    assistant(call('f0', 'fold')),
    note
  ]);
  if (applied) manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-0' });
  for (const message of [
    assistant(call('y', 'bash')),
    result('y', 'bash', 'out', 4),
    assistant(call('f1', 'fold'))
  ]) {
    manager.appendMessage(message);
  }
  return manager.buildSessionProjection().entries;
}

test('a fold reaching behind an earlier fold note covers it', () => {
  const entries = earlierFold(true);

  const merged = planFold(entries, 'f1', 2);
  // Recorded without its text, the note is kept as its result stands.
  assert.deepEqual(merged.covers, [{ handle: 'ep-0', range: 't2', text: 'first note' }]);
  assert.equal(merged.anchorStamp, 1);
  // The note's own stamp starts the span at the response that called the earlier fold.
  assert.deepEqual(
    planFold(entries, 'f1', 3).covers.map((fold) => fold.handle),
    ['ep-0']
  );
  const after = planFold(entries, 'f1', 4);
  assert.deepEqual(after.covers, []);
  assert.equal(after.anchorStamp, 3);
});

// A result as the fold tool writes it: the note between harness lines, then the notes kept.
const noteBody = (handle: string, note: string) =>
  `[fold ${handle}: archived]\n\n${note}\n\nTo go back: recall.`;

/** A session in which each fold starts at the stamp of the note before it, and so covers it. */
function foldChain() {
  const { manager } = session([
    user('go', 1),
    assistant(call('a', 'bash')),
    result('a', 'bash', 'alpha-out', 2)
  ]);
  let stamp = 2;
  let from = 2;
  return (
    id: string,
    note: string,
    options: {
      /** The note as a handler left it in the result. */
      shown?: string;
      recorded?: boolean;
      /** What something did to all of the result afterwards. */
      edit?: (text: string) => string;
      /** A block before the result's own that holds the kept notes too, as written. */
      copy?: boolean;
    } = {}
  ) => {
    const since = from;
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(manager.buildSessionProjection().entries, id, since);
    const kept = renderCovered(plan.covers);
    const text = noteBody(plan.handle, options.shown ?? note);
    const whole = kept ? `${text}\n\n${kept}` : text;
    const message = result(id, 'fold', options.edit ? options.edit(whole) : whole, ++stamp);
    const copied = options.copy && kept ? [{ type: 'text' as const, text: kept }] : [];
    manager.appendMessage({
      ...message,
      content: [...copied, ...message.content],
      details: {
        kind: 'fold',
        handle: plan.handle,
        headline: id,
        since: `t${since}`,
        ...(stamp - 1 === since ? {} : { until: `t${stamp - 1}` }),
        covers: plan.covers.map((covered) => covered.handle),
        ...(kept ? { kept: keptMark(kept, copied.length) } : {}),
        ...(options.recorded === false ? {} : { note })
      }
    } as Message);
    apply(manager, plan.edits);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
    from = stamp;
    manager.appendMessage(assistant(call(`${id}-work`, 'bash')));
    manager.appendMessage(result(`${id}-work`, 'bash', 'out', ++stamp));
    return plan;
  };
}

test('a fold keeps the notes it covers and the notes under them, each once and as it stands', () => {
  const fold = foldChain();
  const body = noteBody;
  const alpha = 'Learned: alpha is 7.\n\nOpen: none.';
  const first = fold('f1', alpha);
  const second = fold('f2', 'Learned: beta is 8.');
  const third = fold('f3', 'third');
  assert.deepEqual(first.covers, []);
  assert.deepEqual(second.covers, [{ handle: first.handle, range: 't2', text: alpha }]);
  // The first note is out of context by now: it is read from under the second.
  const two = [
    { handle: first.handle, range: 't2', text: alpha },
    { handle: second.handle, range: 't3-t4', text: 'Learned: beta is 8.' }
  ];
  assert.deepEqual(third.covers, two);
  assert.equal(
    renderCovered(third.covers),
    [
      'Notes of the earlier folds covered here, oldest first, as they were written:',
      '',
      `[note of ${first.handle}, t2, ${alpha.length} characters]`,
      alpha,
      '',
      `[note of ${second.handle}, t3-t4, 19 characters]`,
      'Learned: beta is 8.'
    ].join('\n')
  );
  assert.equal(coveredTokens(third.covers), Math.ceil(alpha.length / 4) + 5);
  assert.equal(renderCovered([]), undefined);

  // A note that a handler rewrote, and one of a fold made before notes were recorded, are
  // kept as their results stand: what the agent wrote does not come back over the handler's
  // text, and nothing of the older fold's result is dropped.
  const fourth = fold('f4', 'The key is 42.', { shown: 'The key is [redacted].' });
  const fifth = fold('f5', 'An older note.', { recorded: false });
  const sixth = fold('f6', 'sixth');
  assert.deepEqual(sixth.covers, [
    ...two,
    { handle: third.handle, range: 't5-t6', text: 'third' },
    { handle: fourth.handle, range: 't7-t8', text: body(fourth.handle, 'The key is [redacted].') },
    { handle: fifth.handle, range: 't9-t10', text: body(fifth.handle, 'An older note.') }
  ]);
});

test('a note that reads like the notes kept under one stays one note, through later folds', () => {
  const fold = foldChain();
  const title = 'Notes of the earlier folds covered here, oldest first, as they were written:';
  const first = fold('f1', 'Learned: alpha is 7.');
  // A note that quotes the title, with its own words after it; one that quotes a kept note
  // with the line over it, naming a fold that is kept too; and one that ends like kept notes.
  const quoting = `Saw this in a note:\n\n${title}\n\nand nothing under it.`;
  const naming = `It said:\n\n[note of ${first.handle}, t2, 4 characters]\nfake\n\nand then more.`;
  const ending = `Ends so:\n\n${title}\n\n[note of ${first.handle}, t2, 4 characters]\nfake`;
  const second = fold('f2', quoting);
  const third = fold('f3', naming, { recorded: false });
  const fourth = fold('f4', ending);
  const fifth = fold('f5', 'fifth');
  const sixth = fold('f6', 'sixth');
  const four = [
    { handle: first.handle, range: 't2', text: 'Learned: alpha is 7.' },
    { handle: second.handle, range: 't3-t4', text: quoting },
    { handle: third.handle, range: 't5-t6', text: noteBody(third.handle, naming) },
    { handle: fourth.handle, range: 't7-t8', text: ending }
  ];
  assert.deepEqual(third.covers, four.slice(0, 2));
  assert.deepEqual(fifth.covers, four);
  assert.deepEqual(sixth.covers, [
    ...four,
    { handle: fifth.handle, range: 't9-t10', text: 'fifth' }
  ]);
  // And so on: what a fold keeps is what the one before it kept, and its note.
  let last = sixth;
  for (let round = 7; round < 12; round++) {
    const next = fold(`f${round}`, `note ${round}`);
    assert.deepEqual(next.covers.slice(0, -1), last.covers);
    last = next;
  }
});

test('what does not read back as the fold wrote it is kept as it stands', () => {
  const fold = foldChain();
  const first = fold('f1', 'Ran it. alpha is 7.');
  // A handler rewrote this note, whose words as the agent wrote them stand in the note kept
  // under it: they do not come back over the handler's.
  const second = fold('f2', 'Ran it.', { shown: 'Checked.' });
  // Something redacted a note kept under this one, so the notes under it no longer read back
  // by their lengths: they stay in its text, and its own note does not stand for all of it.
  const third = fold('f3', 'third', {
    edit: (text) => text.replace('alpha is 7', 'alpha is [redacted]')
  });
  const fourth = fold('f4', 'fourth');
  // All of this result was replaced, by text that happens to hold the short note: the
  // notes it kept are out of context with it.
  const fifth = fold('f5', 'ok', { edit: () => 'The token was revoked.' });
  const sixth = fold('f6', 'sixth');
  assert.deepEqual(third.covers, [
    { handle: first.handle, range: 't2', text: 'Ran it. alpha is 7.' },
    { handle: second.handle, range: 't3-t4', text: noteBody(second.handle, 'Checked.') }
  ]);
  assert.equal(fourth.covers.length, 1);
  assert.equal(fourth.covers[0]!.handle, third.handle);
  for (const fact of ['alpha is [redacted]', 'Checked.', 'third\n']) {
    assert.equal(fourth.covers[0]!.text.split(fact).length - 1, 1, fact);
  }
  // A later fold keeps that text whole, under the one handle.
  assert.deepEqual(fifth.covers, [
    ...fourth.covers,
    { handle: fourth.handle, range: 't7-t8', text: 'fourth' }
  ]);
  assert.deepEqual(sixth.covers, [
    { handle: fifth.handle, range: 't9-t10', text: 'The token was revoked.' }
  ]);

  // A note that ends in kept notes of its own making, right to the last character and for
  // the very folds kept, under a fold whose real kept notes no longer stand as written: the
  // made-up ones are not taken for them.
  const made = foldChain();
  const one = made('m1', 'Learned: gamma is 9.');
  const title = 'Notes of the earlier folds covered here, oldest first, as they were written:';
  const own = `ep-${createHash('sha256').update('m2').digest('hex').slice(0, 8)}`;
  const quoted = `Prose before.\n\n${title}\n\n[note of ${one.handle}, t2, 4 characters]\nfake\n\n[note of ${own}, t3-t4, 4 characters]\nfake`;
  const two = made('m2', quoted);
  assert.equal(two.handle, own);
  const holder = made('m3', 'holder', {
    edit: (text) => text.replace('gamma is 9', 'gamma is [redacted]')
  });
  const next = made('m4', 'next');
  assert.deepEqual(
    holder.covers.map((note) => note.handle),
    [one.handle, two.handle]
  );
  assert.equal(next.covers.length, 1);
  for (const fact of ['gamma is [redacted]', 'Prose before.', 'fake\n', 'holder\n']) {
    assert.equal(next.covers[0]!.text.split(fact).length - 1, 1, fact);
  }

  // A result that holds the start of the note, cut short, before the whole note, as the
  // output cap leaves a script's result: the agent's note stands for it.
  const capped = foldChain();
  capped('p1', 'Learned: eta is 5. And more.', {
    edit: (text) =>
      `${text.slice(0, 44)}\n\n[episodes] 9 of 99 characters left out here.\n\nrow row\n${text}`
  });
  assert.deepEqual(
    capped('p2', 'second').covers.map((note) => note.text),
    ['Learned: eta is 5. And more.']
  );

  // The kept notes changed where the fold wrote them, and an exact copy of them as written
  // stands in another block of the result: the copy is not read in their place.
  const twice = foldChain();
  twice('d1', 'Learned: zeta is 3.');
  twice('d2', 'second', {
    copy: true,
    edit: (text) => text.replace(/zeta is 3/, 'zeta is [redacted]')
  });
  const moved = twice('d3', 'third');
  assert.equal(moved.covers.length, 1);
  assert.match(moved.covers[0]!.text, /zeta is \[redacted\]/);
  // Unchanged, they are read from the block the fold recorded, and the copy is not kept.
  const again = foldChain();
  again('g1', 'Learned: zeta is 3.');
  again('g2', 'second', { copy: true });
  assert.deepEqual(
    again('g3', 'third').covers.map((note) => note.text),
    ['Learned: zeta is 3.', 'second']
  );

  // Changed by one character, at the same length, the kept notes stay whole as well.
  const same = foldChain();
  same('s1', 'Learned: delta is 1.');
  same('s2', 'second');
  const changed = same('s3', 'third', { edit: (text) => text.replace('delta is 1', 'delta is 2') });
  const later = same('s4', 'later');
  assert.deepEqual(
    later.covers.map((note) => note.handle),
    [changed.handle]
  );
  assert.match(later.covers[0]!.text, /, 20 characters\]\nLearned: delta is 2\.\n\n\[note of /);

  // A note that is empty reads back, also as the last one.
  const empty = foldChain();
  empty('e1', 'x', { edit: () => '' });
  empty('e2', 'second');
  assert.deepEqual(
    empty('e3', 'third').covers.map((note) => note.text),
    ['', 'second']
  );

  // A result that lost its end: the last note under it is shorter than its line says.
  const cut = foldChain();
  cut('c1', 'Learned: beta is 8.');
  const short = cut('c2', 'short', { edit: (text) => text.slice(0, -3) });
  const after = cut('c3', 'after');
  assert.equal(after.covers.length, 1);
  assert.equal(after.covers[0]!.handle, short.handle);
  assert.match(after.covers[0]!.text, /\n\nshort\n\n[\s\S]*, 19 characters\]\nLearned: beta is$/);
});

test('kept notes stand in the order they were written, also when a fold with until kept an older one', () => {
  const { manager } = session([user('go', 1)]);
  let stamp = 1;
  const work = (id: string) => {
    manager.appendMessage(assistant(call(id, 'bash')));
    manager.appendMessage(result(id, 'bash', 'out', ++stamp));
  };
  const fold = (id: string, since: number, until?: number) => {
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(
      manager.buildSessionProjection().entries,
      id,
      since,
      until === undefined ? {} : { until }
    );
    const kept = renderCovered(plan.covers);
    manager.appendMessage({
      ...result(id, 'fold', [`${id} text`, kept ?? ''].join('\n\n'), ++stamp),
      details: {
        kind: 'fold',
        handle: plan.handle,
        headline: id,
        since: `t${since}`,
        covers: plan.covers.map((covered) => covered.handle),
        ...(kept ? { kept: keptMark(kept) } : {}),
        note: id
      }
    } as Message);
    apply(manager, plan.edits);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
    return plan;
  };
  work('a');
  fold('oldest', 2);
  work('b');
  fold('middle', 4);
  // Covers the oldest note only, and stands after the middle one.
  assert.deepEqual(
    fold('newest', 3, 3).covers.map((covered) => covered.text),
    ['oldest text']
  );
  // From the stamp of the middle note: it comes first in the span, and was written second.
  assert.deepEqual(
    fold('last', 5).covers.map((covered) => covered.text),
    ['oldest text', 'middle text', 'newest text']
  );
});

test('a fold over a span that an earlier fold with until emptied holds only what it removed', async () => {
  const { manager } = session([
    user('go', 1),
    assistant(call('a', 'bash')),
    result('a', 'bash', 'alpha-out', 2),
    assistant(call('b', 'bash')),
    result('b', 'bash', 'beta-out', 3),
    assistant(call('c', 'bash')),
    result('c', 'bash', 'gamma-out', 4)
  ]);
  // Each fold stops before its own call, so the first note is after the second span.
  const fold = (id: string, since: number, until: number, stamp: number) => {
    manager.appendMessage(assistant(call(id, 'fold')));
    const plan = planFold(manager.buildSessionProjection().entries, id, since, { until });
    manager.appendMessage({
      ...result(id, 'fold', `${id} note`, stamp),
      details: {
        kind: 'fold',
        handle: plan.handle,
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        covers: plan.covers.map((covered) => covered.handle)
      }
    } as Parameters<typeof manager.appendMessage>[0]);
    apply(manager, plan.edits);
    manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });
    return plan;
  };
  const inner = fold('f1', 3, 3, 5);
  const outer = fold('f2', 2, 4, 6);
  assert.deepEqual(outer.covers, []);

  const threads = await new ThreadIndex(manager, new Map()).onBranch();
  const held = (handle: string) => {
    const thread = threads.find((candidate) => candidate.handle === handle)!;
    return {
      archived: JSON.stringify(thread.messages).match(/(alpha|beta|gamma)-out/g),
      asked: JSON.stringify(thread.context!()).match(/(alpha|beta|gamma)-out/g)
    };
  };
  assert.deepEqual(held(inner.handle), {
    archived: ['beta-out'],
    asked: ['alpha-out', 'beta-out']
  });
  assert.deepEqual(held(outer.handle), {
    archived: ['alpha-out', 'gamma-out'],
    asked: ['alpha-out', 'gamma-out']
  });
});

test('a fold whose edits never landed is not covered', () => {
  // Its span is still in context, so its result is only a tool result.
  const plan = planFold(earlierFold(false), 'f1', 2);
  assert.equal(plan.anchorStamp, 1);
  assert.deepEqual(plan.covers, []);
});

test('a user message an earlier fold capped is not capped again', () => {
  const { manager, ids } = session([
    user('x'.repeat(USER_TEXT_CAP * 2), 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    assistant(call('f0', 'fold'))
  ]);
  const first = planFold(manager.buildSessionProjection().entries, 'f0', 1);
  assert.ok(first.edits.some((edit) => edit.targetId === ids[0]));
  apply(manager, first.edits);
  manager.appendMessage(
    Object.assign(result('f0', 'fold', 'first note', 3), {
      details: { kind: 'fold', handle: first.handle, headline: 'first note' }
    })
  );
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: first.handle });
  for (const message of [
    assistant(call('y', 'bash')),
    result('y', 'bash', 'out', 4),
    assistant(call('f1', 'fold'))
  ]) {
    manager.appendMessage(message);
  }
  const second = planFold(manager.buildSessionProjection().entries, 'f1', 1);
  assert.deepEqual(
    second.covers.map((fold) => fold.handle),
    [first.handle]
  );
  assert.ok(!second.edits.some((edit) => edit.targetId === ids[0]));
});

test('rejects stamps that are missing, after the call, or leave nothing to fold', () => {
  const { manager } = session(buildLoop);
  const entries = manager.buildSessionProjection().entries;

  assert.throws(
    () => planFold(entries, 'f1', 99, { latestStamp: 5 }),
    /t99 does not exist; the latest stamp is t5\. The earliest stamp you can fold from is t1\./
  );
  assert.throws(() => planFold(entries, 'missing', 2), /called fold is not in the current context/);

  const { manager: late } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    user('one more thing', 3),
    assistant(call('f', 'fold'))
  ]);
  assert.throws(
    () => planFold(late.buildSessionProjection().entries, 'f', 3),
    /holds no responses/
  );
});

test('rejects a second fold call in the same response', () => {
  const { manager } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    assistant(call('f1', 'fold'), call('f2', 'fold'))
  ]);
  const entries = manager.buildSessionProjection().entries;

  assert.throws(() => planFold(entries, 'f2', 2), /Only one fold per response/);
  assert.equal(planFold(entries, 'f1', 2).anchorStamp, 1);
});

test('caps long user messages in the span but keeps their stamp', () => {
  const paste = 'x'.repeat(USER_TEXT_CAP + 500);
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2),
    user(paste, 3),
    assistant(call('y', 'bash')),
    result('y', 'bash', 'out', 4),
    assistant(call('f', 'fold'))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f', 2);
  const edit = plan.edits.find((candidate) => candidate.targetId === ids[3]);

  assert.ok(edit?.replacement);
  const content = edit.replacement.content as { type: string; text: string }[];
  assert.match(content[0]!.text, /500 more characters of this message are in ep-/);
  assert.equal(content.at(-1)!.text, '[t3]');
});

test('omits extension messages inside the span', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'out', 2)
  ]);
  manager.appendCustomMessageEntry('episodes-fold-request', 'fold now', true);
  const foldId = manager.appendMessage(assistant(call('f', 'fold')));
  const plan = planFold(manager.buildSessionProjection().entries, 'f', 1);

  assert.equal(plan.firstEntryId, ids[0]);
  assert.equal(plan.anchorStamp, undefined);
  assert.equal(plan.edits.length, 3);
  assert.ok(!plan.edits.some((edit) => edit.targetId === foldId));
});

test('renders the note with disk changes, kept turns and the handle', () => {
  const text = renderNote({
    handle: 'ep-12345678',
    since: 't2',
    until: 't9',
    messages: 4,
    tokens: 41_000,
    note: '## Changed\nregenerated src/client',
    changes: [{ status: 'M', path: '/repo/src/client/api.ts' }],
    touched: [],
    keptUserTurns: ['t4'],
    keptFrom: 't10',
    skills: ['/home/me/skills/tdd/SKILL.md'],
    cwd: '/repo'
  });

  assert.match(text, /^\[fold ep-12345678: 4 messages, t2-t9, ~41k tokens, archived\]/);
  assert.match(text, /Changed on disk \(working tree diff\):\nM src\/client\/api\.ts/);
  assert.match(text, /stay above with their stamps: t4\./);
  assert.match(text, /The turns from t10 on were not folded and stay above\./);
  assert.match(
    text,
    /Skills read inside the span, .*: \/home\/me\/skills\/tdd\/SKILL\.md\. Read a skill again/
  );
  assert.match(text, /ask\("ep-12345678"/);
  assert.equal(headline('## Changed\nregenerated'), 'Changed');
});

test('names the skill files a span read, not the ones it wrote', () => {
  const messages = [
    assistant(call('a', 'read', { path: '/skills/tdd/SKILL.md' })),
    assistant(call('b', 'read', { path: 'docs/SKILL.md.bak' })),
    assistant(call('c', 'write', { path: '/skills/new/SKILL.md' })),
    assistant(call('d', 'read', { path: 'skills/review/SKILL.md' })),
    // As pi's read takes it: under the home directory, not under the working directory.
    assistant(call('e', 'read', { path: '~/.pi/skills/plan/SKILL.md' }))
  ];
  assert.deepEqual(skillPaths(messages, '/repo'), [
    '/skills/tdd/SKILL.md',
    '/repo/skills/review/SKILL.md',
    join(homedir(), '.pi/skills/plan/SKILL.md')
  ]);
});

test('a call that failed touched no file', () => {
  const failed = { ...result('b', 'write', 'EACCES'), isError: true };
  const script = {
    ...result('c', 'codemode', 'done'),
    nestedCalls: {
      calls: [
        { id: 'n1', name: 'write', arguments: { path: 'c.ts' }, status: 'error' as const },
        { id: 'n2', name: 'write', arguments: { path: 'd.ts' }, status: 'ok' as const }
      ],
      complete: true
    }
  };
  const messages = [
    assistant(call('a', 'read', { path: 'a.ts' }), call('b', 'write', { path: 'a.ts' })),
    result('a', 'read', 'a'),
    failed,
    assistant(call('c', 'codemode', {})),
    script
  ];
  assert.deepEqual(touchedPaths(messages, '/repo'), ['/repo/a.ts', '/repo/d.ts']);
  // The write of a.ts failed, so the span only read it.
  assert.deepEqual(touchedPaths(messages, '/repo', new Set(['write'])), ['/repo/d.ts']);
});

test('reads the Open section of a note, in the forms the template produces', () => {
  const note = [
    'Build fails with TS2322: fixed by regenerating the client.',
    '- Changed: regenerated src/client.',
    '- Open: whether CI runs generate before build;',
    '  remove the debug flag in vite.config.ts.',
    '- Learned: nothing new.'
  ].join('\n');
  assert.equal(
    openItems(note),
    'whether CI runs generate before build; remove the debug flag in vite.config.ts.'
  );
  assert.equal(openItems('**Open:** check the migration'), 'check the migration');
  assert.equal(openItems('- Open: none.'), undefined);
  assert.equal(openItems('Changed: everything'), undefined);
});

test('after a compaction, stamps the summary mentions are gone and the kept ones can be folded', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('x', 'bash')),
    result('x', 'bash', 'build failed', 2),
    user('next', 3),
    assistant(call('y', 'bash')),
    result('y', 'bash', 'out', 4)
  ]);
  manager.appendCompaction('Earlier: at t2 the build failed.', ids[3]!, 1000);
  for (const message of [
    assistant(call('z', 'bash')),
    result('z', 'bash', 'out', 5),
    assistant(call('f', 'fold'))
  ]) {
    manager.appendMessage(message);
  }
  const entries = manager.buildSessionProjection().entries;

  assert.throws(
    () => planFold(entries, 'f', 2, { latestStamp: 5 }),
    /t2 is no longer in your context: it was compacted or already folded\. The earliest stamp you can fold from is t3\./
  );
  const plan = planFold(entries, 'f', 3, { latestStamp: 5 });
  assert.equal(plan.anchorStamp, undefined);
  assert.equal(plan.keptUserTurns.join(), 't3');
});

test('lists the turns a fold can start from, newest first, after the latest compaction', () => {
  const { manager } = session(buildLoop);
  assert.deepEqual(
    foldableTurns(manager.buildSessionProjection().entries).map((turn) => turn.label),
    [
      't5  bash: build passed',
      't4  you: do not edit generated files',
      't3  bash: src/client/api.ts(12,5): error TS2322',
      't2  read: export function claim() {}',
      't1  you: add the claim endpoint'
    ]
  );
  assert.deepEqual(
    foldableTurns(earlierFold(true)).map((turn) => turn.stamp),
    [4, 3, 2, 1]
  );
});

test('a fold inside a fork that covers an earlier one holds only what it removed itself', async () => {
  const fold = (handle: string, covers?: string[]) => ({
    kind: 'fold' as const,
    handle,
    since: 't1',
    firstEntryId: '',
    lastEntryId: '',
    messages: 0,
    tokens: 0,
    headline: handle,
    keptUserTurns: [],
    files: [],
    ...(covers ? { covers } : {})
  });
  const messages = [
    assistant(call('a', 'bash')),
    result('a', 'bash', 'raw-a', 2),
    assistant(call('f0', 'fold')),
    result('f0', 'fold', 'note-a', 3),
    assistant(call('b', 'bash')),
    result('b', 'bash', 'raw-b', 4),
    assistant(call('f1', 'fold')),
    result('f1', 'fold', 'note-b', 5)
  ];
  const archive: ForkArchive = {
    handle: 'fk-1',
    task: 't',
    parentEntryId: 'p',
    messages,
    folds: [
      { record: fold('ep-a'), first: 0, last: 1, omit: [0, 1] },
      { record: fold('ep-b', ['ep-a']), first: 0, last: 5, omit: [2, 3, 4, 5] }
    ]
  } as ForkArchive;
  const { manager } = session([
    user('go', 1),
    Object.assign(result('k', 'fork', 'done', 6), {
      details: {
        kind: 'fork',
        handle: 'fk-1',
        task: 't',
        parentEntryId: 'p',
        turns: 1,
        headline: 'h',
        files: []
      }
    })
  ]);
  const threads = await new ThreadIndex(manager, new Map([['fk-1', archive]])).onBranch();
  const texts = (handle: string) =>
    threads
      .find((thread) => thread.handle === handle)!
      .messages.flatMap((message) => (message.role === 'toolResult' ? [readStamp(message)] : []));
  assert.deepEqual(texts('ep-a'), [2]);
  assert.deepEqual(texts('ep-b'), [3, 4]);
});
