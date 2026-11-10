import assert from 'node:assert/strict';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
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
  cutShortFold,
  foldableTurns,
  headline,
  openItems,
  planFold,
  renderNote,
  USER_TEXT_CAP
} from '../src/fold.js';
import { skillPaths, touchedPaths } from '../src/messages.js';
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
    /Nothing to fold from t4 to t4: the span holds no responses/
  );
});
function earlierFold(applied: boolean) {
  const note = {
    ...result('f0', 'fold', 'first note', 3),
    details: { kind: 'fold', handle: 'ep-0', headline: 'first note' }
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
  assert.deepEqual(merged.covers, [{ handle: 'ep-0', headline: 'first note' }]);
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
