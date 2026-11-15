import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  type Message,
  type TextContent,
  type ThinkingContent,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import {
  convertToLlm,
  createAgentSession,
  createCodemodeExtension,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager
} from '@earendil-works/pi-coding-agent';
import {
  appliedFolds,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_STARTED_TYPE,
  FOLD_UNFOLD_STARTED_TYPE,
  FOLD_UNFOLDED_TYPE,
  type FoldDetails,
  type ForkArchive,
  type ForkDetails,
  foldRecords,
  ThreadIndex,
  threadRecords
} from '../src/archive.js';
import { cacheAnchor } from '../src/cache.js';
import {
  completeCutShortFolds,
  cutShortFold,
  planFold,
  planUnfold,
  USER_TEXT_CAP,
  workSince
} from '../src/fold.js';
import { ForkFolds } from '../src/fork-fold.js';
import episodes from '../src/index.js';
import { nudgeGrowth } from '../src/nudge.js';
import { EpisodeState } from '../src/state.js';
import { episodeStats } from '../src/stats.js';
import { assistant, call, result, session, user } from './helpers.js';

function assertPaired(messages: readonly AgentMessage[]) {
  const llm = convertToLlm([...messages]);
  const open = new Set<string>();
  for (const message of llm) {
    if (message.role === 'assistant') {
      for (const block of message.content) {
        if (block.type === 'toolCall') open.add(block.id);
      }
    } else if (message.role === 'toolResult') {
      assert.ok(open.has(message.toolCallId), `unpaired toolResult ${message.toolCallId}`);
    }
  }
}

function applyDrafts(
  manager: SessionManager,
  drafts: readonly (
    | { type: 'context_edit'; targetId: string; replacement: { content: unknown } | null }
    | { type: 'custom'; customType: string; data: unknown }
  )[]
) {
  for (const draft of drafts) {
    if (draft.type === 'context_edit') {
      manager.appendContextEdit(
        draft.targetId,
        draft.replacement as Parameters<SessionManager['appendContextEdit']>[1]
      );
    } else {
      manager.appendCustomEntry(draft.customType, draft.data);
    }
  }
}

test('unfold happy path restores assistant, toolResult, custom_message and capped user entries as they were, and replaces the note', () => {
  const longUser = `${'u'.repeat(USER_TEXT_CAP + 500)}\n`;
  const { manager, ids } = session([
    user('before', 1),
    user(longUser, 2),
    {
      ...assistant('inside'),
      content: [
        { type: 'thinking', thinking: 'inner reasoning', thinkingSignature: 'sig-inner' },
        { type: 'toolCall', id: 'c1|fc_inner', name: 'bash', arguments: { command: 'echo hi' } }
      ]
    },
    result('c1|fc_inner', 'bash', 'hi', 3)
  ]);
  const customId = manager.appendCustomMessageEntry('episodes-nudge', 'nudge inside span', true);
  manager.appendMessage(assistant(call('f1', 'fold', { since: 't2', note: 'done' })));

  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const noteMsg = Object.assign(
    result(
      'f1',
      'fold',
      `[fold ${plan.handle}: 3 messages, t2-t3, ~1k tokens, archived]\n\ndone`,
      4
    ),
    {
      details: {
        kind: 'fold' as const,
        handle: plan.handle,
        since: 't2',
        until: 't3',
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        messages: plan.omitted.length,
        tokens: 100,
        headline: 'done',
        keptUserTurns: plan.keptUserTurns,
        files: []
      }
    }
  );
  manager.appendMessage(noteMsg);
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_STARTED_TYPE,
      data: { handle: plan.handle, edits: plan.edits }
    },
    ...plan.edits,
    { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: plan.handle } }
  ]);

  // Add a later signed assistant turn and an unfold call.
  const laterId = manager.appendMessage({
    ...assistant('after fold'),
    content: [
      {
        type: 'thinking',
        thinking: '[Reasoning redacted]',
        thinkingSignature: 'opaque',
        redacted: true
      }
    ]
  });
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));

  assert.ok(appliedFolds(manager.getBranch()).has(plan.handle));
  const unfold = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    plan.handle,
    { callerId: 'u1' }
  );
  manager.appendMessage(result('u1', 'unfold', `[unfolded ${plan.handle}]`, 5));
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_UNFOLD_STARTED_TYPE,
      data: { handle: plan.handle, edits: unfold.edits }
    },
    ...unfold.edits,
    { type: 'custom', customType: FOLD_UNFOLDED_TYPE, data: { handle: plan.handle } }
  ]);

  assert.equal(appliedFolds(manager.getBranch()).has(plan.handle), false);
  assert.equal(foldRecords(manager.getBranch()).length, 0);
  assert.equal(threadRecords(manager.getBranch()).length, 0);

  const projected = manager.buildSessionProjection();
  const byId = new Map(projected.entries.map((e) => [e.sourceEntry.id, e.messages]));
  // Capped user message is restored in full with its stamp.
  const restoredUser = byId.get(ids[1]!)![0] as Message & { role: 'user' };
  const userText = Array.isArray(restoredUser.content)
    ? restoredUser.content.map((b) => (b.type === 'text' ? b.text : '')).join('')
    : restoredUser.content;
  assert.ok(userText.includes('u'.repeat(USER_TEXT_CAP + 500)));
  assert.ok(!userText.includes('more characters of this message are in'));

  // Custom message inside the span is restored.
  assert.equal(byId.get(customId)!.length, 1);

  // The restored response is as the provider sent it, signature included.
  const restoredAsst = byId.get(ids[2]!)![0] as AssistantMessage;
  assert.deepEqual(restoredAsst.content, [
    { type: 'thinking', thinking: 'inner reasoning', thinkingSignature: 'sig-inner' },
    { type: 'toolCall', id: 'c1|fc_inner', name: 'bash', arguments: { command: 'echo hi' } }
  ]);
  assertPaired(projected.messages);

  // Old fold note is replaced with [unfolded ep-...] + its [t4] stamp, keeping toolCall f1 paired.
  const noteAfter = byId.get(unfold.noteEntryId)![0] as Message & { role: 'toolResult' };
  assert.deepEqual(noteAfter.content, [
    { type: 'text', text: `[unfolded ${plan.handle}]` },
    { type: 'text', text: '[t4]' }
  ]);
  assert.deepEqual(cacheAnchor(projected.messages, appliedFolds(manager.getBranch())), {
    stamp: '[t2]'
  });

  // Neither the fold nor the unfold edited a response outside the span.
  assert.deepEqual((byId.get(laterId)![0] as AssistantMessage).content, [
    {
      type: 'thinking',
      thinking: '[Reasoning redacted]',
      thinkingSignature: 'opaque',
      redacted: true
    }
  ]);
  const edited = new Set(
    manager.getBranch().flatMap((entry) => (entry.type === 'context_edit' ? [entry.targetId] : []))
  );
  assert.equal(edited.has(laterId), false);
});

test('unfold with covering folds: ep-A is refused while ep-B covers it; unfolding ep-B keeps ep-A folded; then unfolding ep-A works', () => {
  const longA = 'a'.repeat(USER_TEXT_CAP + 200);
  const { manager } = session([
    user(longA, 1),
    assistant(call('c1', 'bash', { command: 'echo a' })),
    result('c1', 'bash', 'alpha', 2),
    assistant(call('fA', 'fold', { since: 't1', note: 'note A' }))
  ]);

  // Apply ep-A.
  const planA = planFold(manager.buildSessionProjection().entries, 'fA', 1);
  const noteAText = `[fold ${planA.handle}: 2 messages, t1-t2, ~10 tokens, archived]\n\nnote A`;
  const noteAId = manager.appendMessage(
    Object.assign(result('fA', 'fold', noteAText, 3), {
      details: {
        kind: 'fold' as const,
        handle: planA.handle,
        since: 't1',
        until: 't2',
        firstEntryId: planA.firstEntryId,
        lastEntryId: planA.lastEntryId,
        messages: planA.omitted.length,
        tokens: 10,
        headline: 'note A',
        note: 'note A',
        keptUserTurns: planA.keptUserTurns,
        files: []
      }
    })
  );
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_STARTED_TYPE,
      data: { handle: planA.handle, edits: planA.edits }
    },
    ...planA.edits,
    { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: planA.handle } }
  ]);

  // More work, then ep-B covers ep-A (starting from t1).
  manager.appendMessage(assistant(call('c2', 'bash', { command: 'echo b' })));
  manager.appendMessage(result('c2', 'bash', 'beta', 4));
  manager.appendMessage(assistant(call('fB', 'fold', { since: 't1', note: 'note B' })));

  const planB = planFold(manager.buildSessionProjection().entries, 'fB', 1);
  assert.deepEqual(
    planB.covers.map((c) => c.handle),
    [planA.handle]
  );
  manager.appendMessage(
    Object.assign(result('fB', 'fold', `[fold ${planB.handle}]\n\nnote B`, 5), {
      details: {
        kind: 'fold' as const,
        handle: planB.handle,
        since: 't1',
        until: 't4',
        firstEntryId: planB.firstEntryId,
        lastEntryId: planB.lastEntryId,
        covers: [planA.handle],
        messages: planB.omitted.length,
        tokens: 20,
        headline: 'note B',
        note: 'note B',
        keptUserTurns: planB.keptUserTurns,
        files: []
      }
    })
  );
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_STARTED_TYPE,
      data: { handle: planB.handle, edits: planB.edits }
    },
    ...planB.edits,
    { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: planB.handle } }
  ]);

  // 1. Unfolding ep-A while ep-B covers it is rejected naming ep-B.
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: planA.handle })));
  assert.throws(
    () =>
      planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, planA.handle, {
        callerId: 'u1'
      }),
    new RegExp(`${planA.handle} is covered by ${planB.handle}; unfold ${planB.handle} first\\.`)
  );
  manager.appendMessage(result('u1', 'unfold', 'err', 6));

  // 2. Unfold ep-B: restores ep-B's own turns AND ep-A's note, while keeping ep-A's span omitted
  // and ep-A's user message capped!
  manager.appendMessage(assistant(call('uB', 'unfold', { handle: planB.handle })));
  const unfoldB = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    planB.handle,
    { callerId: 'uB' }
  );
  manager.appendMessage(result('uB', 'unfold', `[unfolded ${planB.handle}]`, 7));
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_UNFOLD_STARTED_TYPE,
      data: { handle: planB.handle, edits: unfoldB.edits }
    },
    ...unfoldB.edits,
    { type: 'custom', customType: FOLD_UNFOLDED_TYPE, data: { handle: planB.handle } }
  ]);

  assert.deepEqual([...appliedFolds(manager.getBranch())], [planA.handle]);
  const afterB = manager.buildSessionProjection();
  const textsAfterB = afterB.messages.map((m) => JSON.stringify(m)).join('\n');
  assert.ok(textsAfterB.includes('beta'), 'ep-B work restored');
  assert.ok(textsAfterB.includes('note A'), 'ep-A note restored');
  assert.ok(!textsAfterB.includes('alpha'), 'ep-A span stays folded');
  assert.ok(
    textsAfterB.includes(`more characters of this message are in ${planA.handle}`),
    'ep-A user cap stands'
  );
  const noteAEntry = afterB.entries.find((e) => e.sourceEntry.id === noteAId);
  assert.equal(noteAEntry?.messages.length, 1);
  assertPaired(afterB.messages);

  // 3. Now unfolding ep-A succeeds and restores alpha + uncapped user message.
  manager.appendMessage(assistant(call('uA', 'unfold', { handle: planA.handle })));
  const unfoldA = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    planA.handle,
    { callerId: 'uA' }
  );
  manager.appendMessage(result('uA', 'unfold', `[unfolded ${planA.handle}]`, 8));
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_UNFOLD_STARTED_TYPE,
      data: { handle: planA.handle, edits: unfoldA.edits }
    },
    ...unfoldA.edits,
    { type: 'custom', customType: FOLD_UNFOLDED_TYPE, data: { handle: planA.handle } }
  ]);

  assert.deepEqual([...appliedFolds(manager.getBranch())], []);
  const afterA = manager.buildSessionProjection();
  const textsAfterA = afterA.messages.map((m) => JSON.stringify(m)).join('\n');
  assert.ok(textsAfterA.includes('alpha'), 'ep-A span restored');
  assert.ok(textsAfterA.includes(longA), 'ep-A user message uncapped');
  assertPaired(afterA.messages);
});

test('unfold rejects pre-compaction folds, unknown handles, and fork or compaction handles', () => {
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('c1', 'bash')),
    result('c1', 'bash', 'early', 2),
    assistant(call('f1', 'fold', { since: 't1', note: 'early note' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 1);
  manager.appendMessage(
    Object.assign(result('f1', 'fold', 'early note', 3), {
      details: {
        kind: 'fold' as const,
        handle: plan.handle,
        since: 't1',
        until: 't2',
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        messages: plan.omitted.length,
        tokens: 10,
        headline: 'early note',
        keptUserTurns: plan.keptUserTurns,
        files: []
      }
    })
  );
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_STARTED_TYPE,
      data: { handle: plan.handle, edits: plan.edits }
    },
    ...plan.edits,
    { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: plan.handle } }
  ]);

  const keptId = manager.appendMessage(user('after', 4));
  manager.appendCompaction('Summary of early work', keptId, 500);
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));

  assert.throws(
    () =>
      planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, plan.handle, {
        callerId: 'u1'
      }),
    /lies before the latest compaction.*recall and ask/
  );
  assert.throws(
    () =>
      planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, 'ep-deadbeef', {
        callerId: 'u1'
      }),
    /No active fold ep-deadbeef/
  );
  assert.throws(
    () =>
      planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, 'fk-12345678', {
        callerId: 'u1'
      }),
    /is a fork, not an inline fold/
  );
  assert.throws(
    () =>
      planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, 'cp-12345678', {
        callerId: 'u1'
      }),
    /is a compaction archive/
  );
  void ids;
});

test('cut-short unfold batch is completed cleanly on the next boundary', () => {
  const { manager } = session([
    user('go', 1),
    assistant(call('c1', 'bash')),
    result('c1', 'bash', 'one', 2),
    assistant(call('c2', 'bash')),
    result('c2', 'bash', 'two', 3),
    assistant(call('f1', 'fold', { since: 't2', note: 'folded' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  manager.appendMessage(
    Object.assign(result('f1', 'fold', 'folded', 4), {
      details: {
        kind: 'fold' as const,
        handle: plan.handle,
        since: 't2',
        until: 't3',
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        messages: plan.omitted.length,
        tokens: 10,
        headline: 'folded',
        keptUserTurns: plan.keptUserTurns,
        files: []
      }
    })
  );
  applyDrafts(manager, [
    {
      type: 'custom',
      customType: FOLD_STARTED_TYPE,
      data: { handle: plan.handle, edits: plan.edits }
    },
    ...plan.edits,
    { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: plan.handle } }
  ]);

  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));
  const unfold = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    plan.handle,
    { callerId: 'u1' }
  );
  manager.appendMessage(result('u1', 'unfold', `[unfolded ${plan.handle}]`, 5));

  // Crash after writing FOLD_UNFOLD_STARTED_TYPE and only the first edit.
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, {
    handle: plan.handle,
    edits: unfold.edits
  });
  applyDrafts(manager, unfold.edits.slice(0, 1));

  assert.equal(cutShortFold(manager.getBranch()), plan.handle);
  assert.ok(appliedFolds(manager.getBranch()).has(plan.handle));
  // Every prefix of an unfold keeps toolResults paired because edits run in forward order.
  assertPaired(manager.buildSessionProjection().messages);

  const rest = completeCutShortFolds(manager.getBranch());
  assert.deepEqual(rest.at(-1), {
    type: 'custom',
    customType: FOLD_UNFOLDED_TYPE,
    data: { handle: plan.handle }
  });
  applyDrafts(manager, rest);

  assert.equal(cutShortFold(manager.getBranch()), undefined);
  assert.equal(appliedFolds(manager.getBranch()).has(plan.handle), false);
  assert.deepEqual(completeCutShortFolds(manager.getBranch()), []);
  const texts = manager
    .buildSessionProjection()
    .messages.map((m) => JSON.stringify(m))
    .join('\n');
  assert.ok(texts.includes('one') && texts.includes('two'));
  assertPaired(manager.buildSessionProjection().messages);
});

test('ForkFolds folds with until and unfolds in place, and leaves the responses it keeps as they are', async () => {
  const state = new EpisodeState();
  state.nextStamp = 13;
  const parentPrefix: Message[] = [
    user('parent', 1),
    {
      ...assistant('parent signed'),
      content: [
        { type: 'thinking', thinking: 'parent thought', thinkingSignature: 'parent-sig' },
        { type: 'toolCall', id: 'fk1|fc_fk1', name: 'fork', arguments: { task: 'subtask' } }
      ]
    }
  ];
  const { manager } = session(parentPrefix);
  const parentEntries = manager.buildSessionProjection().entries;
  const taskMsg = result('fk1|fc_fk1', 'fork', '[fork fk-test] task', 10);
  const messages: Message[] = [...parentPrefix, taskMsg];

  const folds = new ForkFolds({
    forkHandle: 'fk-test',
    state,
    parentEntries,
    prefixLength: parentPrefix.length,
    siblings: 0
  });

  const foldTool = folds.tool(
    { name: 'fold', description: '', parameters: {} as never },
    () => messages
  );
  const unfoldTool = folds.tool(
    { name: 'unfold', description: '', parameters: {} as never },
    () => messages
  );

  // Child work: turn 11 (folded) and turn 12 (kept via until), plus fold call with encrypted reasoning.
  messages.push(assistant(call('c1', 'bash', { command: 'echo secret-one' })));
  messages.push(result('c1', 'bash', 'secret-one', 11));
  messages.push({
    ...assistant('kept child'),
    content: [
      {
        type: 'thinking',
        thinking: '',
        thinkingSignature: '{"id":"rs_child","type":"reasoning","encrypted_content":"xyz"}'
      },
      {
        type: 'toolCall',
        id: 'c2|fc_c2',
        name: 'bash',
        arguments: { command: 'echo kept-two' },
        thoughtSignature: 'Z29vZ2xl'
      }
    ]
  });
  messages.push(result('c2|fc_c2', 'bash', 'kept-two', 12));
  messages.push({
    ...assistant('calling fold'),
    content: [
      {
        type: 'thinking',
        thinking: '[Reasoning redacted]',
        thinkingSignature: 'redacted',
        redacted: true
      },
      { type: 'toolCall', id: 'f_child|fc_fc', name: 'fold', arguments: {} }
    ]
  });

  const folded = await foldTool.execute('f_child|fc_fc', {
    since: 't11',
    until: 't11',
    note: 'child folded turn 11'
  });
  const handle = (folded.details as { handle: string }).handle;
  messages.push(
    Object.assign(result('f_child|fc_fc', 'fold', (folded.content[0] as TextContent).text, 13), {
      details: folded.details
    })
  );

  const viewAfterFold = folds.view(messages);
  // Parent prefix before the fork's mutation keeps its signature.
  const parentAsst = viewAfterFold[1] as AssistantMessage;
  assert.equal((parentAsst.content[0] as ThinkingContent).thinkingSignature, 'parent-sig');

  // The folded turn is omitted; the child's responses that stay are the messages themselves.
  assert.ok(!JSON.stringify(viewAfterFold).includes('secret-one'));
  const keptAsst = viewAfterFold.find(
    (m): m is AssistantMessage =>
      m.role === 'assistant' && m.content.some((b) => b.type === 'toolCall' && b.name === 'bash')
  )!;
  assert.equal(keptAsst, messages[5]);
  const foldCallerView = viewAfterFold.find(
    (m): m is AssistantMessage =>
      m.role === 'assistant' && m.content.some((b) => b.type === 'toolCall' && b.name === 'fold')
  )!;
  assert.equal(foldCallerView, messages[7]);
  assertPaired(viewAfterFold);
  assert.deepEqual(cacheAnchor(viewAfterFold, folds.activeHandles()), {
    toolUseId: 'f_child|fc_fc'
  });

  // A child turn appended after the fold is sent as it is too.
  const freshAfterFold: AssistantMessage = {
    ...assistant('fresh'),
    content: [
      { type: 'thinking', thinking: 'fresh thought', thinkingSignature: 'fresh-sig' },
      { type: 'toolCall', id: 'c_fresh|fc_fresh', name: 'read', arguments: { path: 'x.ts' } }
    ]
  };
  messages.push(freshAfterFold);
  messages.push(result('c_fresh|fc_fresh', 'read', 'x', 14));
  const viewWithFresh = folds.view(messages);
  const freshSeen = viewWithFresh.find(
    (m): m is AssistantMessage =>
      m.role === 'assistant' &&
      m.content.some((b) => b.type === 'toolCall' && b.id === 'c_fresh|fc_fresh')
  )!;
  assert.equal((freshSeen.content[0] as ThinkingContent).thinkingSignature, 'fresh-sig');

  // Unfold inside the fork restores secret-one and replaces the note.
  messages.push(assistant(call('u_child|fc_uc', 'unfold', { handle })));
  const unfolded = await unfoldTool.execute('u_child|fc_uc', { handle });
  messages.push(result('u_child|fc_uc', 'unfold', (unfolded.content[0] as TextContent).text, 15));

  assert.equal(folds.activeFolds.length, 0);
  assert.equal(folds.activeHandles().has(handle), false);
  const viewAfterUnfold = folds.view(messages);
  assertPaired(viewAfterUnfold);
  const contentTexts = viewAfterUnfold
    .flatMap((m) => (m.role === 'toolResult' ? m.content : []))
    .map((b) => (b.type === 'text' ? b.text : ''))
    .join('\n');
  assert.ok(contentTexts.includes('secret-one'));
  assert.ok(!contentTexts.includes('child folded turn 11'));
  assert.ok(contentTexts.includes(`[unfolded ${handle}]`));

  // The restored response and a turn appended after the unfold are the messages themselves.
  assert.equal(viewAfterUnfold[3], messages[3]);
  const postUnfoldTurn: AssistantMessage = {
    ...assistant('post-unfold'),
    content: [
      { type: 'thinking', thinking: 'post', thinkingSignature: 'post-unfold-sig' },
      { type: 'toolCall', id: 'c_post|fc_post', name: 'read', arguments: { path: 'y.ts' } }
    ]
  };
  messages.push(postUnfoldTurn);
  messages.push(result('c_post|fc_post', 'read', 'y', 16));
  const postSeen = folds.view(messages).at(-2) as AssistantMessage;
  assert.equal((postSeen.content[0] as ThinkingContent).thinkingSignature, 'post-unfold-sig');
});

test('live session unfold works both directly and through codemode, updates archive/status, and leaves the responses as they are', async () => {
  for (const mode of ['off', 'only'] as const) {
    const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-unfold-live-'));
    const faux = fauxProvider();
    const requests: TranscriptContext[] = [];
    let handle = '';

    const steps: ((ctx: TranscriptContext) => AssistantMessage)[] = [
      () =>
        mode === 'only'
          ? fauxAssistantMessage(
              fauxToolCall('codemode', {
                code: 'text(await tools.bash({ command: "echo UNIQUE_PAYLOAD_42" }));'
              }),
              { stopReason: 'toolUse' }
            )
          : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo UNIQUE_PAYLOAD_42' }), {
              stopReason: 'toolUse'
            }),
      () => {
        const msg =
          mode === 'only'
            ? fauxAssistantMessage(
                fauxToolCall('codemode', {
                  code: 'text(await tools.fold({ since: "t1", note: "folded payload" }));'
                }),
                { stopReason: 'toolUse' }
              )
            : fauxAssistantMessage(
                fauxToolCall(
                  'fold',
                  { since: 't1', note: 'folded payload' },
                  { id: 'f_live|fc_live' }
                ),
                { stopReason: 'toolUse' }
              );
        msg.content.unshift({
          type: 'thinking',
          thinking: '',
          thinkingSignature: '{"id":"rs_live_fold","type":"reasoning","encrypted_content":"secret"}'
        });
        return msg;
      },
      (context) => {
        const conv = context.messages.filter((m) => m.role !== 'system');
        const full = JSON.stringify(conv);
        assert.ok(!full.includes('UNIQUE_PAYLOAD_42'), `${mode}: payload omitted after fold`);
        assert.ok(full.includes('rs_live_fold'), `${mode}: fold turn reasoning kept`);
        handle = /ep-[0-9a-f]{8}/.exec(full)![0];
        const msg =
          mode === 'only'
            ? fauxAssistantMessage(
                fauxToolCall('codemode', {
                  code: `text(await tools.unfold({ handle: "${handle}" }));`
                }),
                { stopReason: 'toolUse' }
              )
            : fauxAssistantMessage(fauxToolCall('unfold', { handle }), {
                stopReason: 'toolUse'
              });
        msg.content.unshift({
          type: 'thinking',
          thinking: 'unfold reasoning',
          thinkingSignature: 'sig-unfold-turn'
        });
        return msg;
      },
      (context) => {
        const conv = context.messages.filter((m) => m.role !== 'system');
        const full = JSON.stringify(conv);
        assert.ok(full.includes('UNIQUE_PAYLOAD_42'), `${mode}: payload restored after unfold`);
        assert.ok(!full.includes('folded payload'), `${mode}: old fold note replaced`);
        assert.ok(full.includes(`[unfolded ${handle}]`), `${mode}: unfolded marker present`);
        assert.ok(full.includes('sig-unfold-turn'), `${mode}: unfold turn signature kept`);
        assertPaired(conv);
        return fauxAssistantMessage('All verified.');
      }
    ];

    faux.setResponses(
      steps.map((step) => (context) => {
        requests.push(context);
        return step(context);
      })
    );

    const resourceLoader = new DefaultResourceLoader({
      cwd: dir,
      agentDir: dir,
      noSkills: true,
      extensionFactories: [
        (pi) => pi.registerProvider(faux.provider),
        ...(mode === 'off' ? [] : [createCodemodeExtension({ models: false })]),
        episodes
      ]
    });
    await resourceLoader.reload();

    const { session: agentSession } = await createAgentSession({
      cwd: dir,
      agentDir: dir,
      model: faux.getModel(),
      thinkingLevel: 'off',
      modelRuntime: await ModelRuntime.create({
        modelsPath: null,
        authPath: join(dir, 'auth.json')
      }),
      resourceLoader,
      tools: [
        'bash',
        'read',
        'fold',
        'unfold',
        'recall',
        'ask',
        'fork',
        ...(mode === 'off' ? [] : ['codemode'])
      ],
      sessionManager: SessionManager.inMemory(dir),
      settingsManager: SettingsManager.inMemory({
        compaction: { enabled: false },
        ...(mode === 'off' ? {} : { codemode: { mode } })
      })
    });
    await agentSession.bindExtensions({});

    try {
      await agentSession.prompt('run');
      assert.equal(faux.getPendingResponseCount(), 0);
      const branch = agentSession.sessionManager.getBranch();
      assert.equal(appliedFolds(branch).has(handle), false);
      const threads = await new ThreadIndex(agentSession.sessionManager, new Map()).onBranch();
      assert.equal(threads.length, 0, `${mode}: unfolded episode is no longer active on branch`);
      // Still searchable by direct handle in find() across session history? Wait: foldRecords checks appliedFolds!
      // Verify active threadRecords on branch is empty.
      assert.equal(threadRecords(branch).length, 0);
      // The fold and the unfold edited what the fold removed and its note, nothing else.
      const edited = new Set(
        branch.flatMap((entry) => (entry.type === 'context_edit' ? [entry.targetId] : []))
      );
      const responses = branch.filter(
        (entry) => entry.type === 'message' && entry.message.role === 'assistant'
      );
      assert.deepEqual(
        responses.map((entry) => edited.has(entry.id)),
        [true, false, false, false],
        mode
      );
    } finally {
      agentSession.dispose();
      await rm(dir, { recursive: true, force: true });
    }
  }
});

test('codemode script fold note with sibling script output keeps script output when unfolded', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-unfold-cm-'));
  const faux = fauxProvider();
  let handle = '';

  faux.setResponses([
    () =>
      fauxAssistantMessage(
        fauxToolCall('codemode', {
          code: 'text(await tools.bash({ command: "echo INNER_SPAN_DATA" }));'
        }),
        { stopReason: 'toolUse' }
      ),
    () =>
      fauxAssistantMessage(
        fauxToolCall('codemode', {
          code: 'text("SCRIPT_SIDE_OUTPUT"); const n = await tools.fold({ since: "t1", note: "cm fold note" }); text(n);'
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const conv = context.messages.filter((m) => m.role !== 'system');
      const full = JSON.stringify(conv);
      assert.ok(full.includes('SCRIPT_SIDE_OUTPUT'));
      assert.ok(!full.includes('INNER_SPAN_DATA'));
      handle = /ep-[0-9a-f]{8}/.exec(full)![0];
      return fauxAssistantMessage(
        fauxToolCall('codemode', {
          code: `text(await tools.unfold({ handle: "${handle}" }));`
        }),
        { stopReason: 'toolUse' }
      );
    },
    (context) => {
      const conv = context.messages.filter((m) => m.role !== 'system');
      const full = JSON.stringify(conv);
      assert.ok(full.includes('INNER_SPAN_DATA'), 'span restored');
      assert.ok(full.includes('SCRIPT_SIDE_OUTPUT'), 'script side output kept');
      assert.ok(!full.includes('cm fold note'), 'fold note removed');
      assert.ok(full.includes(`[unfolded ${handle}]`), 'unfold marker added');
      assertPaired(conv);
      return fauxAssistantMessage('done');
    }
  ]);

  const resourceLoader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: dir,
    noSkills: true,
    extensionFactories: [
      (pi) => pi.registerProvider(faux.provider),
      createCodemodeExtension({ models: false }),
      episodes
    ]
  });
  await resourceLoader.reload();

  const { session: agentSession } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model: faux.getModel(),
    thinkingLevel: 'off',
    modelRuntime: await ModelRuntime.create({
      modelsPath: null,
      authPath: join(dir, 'auth.json')
    }),
    resourceLoader,
    tools: ['bash', 'read', 'fold', 'unfold', 'recall', 'ask', 'fork', 'codemode'],
    sessionManager: SessionManager.inMemory(dir),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      codemode: { mode: 'only' }
    })
  });
  await agentSession.bindExtensions({});

  try {
    await agentSession.prompt('go');
    assert.equal(faux.getPendingResponseCount(), 0);
  } finally {
    agentSession.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

test('forkThreads replays archived fork folds, unfold stubs and historical nested snapshots', async () => {
  const { manager } = session([user('root', 1)]);
  const parentAsstId = manager.appendMessage(
    assistant(call('fk_call', 'fork', { task: 'investigate' }))
  );
  const forkDetails: ForkDetails = {
    kind: 'fork',
    handle: 'fk-arch0001',
    task: 'investigate',
    parentEntryId: parentAsstId,
    turns: 4,
    headline: 'done',
    files: []
  };
  const forkResult = Object.assign(result('fk_call', 'fork', '[fork fk-arch0001 done]', 20), {
    details: forkDetails as never
  });
  manager.appendMessage(forkResult);

  const state = new EpisodeState();
  state.nextStamp = 10;
  const prefix = manager.buildSessionProjection().messages.slice(0, -1);
  const taskMsg = result('fk_call', 'fork', 'task brief');
  const folds = new ForkFolds({
    forkHandle: 'fk-arch0001',
    state,
    parentEntries: manager.buildSessionProjection().entries.slice(0, -1),
    prefixLength: prefix.length,
    siblings: 0
  });
  const foldTool = folds.tool(
    { name: 'fold', description: '', parameters: {} as never },
    () => msgs
  );
  const unfoldTool = folds.tool(
    { name: 'unfold', description: '', parameters: {} as never },
    () => msgs
  );

  const msgs: AgentMessage[] = [...prefix, taskMsg];
  // Child turn 1 (t10)
  msgs.push({
    ...assistant('t1'),
    content: [
      { type: 'thinking', thinking: 't1', thinkingSignature: 'sig1' },
      { type: 'toolCall', id: 'k1|fc_k1', name: 'bash', arguments: { command: 'echo early-fact' } }
    ]
  });
  const r1 = result('k1|fc_k1', 'bash', 'early-fact');
  folds.stamp(r1);
  msgs.push(r1); // t10

  // Fold ep-early covering t10
  msgs.push(assistant(call('f_early|fc_fe', 'fold', { since: 't10', note: 'early note' })));
  const earlyRes = await foldTool.execute('f_early|fc_fe', { since: 't10', note: 'early note' });
  const earlyHandle = (earlyRes.details as FoldDetails).handle;
  const earlyNoteMsg = Object.assign(
    result('f_early|fc_fe', 'fold', (earlyRes.content[0] as TextContent).text),
    { details: earlyRes.details }
  );
  folds.stamp(earlyNoteMsg);
  msgs.push(earlyNoteMsg); // t11

  // Child turn 2 (t12) while ep-early is folded
  msgs.push({
    ...assistant('t2'),
    content: [
      { type: 'thinking', thinking: 't2', thinkingSignature: 'sig2' },
      { type: 'toolCall', id: 'k2|fc_k2', name: 'bash', arguments: { command: 'echo mid-fact' } }
    ]
  });
  const r2 = result('k2|fc_k2', 'bash', 'mid-fact');
  folds.stamp(r2);
  msgs.push(r2); // t12

  // Fold ep-mid covering ONLY t12 (ep-early is still folded at this moment!)
  msgs.push(assistant(call('f_mid|fc_fm', 'fold', { since: 't12', note: 'mid note' })));
  const midRes = await foldTool.execute('f_mid|fc_fm', { since: 't12', note: 'mid note' });
  const midHandle = (midRes.details as FoldDetails).handle;
  const midNoteMsg = Object.assign(
    result('f_mid|fc_fm', 'fold', (midRes.content[0] as TextContent).text),
    { details: midRes.details }
  );
  folds.stamp(midNoteMsg);
  msgs.push(midNoteMsg); // t13

  // Now unfold ep-early AFTER ep-mid was created!
  msgs.push(assistant(call('u_early|fc_ue', 'unfold', { handle: earlyHandle })));
  const uRes = await unfoldTool.execute('u_early|fc_ue', { handle: earlyHandle });
  const uMsg = result('u_early|fc_ue', 'unfold', (uRes.content[0] as TextContent).text);
  folds.stamp(uMsg);
  msgs.push(uMsg);

  // Final turn AFTER unfold (keeps its signature!)
  msgs.push({
    ...assistant('final'),
    content: [
      { type: 'thinking', thinking: 'final-thought', thinkingSignature: 'sig_final' },
      { type: 'toolCall', id: 'k_final|fc_kf', name: 'read', arguments: { path: 'z.ts' } }
    ]
  });
  const rFinal = result('k_final|fc_kf', 'read', 'z');
  folds.stamp(rFinal);
  msgs.push(rFinal);

  const archive: ForkArchive = {
    handle: 'fk-arch0001',
    task: 'investigate',
    parentEntryId: parentAsstId,
    messages: msgs.slice(prefix.length),
    folds: folds.folds.map((f) => ({ ...f }))
  };
  const archiveMap = new Map([['fk-arch0001', archive]]);
  const index = new ThreadIndex(manager, archiveMap);

  // 1. Unfolded nested fold is not returned by find().
  assert.equal(await index.find(earlyHandle), undefined);

  // 2. Fork thread context() has early-fact restored, early note replaced with [unfolded ...],
  //    and every response that stays as the child received it.
  const forkThread = (await index.find('fk-arch0001'))!;
  const forkCtx = forkThread.context!();
  const forkText = JSON.stringify(forkCtx);
  assert.ok(forkText.includes('early-fact'));
  assert.ok(!forkText.includes('mid-fact'));
  const forkResultText = forkCtx
    .flatMap((m) => (m.role === 'toolResult' ? m.content : []))
    .map((b) => (b.type === 'text' ? b.text : ''))
    .join('\n');
  assert.ok(!forkResultText.includes('early note'));
  assert.ok(forkResultText.includes(`[unfolded ${earlyHandle}]`));
  assert.ok(forkText.includes('sig1'));
  assert.ok(forkText.includes('sig_final'));

  // 3. Nested fold ep-mid's historical context() at the time ep-mid ran STILL hides early-fact
  //    and shows early note (because ep-early was unfolded AFTER ep-mid was created!).
  const midThread = (await index.find(midHandle))!;
  const midCtxText = JSON.stringify(midThread.context!());
  assert.ok(!midCtxText.includes('early-fact'));
  assert.ok(midCtxText.includes('early note'));
  assert.ok(midCtxText.includes('mid-fact'));
});

test('ThreadIndex.find resolves fold active on Branch 1 even when unfolded on Branch 2', async () => {
  const { manager } = session([
    user('root', 1),
    assistant(call('c1', 'bash', { command: 'echo branch-secret' })),
    result('c1', 'bash', 'branch-secret', 2),
    assistant(call('f1', 'fold', { since: 't2', note: 'branch fold' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const noteMsg = Object.assign(result('f1', 'fold', '[fold note]', 3), {
    details: {
      kind: 'fold',
      handle: plan.handle,
      since: 't2',
      firstEntryId: plan.firstEntryId,
      lastEntryId: plan.lastEntryId,
      messages: plan.omitted.length,
      tokens: 20,
      headline: 'branch fold',
      keptUserTurns: [],
      files: [],
      note: 'branch fold'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(noteMsg);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
  applyDrafts(manager, plan.edits);
  const branch1Leaf = manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });

  // Continue onto Branch 2 and unfold plan.handle there.
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));
  const uPlan = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    plan.handle
  );
  manager.appendMessage(result('u1', 'unfold', '[unfolded]', 4));
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, { handle: plan.handle, edits: uPlan.edits });
  applyDrafts(manager, uPlan.edits);
  const branch2Leaf = manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: plan.handle });

  const index = new ThreadIndex(manager, new Map());
  // On Branch 2, the fold is unfolded so find() returns undefined.
  assert.equal(await index.find(plan.handle), undefined);

  // Switch back to Branch 1: find() resolves the still-folded thread!
  manager.branch(branch1Leaf);
  const foundOnB1 = await index.find(plan.handle);
  assert.notEqual(foundOnB1, undefined);
  assert.equal(foundOnB1!.handle, plan.handle);

  // Switch back to Branch 2 again: undefined.
  manager.branch(branch2Leaf);
  assert.equal(await index.find(plan.handle), undefined);
});

test('planUnfold rejects partial span overlap without covers and preserves newer external edits & interrupted-batch edits', () => {
  // 1. Partial span overlap without covers: Fold A (t2..t2, note at t4), Fold B (t2..t3, stops before A's note)
  const longUser = `${'u'.repeat(USER_TEXT_CAP + 400)}TAIL`;
  const { manager, ids } = session([
    user(longUser, 1),
    assistant(call('c1', 'bash', { command: 'echo 1' })),
    result('c1', 'bash', '1', 2),
    assistant(call('c2', 'bash', { command: 'echo 2' })),
    result('c2', 'bash', '2', 3),
    assistant(call('fA', 'fold', { since: 't2', until: 't2', note: 'A' }))
  ]);
  const planA = planFold(manager.buildSessionProjection().entries, 'fA', 2, { until: 2 });
  const noteA = Object.assign(result('fA', 'fold', '[fold A]', 4), {
    details: {
      kind: 'fold',
      handle: planA.handle,
      since: 't2',
      until: 't2',
      firstEntryId: planA.firstEntryId,
      lastEntryId: planA.lastEntryId,
      messages: planA.omitted.length,
      tokens: 10,
      headline: 'A',
      keptUserTurns: [],
      files: [],
      note: 'A'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(noteA);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: planA.handle, edits: planA.edits });
  applyDrafts(manager, planA.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: planA.handle });

  // Fold B starts at t1 and stops at until: t3 (before noteA at t4!), so !B.covers.includes(A),
  // yet B's [firstEntryId..lastEntryId] spans across A's [firstEntryId..lastEntryId]!
  manager.appendMessage(assistant(call('fB', 'fold', { since: 't1', until: 't3', note: 'B' })));
  const planB = planFold(manager.buildSessionProjection().entries, 'fB', 1, { until: 3 });
  assert.equal(
    planB.covers.some((c) => c.handle === planA.handle),
    false
  );
  const noteB = Object.assign(result('fB', 'fold', '[fold B]', 5), {
    details: {
      kind: 'fold',
      handle: planB.handle,
      since: 't1',
      until: 't3',
      firstEntryId: planB.firstEntryId,
      lastEntryId: planB.lastEntryId,
      messages: planB.omitted.length,
      tokens: 50,
      headline: 'B',
      keptUserTurns: ['t1'],
      files: [],
      note: 'B'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(noteB);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: planB.handle, edits: planB.edits });
  applyDrafts(manager, planB.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: planB.handle });

  assert.throws(
    () => planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, planA.handle),
    new RegExp(`overlaps active fold ${planB.handle}`)
  );

  // 2. External edit on u1 written AFTER Fold B stands when B is unfolded!
  manager.appendContextEdit(ids[0]!, {
    content: [{ type: 'text', text: 'EXTERNAL OVERRIDE [t1]' }]
  });
  const unfoldB = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    planB.handle
  );
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, {
    handle: planB.handle,
    edits: unfoldB.edits
  });
  applyDrafts(manager, unfoldB.edits);
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: planB.handle });
  const u1AfterB = manager.buildSessionProjection().entries[0]!.messages[0] as {
    content: TextContent[];
  };
  assert.equal(u1AfterB.content[0]!.text, 'EXTERNAL OVERRIDE [t1]');

  // 3. Interrupted-batch external edit is not claimed by fold:H.
  const { manager: m2, ids: ids2 } = session([
    user('hello', 1),
    assistant(call('x1', 'bash', { command: 'echo x' })),
    result('x1', 'bash', 'x', 2),
    assistant(call('fx', 'fold', { since: 't2', note: 'cut' }))
  ]);
  const pCut = planFold(m2.buildSessionProjection().entries, 'fx', 2);
  const nCut = Object.assign(result('fx', 'fold', '[fold cut]', 3), {
    details: {
      kind: 'fold',
      handle: pCut.handle,
      since: 't2',
      firstEntryId: pCut.firstEntryId,
      lastEntryId: pCut.lastEntryId,
      messages: pCut.omitted.length,
      tokens: 10,
      headline: 'cut',
      keptUserTurns: [],
      files: [],
      note: 'cut'
    } satisfies FoldDetails as never
  });
  m2.appendMessage(nCut);
  m2.appendCustomEntry(FOLD_STARTED_TYPE, { handle: pCut.handle, edits: pCut.edits });
  applyDrafts(m2, pCut.edits.slice(0, 1));
  // External extension edits user message DURING interrupted window!
  m2.appendContextEdit(ids2[0]!, { content: [{ type: 'text', text: 'DURING CRASH [t1]' }] });
  applyDrafts(m2, completeCutShortFolds(m2.getBranch()));
  assert.equal(cutShortFold(m2.getBranch()), undefined);
  // Now unfold pCut across a span starting at t1? Wait: check user message & span restoration:
  // Even if an external edit targeted an entry in the span, it stands!
  m2.appendContextEdit(ids2[2]!, { content: [{ type: 'text', text: 'EXT TOOL RESULT [t2]' }] });
  const uCut = planUnfold(m2.getBranch(), m2.buildSessionProjection().entries, pCut.handle);
  applyDrafts(m2, uCut.edits);
  m2.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: pCut.handle });
  const r2Restored = m2.buildSessionProjection().entries[2]!.messages[0] as {
    content: TextContent[];
  };
  assert.equal(r2Restored.content[0]!.text, 'EXT TOOL RESULT [t2]');
});

test('a fold with until edits only its span, and a repair-only turn_end leaves the current response as it is', async () => {
  const { manager } = session([
    user('u1', 1),
    assistant(call('c1', 'bash', { command: 'echo 1' })),
    result('c1', 'bash', '1', 2),
    {
      ...assistant('surviving'),
      content: [
        { type: 'thinking', thinking: 'surviving thought', thinkingSignature: 'sig-surv' },
        {
          type: 'toolCall',
          id: 'f1',
          name: 'fold',
          arguments: { since: 't2', until: 't2', note: 'n' }
        }
      ]
    }
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 2 });
  // The response that called the fold stays as it is: the edits remove the span, newest first.
  const spanIds = manager
    .getBranch()
    .slice(1, 3)
    .map((entry) => entry.id);
  assert.deepEqual(
    plan.edits,
    spanIds.reverse().map((targetId) => ({ type: 'context_edit', targetId, replacement: null }))
  );

  // A turn_end that only completes a cut-short fold edits nothing but that fold's span.
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-repair-retire-'));
  const faux = fauxProvider();
  faux.setResponses([
    () => ({
      ...fauxAssistantMessage('repaired turn answer'),
      content: [
        { type: 'thinking', thinking: 'repair turn thought', thinkingSignature: 'sig-repair-turn' },
        { type: 'text', text: 'done' }
      ]
    })
  ]);
  const settingsManager = SettingsManager.inMemory({ episodes: { snapshots: false } } as never);
  const sm = SessionManager.inMemory(dir);
  const uId = sm.appendMessage(user('hi', 1));
  const aId = sm.appendMessage(assistant(call('c1', 'bash', { command: 'echo 1' })));
  const rId = sm.appendMessage(result('c1', 'bash', '1', 2));
  const planned = [
    { type: 'context_edit' as const, targetId: rId, replacement: null },
    { type: 'context_edit' as const, targetId: aId, replacement: null }
  ];
  sm.appendCustomEntry(FOLD_STARTED_TYPE, { handle: 'ep-cut00001', edits: planned });
  sm.appendContextEdit(rId, null); // cut short midway!
  void uId;

  const loader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: dir,
    noSkills: true,
    settingsManager,
    extensionFactories: [(pi) => pi.registerProvider(faux.provider), episodes]
  });
  await loader.reload();
  const { session: agentSession } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model: faux.getModel(),
    thinkingLevel: 'off',
    sessionManager: sm,
    settingsManager,
    modelRuntime: await ModelRuntime.create({
      modelsPath: null,
      authPath: join(dir, 'auth.json')
    }),
    resourceLoader: loader
  });
  await agentSession.bindExtensions({});
  try {
    await agentSession.prompt('finish repair');
    assert.equal(cutShortFold(sm.getBranch()), undefined);
    const projMsgs = sm.buildSessionProjection().messages;
    assert.equal(JSON.stringify(projMsgs).includes('sig-repair-turn'), true);
    assert.deepEqual(
      sm.getBranch().flatMap((entry) => (entry.type === 'context_edit' ? [entry.targetId] : [])),
      [rId, aId]
    );
  } finally {
    agentSession.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

test('codemode unfold preserves script images, drops carried images, unescapes multiline notes, and round-trips re-folds', () => {
  const scriptImg = { type: 'image' as const, mimeType: 'image/png', data: 'U0NSSVBUX0lNRw==' };
  const carriedImg = { type: 'image' as const, mimeType: 'image/png', data: 'Q0FSUklFRF9JTUc=' };
  const multilineNote = 'Line 1 summary\nOpen:\n- item two';
  const rendered = `[fold ep-cm000001: 2 messages, t2, ~50 tokens]\n${multilineNote}`;
  const escapedRendered = JSON.stringify(rendered); // includes outer quotes + \n escapes

  const { manager } = session([
    user('start', 1),
    assistant(call('c1', 'bash', { command: 'echo secret-cm' })),
    result('c1', 'bash', 'secret-cm', 2),
    assistant(call('cm1', 'codemode', { code: 'fold()' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'cm1', 2);
  const details: FoldDetails = {
    kind: 'fold',
    handle: 'ep-cm000001',
    since: 't2',
    firstEntryId: plan.firstEntryId,
    lastEntryId: plan.lastEntryId,
    messages: plan.omitted.length,
    tokens: 50,
    headline: 'Line 1 summary',
    keptUserTurns: [],
    files: [],
    note: multilineNote,
    renderedNote: rendered,
    carriedImages: 1,
    written: { blocks: 2 }
  };
  const cmMsg: Message = {
    role: 'toolResult',
    toolCallId: 'cm1',
    toolName: 'codemode',
    content: [
      scriptImg,
      { type: 'text', text: `KEEP_SCRIPT_OUT\n${escapedRendered}` },
      carriedImg,
      { type: 'text', text: '[t3]' }
    ],
    details: { [FOLD_DETAILS_KEY]: details } as never,
    isError: false,
    timestamp: Date.now()
  };
  manager.appendMessage(cmMsg);
  const patchedEdits = plan.edits.map((e) => e);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: details.handle, edits: patchedEdits });
  applyDrafts(manager, patchedEdits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: details.handle });

  // Unfold ep-cm000001: scriptImg stays, carriedImg is dropped, escaped multiline note becomes [unfolded ep-cm000001]!
  const u1 = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    details.handle
  );
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, { handle: details.handle, edits: u1.edits });
  applyDrafts(manager, u1.edits);
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: details.handle });

  const afterU1 = manager.buildSessionProjection().messages.at(-1)! as {
    content: (TextContent | typeof scriptImg)[];
  };
  assert.equal(
    afterU1.content.some((b) => b.type === 'image' && b.data === scriptImg.data),
    true
  );
  assert.equal(
    afterU1.content.some((b) => b.type === 'image' && b.data === carriedImg.data),
    false
  );
  const textAfterU1 = afterU1.content
    .flatMap((b) => (b.type === 'text' ? [b.text] : []))
    .join('\n');
  assert.ok(textAfterU1.includes('KEEP_SCRIPT_OUT'));
  assert.ok(!textAfterU1.includes('Line 1 summary'));
  assert.ok(textAfterU1.includes('[unfolded ep-cm000001]'));

  // workSince & nudgeGrowth now count the unfolded codemode output in `left`!
  const ws = workSince(manager.buildSessionProjection().entries, 0);
  assert.ok(ws.left.some((e) => JSON.stringify(e.messages).includes('KEEP_SCRIPT_OUT')));
  assert.ok(nudgeGrowth(manager.buildSessionProjection().entries) > 0);

  // Re-fold across t2..t3 (covering the already-unfolded stub) and unfold the second fold:
  // the stub stays `[unfolded ep-cm000001]` and does NOT resurrect `Line 1 summary`!
  manager.appendMessage(assistant(call('f2', 'fold', { since: 't2', note: 'second fold' })));
  const p2 = planFold(manager.buildSessionProjection().entries, 'f2', 2);
  const n2 = Object.assign(result('f2', 'fold', '[fold second]', 4), {
    details: {
      kind: 'fold',
      handle: p2.handle,
      since: 't2',
      firstEntryId: p2.firstEntryId,
      lastEntryId: p2.lastEntryId,
      messages: p2.omitted.length,
      tokens: 60,
      headline: 'second fold',
      keptUserTurns: [],
      files: [],
      note: 'second fold'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(n2);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: p2.handle, edits: p2.edits });
  applyDrafts(manager, p2.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: p2.handle });

  const u2 = planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, p2.handle);
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, { handle: p2.handle, edits: u2.edits });
  applyDrafts(manager, u2.edits);
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: p2.handle });

  const allResultText = manager
    .buildSessionProjection()
    .messages.flatMap((m) => (m.role === 'toolResult' ? m.content : []))
    .flatMap((b) => (b.type === 'text' ? [b.text] : []))
    .join('\n');
  assert.ok(allResultText.includes('[unfolded ep-cm000001]'));
  assert.ok(!allResultText.includes('Line 1 summary'));
});

test('episodeStats preserves historical savings accrued before unfold and stops accruing after unfold', () => {
  const { manager } = session([
    user('u1', 1),
    assistant(call('c1', 'bash', { command: 'echo heavy' })),
    result('c1', 'bash', 'x'.repeat(4000), 2),
    assistant(call('f1', 'fold', { since: 't2', note: 'tiny' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const noteMsg = Object.assign(result('f1', 'fold', '[fold tiny]', 3), {
    details: {
      kind: 'fold',
      handle: plan.handle,
      since: 't2',
      firstEntryId: plan.firstEntryId,
      lastEntryId: plan.lastEntryId,
      messages: plan.omitted.length,
      tokens: 1000,
      noteTokens: 20,
      headline: 'tiny',
      keptUserTurns: [],
      files: [],
      note: 'tiny'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(noteMsg);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
  applyDrafts(manager, plan.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });

  // Two model requests while the fold is active save 980 tokens each (1960 total).
  const usageMsg = (): AssistantMessage => ({
    ...assistant('ok'),
    usage: {
      input: 500,
      output: 50,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 550,
      cost: { input: 0.001, output: 0.001, cacheRead: 0, cacheWrite: 0, total: 0.002 }
    }
  });
  manager.appendMessage(usageMsg());
  manager.appendMessage(usageMsg());
  const beforeUnfold = episodeStats(manager.getBranch(), '/tmp');
  assert.equal(beforeUnfold.folds.saved, 1960);
  assert.equal(beforeUnfold.folds.count, 1);

  // Unfold the episode, then make two more requests.
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: plan.handle });
  manager.appendMessage(usageMsg());
  manager.appendMessage(usageMsg());

  const afterUnfold = episodeStats(manager.getBranch(), '/tmp');
  assert.equal(afterUnfold.folds.count, 1);
  assert.equal(afterUnfold.folds.saved, 1960);
});

test('planUnfold edits only the span and the note, unfoldEditsOf preserves earlier turn_end boundary edits, and Thread exposes appliedFolds', async () => {
  // 1. planUnfold restores the span, then replaces the note, and edits no response outside it
  const { manager } = session([
    user('u1', 1),
    assistant(call('c1', 'bash', { command: 'echo SECRET' })),
    result('c1', 'bash', 'SECRET', 2),
    assistant(call('f1', 'fold', { since: 't2', note: 'n' }))
  ]);
  const p = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const n = Object.assign(result('f1', 'fold', '[fold n]', 3), {
    details: {
      kind: 'fold',
      handle: p.handle,
      since: 't2',
      firstEntryId: p.firstEntryId,
      lastEntryId: p.lastEntryId,
      messages: p.omitted.length,
      tokens: 20,
      headline: 'n',
      keptUserTurns: [],
      files: [],
      note: 'n'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(n);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: p.handle, edits: p.edits });
  applyDrafts(manager, p.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: p.handle });

  // Add a signed surviving assistant message after the fold, then planUnfold
  const survId = manager.appendMessage({
    ...assistant('surv'),
    content: [
      { type: 'thinking', thinking: 'post-fold thought', thinkingSignature: 'sig-post-fold' },
      { type: 'text', text: 'ready to unfold' }
    ]
  });
  const uPlan = planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, p.handle);
  const ids = manager.getBranch().map((entry) => entry.id);
  assert.deepEqual(
    uPlan.edits.map((edit) => edit.targetId),
    [ids[1], ids[2], uPlan.noteEntryId]
  );
  assert.equal(
    uPlan.edits.some((edit) => edit.targetId === survId),
    false
  );

  // ThreadIndex.find exposes appliedFolds() for historical fold context
  const index = new ThreadIndex(manager, new Map());
  const foldThread = (await index.find(p.handle))!;
  assert.ok(foldThread.appliedFolds !== undefined);
  assert.equal(foldThread.appliedFolds!().size, 0);

  // 2. Live session: earlier turn_end handler queues a context_edit redacting a restored toolResult
  //    in event.entries — unfold preserves 'REDACTED BY EARLIER HOOK' instead of overwriting with 'SECRET'!
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-unfold-boundary-'));
  const faux = fauxProvider();
  let unfoldedHandle = '';
  let secretResultEntryId = '';
  faux.setResponses([
    () => fauxAssistantMessage([fauxToolCall('bash', { command: 'echo SECRET' })]),
    () => fauxAssistantMessage([fauxToolCall('fold', { since: 't2', note: 'hid secret' })]),
    () => fauxAssistantMessage('folded ok'),
    () => ({
      ...fauxAssistantMessage('unfolding'),
      content: [
        { type: 'thinking', thinking: 'unfold thought', thinkingSignature: 'sig-unfold-call' },
        fauxToolCall('unfold', { handle: unfoldedHandle })
      ]
    }),
    () => fauxAssistantMessage('unfolded ok')
  ]);
  const settingsManager = SettingsManager.inMemory({ episodes: { snapshots: false } } as never);
  const sm = SessionManager.inMemory(dir);
  const earlyRedactor = (pi: Parameters<typeof episodes>[0]) => {
    pi.on('turn_end', (event) => {
      if (!secretResultEntryId) return undefined;
      const isUnfoldTurn = event.toolResults.some((tr) => tr.toolName === 'unfold' && !tr.isError);
      if (!isUnfoldTurn) return undefined;
      return {
        entries: [
          ...event.entries,
          {
            type: 'context_edit' as const,
            targetId: secretResultEntryId,
            replacement: { content: [{ type: 'text' as const, text: 'REDACTED BY EARLIER HOOK' }] }
          }
        ]
      };
    });
  };
  const loader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: dir,
    noSkills: true,
    settingsManager,
    extensionFactories: [(pi) => pi.registerProvider(faux.provider), earlyRedactor, episodes]
  });
  await loader.reload();
  const { session: agentSession } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model: faux.getModel(),
    thinkingLevel: 'off',
    sessionManager: sm,
    settingsManager,
    modelRuntime: await ModelRuntime.create({
      modelsPath: null,
      authPath: join(dir, 'auth.json')
    }),
    resourceLoader: loader
  });
  await agentSession.bindExtensions({});
  try {
    await agentSession.prompt('run secret and fold');
    const branchAfterFold = sm.getBranch();
    const secretEntry = branchAfterFold.find(
      (e) =>
        e.type === 'message' && e.message.role === 'toolResult' && e.message.toolName === 'bash'
    )!;
    secretResultEntryId = secretEntry.id;
    const appliedEntry = branchAfterFold.find(
      (e) => e.type === 'custom' && e.customType === FOLD_APPLIED_TYPE
    ) as { data?: { handle?: string } } | undefined;
    unfoldedHandle = appliedEntry?.data?.handle ?? '';
    assert.ok(unfoldedHandle.startsWith('ep-'));

    await agentSession.prompt('now unfold');
    const projMsgs = sm.buildSessionProjection().messages;
    const toolResultTexts = projMsgs
      .filter((m) => m.role === 'toolResult')
      .map((m) => JSON.stringify(m.content))
      .join('\n');
    assert.ok(toolResultTexts.includes('REDACTED BY EARLIER HOOK'));
    assert.ok(!toolResultTexts.includes('SECRET'));
    assert.ok(JSON.stringify(projMsgs).includes('sig-unfold-call'));
  } finally {
    agentSession.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

test('completeCutShortFolds completes an interrupted unfold and leaves pending boundary edits of restored entries standing', () => {
  const { manager, ids } = session([
    user('u1', 1),
    {
      ...assistant('sec-call'),
      content: [
        { type: 'thinking', thinking: 'orig-t', thinkingSignature: 'sig-orig' },
        call('c1', 'bash', { command: 'echo ok' })
      ]
    },
    result('c1', 'bash', 'SECRET', 2),
    assistant(call('f1', 'fold', { since: 't2', note: 'n' }))
  ]);
  const p = planFold(manager.buildSessionProjection().entries, 'f1', 2);
  const note = Object.assign(result('f1', 'fold', '[fold ep-cutunf: 2 messages]\nn', 3), {
    details: {
      kind: 'fold',
      handle: p.handle,
      since: 't2',
      firstEntryId: p.firstEntryId,
      lastEntryId: p.lastEntryId,
      messages: p.omitted.length,
      tokens: 20,
      headline: 'n',
      keptUserTurns: [],
      files: [],
      note: 'n',
      renderedNote: '[fold ep-cutunf: 2 messages]\nn'
    } satisfies FoldDetails as never
  });
  manager.appendMessage(note);
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: p.handle, edits: p.edits });
  applyDrafts(manager, p.edits);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: p.handle });

  // A later response, which the unfold does not touch; the unfold is cut short after one edit.
  manager.appendMessage({
    ...assistant('surv'),
    content: [
      { type: 'thinking', thinking: 'surv-t', thinkingSignature: 'sig-surv' },
      { type: 'text', text: 'unfold now' }
    ]
  });
  const uPlan = planUnfold(manager.getBranch(), manager.buildSessionProjection().entries, p.handle);
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, { handle: p.handle, edits: uPlan.edits });
  applyDrafts(manager, uPlan.edits.slice(0, 1));
  assert.equal(cutShortFold(manager.getBranch()), p.handle);

  // At recovery boundary, an earlier hook passes pending replacements on BOTH the restored toolResult
  // AND the restored assistant message.
  const pendingBoundary = [
    {
      type: 'context_edit' as const,
      targetId: ids[2]!,
      replacement: { content: [{ type: 'text' as const, text: 'REDACTED BY EARLIER HOOK [t2]' }] }
    },
    {
      type: 'context_edit' as const,
      targetId: ids[1]!,
      replacement: {
        content: [
          { type: 'thinking' as const, thinking: 'hook-t', thinkingSignature: 'sig-from-hook' },
          call('c1', 'bash', { command: 'echo ok' })
        ]
      }
    }
  ];
  const recoveryDrafts = completeCutShortFolds(manager.getBranch(), pendingBoundary);
  applyDrafts(manager, [...pendingBoundary, ...recoveryDrafts]);
  assert.equal(cutShortFold(manager.getBranch()), undefined);

  const projText = JSON.stringify(manager.buildSessionProjection().messages);
  assert.ok(projText.includes('REDACTED BY EARLIER HOOK'));
  assert.ok(!projText.includes('SECRET'));
  assert.ok(projText.includes('hook-t'));
  assert.ok(projText.includes('sig-from-hook'));
  assert.ok(projText.includes('sig-surv'));
  assert.ok(projText.includes(`[unfolded ${p.handle}]`));
});

test('running fork callingFork and prefixEntryId integrate with nested unfold', async () => {
  const { manager, ids } = session([
    user('parent start', 1),
    assistant(call('p1', 'bash', { command: 'echo p' })),
    result('p1', 'bash', 'p', 2),
    {
      ...assistant('fold-and-fork'),
      content: [
        {
          type: 'thinking',
          thinking: 'parent-t',
          thinkingSignature: JSON.stringify({ id: 'rs_p' })
        },
        call('f_par|fc_fp', 'fold', { since: 't2', note: 'parent folded' }),
        call('fk_call|fc_fk', 'fork', { task: 'child work' })
      ]
    }
  ]);
  const parPlan = planFold(manager.buildSessionProjection().entries, 'f_par|fc_fp', 2);
  const parNoteId = manager.appendMessage(
    Object.assign(result('f_par|fc_fp', 'fold', '[fold parent]', 3), {
      details: {
        kind: 'fold',
        handle: parPlan.handle,
        since: 't2',
        firstEntryId: parPlan.firstEntryId,
        lastEntryId: parPlan.lastEntryId,
        messages: parPlan.omitted.length,
        tokens: 10,
        headline: 'parent folded',
        keptUserTurns: [],
        files: [],
        note: 'parent folded'
      } satisfies FoldDetails as never
    })
  );
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: parPlan.handle, edits: parPlan.edits });
  applyDrafts(manager, parPlan.edits);
  const leafBeforeFork = manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: parPlan.handle });
  void parNoteId;

  const state = new EpisodeState();
  state.nextStamp = 10;
  const proj = manager.buildSessionProjection();
  const taskMsg = result('fk_call|fc_fk', 'fork', 'Task: child work', 3);
  const childMsgs = [...proj.messages, taskMsg];
  const folds = new ForkFolds({
    forkHandle: 'fk-run00001',
    state,
    parentEntries: proj.entries,
    prefixLength: proj.messages.length,
    siblings: 0
  });
  const foldTool = folds.tool(
    { name: 'fold', description: '', parameters: {} as never },
    () => childMsgs
  );
  const unfoldTool = folds.tool(
    { name: 'unfold', description: '', parameters: {} as never },
    () => childMsgs
  );

  // Child runs bash -> folds it -> running ThreadIndex exposes the nested fold with prefixEntryId state!
  childMsgs.push(assistant(call('cb1|fc_cb1', 'bash', { command: 'echo RUN_SECRET' })));
  const cr1 = result('cb1|fc_cb1', 'bash', 'RUN_SECRET');
  folds.stamp(cr1);
  childMsgs.push(cr1);

  childMsgs.push(assistant(call('cf1|fc_cf1', 'fold', { since: 't10', note: 'child folded' })));
  const cfRes = await foldTool.execute('cf1|fc_cf1', {
    since: 't10',
    note: 'child folded'
  });
  const childHandle = (cfRes.details as FoldDetails).handle;
  const cfNote = Object.assign(
    result('cf1|fc_cf1', 'fold', (cfRes.content[0] as TextContent).text),
    { details: cfRes.details }
  );
  folds.stamp(cfNote);
  childMsgs.push(cfNote);

  const snapshot = (): ForkArchive => ({
    handle: 'fk-run00001',
    task: 'child work',
    parentEntryId: ids[3]!,
    prefixEntryId: leafBeforeFork,
    messages: childMsgs.slice(proj.messages.length),
    folds: folds.folds.map((f) => ({ ...f }))
  });

  const idxFolded = new ThreadIndex(manager, new Map(), snapshot());
  const foundFolded = await idxFolded.find(childHandle);
  assert.notEqual(foundFolded, undefined);
  assert.equal(foundFolded!.appliedFolds!().has(parPlan.handle), true);
  assert.ok(JSON.stringify(foundFolded!.context!()).includes('rs_p'));

  // Now the running fork unfolds childHandle: it disappears from running ThreadIndex!
  childMsgs.push(assistant(call('cu1|fc_cu1', 'unfold', { handle: childHandle })));
  const cuRes = await unfoldTool.execute('cu1|fc_cu1', { handle: childHandle });
  const cuMsg = result('cu1|fc_cu1', 'unfold', (cuRes.content[0] as TextContent).text);
  folds.stamp(cuMsg);
  childMsgs.push(cuMsg);

  const idxAfterUnfold = new ThreadIndex(manager, new Map(), snapshot());
  assert.equal(await idxAfterUnfold.find(childHandle), undefined);
  assert.deepEqual(
    (await idxAfterUnfold.onBranch()).map((t) => t.handle),
    [parPlan.handle]
  );
});
