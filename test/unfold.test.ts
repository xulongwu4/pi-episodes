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
  type Model,
  normalizeContext,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import { convertMessages as convertGoogleMessages } from '@earendil-works/pi-ai/api/google-shared';
import { convertResponsesMessages } from '@earendil-works/pi-ai/api/openai-responses-shared';
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
  branchRetiredResponsesCallIds,
  completeCutShortFolds,
  cutShortFold,
  GEMINI_SKIP_THOUGHT_SIGNATURE,
  planFold,
  planUnfold,
  retireAssistantContent,
  retireAssistantMessage,
  retiredResponsesCallIds,
  sanitizeProviderPayload,
  stripRetiredResponsesIds,
  USER_TEXT_CAP,
  workSince
} from '../src/fold.js';
import { ForkFolds } from '../src/fork-fold.js';
import episodes from '../src/index.js';
import { nudgeGrowth } from '../src/nudge.js';
import { requestOptions } from '../src/requests.js';
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

test('retireAssistantContent sanitizes signed, unsigned, empty, redacted and thinking-only blocks, textSignature and preserves toolCall ids', () => {
  // Plain unsigned non-empty thinking, plain text and tool calls (including call_id|fc_* ids) stay untouched.
  const clean: AssistantMessage['content'] = [
    { type: 'thinking', thinking: 'plain thought' },
    { type: 'text', text: 'hello' },
    { type: 'toolCall', id: 'call_1|fc_1', name: 'bash', arguments: { command: 'ls' } }
  ];
  assert.equal(retireAssistantContent(clean), undefined);

  // Signed non-empty thinking keeps its text without thinkingSignature; empty signed thinking and
  // redacted thinking are dropped.
  const signedThinking: ThinkingContent = {
    type: 'thinking',
    thinking: 'step 1',
    thinkingSignature: 'sig-anth'
  };
  const emptyEncrypted: ThinkingContent = {
    type: 'thinking',
    thinking: '',
    thinkingSignature: '{"id":"rs_123","type":"reasoning","encrypted_content":"abc"}'
  };
  const redactedBlock: ThinkingContent = {
    type: 'thinking',
    thinking: '[Reasoning redacted]',
    thinkingSignature: 'opaque-redacted-bytes',
    redacted: true
  };
  const phaseText: TextContent = {
    type: 'text',
    text: 'final reply',
    textSignature: JSON.stringify({ v: 1, id: 'msg_abc123', phase: 'final_answer' })
  };
  const idOnlyText: TextContent = {
    type: 'text',
    text: 'plain',
    textSignature: JSON.stringify({ v: 1, id: 'msg_def456' })
  };
  const geminiText: TextContent = {
    type: 'text',
    text: 'gemini',
    textSignature: 'Z2VtaW5pLXNpZw=='
  };
  const signedToolCall: ToolCall = {
    type: 'toolCall',
    id: 'call_99|fc_item99',
    name: 'read',
    arguments: { path: 'a.ts' },
    thoughtSignature: 'Z29vZ2xlLXRjLXNpZw=='
  };

  const retired = retireAssistantContent([
    signedThinking,
    emptyEncrypted,
    redactedBlock,
    phaseText,
    idOnlyText,
    geminiText,
    signedToolCall
  ]);
  assert.deepEqual(retired, [
    { type: 'thinking', thinking: 'step 1' },
    {
      type: 'text',
      text: 'final reply',
      textSignature: JSON.stringify({ v: 1, id: '', phase: 'final_answer' })
    },
    { type: 'text', text: 'plain' },
    { type: 'text', text: 'gemini' },
    { type: 'toolCall', id: 'call_99|fc_item99', name: 'read', arguments: { path: 'a.ts' } }
  ]);

  // CRITICAL edge case: assistant message that held ONLY redacted or empty encrypted thinking
  // never becomes content: [].
  const redactedOnly = retireAssistantContent([redactedBlock]);
  assert.deepEqual(redactedOnly, [{ type: 'text', text: '[thinking retired]' }]);
  const encryptedOnly = retireAssistantContent([emptyEncrypted]);
  assert.deepEqual(encryptedOnly, [{ type: 'text', text: '[thinking retired]' }]);

  // Thinking-only turns convert non-empty thinking to plain text so OpenAI Responses (and all
  // other adapters) emit a non-empty assistant message instead of dropping the turn.
  const thinkingOnly = retireAssistantContent([signedThinking]);
  assert.deepEqual(thinkingOnly, [{ type: 'text', text: 'step 1' }]);
  const unsignedThinkingOnly = retireAssistantContent([
    { type: 'thinking', thinking: 'unsigned step' }
  ]);
  assert.deepEqual(unsignedThinkingOnly, [{ type: 'text', text: 'unsigned step' }]);
  const whitespaceTextAndThinking = retireAssistantContent([
    { type: 'thinking', thinking: 'thought body', thinkingSignature: 'sig' },
    { type: 'text', text: '   ' }
  ]);
  assert.deepEqual(whitespaceTextAndThinking, [{ type: 'text', text: 'thought body' }]);
});

test('planFold retires signatures on surviving turns after until and on the fold response itself', () => {
  const preFold = {
    ...assistant('early'),
    content: [
      { type: 'thinking' as const, thinking: 'keep signature', thinkingSignature: 'sig-before' },
      { type: 'text' as const, text: 'early' },
      { type: 'toolCall' as const, id: 'c0|fc_0', name: 'read', arguments: { path: 'a.ts' } }
    ]
  };
  const foldedTurn = assistant(call('c1', 'bash', { command: 'echo 1' }));
  const survivingTurn = {
    ...assistant('kept'),
    content: [
      {
        type: 'thinking' as const,
        thinking: '',
        thinkingSignature: '{"id":"rs_surviving","type":"reasoning","encrypted_content":"enc"}'
      },
      {
        type: 'text' as const,
        text: 'checking',
        textSignature: JSON.stringify({ v: 1, id: 'msg_surviving', phase: 'commentary' })
      },
      {
        type: 'toolCall' as const,
        id: 'c2|fc_surviving',
        name: 'bash',
        arguments: { command: 'echo 2' },
        thoughtSignature: 'dGhvdWdodA=='
      }
    ]
  };
  const foldCaller = {
    ...assistant('folding'),
    content: [
      {
        type: 'thinking' as const,
        thinking: '[Reasoning redacted]',
        thinkingSignature: 'redacted-blob',
        redacted: true
      },
      { type: 'toolCall' as const, id: 'f1|fc_fold', name: 'fold', arguments: { since: 't3' } }
    ]
  };

  const { manager } = session([
    user('start', 1),
    preFold,
    result('c0|fc_0', 'read', 'file', 2),
    foldedTurn,
    result('c1', 'bash', 'one', 3),
    survivingTurn,
    result('c2|fc_surviving', 'bash', 'two', 4),
    foldCaller
  ]);

  const plan = planFold(manager.buildSessionProjection().entries, 'f1|fc_fold', 3, { until: 3 });
  applyDrafts(manager, plan.edits);
  manager.appendMessage(result('f1|fc_fold', 'fold', '[fold note]', 5));

  const view = manager.buildSessionProjection().messages;
  const assistants = view.filter((m): m is AssistantMessage => m.role === 'assistant');
  assert.equal(assistants.length, 3);

  // Prefix before the folded span is untouched (cache preserved).
  assert.equal((assistants[0]!.content[0] as ThinkingContent).thinkingSignature, 'sig-before');
  assert.equal((assistants[0]!.content[2] as ToolCall).id, 'c0|fc_0');

  // Surviving assistant after `until` had encrypted thinking dropped, textSignature sanitized,
  // and thoughtSignature retired, while ToolCall.id stays intact to match ToolResultMessage.toolCallId.
  assert.deepEqual(assistants[1]!.content, [
    {
      type: 'text',
      text: 'checking',
      textSignature: JSON.stringify({ v: 1, id: '', phase: 'commentary' })
    },
    { type: 'toolCall', id: 'c2|fc_surviving', name: 'bash', arguments: { command: 'echo 2' } }
  ]);

  // Fold caller had redacted thinking dropped and keeps its full ToolCall.id; branch retirement
  // records c2 and f1 (leaving pre-fold c0 untouched) for wire-level fc_* stripping.
  assert.deepEqual(assistants[2]!.content, [
    { type: 'toolCall', id: 'f1|fc_fold', name: 'fold', arguments: { since: 't3' } }
  ]);
  assertPaired(view);
  assert.deepEqual(
    branchRetiredResponsesCallIds(manager.getBranch(), manager.buildSessionProjection().entries),
    new Set(['c2', 'f1'])
  );
});

test('unfold happy path restores assistant, toolResult, custom_message and capped user entries, replaces note and retires signatures', () => {
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
  manager.appendMessage({
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

  // Restored assistant message had its signature retired while keeping ToolCall.id === 'c1|fc_inner'
  // so it pairs cleanly with result('c1|fc_inner').
  const restoredAsst = byId.get(ids[2]!)![0] as AssistantMessage;
  assert.deepEqual(restoredAsst.content, [
    { type: 'thinking', thinking: 'inner reasoning' },
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

  // Thinking-only redacted assistant turn after the fold became [thinking retired], not [].
  const retiredRedacted = projected.messages.find(
    (m) =>
      m.role === 'assistant' &&
      m.content.length === 1 &&
      m.content[0]?.type === 'text' &&
      m.content[0].text === '[thinking retired]'
  );
  assert.ok(retiredRedacted);
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

test('ForkFolds retires stale signatures after fold and supports in-place unfold', async () => {
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

  // Folded turn is omitted; surviving child assistant messages have signatures retired.
  assert.ok(!JSON.stringify(viewAfterFold).includes('secret-one'));
  const keptAsst = viewAfterFold.find(
    (m): m is AssistantMessage =>
      m.role === 'assistant' && m.content.some((b) => b.type === 'toolCall' && b.name === 'bash')
  )!;
  assert.deepEqual(keptAsst.content, [
    { type: 'toolCall', id: 'c2|fc_c2', name: 'bash', arguments: { command: 'echo kept-two' } }
  ]);
  const foldCallerView = viewAfterFold.find(
    (m): m is AssistantMessage =>
      m.role === 'assistant' && m.content.some((b) => b.type === 'toolCall' && b.name === 'fold')
  )!;
  assert.deepEqual(foldCallerView.content, [
    { type: 'toolCall', id: 'f_child|fc_fc', name: 'fold', arguments: {} }
  ]);
  assertPaired(viewAfterFold);
  assert.deepEqual(folds.retiredCallIds(messages), new Set(['c2', 'f_child']));
  assert.deepEqual(cacheAnchor(viewAfterFold, folds.activeHandles()), {
    toolUseId: 'f_child|fc_fc'
  });

  // New child turn appended AFTER fold keeps its signatures and is excluded from retiredCallIds.
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
  assert.equal(folds.retiredCallIds(messages).has('c_fresh'), false);

  // Unfold inside the fork restores secret-one, retires the note, and bounds retirement up to unfold.
  messages.push(assistant(call('u_child|fc_uc', 'unfold', { handle })));
  const unfolded = await unfoldTool.execute('u_child|fc_uc', { handle });
  messages.push(result('u_child|fc_uc', 'unfold', (unfolded.content[0] as TextContent).text, 15));

  assert.equal(folds.activeFolds.length, 0);
  assert.equal(folds.activeHandles().has(handle), false);
  assert.deepEqual(
    folds.retiredCallIds(messages),
    new Set(['c1', 'c2', 'f_child', 'c_fresh', 'u_child'])
  );
  const viewAfterUnfold = folds.view(messages);
  assertPaired(viewAfterUnfold);
  const contentTexts = viewAfterUnfold
    .flatMap((m) => (m.role === 'toolResult' ? m.content : []))
    .map((b) => (b.type === 'text' ? b.text : ''))
    .join('\n');
  assert.ok(contentTexts.includes('secret-one'));
  assert.ok(!contentTexts.includes('child folded turn 11'));
  assert.ok(contentTexts.includes(`[unfolded ${handle}]`));

  // A turn appended AFTER unfold stays unretired.
  const postUnfoldTurn: AssistantMessage = {
    ...assistant('post-unfold'),
    content: [
      { type: 'thinking', thinking: 'post', thinkingSignature: 'post-unfold-sig' },
      { type: 'toolCall', id: 'c_post|fc_post', name: 'read', arguments: { path: 'y.ts' } }
    ]
  };
  messages.push(postUnfoldTurn);
  messages.push(result('c_post|fc_post', 'read', 'y', 16));
  assert.equal(folds.retiredCallIds(messages).has('c_post'), false);
  const postSeen = folds.view(messages).at(-2) as AssistantMessage;
  assert.equal((postSeen.content[0] as ThinkingContent).thinkingSignature, 'post-unfold-sig');
});

test('live session unfold works both directly and through codemode, updates archive/status, and retires current turn signatures', async () => {
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
        assert.ok(!full.includes('rs_live_fold'), `${mode}: fold turn reasoning retired`);
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
        assert.ok(!full.includes('sig-unfold-turn'), `${mode}: unfold turn signature retired`);
        assertPaired(conv);
        return fauxAssistantMessage('All verified.');
      }
    ];

    const strippedPayloads: Record<string, unknown>[][] = [];
    faux.setResponses(
      steps.map((step) => async (context, streamOptions, _state, requestModel) => {
        requests.push(context);
        const wire = {
          model: requestModel.id,
          input: convertResponsesMessages(requestModel, context, new Set([requestModel.provider]))
        };
        const out = ((await streamOptions?.onPayload?.(wire, requestModel)) ?? wire) as {
          input: Record<string, unknown>[];
        };
        strippedPayloads.push(out.input);
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
      if (mode === 'off') {
        const afterFoldInput = strippedPayloads[2]!;
        const liveFoldCalls = afterFoldInput.filter(
          (item) => item.type === 'function_call' && item.call_id === 'f_live'
        );
        assert.equal(liveFoldCalls.length, 1);
        assert.equal('id' in liveFoldCalls[0]!, false);
        assert.ok(!JSON.stringify(afterFoldInput).includes('No result provided'));
      }
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

test('OpenAI Responses adapter pairs retired call_abc|fc_123 without synthetic error output and strips stale wire item ids', () => {
  const model: Model<'openai-responses'> = {
    id: 'gpt-5',
    name: 'GPT-5',
    api: 'openai-responses',
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    reasoning: true,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 16384
  };
  const allowedProviders = new Set(['openai', 'openai-codex', 'opencode']);

  const prefixTurn: AssistantMessage = {
    ...assistant('prefix'),
    api: 'openai-responses',
    provider: 'openai',
    model: 'gpt-5',
    content: [
      {
        type: 'thinking',
        thinking: '',
        thinkingSignature: JSON.stringify({
          id: 'rs_prefix',
          type: 'reasoning',
          encrypted_content: 'enc_prefix',
          summary: []
        })
      },
      { type: 'toolCall', id: 'call_pre|fc_pre', name: 'read', arguments: { path: 'a.ts' } }
    ]
  };
  const prefixResult = result('call_pre|fc_pre', 'read', 'contents');

  const postFoldTurn: AssistantMessage = {
    ...assistant('post'),
    api: 'openai-responses',
    provider: 'openai',
    model: 'gpt-5',
    content: [
      {
        type: 'thinking',
        thinking: '',
        thinkingSignature: JSON.stringify({
          id: 'rs_123',
          type: 'reasoning',
          encrypted_content: 'enc_123',
          summary: []
        })
      },
      {
        type: 'toolCall',
        id: 'call_abc|fc_123',
        name: 'bash',
        arguments: { command: 'echo ok' }
      }
    ]
  };
  const postFoldResult = result('call_abc|fc_123', 'bash', 'ok');

  const thinkingOnlyTurn: AssistantMessage = {
    ...assistant('thought-only'),
    api: 'openai-responses',
    provider: 'openai',
    model: 'gpt-5',
    content: [
      {
        type: 'thinking',
        thinking: 'deliberating on final answer',
        thinkingSignature: JSON.stringify({
          id: 'rs_think',
          type: 'reasoning',
          encrypted_content: 'enc_think',
          summary: [{ type: 'summary_text', text: 'deliberating on final answer' }]
        })
      }
    ]
  };

  const retiredTurn = retireAssistantMessage(postFoldTurn);
  const retiredThinkingOnly = retireAssistantMessage(thinkingOnlyTurn);

  // ToolCall.id remains intact so transformMessages pairs it with ToolResultMessage.toolCallId.
  const retiredCall = retiredTurn.content.find((b): b is ToolCall => b.type === 'toolCall')!;
  assert.equal(retiredCall.id, 'call_abc|fc_123');

  const messages: Message[] = [
    user('run check'),
    prefixTurn,
    prefixResult,
    retiredTurn,
    postFoldResult,
    retiredThinkingOnly,
    user('follow up')
  ];

  const rawInput = convertResponsesMessages(
    model,
    normalizeContext({ messages }),
    allowedProviders
  );
  const retiredIds = retiredResponsesCallIds(messages, 3);
  assert.deepEqual(retiredIds, new Set(['call_abc']));

  const rawPayload = { model: model.id, input: rawInput };
  const stripped = stripRetiredResponsesIds(rawPayload, retiredIds) as {
    model: string;
    input: Record<string, unknown>[];
  };

  // Cached prefix before the fold keeps its reasoning item and fc_pre item id untouched.
  const preCall = stripped.input.find(
    (item) => item.type === 'function_call' && item.call_id === 'call_pre'
  )!;
  assert.equal(preCall.id, 'fc_pre');
  assert.ok(stripped.input.some((item) => item.type === 'reasoning' && item.id === 'rs_prefix'));

  // Post-fold call has id omitted (undefined), call_id === 'call_abc', and a single output 'ok'
  // (no synthetic 'No result provided' error output injected by transformMessages).
  const postCalls = stripped.input.filter(
    (item) => item.type === 'function_call' && item.call_id === 'call_abc'
  );
  assert.equal(postCalls.length, 1);
  assert.equal(postCalls[0]!.id, undefined);
  assert.equal('id' in postCalls[0]!, false);
  assert.equal(postCalls[0]!.call_id, 'call_abc');

  const postOutputs = stripped.input.filter(
    (item) => item.type === 'function_call_output' && item.call_id === 'call_abc'
  );
  assert.equal(postOutputs.length, 1);
  assert.equal(postOutputs[0]!.output, 'ok');
  assert.ok(!JSON.stringify(stripped.input).includes('No result provided'));

  // Thinking-only retired turn survived as an assistant message item instead of being dropped.
  const assistantMessages = stripped.input.filter(
    (item) => item.type === 'message' && item.role === 'assistant'
  );
  assert.equal(assistantMessages.length, 1);
  assert.ok(JSON.stringify(assistantMessages[0]).includes('deliberating on final answer'));

  // Custom grammar tool calls (ctc_*) are stripped too, and untouched payloads keep identity.
  const withCustom = {
    input: [
      { type: 'custom_tool_call', id: 'ctc_9', call_id: 'call_abc', name: 'patch', input: '' }
    ]
  };
  const strippedCustom = stripRetiredResponsesIds(withCustom, retiredIds) as {
    input: Record<string, unknown>[];
  };
  assert.equal('id' in strippedCustom.input[0]!, false);
  assert.equal(stripRetiredResponsesIds(stripped, retiredIds), stripped);
  assert.equal(stripRetiredResponsesIds(rawPayload, new Set()), rawPayload);
});

test('requestOptions onPayload strips branch and fork retired Responses ids even when cacheAnchor is disabled', async () => {
  const { manager } = session([
    user('start', 1),
    assistant(call('c_pre|fc_pre', 'read', { path: 'a.ts' })),
    result('c_pre|fc_pre', 'read', 'a', 2),
    assistant(call('c_span|fc_span', 'bash', { command: 'echo 1' })),
    result('c_span|fc_span', 'bash', '1', 3),
    assistant(call('f1|fc_fold', 'fold', { since: 't3', note: 'folded' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1|fc_fold', 3);
  applyDrafts(manager, plan.edits);
  manager.appendMessage(result('f1|fc_fold', 'fold', '[fold note]', 4));

  const piStub = {
    getSettings: () => ({ episodes: { cacheAnchor: false } }),
    getThinkingLevel: () => 'off'
  } as never;
  const ctxStub = {
    sessionManager: manager,
    thinkingLevel: 'off'
  } as never;

  const opts = requestOptions(
    piStub,
    ctxStub,
    () => manager.buildSessionProjection().messages,
    undefined,
    () => new Set(['c_fork'])
  );
  const wire = {
    model: 'gpt-5',
    input: [
      { type: 'function_call', id: 'fc_pre', call_id: 'c_pre', name: 'read', arguments: '{}' },
      { type: 'function_call', id: 'fc_fold', call_id: 'f1', name: 'fold', arguments: '{}' },
      { type: 'custom_tool_call', id: 'ctc_fork', call_id: 'c_fork', name: 'patch', input: 'x' }
    ]
  };
  const transformed = (await opts.onPayload!(wire, {} as never)) as typeof wire;
  assert.notEqual(transformed, undefined);
  assert.equal(transformed.input[0]!.id, 'fc_pre');
  assert.equal('id' in transformed.input[1]!, false);
  assert.equal('id' in transformed.input[2]!, false);
  // Calling again when nothing matches returns undefined so pi-ai keeps the payload reference.
  assert.equal(await opts.onPayload!(transformed, {} as never), undefined);
});

test('branchRetiredResponsesCallIds excludes new assistant turns created after the latest edit', () => {
  const { manager } = session([
    user('u1', 1),
    assistant(call('c1|fc_1', 'bash', { command: 'echo 1' })),
    result('c1|fc_1', 'bash', '1', 2),
    assistant(call('f1|fc_f1', 'fold', { since: 't2', note: 'n' }))
  ]);
  const plan = planFold(manager.buildSessionProjection().entries, 'f1|fc_f1', 2);
  applyDrafts(manager, plan.edits);
  manager.appendMessage(result('f1|fc_f1', 'fold', '[fold note]', 3));
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: plan.handle });

  // New turn appended AFTER the fold mutation keeps its Responses call_id unretired.
  manager.appendMessage(assistant(call('c_new|fc_new', 'read', { path: 'fresh.ts' })));
  manager.appendMessage(result('c_new|fc_new', 'read', 'fresh', 4));

  const retired = branchRetiredResponsesCallIds(
    manager.getBranch(),
    manager.buildSessionProjection().entries
  );
  assert.deepEqual(retired, new Set(['f1']));
  assert.equal(retired.has('c_new'), false);
});

test('sanitizeProviderPayload handles Gemini 3/4 skip_thought_signature_validator and Anthropic/Bedrock budget thinking', () => {
  // 1. Google Gemini 3/4 via real convertGoogleMessages
  const geminiModel: Model<'google-generative-ai'> = {
    id: 'gemini-3-pro',
    name: 'Gemini 3 Pro',
    api: 'google-generative-ai',
    provider: 'google',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    reasoning: true,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1000000,
    maxTokens: 16384
  };
  const validB64Sig = Buffer.from('valid-gemini-sig').toString('base64');
  const priorTurn = retireAssistantMessage({
    ...assistant('prior'),
    api: 'google-generative-ai',
    provider: 'google',
    model: 'gemini-3-pro',
    content: [
      {
        type: 'toolCall',
        id: 'g_old',
        name: 'read',
        arguments: { path: 'old.ts' },
        thoughtSignature: validB64Sig
      }
    ]
  });
  const currentRetiredParallel = retireAssistantMessage({
    ...assistant('parallel'),
    api: 'google-generative-ai',
    provider: 'google',
    model: 'gemini-3-pro',
    content: [
      {
        type: 'toolCall',
        id: 'g1',
        name: 'read',
        arguments: { path: 'a.ts' },
        thoughtSignature: validB64Sig
      },
      { type: 'toolCall', id: 'g2', name: 'bash', arguments: { command: 'ls' } }
    ]
  });
  const currentSignedTurn: AssistantMessage = {
    ...assistant('signed'),
    api: 'google-generative-ai',
    provider: 'google',
    model: 'gemini-3-pro',
    content: [
      {
        type: 'toolCall',
        id: 'g3',
        name: 'read',
        arguments: { path: 'b.ts' },
        thoughtSignature: validB64Sig
      }
    ]
  };
  const contents = convertGoogleMessages(
    geminiModel,
    normalizeContext({
      messages: [
        user('first prompt'),
        priorTurn,
        result('g_old', 'read', 'old'),
        user('second prompt'),
        currentRetiredParallel,
        result('g1', 'read', 'a'),
        result('g2', 'bash', 'files'),
        currentSignedTurn,
        result('g3', 'read', 'b')
      ]
    })
  );
  const sanitizedGoogle = sanitizeProviderPayload(
    { model: geminiModel.id, contents },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { contents: { role: string; parts: Record<string, unknown>[] }[] };
  // Earlier turn before 'second prompt' stays untouched.
  assert.equal(sanitizedGoogle.contents[1]!.parts[0]!.thoughtSignature, undefined);
  // Retired parallel turn in current user turn gets skip_thought_signature_validator ONLY on first functionCall.
  const parallelModelParts = sanitizedGoogle.contents[4]!.parts;
  assert.equal(parallelModelParts[0]!.thoughtSignature, GEMINI_SKIP_THOUGHT_SIGNATURE);
  assert.equal(parallelModelParts[1]!.thoughtSignature, undefined);
  // Fresh signed turn keeps its real base64 signature.
  assert.equal(sanitizedGoogle.contents[6]!.parts[0]!.thoughtSignature, validB64Sig);

  // 2. Anthropic budget thinking (`thinking.type === 'enabled'`)
  const anthropicToolCont = {
    model: 'claude-sonnet-4-20250514',
    thinking: { type: 'enabled', budget_tokens: 2048 },
    betas: ['claude-code-20250219', 'interleaved-thinking-2025-05-14'],
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'do work' }] },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'retired thought' },
          { type: 'tool_use', id: 'tu_1', name: 'fold', input: {} }
        ]
      },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_1', content: 'ok' }] }
    ]
  };
  const sanitizedAnth = sanitizeProviderPayload(anthropicToolCont, {
    retiredCallIds: new Set(),
    hasMutatedHistory: true
  }) as Record<string, unknown>;
  assert.equal('thinking' in sanitizedAnth, false);
  assert.deepEqual(sanitizedAnth.betas, ['claude-code-20250219']);
  // Once the assistant finishes that turn and a plain user prompt follows, budget thinking stays enabled.
  const anthropicNextTurn = {
    ...anthropicToolCont,
    messages: [
      ...anthropicToolCont.messages,
      { role: 'assistant', content: [{ type: 'text', text: 'all done' }] },
      { role: 'user', content: 'next question' }
    ]
  };
  assert.equal(
    sanitizeProviderPayload(anthropicNextTurn, {
      retiredCallIds: new Set(),
      hasMutatedHistory: true
    }),
    anthropicNextTurn
  );
  // Adaptive thinking stays untouched.
  const anthropicAdaptive = { ...anthropicToolCont, thinking: { type: 'adaptive' } };
  assert.equal(
    sanitizeProviderPayload(anthropicAdaptive, {
      retiredCallIds: new Set(),
      hasMutatedHistory: true
    }),
    anthropicAdaptive
  );

  // 3. Bedrock Converse budget thinking
  const bedrockCont = {
    modelId: 'anthropic.claude-sonnet-4-v1:0',
    additionalModelRequestFields: {
      thinking: { type: 'enabled', budget_tokens: 2048 },
      anthropic_beta: ['interleaved-thinking-2025-05-14']
    },
    messages: [
      { role: 'user', content: [{ text: 'go' }] },
      { role: 'assistant', content: [{ text: 'retired' }, { toolUse: { toolUseId: 'b1' } }] },
      { role: 'user', content: [{ toolResult: { toolUseId: 'b1', content: [] } }] }
    ]
  };
  const sanitizedBedrock = sanitizeProviderPayload(bedrockCont, {
    retiredCallIds: new Set(),
    hasMutatedHistory: true
  }) as Record<string, unknown>;
  assert.equal('additionalModelRequestFields' in sanitizedBedrock, false);
});

test('forkThreads replays archived fork folds, unfold stubs, historical nested snapshots, and retired call ids', async () => {
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
  //    retired turns stripped, and post-unfold turn sig_final preserved!
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
  assert.ok(!forkText.includes('sig1'));
  assert.ok(forkText.includes('sig_final'));
  const retiredIds = forkThread.retiredCallIds!();
  assert.equal(retiredIds.has('k1'), true);
  assert.equal(retiredIds.has('k_final'), false);
  assert.equal(forkThread.hasMutatedHistory!(), true);

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

test('planFold orders signature retirement before removals and repair-only turn_end retires current response', async () => {
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
  // First edit retires surviving assistant signature; remaining edits remove span entries (replacement === null).
  assert.notEqual(plan.edits[0]!.replacement, null);
  assert.equal(plan.edits.at(-1)!.replacement, null);

  // Live session repair-only turn_end retires current assistant response's signature.
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
    assert.equal(JSON.stringify(projMsgs).includes('sig-repair-turn'), false);
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

test('Bedrock cachePoint blocks, Anthropic/Bedrock steer continuations, and Chat Completions reasoning preservation', () => {
  // 1. Bedrock Converse with default prompt caching appends `{ cachePoint: { type: 'default' } }`
  //    to user messages: active tool continuation omits `thinking`, while a new plain prompt keeps it.
  const bedrockToolCont = sanitizeProviderPayload(
    {
      messages: [
        { role: 'user', content: [{ text: 'run fold' }, { cachePoint: { type: 'default' } }] },
        {
          role: 'assistant',
          content: [
            { text: 'folding now' },
            { toolUse: { toolUseId: 'f1', name: 'fold', input: {} } }
          ]
        },
        {
          role: 'user',
          content: [
            { toolResult: { toolUseId: 'f1', content: [{ text: '[fold]' }], status: 'success' } },
            { cachePoint: { type: 'default' } }
          ]
        }
      ],
      additionalModelRequestFields: {
        thinking: { type: 'enabled', budgetTokens: 2048 },
        anthropic_beta: ['interleaved-thinking-2025-05-14']
      }
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { additionalModelRequestFields?: unknown };
  assert.equal(bedrockToolCont.additionalModelRequestFields, undefined);

  const bedrockFreshPrompt = sanitizeProviderPayload(
    {
      messages: [
        { role: 'user', content: [{ text: 'run fold' }] },
        { role: 'assistant', content: [{ text: 'done' }] },
        { role: 'user', content: [{ text: 'next question' }, { cachePoint: { type: 'default' } }] }
      ],
      additionalModelRequestFields: {
        thinking: { type: 'enabled', budgetTokens: 2048 }
      }
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { additionalModelRequestFields?: { thinking?: unknown } };
  assert.deepEqual(bedrockFreshPrompt.additionalModelRequestFields?.thinking, {
    type: 'enabled',
    budgetTokens: 2048
  });

  // 2. Anthropic steer continuation (`[assistant(tool_use), user([tool_result, text('steer')])]`)
  const anthropicSteered = sanitizeProviderPayload(
    {
      thinking: { type: 'enabled', budget_tokens: 4096 },
      betas: ['interleaved-thinking-2025-05-14'],
      messages: [
        { role: 'user', content: 'start' },
        { role: 'assistant', content: [{ type: 'tool_use', id: 'f1', name: 'fold', input: {} }] },
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'f1', content: 'folded' },
            { type: 'text', text: 'also check X' }
          ]
        }
      ]
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { thinking?: unknown; betas?: unknown };
  assert.equal(anthropicSteered.thinking, undefined);
  assert.equal(anthropicSteered.betas, undefined);

  // 3. Chat Completions plain field selectors ('reasoning_content', 'reasoning', 'reasoning_text')
  //    stay intact while cryptographic signatures are retired.
  const deepseekUnchanged = retireAssistantContent([
    { type: 'thinking', thinking: 'deepseek chain', thinkingSignature: 'reasoning_content' },
    { type: 'toolCall', id: 'c1', name: 'bash', arguments: { command: 'ls' } }
  ]);
  assert.equal(deepseekUnchanged, undefined);

  // 4. OpenRouter Gemini & Google OpenAI-compat Gemini & OpenRouter Claude Chat Completions payloads
  const orGemini = sanitizeProviderPayload(
    {
      model: 'google/gemini-3-pro-preview',
      reasoning: { effort: 'high' },
      messages: [
        { role: 'user', content: 'fold' },
        {
          role: 'assistant',
          content: 'folding',
          tool_calls: [
            { id: 'call_g1', type: 'function', function: { name: 'fold', arguments: '{}' } }
          ]
        },
        { role: 'tool', tool_call_id: 'call_g1', content: '[fold]' }
      ]
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { messages: Array<{ reasoning_details?: Array<{ data?: string; id?: string }> }> };
  assert.equal(
    orGemini.messages[1]!.reasoning_details?.[0]?.data,
    'skip_thought_signature_validator'
  );
  assert.equal(orGemini.messages[1]!.reasoning_details?.[0]?.id, 'call_g1');

  const compatGemini = sanitizeProviderPayload(
    {
      model: 'gemini-3-flash-preview',
      messages: [
        { role: 'user', content: 'fold' },
        {
          role: 'assistant',
          tool_calls: [
            { id: 'call_g2', type: 'function', function: { name: 'fold', arguments: '{}' } }
          ]
        },
        { role: 'tool', tool_call_id: 'call_g2', content: '[fold]' }
      ]
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as {
    messages: Array<{
      tool_calls?: Array<{ extra_content?: { google?: { thought_signature?: string } } }>;
    }>;
  };
  assert.equal(
    compatGemini.messages[1]!.tool_calls?.[0]?.extra_content?.google?.thought_signature,
    'skip_thought_signature_validator'
  );

  const orClaude = sanitizeProviderPayload(
    {
      model: 'anthropic/claude-opus-4-6',
      reasoning: { effort: 'high' },
      messages: [
        { role: 'user', content: 'fold' },
        {
          role: 'assistant',
          content: 'folding',
          tool_calls: [
            { id: 'call_c1', type: 'function', function: { name: 'fold', arguments: '{}' } }
          ]
        },
        { role: 'tool', tool_call_id: 'call_c1', content: '[fold]' }
      ]
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { reasoning?: unknown };
  assert.equal(orClaude.reasoning, undefined);
});

test('retiredResponsesCallIds preserves intact (rs_*, fc_*) pairs between two folds while stripping retired turns', () => {
  const msgs = [
    {
      ...assistant('retired-early'),
      content: [
        { type: 'thinking' as const, thinking: 'retired thought' },
        { type: 'toolCall' as const, id: 'c_ret|fc_ret', name: 'bash', arguments: {} }
      ]
    },
    {
      ...assistant('intact-middle'),
      content: [
        {
          type: 'thinking' as const,
          thinking: 'intact thought',
          thinkingSignature: JSON.stringify({ id: 'rs_mid', type: 'reasoning' })
        },
        { type: 'toolCall' as const, id: 'c_mid|fc_mid', name: 'bash', arguments: {} }
      ]
    }
  ];
  const ids = retiredResponsesCallIds(msgs, 0);
  assert.equal(ids.has('c_ret'), true);
  assert.equal(ids.has('c_mid'), false);
});

test('planUnfold retires signatures first, unfoldEditsOf preserves earlier turn_end boundary edits, and Thread exposes appliedFolds', async () => {
  // 1. planUnfold orders surviving-assistant signature retirement BEFORE span restoration & note stub
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
  assert.equal(uPlan.edits[0]!.targetId, survId);

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
    assert.ok(!JSON.stringify(projMsgs).includes('sig-unfold-call'));
  } finally {
    agentSession.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

test('completeCutShortFolds reconciles interrupted unfold recovery with pending boundary edits and split-message steers', () => {
  // 1. Split-message steer shape produced by pi-ai (`user([tool_result])` followed by `user(text)`)
  const anthropicSplitSteer = sanitizeProviderPayload(
    {
      thinking: { type: 'enabled', budget_tokens: 4096 },
      betas: ['interleaved-thinking-2025-05-14'],
      messages: [
        { role: 'user', content: 'start' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'folding' },
            { type: 'tool_use', id: 'f1', name: 'fold', input: {} }
          ]
        },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'f1', content: 'folded' }] },
        { role: 'user', content: [{ type: 'text', text: 'also check X' }] }
      ]
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { thinking?: unknown; betas?: unknown };
  assert.equal(anthropicSplitSteer.thinking, undefined);
  assert.equal(anthropicSplitSteer.betas, undefined);

  const bedrockSplitSteer = sanitizeProviderPayload(
    {
      messages: [
        { role: 'user', content: [{ text: 'start' }] },
        {
          role: 'assistant',
          content: [{ text: 'folding' }, { toolUse: { toolUseId: 'f1', name: 'fold', input: {} } }]
        },
        {
          role: 'user',
          content: [
            { toolResult: { toolUseId: 'f1', content: [{ text: 'folded' }], status: 'success' } }
          ]
        },
        { role: 'user', content: [{ text: 'also check X' }, { cachePoint: { type: 'default' } }] }
      ],
      additionalModelRequestFields: {
        thinking: { type: 'enabled', budgetTokens: 2048 }
      }
    },
    { retiredCallIds: new Set(), hasMutatedHistory: true }
  ) as { additionalModelRequestFields?: unknown };
  assert.equal(bedrockSplitSteer.additionalModelRequestFields, undefined);

  // 2. Interrupted unfold recovery preserves pending boundary replacement on toolResult and retires
  //    signatures on pending boundary replacement of a restored assistant message!
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

  // Add surviving signed assistant message so uPlan.edits[0] is its retirement, then interrupt after edits[0]!
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
  // AND the restored assistant message (with a fresh stale signature)!
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
  assert.ok(!projText.includes('sig-from-hook'));
});
test('running fork callingFork and prefixEntryId integrate with nested unfold and retired signatures', async () => {
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
  assert.equal(foundFolded!.hasMutatedHistory!(), true);
  assert.equal(foundFolded!.retiredCallIds!().has('f_par'), true);

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
