import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  type ImageContent,
  type Message,
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
  FOLD_STARTED_TYPE,
  type FoldDetails
} from '../src/archive.js';
import {
  attributeEpisodesEdits,
  completeCutShortFolds,
  FoldError,
  mergeSpanDiskChanges,
  normalizeFoldSpans,
  planFoldBatch,
  planUnfold,
  USER_TEXT_CAP
} from '../src/fold.js';
import { ForkFolds } from '../src/fork-fold.js';
import { FOLD_ABORTED_MESSAGE } from '../src/images.js';
import episodes from '../src/index.js';
import { planFoldTailPruning } from '../src/prune.js';
import { withStamp } from '../src/stamps.js';
import { EpisodeState } from '../src/state.js';
import { episodeStats } from '../src/stats.js';
import { foldRenderers, unfoldRenderers } from '../src/ui.js';
import { assistant, call, result, session, user } from './helpers.js';

// 1x1 transparent PNG
const TINY_PNG =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

function text(message: AgentMessage | Message): string {
  if (!('content' in message)) return '';
  if (typeof message.content === 'string') return message.content;
  return message.content.flatMap((block) => (block.type === 'text' ? [block.text] : [])).join('\n');
}

/** No toolResult may ever appear without its preceding assistant toolCall. */
function assertResultsHaveCallers(messages: readonly AgentMessage[], label = ''): void {
  const llm = convertToLlm([...messages]);
  const called = new Set<string>();
  for (const message of llm) {
    if (message.role === 'assistant') {
      for (const block of message.content) {
        if (block.type === 'toolCall') called.add(block.id);
      }
    } else if (message.role === 'toolResult') {
      assert.ok(
        called.has(message.toolCallId),
        `${label}: orphaned toolResult ${message.toolCallId}`
      );
    }
  }
}

function signedAssistant(id: string, toolCall: ReturnType<typeof call>): AssistantMessage {
  return {
    ...assistant(toolCall),
    content: [
      {
        type: 'thinking',
        thinking: `reasoning-${id}`,
        thinkingSignature: JSON.stringify({ id: `rs_${id}`, type: 'reasoning' })
      },
      toolCall
    ]
  };
}

function hasLiveSignature(message: AgentMessage): boolean {
  if (message.role !== 'assistant') return false;
  return message.content.some(
    (block) => block.type === 'thinking' && Boolean(block.thinkingSignature)
  );
}

type Step = (context: TranscriptContext) => AssistantMessage | Promise<AssistantMessage>;

async function createLiveSession(
  steps: Step[],
  options: { git?: boolean; persist?: boolean; codemodeOnly?: boolean } = {}
) {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-batch-'));
  if (options.git) {
    execFileSync('git', ['init', '-q'], { cwd: dir });
    execFileSync('git', ['config', 'user.email', 'test@example.com'], { cwd: dir });
    execFileSync('git', ['config', 'user.name', 'Test'], { cwd: dir });
  }
  const faux = fauxProvider({ models: [{ id: 'faux-batch', reasoning: true }] });
  const requests: TranscriptContext[] = [];
  const failures: unknown[] = [];
  faux.setResponses(
    steps.map((step) => async (context) => {
      requests.push(context);
      try {
        return await step(context);
      } catch (error) {
        failures.push(error);
        throw error;
      }
    })
  );

  const resourceLoader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: dir,
    noSkills: true,
    extensionFactories: [
      (pi) => pi.registerProvider(faux.provider),
      ...(options.codemodeOnly ? [createCodemodeExtension({ models: false })] : []),
      episodes
    ]
  });
  await resourceLoader.reload();

  const { session: agentSession } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model: faux.getModel(),
    thinkingLevel: 'off',
    modelRuntime: await ModelRuntime.create({ modelsPath: null, authPath: join(dir, 'auth.json') }),
    resourceLoader,
    tools: [
      'bash',
      'read',
      'fold',
      'unfold',
      'recall',
      'ask',
      'fork',
      ...(options.codemodeOnly ? ['codemode'] : [])
    ],
    sessionManager: options.persist
      ? SessionManager.create(dir, join(dir, 'sessions'))
      : SessionManager.inMemory(dir),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      ...(options.codemodeOnly ? { codemode: { mode: 'only' } } : {})
    })
  });
  await agentSession.bindExtensions({});

  return {
    dir,
    session: agentSession,
    requests,
    failures,
    async cleanup() {
      agentSession.dispose();
      await rm(dir, { recursive: true, force: true });
    }
  };
}

test('normalizeFoldSpans and planFoldBatch validate inputs and reject overlapping spans including parallel-tool expansion', () => {
  assert.throws(
    () => normalizeFoldSpans({ spans: [] }),
    (err: unknown) => err instanceof FoldError && err.reason === 'bad-spans'
  );
  assert.throws(
    () =>
      normalizeFoldSpans({
        since: 't1',
        note: 'single',
        spans: [{ since: 't1', note: 'batch' }]
      }),
    (err: unknown) => err instanceof FoldError && err.reason === 'bad-spans'
  );
  assert.throws(
    () =>
      normalizeFoldSpans({
        spans: [
          { since: 't1', until: 't2', note: 'valid note' },
          { since: 'nope', note: 'bad stamp' }
        ]
      }),
    (err: unknown) => err instanceof FoldError && err.reason === 'bad-stamp'
  );
  assert.throws(
    () =>
      normalizeFoldSpans({
        spans: [
          { since: 't1', until: 't2', note: 'valid note' },
          { since: 't3', note: '   ' }
        ]
      }),
    (err: unknown) => err instanceof FoldError && err.reason === 'empty-note'
  );

  // Parallel tool calls in a single assistant response get distinct stamps t2 and t3,
  // so [since: t2, until: t2] and [since: t3] expand to the SAME assistant response and overlap!
  const { manager } = session([
    user('go', 1),
    assistant(call('p1', 'read', { path: 'a.ts' }), call('p2', 'read', { path: 'b.ts' })),
    result('p1', 'read', 'aaa', 2),
    result('p2', 'read', 'bbb', 3),
    assistant(call('c3', 'bash', { command: 'true' })),
    result('c3', 'bash', 'ok', 4),
    assistant(call('f1', 'fold', {}))
  ]);
  const entries = manager.buildSessionProjection().entries;

  assert.throws(
    () =>
      planFoldBatch(entries, 'f1', [
        { since: 2, until: 2, note: 'first parallel result' },
        { since: 3, note: 'second parallel result in same turn' }
      ]),
    (err: unknown) => err instanceof FoldError && err.reason === 'overlapping-spans'
  );
});

test('planFoldBatch sorts disjoint spans chronologically, retains interior gaps, and keeps every edit prefix valid', () => {
  const longUser = `important constraint ${'x'.repeat(USER_TEXT_CAP + 400)}`;
  const { manager, ids } = session([
    user('initial prompt', 1),
    signedAssistant('s1', call('c1', 'bash', { command: 'step1' })),
    result('c1', 'bash', 'span1-out', 2),
    user(longUser, 3),
    signedAssistant('s2', call('c2', 'bash', { command: 'step2' })),
    result('c2', 'bash', 'span1-end', 4),
    // Interior gap [t5..t6] stays active in context:
    signedAssistant('gap', call('cg', 'read', { path: 'gap.ts' })),
    result('cg', 'read', 'active-gap-code', 5),
    user('keep going', 6),
    // Second span [t7..t8]:
    signedAssistant('s3', call('c3', 'bash', { command: 'step3' })),
    result('c3', 'bash', 'span2-out', 7),
    signedAssistant('s4', call('c4', 'bash', { command: 'step4' })),
    result('c4', 'bash', 'span2-end', 8),
    signedAssistant('fold_caller', call('f1', 'fold', {}))
  ]);

  // Pass spans in reverse order to verify chronological sorting.
  const batch = planFoldBatch(manager.buildSessionProjection().entries, 'f1', [
    { since: 7, note: 'Span 2 summary\nOpen: test edge case' },
    { since: 2, until: 4, note: 'Span 1 summary\nOpen: none' }
  ]);

  assert.match(batch.handle, /^ep-[0-9a-f]{8}$/);
  assert.equal(batch.spans.length, 2);
  assert.equal(batch.spans[0]!.since, 2);
  assert.equal(batch.spans[0]!.until, 4);
  assert.equal(batch.spans[1]!.since, 7);
  assert.equal(batch.spans[1]!.until, undefined);
  assert.equal(batch.keptFrom, 't5');
  assert.deepEqual(batch.keptUserTurns, ['t3']);
  assert.equal(batch.open, 'test edge case');

  // Every prefix of batch.edits never orphans a toolResult AND never leaves a live signature
  // after an already-mutated entry.
  for (let k = 0; k <= batch.edits.length; k++) {
    const { manager: copy, ids: copyIds } = session([
      user('initial prompt', 1),
      signedAssistant('s1', call('c1', 'bash', { command: 'step1' })),
      result('c1', 'bash', 'span1-out', 2),
      user(longUser, 3),
      signedAssistant('s2', call('c2', 'bash', { command: 'step2' })),
      result('c2', 'bash', 'span1-end', 4),
      signedAssistant('gap', call('cg', 'read', { path: 'gap.ts' })),
      result('cg', 'read', 'active-gap-code', 5),
      user('keep going', 6),
      signedAssistant('s3', call('c3', 'bash', { command: 'step3' })),
      result('c3', 'bash', 'span2-out', 7),
      signedAssistant('s4', call('c4', 'bash', { command: 'step4' })),
      result('c4', 'bash', 'span2-end', 8),
      signedAssistant('fold_caller', call('f1', 'fold', {}))
    ]);
    const idMap = new Map(ids.map((orig, idx) => [orig, copyIds[idx]!]));
    let earliestMutatedNonAssistant = Infinity;
    for (const edit of batch.edits.slice(0, k)) {
      const idx = ids.indexOf(edit.targetId);
      if (edit.replacement === null || idx === 3) {
        earliestMutatedNonAssistant = Math.min(earliestMutatedNonAssistant, idx);
      }
      copy.appendContextEdit(idMap.get(edit.targetId)!, edit.replacement);
    }
    copy.appendMessage(result('f1', 'fold', 'batch note', 9));
    const proj = copy.buildSessionProjection();
    assertResultsHaveCallers(proj.messages, `prefix ${k}`);
    if (earliestMutatedNonAssistant < Infinity) {
      for (const entry of proj.entries) {
        const origIdx = copyIds.indexOf(entry.sourceEntry.id);
        if (origIdx > earliestMutatedNonAssistant && entry.messages[0]) {
          assert.equal(
            hasLiveSignature(entry.messages[0]),
            false,
            `prefix ${k}: surviving assistant at index ${origIdx} still had live signature`
          );
        }
      }
    }
  }
});

test('live session batch fold retires disjoint spans, excludes interior gaps from recall/files, tracks stats once, and unfolds cleanly', async () => {
  let batchHandle = '';
  const bigGapRead = 'export const GAP = 42;\n'.repeat(40);
  const harness = await createLiveSession(
    [
      // Span 1 (t2..t3): touches span1.txt
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'echo alpha123 > span1.txt && cat span1.txt' })
        ),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'span1.txt' })),
      // Retained interior gap (t4..t5): touches gap-only.txt and reads gap-shared.txt
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'echo gapsecret999 > gap-only.txt && cat gap-only.txt' })
        ),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'gap-shared.txt' })),
      // Span 2 (t6..t7): touches span2.txt AND reads gap-shared.txt (must NOT prune t5's read!)
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'echo omega789 > span2.txt && cat span2.txt' })
        ),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'gap-shared.txt' })),
      // Trailing retained turn (t8): does NOT touch span files
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo trailing-active' })),
      // Batch fold: retire [t2..t3] and [t6..t7], keeping [t4..t5] and [t8] active!
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            spans: [
              {
                since: 't2',
                until: 't3',
                note: 'Created span1.txt with alpha.\nOpen: check alpha'
              },
              { since: 't6', until: 't7', note: 'Created span2.txt with omega.\nOpen: none' }
            ]
          }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const joined = context.messages
          .filter((m) => m.role !== 'system')
          .map(text)
          .join('\n---\n');
        assert.doesNotMatch(joined, /alpha123/);
        assert.doesNotMatch(joined, /omega789/);
        // Crucially, t5's read of gap-shared.txt was NOT pruned because t7's witness was folded away!
        assert.match(joined, /gapsecret999/);
        assert.match(joined, /GAP = 42/);
        assert.doesNotMatch(joined, /\[duplicate read of gap-shared\.txt/);
        assert.match(joined, /trailing-active/);
        assert.match(joined, /\[fold (ep-[0-9a-f]{8}): .*t2-t3, t6-t7/);
        assert.match(joined, /The turns t4-t5 and from t8 on were not folded and stay above\./);
        // Disk changes attribute span1.txt and span2.txt, NOT gap-only.txt!
        assert.match(joined, /A span1\.txt/);
        assert.match(joined, /A span2\.txt/);
        assert.doesNotMatch(joined, /gap-only\.txt/);
        batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'alpha123' }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const recallAlpha = text(context.messages.at(-1)!);
        assert.match(recallAlpha, new RegExp(batchHandle));
        assert.match(recallAlpha, /alpha123/);
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'omega789' }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const recallOmega = text(context.messages.at(-1)!);
        assert.match(recallOmega, new RegExp(batchHandle));
        assert.match(recallOmega, /omega789/);
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'gapsecret999' }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const recallGap = text(context.messages.at(-1)!);
        assert.match(recallGap, /No match for "gapsecret999"/);
        return fauxAssistantMessage(fauxToolCall('unfold', { handle: batchHandle }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const joined = context.messages
          .filter((m) => m.role !== 'system')
          .map(text)
          .join('\n---\n');
        assert.match(joined, /alpha123/);
        assert.match(joined, /gapsecret999/);
        assert.match(joined, /omega789/);
        assert.match(joined, new RegExp(`\\[unfolded ${batchHandle}\\]`));
        return fauxAssistantMessage('All verified.');
      }
    ],
    { git: true }
  );

  try {
    await writeFile(join(harness.dir, 'gap-shared.txt'), bigGapRead);
    execFileSync('git', ['add', '.'], { cwd: harness.dir });
    execFileSync('git', ['commit', '-m', 'init'], { cwd: harness.dir });

    await harness.session.prompt('run workflow');
    if (harness.failures[0]) throw harness.failures[0];

    const branch = harness.session.sessionManager.getBranch();
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.folds.count, 1);
    assert.equal(stats.folds.each.length, 1);
    assert.notEqual(stats.folds.each[0]?.uncachedAfter, undefined);
    // 1 after batch fold + 1 after unfold:
    assert.equal(stats.requests.filter((r) => r.after === 'fold').length, 2);
  } finally {
    await harness.cleanup();
  }
});

test('interior gap duplicate read IS pruned when witness survives in trailing gap, counted in stats, and restored by unfold', async () => {
  let batchHandle = '';
  const sharedBody = 'export function sharedHelper() { return 1; }\n'.repeat(30);
  const harness = await createLiveSession([
    // Span 1 (t2)
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo span1work' })),
    // Interior gap (t3): reads shared.ts
    () => fauxAssistantMessage(fauxToolCall('read', { path: 'shared.ts' })),
    // Span 2 (t4)
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo span2work' })),
    // Trailing surviving turn (t5): reads shared.ts identically -> witnesses pruning of t3!
    () => fauxAssistantMessage(fauxToolCall('read', { path: 'shared.ts' })),
    // Batch fold [t2..t2] and [t4..t4], keeping t3 and t5!
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Span 1 done.' },
            { since: 't4', until: 't4', note: 'Span 2 done.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const joined = context.messages
        .filter((m) => m.role !== 'system')
        .map(text)
        .join('\n---\n');
      assert.match(joined, /\[duplicate read of shared\.ts, \d+ chars; identical to t5\b/);
      assert.match(joined, /\[turn t2\]/);
      assert.match(joined, /\[turn t4\]/);
      batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
      return fauxAssistantMessage('Done.');
    }
  ]);

  try {
    await writeFile(join(harness.dir, 'shared.ts'), sharedBody);
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];

    const branch = harness.session.sessionManager.getBranch();
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.folds.count, 1);
    assert.ok(stats.folds.saved > 100);
    const unfoldPlan = planUnfold(
      branch,
      harness.session.sessionManager.buildSessionProjection().entries,
      batchHandle
    );
    for (const edit of unfoldPlan.edits) {
      harness.session.sessionManager.appendContextEdit(edit.targetId, edit.replacement);
    }
    const restoredText = harness.session.sessionManager
      .buildSessionProjection()
      .messages.map(text)
      .join('\n');
    assert.doesNotMatch(restoredText, /\[duplicate read of shared\.ts/);
  } finally {
    await harness.cleanup();
  }
});

test('non-overlapping interior-gap fold does not block unfolding outer batch fold; covering fold keeps notes and archives gap', async () => {
  let outerHandle = '';
  let gapHandle = '';
  const harness = await createLiveSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo outer-one' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo middle-gap-item' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo outer-two' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Outer part 1.' },
            { since: 't4', until: 't4', note: 'Outer part 2.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const joined = context.messages.map(text).join('\n');
      outerHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: 't3', until: 't3', note: 'Middle gap folded separately.' }),
        { stopReason: 'toolUse' }
      );
    },
    (context) => {
      const joined = context.messages.map(text).join('\n');
      const matches = [...joined.matchAll(/\[fold (ep-[0-9a-f]{8}):/g)].map((m) => m[1]!);
      gapHandle = matches.find((h) => h !== outerHandle)!;
      assert.ok(gapHandle);
      return fauxAssistantMessage(fauxToolCall('unfold', { handle: outerHandle }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      const joined = context.messages
        .filter((m) => m.role !== 'system')
        .map(text)
        .join('\n');
      assert.match(joined, /outer-one/);
      assert.match(joined, /outer-two/);
      assert.doesNotMatch(joined, /middle-gap-item/);
      assert.match(joined, new RegExp(`\\[fold ${gapHandle}:`));
      return fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Re-folded part 1.' },
            { since: 't4', until: 't4', note: 'Re-folded part 2.' }
          ]
        }),
        { stopReason: 'toolUse' }
      );
    },
    () =>
      fauxAssistantMessage(fauxToolCall('unfold', { handle: gapHandle }), {
        stopReason: 'toolUse'
      }),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', { since: 't3', note: 'Covering fold over gap and batch note.' }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const joined = context.messages
        .filter((m) => m.role !== 'system')
        .map(text)
        .join('\n');
      assert.match(joined, /Re-folded part 1\./);
      assert.match(joined, /Re-folded part 2\./);
      return fauxAssistantMessage(fauxToolCall('recall', { query: '"middle-gap-item"' }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      const recallOut = text(context.messages.at(-1)!);
      assert.match(recallOut, /middle-gap-item/);
      return fauxAssistantMessage('Done.');
    }
  ]);

  try {
    await harness.session.prompt('start');
    if (harness.failures[0]) throw harness.failures[0];
  } finally {
    await harness.cleanup();
  }
});

test('every interrupted journal prefix of a multi-span fold recovers cleanly via completeCutShortFolds', () => {
  const sharedRead = 'const x = 1;\n'.repeat(30);
  const baseMessages: Message[] = [
    user('task', 1),
    signedAssistant('a1', call('c1', 'bash', { command: 'one' })),
    result('c1', 'bash', 'out1', 2),
    signedAssistant('ag', call('cg', 'read', { path: 'dup.ts' })),
    result('cg', 'read', sharedRead, 3),
    signedAssistant('a2', call('c2', 'bash', { command: 'two' })),
    result('c2', 'bash', 'out2', 4),
    signedAssistant('at', call('ct', 'read', { path: 'dup.ts' })),
    result('ct', 'read', sharedRead, 5),
    signedAssistant('afold', call('f1', 'fold', {}))
  ];

  function buildFullTurnEnd() {
    const { manager, ids } = session(baseMessages);
    const batch = planFoldBatch(manager.buildSessionProjection().entries, 'f1', [
      { since: 2, until: 2, note: 'First span.' },
      { since: 4, until: 4, note: 'Second span.' }
    ]);
    const noteDetails: FoldDetails = {
      kind: 'fold',
      handle: batch.handle,
      since: 't2',
      until: 't4',
      spans: batch.spans.map((s) => ({
        since: `t${s.since}`,
        ...(s.until !== undefined ? { until: `t${s.until}` } : {}),
        firstEntryId: s.firstEntryId,
        lastEntryId: s.lastEntryId,
        note: s.note
      })),
      firstEntryId: batch.firstEntryId,
      lastEntryId: batch.lastEntryId,
      messages: batch.omitted.length,
      tokens: batch.tokens,
      headline: 'First span. · Second span.',
      keptUserTurns: [],
      ...(batch.keptFrom ? { keptFrom: batch.keptFrom } : {}),
      note: batch.note,
      files: []
    };
    manager.appendMessage(
      Object.assign(result('f1', 'fold', '[fold note]', 6), {
        details: noteDetails as never
      })
    );
    const branch = manager.getBranch();
    const removalDrafts = batch.edits.filter((e) => e.replacement === null);
    const tail = planFoldTailPruning({
      branch,
      repairDrafts: removalDrafts,
      editOwners: attributeEpisodesEdits(branch),
      afterEntryId: batch.spans[0]!.lastEntryId,
      beforeEntryId: ids[9]!,
      cwd: '/repo'
    });
    const fullEdits = [...batch.edits, ...tail.readEdits];
    const journal: Array<
      | { type: 'custom'; customType: string; data: unknown }
      | {
          type: 'context_edit';
          targetId: string;
          replacement: (typeof fullEdits)[number]['replacement'];
        }
    > = [
      {
        type: 'custom',
        customType: FOLD_STARTED_TYPE,
        data: { handle: batch.handle, edits: fullEdits, readDeps: tail.readDeps }
      },
      ...fullEdits,
      { type: 'custom', customType: FOLD_APPLIED_TYPE, data: { handle: batch.handle } }
    ];
    return { manager, ids, batch, journal };
  }

  const ref = buildFullTurnEnd();
  for (const item of ref.journal) {
    if (item.type === 'custom') ref.manager.appendCustomEntry(item.customType, item.data);
    else ref.manager.appendContextEdit(item.targetId, item.replacement);
  }
  const expectedText = ref.manager.buildSessionProjection().messages.map(text);

  for (let cut = 1; cut < ref.journal.length; cut++) {
    const partial = buildFullTurnEnd();
    for (const item of partial.journal.slice(0, cut)) {
      if (item.type === 'custom') partial.manager.appendCustomEntry(item.customType, item.data);
      else partial.manager.appendContextEdit(item.targetId, item.replacement);
    }
    const midProj = partial.manager.buildSessionProjection();
    assertResultsHaveCallers(midProj.messages, `cut ${cut}`);

    const repairs = completeCutShortFolds(partial.manager.getBranch());
    assert.ok(repairs.length > 0, `cut ${cut} should emit recovery entries`);
    for (const item of repairs) {
      if (item.type === 'custom') partial.manager.appendCustomEntry(item.customType, item.data);
      else partial.manager.appendContextEdit(item.targetId, item.replacement);
    }
    assert.deepEqual(partial.manager.buildSessionProjection().messages.map(text), expectedText);
    assert.ok(appliedFolds(partial.manager.getBranch()).has(partial.batch.handle));
  }
});

test('child fork batch fold excludes retained child gaps from recall; covered media allocates sequential indices; abort leaves zero state', async () => {
  const imgBlock: ImageContent = { type: 'image', mimeType: 'image/png', data: TINY_PNG };
  let batchHandle = '';
  const harness = await createLiveSession(
    [
      () =>
        fauxAssistantMessage(
          fauxToolCall('fork', { task: 'Investigate two modules and fold both spans.' }),
          { stopReason: 'toolUse' }
        ),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo childspanalpha111' })),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo childgapkeep999' })),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo childspanomega222' })),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            spans: [
              { since: 't2', until: 't2', note: 'Child span 1 archived.' },
              { since: 't4', until: 't4', note: 'Child span 2 archived.' }
            ]
          }),
          { stopReason: 'toolUse' }
        ),
      (childCtx) => {
        const joined = childCtx.messages
          .filter((m) => m.role !== 'system')
          .map(text)
          .join('\n');
        assert.doesNotMatch(joined, /childspanalpha111/);
        assert.doesNotMatch(joined, /childspanomega222/);
        assert.match(joined, /childgapkeep999/);
        batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
        return fauxAssistantMessage('Fork task completed with clean batch fold.');
      },
      () =>
        fauxAssistantMessage(
          fauxToolCall('recall', { query: 'childspanomega222', handle: batchHandle }),
          { stopReason: 'toolUse' }
        ),
      (parentCtx) => {
        assert.match(text(parentCtx.messages.at(-1)!), /childspanomega222/);
        return fauxAssistantMessage(
          fauxToolCall('recall', { query: 'childgapkeep999', handle: batchHandle }),
          { stopReason: 'toolUse' }
        );
      },
      (parentCtx) => {
        assert.match(text(parentCtx.messages.at(-1)!), /No match for "childgapkeep999"/);
        return fauxAssistantMessage('Parent verified fork batch fold.');
      }
    ],
    { persist: true }
  );

  try {
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];

    const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-fork-media-'));
    try {
      const state = new EpisodeState();
      state.cwd = dir;
      state.nextStamp = 10;
      const msgs: AgentMessage[] = [user('task', 1)];
      const forks = new ForkFolds({
        forkHandle: 'fk-00000001',
        state,
        parentEntries: [],
        prefixLength: 0,
        siblings: 0,
        archiveDir: dir
      });
      const foldTool = forks.tool(
        { name: 'fold', description: '', parameters: {} as never },
        () => msgs
      );

      msgs.push(
        assistant(call('c1', 'bash', { command: 'one' })),
        result('c1', 'bash', 'o1', 2),
        assistant(call('f_old1', 'fold', { since: 't2', note: 'old1' }))
      );
      const r1 = await foldTool.execute('f_old1', { since: 't2', note: 'old1' });
      msgs.push(
        Object.assign(result('f_old1', 'fold', '', 3), {
          content: withStamp([...r1.content, imgBlock], 3),
          details: r1.details as never
        })
      );
      msgs.push(
        assistant(call('cg', 'bash', { command: 'gap' })),
        result('cg', 'bash', 'gap-out', 4),
        assistant(call('c2', 'bash', { command: 'two' })),
        result('c2', 'bash', 'o2', 5),
        assistant(call('f_old2', 'fold', { since: 't5', note: 'old2' }))
      );
      const r2 = await foldTool.execute('f_old2', { since: 't5', note: 'old2' });
      msgs.push(
        Object.assign(result('f_old2', 'fold', '', 6), {
          content: withStamp([...r2.content, imgBlock], 6),
          details: r2.details as never
        })
      );
      msgs.push(assistant(call('f_batch', 'fold', {})));

      const aborted = new AbortController();
      aborted.abort();
      const countBefore = forks.folds.length;
      await assert.rejects(
        () =>
          foldTool.execute(
            'f_batch',
            {
              spans: [
                { since: 't3', until: 't3', note: 'Cover old1.' },
                { since: 't6', until: 't6', note: 'Cover old2.' }
              ]
            },
            aborted.signal
          ),
        (err: unknown) => err instanceof Error && err.message === FOLD_ABORTED_MESSAGE
      );
      assert.equal(forks.folds.length, countBefore);

      const batchRes = await foldTool.execute('f_batch', {
        spans: [
          { since: 't3', until: 't3', note: 'Cover old1.' },
          { since: 't6', until: 't6', note: 'Cover old2.' }
        ]
      });
      const batchDetails = batchRes.details as FoldDetails;
      assert.deepEqual(
        batchDetails.archivedImages?.map((img) => img.index),
        [1, 2]
      );
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  } finally {
    await harness.cleanup();
  }
});

test('codemode script batch fold retires disjoint spans, keeps interior gap, and unfolds cleanly', async () => {
  let handle = '';
  const harness = await createLiveSession(
    [
      () =>
        fauxAssistantMessage(
          fauxToolCall('codemode', {
            code: 'text(await tools.bash({ command: "echo CM_SPAN_ONE" }));'
          }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(
          fauxToolCall('codemode', {
            code: 'text(await tools.bash({ command: "echo CM_GAP_ALIVE" }));'
          }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(
          fauxToolCall('codemode', {
            code: 'text(await tools.bash({ command: "echo CM_SPAN_TWO" }));'
          }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(
          fauxToolCall('codemode', {
            code: `text(await tools.fold({
              spans: [
                { since: "t2", until: "t2", note: "Codemode span 1 note." },
                { since: "t4", until: "t4", note: "Codemode span 2 note." }
              ]
            }));`
          }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const joined = context.messages
          .filter((m) => m.role !== 'system')
          .map(text)
          .join('\n');
        assert.doesNotMatch(joined, /CM_SPAN_ONE/);
        assert.doesNotMatch(joined, /CM_SPAN_TWO/);
        assert.match(joined, /CM_GAP_ALIVE/);
        assert.match(joined, /Codemode span 1 note\./);
        assert.match(joined, /Codemode span 2 note\./);
        handle = /ep-[0-9a-f]{8}/.exec(joined)![0];
        return fauxAssistantMessage(
          fauxToolCall('codemode', {
            code: `text(await tools.unfold({ handle: "${handle}" }));`
          }),
          { stopReason: 'toolUse' }
        );
      },
      (context) => {
        const joined = context.messages
          .filter((m) => m.role !== 'system')
          .map(text)
          .join('\n');
        assert.match(joined, /CM_SPAN_ONE/);
        assert.match(joined, /CM_GAP_ALIVE/);
        assert.match(joined, /CM_SPAN_TWO/);
        assert.match(joined, new RegExp(`\\[unfolded ${handle}\\]`));
        return fauxAssistantMessage('Codemode verified.');
      }
    ],
    { codemodeOnly: true }
  );

  try {
    await harness.session.prompt('run codemode batch');
    if (harness.failures[0]) throw harness.failures[0];
  } finally {
    await harness.cleanup();
  }
});

test('adjacent spans omit unfolded notice, ask queries batch thread, and TUI renderers format batch spans', async () => {
  let batchHandle = '';
  const harness = await createLiveSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ADJ_ONE_42' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ADJ_TWO_99' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Closed adjacent episode one.' },
            { since: 't3', note: 'Closed adjacent episode two.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const joined = context.messages
        .filter((m) => m.role !== 'system')
        .map(text)
        .join('\n');
      assert.doesNotMatch(joined, /were not folded and stay above/);
      batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
      return fauxAssistantMessage(
        fauxToolCall('ask', {
          handle: batchHandle,
          question: 'What were the two ADJ tokens?'
        }),
        { stopReason: 'toolUse' }
      );
    },
    (askCtx) => {
      const askJoined = askCtx.messages.map(text).join('\n');
      assert.match(askJoined, /ADJ_ONE_42/);
      assert.match(askJoined, /ADJ_TWO_99/);
      return fauxAssistantMessage('The tokens were ADJ_ONE_42 and ADJ_TWO_99.');
    },
    (parentCtx) => {
      assert.match(text(parentCtx.messages.at(-1)!), /ADJ_ONE_42 and ADJ_TWO_99/);
      return fauxAssistantMessage('Verified adjacent spans and ask.');
    }
  ]);

  try {
    await harness.session.prompt('run adjacent');
    if (harness.failures[0]) throw harness.failures[0];

    const branch = harness.session.sessionManager.getBranch();
    const record = branch
      .flatMap((e) =>
        e.type === 'message' && e.message.role === 'toolResult' && e.message.toolName === 'fold'
          ? [e.message.details as unknown as FoldDetails]
          : []
      )
      .at(-1)!;
    assert.equal(record.keptFrom, undefined);
    assert.equal(record.spans?.length, 2);
    const plainTheme = {
      fg: (_c: string, s: string) => s,
      bold: (s: string) => s
    } as never;
    const callRendered = foldRenderers
      .renderCall(
        {
          spans: [
            { since: 't2', until: 't4', note: 'n1' },
            { since: 't7', note: 'n2' }
          ]
        },
        plainTheme
      )
      .render(120)
      .join('\n');
    assert.match(callRendered, /fold t2-t4, from t7/);
    const resRendered = foldRenderers
      .renderResult(
        { content: [{ type: 'text', text: '[fold]\nnote' }], details: record },
        { expanded: false },
        plainTheme
      )
      .render(120)
      .join('\n');
    assert.match(resRendered, /t2, from t3 on/);
    const unfoldRendered = unfoldRenderers
      .renderResult(
        {
          content: [{ type: 'text', text: 'Unfolded' }],
          details: {
            kind: 'unfold',
            handle: record.handle,
            since: 't2',
            spans: [{ since: 't2', until: 't4' }, { since: 't7' }],
            messages: 4,
            tokens: 200
          }
        },
        { expanded: false },
        plainTheme
      )
      .render(120)
      .join('\n');
    assert.match(unfoldRendered, /t2-t4, from t7 on 4 messages/);
  } finally {
    await harness.cleanup();
  }
});

test('staleness detects interior-gap edits to files touched only in an earlier batch span', async () => {
  let batchHandle = '';
  const harness = await createLiveSession(
    [
      // t2 (Span 0): read and write early.txt
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', {
            command: 'echo v1 > early.txt && cat early.txt && echo EARLY_MARK_77'
          })
        ),
      // t3 (Interior retained gap): modify early.txt AFTER Span 0 ended!
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo v2 > early.txt' })),
      // t4 (Span 1): touch only later.txt (early.txt already has v2 at t4!)
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'echo later > later.txt && echo LATER_MARK_88' })
        ),
      // Fold Span 0 (t2) and Span 1 (t4), keeping interior gap t3
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            spans: [
              { since: 't2', until: 't2', note: 'Created early.txt v1.' },
              { since: 't4', note: 'Created later.txt.' }
            ]
          }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const joined = context.messages.map(text).join('\n');
        batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
        return fauxAssistantMessage(fauxToolCall('recall', { query: '"EARLY_MARK_77"' }), {
          stopReason: 'toolUse'
        });
      },
      (afterRecall) => {
        const recallOut = text(afterRecall.messages.at(-1)!);
        // early.txt was modified in t3 after Span 0 ended, so recall must warn it is stale!
        assert.match(recallOut, /predates changes to: .*early\.txt/);
        assert.doesNotMatch(recallOut, /later\.txt/);
        return fauxAssistantMessage('Verified per-span staleness.');
      }
    ],
    { git: true }
  );

  try {
    execFileSync('git', ['commit', '--allow-empty', '-m', 'init'], { cwd: harness.dir });
    await harness.session.prompt('check staleness');
    if (harness.failures[0]) throw harness.failures[0];
    assert.ok(batchHandle.startsWith('ep-'));
  } finally {
    await harness.cleanup();
  }
});

test('disjoint span boundaries do not bleed Span 2 opening assistant turn into Span 1 turn readback or search attribution', async () => {
  let batchHandle = '';
  const harness = await createLiveSession([
    // t2 (Span 0)
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo SPAN_ONE_OUT' })),
    // t3 (Interior gap)
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo GAP_OUT' })),
    // t4 (Span 1): unique token ONLY in Span 1 opening assistant toolCall command!
    () =>
      fauxAssistantMessage(
        fauxToolCall('bash', {
          command: 'echo SPAN_TWO_CMD_ONLY_TOKEN >/dev/null && echo SPAN_TWO_RES'
        })
      ),
    // Batch fold t2 and t4, keeping t3
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Ran span one.' },
            { since: 't4', note: 'Ran span two.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const joined = context.messages.map(text).join('\n');
      batchHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(joined)![1]!;
      return fauxAssistantMessage(
        [
          fauxToolCall('recall', { handle: batchHandle, turn: 't2' }),
          fauxToolCall('recall', { handle: batchHandle, turn: 't2-t4' }),
          fauxToolCall('recall', { query: '"SPAN_TWO_CMD_ONLY_TOKEN"' }),
          fauxToolCall('recall', { query: 'SPAN_TWO_CMD_ONLY_TOKEN' })
        ],
        { stopReason: 'toolUse' }
      );
    },
    (afterRecall) => {
      const recent = afterRecall.messages.slice(-4).map(text);
      const turnT2Out = recent[0]!;
      const rangeOut = recent[1]!;
      const grepOut = recent[2]!;
      const bm25Out = recent[3]!;
      // 1. Reading back turn t2 must NOT include Span 2's opening assistant message:
      assert.match(turnT2Out, /SPAN_ONE_OUT/);
      assert.doesNotMatch(turnT2Out, /SPAN_TWO_CMD_ONLY_TOKEN/);
      // 2. Cross-span range t2-t4 resets label at Span 2's unstamped assistant boundary:
      assert.match(rangeOut, /SPAN_ONE_OUT/);
      assert.match(rangeOut, /SPAN_TWO_RES/);
      assert.doesNotMatch(rangeOut, /GAP_OUT/);
      assert.match(rangeOut, new RegExp(`${batchHandle} - assistant`));
      assert.doesNotMatch(rangeOut, new RegExp(`${batchHandle} t2 assistant`));
      // 3. Both quoted grep and MiniSearch must NOT misattribute Span 2's opening assistant to t2:
      assert.doesNotMatch(grepOut, new RegExp(`${batchHandle} t2 assistant:`));
      assert.doesNotMatch(bm25Out, new RegExp(`${batchHandle} t2 assistant:`));
      return fauxAssistantMessage('Verified disjoint span boundaries.');
    }
  ]);

  try {
    await harness.session.prompt('check boundaries');
    if (harness.failures[0]) throw harness.failures[0];
  } finally {
    await harness.cleanup();
  }
});

test('planFoldBatch orders transitive covered notes oldest-first by written order across spans', async () => {
  const harness = await createLiveSession([
    // t2: work 1
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo w1' })),
    // t3: ep-1 folds t2 (oldest note, written first!)
    () =>
      fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'OLDEST_NOTE_EP1' }), {
        stopReason: 'toolUse'
      }),
    // t4: work 2
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo w2' })),
    // t5: ep-2 folds t4 (second note, written second!)
    () =>
      fauxAssistantMessage(fauxToolCall('fold', { since: 't4', note: 'SECOND_NOTE_EP2' }), {
        stopReason: 'toolUse'
      }),
    // t6: gap work to keep active
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo gap_keep' })),
    // t7: ep-3 folds bounded t3..t3 (covers ep-1! Written third, stands at t7 AFTER ep-2 at t5!)
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', { since: 't3', until: 't3', note: 'THIRD_NOTE_EP3' }),
        { stopReason: 'toolUse' }
      ),
    // Batch fold covering Span 0 = t5 (ep-2) and Span 1 = t7 (ep-3, which carries ep-1), keeping gap t6!
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't5', until: 't5', note: 'Batch span covering ep-2.' },
            { since: 't7', until: 't7', note: 'Batch span covering ep-3.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const noteText = context.messages
        .filter((m) => m.role === 'toolResult')
        .map(text)
        .at(-1)!;
      const idx1 = noteText.indexOf('OLDEST_NOTE_EP1');
      const idx2 = noteText.indexOf('SECOND_NOTE_EP2');

      const idx3 = noteText.indexOf('THIRD_NOTE_EP3');
      assert.ok(idx1 > 0 && idx2 > 0 && idx3 > 0);
      assert.ok(idx1 < idx2, `Expected ep-1 (${idx1}) before ep-2 (${idx2})`);
      assert.ok(idx2 < idx3, `Expected ep-2 (${idx2}) before ep-3 (${idx3})`);
      return fauxAssistantMessage('Verified oldest-first covered notes order.');
    }
  ]);

  try {
    await harness.session.prompt('check covers order');
    if (harness.failures[0]) throw harness.failures[0];
  } finally {
    await harness.cleanup();
  }
});
test('H1: planFoldBatch never resurrects earlier-omitted signed assistant entries in gaps or trailing tails', () => {
  const sm = SessionManager.inMemory();
  sm.appendMessage(user('task', 1));
  // t2 (Span 0)
  sm.appendMessage(signedAssistant('s1', call('c1', 'bash', { command: 'one' })));
  sm.appendMessage(result('c1', 'bash', 'out1', 2));
  // Earlier-omitted signed assistant + result right at end of Span 0 / in interior gap:
  const omittedGapAsstId = sm.appendMessage(
    signedAssistant('s_dead_gap', call('c_dead_gap', 'bash', { command: 'dead-gap' }))
  );
  const omittedGapResId = sm.appendMessage(result('c_dead_gap', 'bash', 'dead-gap-out', 3));
  // Active interior gap t4
  sm.appendMessage(
    signedAssistant('s_live_gap', call('c_live_gap', 'bash', { command: 'live-gap' }))
  );
  sm.appendMessage(result('c_live_gap', 'bash', 'live-gap-out', 4));
  // t5 (Span 1)
  sm.appendMessage(signedAssistant('s2', call('c2', 'bash', { command: 'two' })));
  sm.appendMessage(result('c2', 'bash', 'out2', 5));
  // Earlier-omitted signed assistant + result in trailing tail after Span 1:
  const omittedTailAsstId = sm.appendMessage(
    signedAssistant('s_dead_tail', call('c_dead_tail', 'bash', { command: 'dead-tail' }))
  );
  const omittedTailResId = sm.appendMessage(result('c_dead_tail', 'bash', 'dead-tail-out', 6));
  // Active trailing turn t7
  sm.appendMessage(
    signedAssistant('s_live_tail', call('c_live_tail', 'bash', { command: 'live-tail' }))
  );
  sm.appendMessage(result('c_live_tail', 'bash', 'live-tail-out', 7));

  // Hide both dead pairs as an earlier fold would have done:
  sm.appendContextEdit(omittedGapAsstId, null);
  sm.appendContextEdit(omittedGapResId, null);
  sm.appendContextEdit(omittedTailAsstId, null);
  sm.appendContextEdit(omittedTailResId, null);

  sm.appendMessage(signedAssistant('s_fold', call('f_batch', 'fold', {})));
  const preEntries = sm.buildSessionProjection().entries;
  const hiddenBefore = new Set(
    preEntries.filter((e) => e.messages.length === 0).map((e) => e.sourceEntry.id)
  );
  assert.ok(hiddenBefore.has(omittedGapAsstId));
  assert.ok(hiddenBefore.has(omittedTailAsstId));

  // Both single-span bounded and multi-span batch must never emit non-null replacements for hidden entries:
  for (const spans of [
    [{ since: 2, until: 2, note: 'Single bounded.' }],
    [
      { since: 2, until: 2, note: 'Batch span 1.' },
      { since: 5, until: 5, note: 'Batch span 2.' }
    ]
  ]) {
    const batch = planFoldBatch(preEntries, 'f_batch', spans, { latestStamp: 7 });
    for (const edit of batch.edits) {
      if (edit.replacement !== null) {
        assert.ok(
          !hiddenBefore.has(edit.targetId),
          `Resurrected hidden entry ${edit.targetId} with non-null replacement`
        );
      }
    }
  }
});

test('M1/M2/L4: context_edit entries before caller preserve runsToCall; empty top-level fields next to spans work; missing non-final until errors clearly', async () => {
  // 1. M2: empty/null top-level fields alongside spans
  const parsed = normalizeFoldSpans({
    since: '',
    until: '   ',
    note: null,
    spans: [
      { since: 't2', until: 't3', note: 'First.' },
      { since: 't5', note: 'Second.' }
    ]
  });
  assert.equal(parsed.length, 2);

  // 2. L4: missing until on non-final span gives clear bad-spans error, while out-of-order final omit works
  const sm = SessionManager.inMemory();
  sm.appendMessage(user('task', 1));
  sm.appendMessage(assistant(call('c1', 'bash', { command: 'one' })));
  const r1Id = sm.appendMessage(result('c1', 'bash', 'out1', 2));
  sm.appendMessage(assistant(call('c2', 'bash', { command: 'two' })));
  sm.appendMessage(result('c2', 'bash', 'out2', 3));
  sm.appendMessage(assistant(call('c3', 'bash', { command: 'three' })));
  sm.appendMessage(result('c3', 'bash', 'out3', 4));
  // Append context_edit and custom entries right BEFORE the fold caller (M1 reproduction!):
  sm.appendContextEdit(r1Id, { content: withStamp('edited-out1', 2) });
  sm.appendCustomEntry('some-extension-event', { ok: true });
  sm.appendMessage(assistant(call('f1', 'fold', {})));
  const entries = sm.buildSessionProjection().entries;

  assert.throws(
    () =>
      planFoldBatch(
        entries,
        'f1',
        [
          { since: 2, note: 'Missing until on earlier span.' },
          { since: 4, note: 'Final span.' }
        ],
        { latestStamp: 4 }
      ),
    (err: unknown) =>
      err instanceof FoldError && err.reason === 'bad-spans' && /until/i.test(err.message)
  );

  // Out-of-order where chronologically final span (t4) omits until still works AND has keptFrom === undefined (M1):
  const outOfOrder = planFoldBatch(
    entries,
    'f1',
    [
      { since: 4, note: 'Open final span.' },
      { since: 2, until: 2, note: 'Bounded early span.' }
    ],
    { latestStamp: 4 }
  );
  assert.equal(outOfOrder.spans[1]!.keptFrom, undefined);
  assert.match(outOfOrder.note, /\[turns from t4 on\]/);
});

test('L2/L3/GPT2/GPT5: mergeSpanDiskChanges returns undefined if any span is unknown, preserves A+A and T, and excludes A+D from readOnly', async () => {
  // Partial unknown diff must be undefined, never [] ("Working tree unchanged"):
  assert.equal(mergeSpanDiskChanges([undefined, []]), undefined);
  assert.equal(mergeSpanDiskChanges([[{ status: 'M', path: '/a' }], undefined]), undefined);
  // A + A across a deleted gap stays A; T stays T:
  assert.deepEqual(
    mergeSpanDiskChanges([
      [
        { status: 'A', path: '/recreated.ts' },
        { status: 'T', path: '/mode.ts' }
      ],
      [{ status: 'A', path: '/recreated.ts' }]
    ]),
    [
      { status: 'A', path: '/recreated.ts' },
      { status: 'T', path: '/mode.ts' }
    ]
  );

  // Live session where Span 0 creates + reads temp.txt, gap keeps state, Span 1 deletes temp.txt (A + D cancels in net diff):
  const harness = await createLiveSession(
    [
      // t2 (Span 0): create temp.txt via bash AND read it via read tool
      () =>
        fauxAssistantMessage([
          fauxToolCall('bash', { command: 'echo transient > temp.txt' }),
          fauxToolCall('read', { path: 'temp.txt' })
        ]),
      // t4 (Interior gap)
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo gap' })),
      // t5 (Span 1): delete temp.txt
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'rm temp.txt' })),
      // Batch fold Span 0 (t2..t3) and Span 1 (t5)
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            spans: [
              { since: 't2', until: 't3', note: 'Created and read temp.txt.' },
              { since: 't5', note: 'Removed temp.txt.' }
            ]
          }),
          { stopReason: 'toolUse' }
        ),
      () => fauxAssistantMessage('Done.')
    ],
    { git: true }
  );

  try {
    execFileSync('git', ['commit', '--allow-empty', '-m', 'init'], { cwd: harness.dir });
    await harness.session.prompt('run A+D readOnly check');
    if (harness.failures[0]) throw harness.failures[0];
    const branch = harness.session.sessionManager.getBranch();
    const record = branch
      .flatMap((e) =>
        e.type === 'message' && e.message.role === 'toolResult' && e.message.toolName === 'fold'
          ? [e.message.details as unknown as FoldDetails]
          : []
      )
      .at(-1)!;
    assert.ok(
      !(record.readOnly ?? []).some((p) => p.endsWith('temp.txt')),
      `Expected temp.txt not to be classified readOnly, got ${JSON.stringify(record.readOnly)}`
    );
  } finally {
    await harness.cleanup();
  }
});

test('GPT3/GPT4: same-file staleness warns when earlier span is recalled after gap edit; adjacent spans keep contiguous recall parity', async () => {
  let sameFileHandle = '';
  const harness1 = await createLiveSession(
    [
      // t2..t3 (Span 0): write v1 to shared.txt and read it
      () =>
        fauxAssistantMessage([
          fauxToolCall('bash', { command: 'echo SHARED_V1_BODY > shared.txt' }),
          fauxToolCall('read', { path: 'shared.txt' })
        ]),
      // t4 (Retained gap): overwrite shared.txt with v2
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo SHARED_V2_BODY > shared.txt' })),
      // t5 (Span 1): read shared.txt (now v2)
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'shared.txt' })),
      // Batch fold Span 0 (t2..t3) and Span 1 (t5), keeping gap t4
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            spans: [
              { since: 't2', until: 't3', note: 'Read shared.txt v1.' },
              { since: 't5', note: 'Read shared.txt v2.' }
            ]
          }),
          { stopReason: 'toolUse' }
        ),
      (ctx) => {
        sameFileHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(ctx.messages.map(text).join('\n'))![1]!;
        return fauxAssistantMessage(
          [
            fauxToolCall('recall', { query: '"SHARED_V1_BODY"' }),
            fauxToolCall('recall', { handle: sameFileHandle, turn: 't3' }),
            fauxToolCall('recall', { handle: sameFileHandle, turn: 't5' })
          ],
          { stopReason: 'toolUse' }
        );
      },
      (afterRecall) => {
        const outs = afterRecall.messages.slice(-3).map(text);
        // Querying v1 (Span 0) or turn t3 (Span 0) MUST warn shared.txt changed after Span 0:
        assert.match(outs[0]!, /predates changes to: .*shared\.txt/);
        assert.match(outs[1]!, /predates changes to: .*shared\.txt/);
        // Querying turn t5 (Span 1, which saw v2) must NOT falsely warn:
        assert.doesNotMatch(outs[2]!, /predates changes to/);
        return fauxAssistantMessage('Verified same-file span staleness.');
      }
    ],
    { git: true }
  );

  try {
    execFileSync('git', ['commit', '--allow-empty', '-m', 'init'], { cwd: harness1.dir });
    await harness1.session.prompt('same-file staleness');
    if (harness1.failures[0]) throw harness1.failures[0];
  } finally {
    await harness1.cleanup();
  }

  // GPT4: adjacent spans [t2..t2] and [t3..on] with NO gap must keep contiguous turn readback & attribution parity!
  let adjHandle = '';
  const harness2 = await createLiveSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ADJ_A_OUT' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('bash', { command: 'echo ADJ_B_CALL_MARKER >/dev/null && echo ADJ_B_OUT' })
      ),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          spans: [
            { since: 't2', until: 't2', note: 'Span A.' },
            { since: 't3', note: 'Span B.' }
          ]
        }),
        { stopReason: 'toolUse' }
      ),
    (ctx) => {
      adjHandle = /\[fold (ep-[0-9a-f]{8}):/.exec(ctx.messages.map(text).join('\n'))![1]!;
      return fauxAssistantMessage(
        [
          fauxToolCall('recall', { handle: adjHandle, turn: 't2' }),
          fauxToolCall('recall', { query: '"ADJ_B_CALL_MARKER"' }),
          fauxToolCall('recall', { query: 'ADJ_B_CALL_MARKER' })
        ],
        { stopReason: 'toolUse' }
      );
    },
    (afterRecall) => {
      const [turnOut, grepOut, bm25Out] = afterRecall.messages.slice(-3).map(text);
      // Because t2 and t3 were adjacent with no retained gap, assistant B follows t2 directly:
      assert.match(turnOut!, /ADJ_B_CALL_MARKER/);
      assert.match(grepOut!, new RegExp(`${adjHandle} t2 assistant:`));
      assert.match(bm25Out!, new RegExp(`${adjHandle} t2 assistant:`));
      return fauxAssistantMessage('Verified adjacent parity.');
    }
  ]);

  try {
    await harness2.session.prompt('adjacent parity');
    if (harness2.failures[0]) throw harness2.failures[0];
  } finally {
    await harness2.cleanup();
  }
});

test('L6 and interrupted batch unfold recovery: compaction error when early span pruned, and every interrupted unfold prefix recovers', () => {
  // 1. L6: compaction error when Span 0 firstEntryId was pruned before compaction
  const smComp = SessionManager.inMemory();
  smComp.appendMessage(user('task', 1));
  smComp.appendMessage(assistant(call('c1', 'bash', { command: 'one' })));
  const r1Id = smComp.appendMessage(result('c1', 'bash', 'out1', 2));
  const keepId = smComp.appendMessage(user('keep', 3));
  smComp.appendCompaction('Compacted summary', keepId, 500);
  const foldCallId = 'f_pre_comp';
  smComp.appendMessage(assistant(call(foldCallId, 'fold', {})));
  const noteMsg: AgentMessage = {
    role: 'toolResult',
    toolCallId: foldCallId,
    toolName: 'fold',
    content: withStamp('[fold ep-deadbeef]', 4),
    details: {
      kind: 'fold',
      handle: 'ep-deadbeef',
      since: 't2',
      firstEntryId: 'missing-pruned-id',
      lastEntryId: r1Id,
      spans: [
        {
          since: 't2',
          until: 't2',
          firstEntryId: 'missing-pruned-id',
          lastEntryId: r1Id,
          note: 'n1'
        },
        { since: 't3', until: 't3', firstEntryId: keepId, lastEntryId: keepId, note: 'n2' }
      ],
      messages: 2,
      tokens: 20,
      headline: 'h',
      keptUserTurns: [],
      files: []
    } as never,
    isError: false,
    timestamp: Date.now()
  };
  smComp.appendMessage(noteMsg);
  smComp.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-deadbeef' });
  smComp.appendMessage(assistant(call('u_comp', 'unfold', { handle: 'ep-deadbeef' })));
  assert.throws(
    () =>
      planUnfold(smComp.getBranch(), smComp.buildSessionProjection().entries, 'ep-deadbeef', {
        callerId: 'u_comp'
      }),
    /lies before the latest compaction/
  );

  // 2. Every interrupted prefix of a batch unfold recovers cleanly via completeCutShortFolds
  function buildUnfoldReady() {
    const sm = SessionManager.inMemory();
    sm.appendMessage(user('task', 1));
    sm.appendMessage(signedAssistant('s1', call('c1', 'bash', { command: 'one' })));
    sm.appendMessage(result('c1', 'bash', 'out1', 2));
    sm.appendMessage(signedAssistant('sg', call('cg', 'bash', { command: 'gap' })));
    sm.appendMessage(result('cg', 'bash', 'gap-out', 3));
    sm.appendMessage(signedAssistant('s2', call('c2', 'bash', { command: 'two' })));
    sm.appendMessage(result('c2', 'bash', 'out2', 4));
    sm.appendMessage(signedAssistant('sf', call('f_b', 'fold', {})));
    const batch = planFoldBatch(
      sm.buildSessionProjection().entries,
      'f_b',
      [
        { since: 2, until: 2, note: 'n1' },
        { since: 4, until: 4, note: 'n2' }
      ],
      { latestStamp: 4 }
    );
    sm.appendMessage({
      role: 'toolResult',
      toolCallId: 'f_b',
      toolName: 'fold',
      content: withStamp(`[fold ${batch.handle}]`, 5),
      details: {
        kind: 'fold',
        handle: batch.handle,
        since: 't2',
        until: 't4',
        firstEntryId: batch.firstEntryId,
        lastEntryId: batch.lastEntryId,
        spans: batch.spans.map((s) => ({
          since: `t${s.since}`,
          until: `t${s.until ?? s.since}`,
          firstEntryId: s.firstEntryId,
          lastEntryId: s.lastEntryId,
          note: s.note
        })),
        messages: batch.omitted.length,
        tokens: batch.tokens,
        headline: batch.headline,
        keptUserTurns: [],
        files: []
      } as never,
      isError: false,
      timestamp: Date.now()
    });
    sm.appendCustomEntry(FOLD_STARTED_TYPE, { handle: batch.handle, edits: batch.edits });
    for (const e of batch.edits) sm.appendContextEdit(e.targetId, e.replacement);
    sm.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: batch.handle });
    sm.appendMessage(signedAssistant('su', call('u_b', 'unfold', { handle: batch.handle })));
    const unfoldPlan = planUnfold(
      sm.getBranch(),
      sm.buildSessionProjection().entries,
      batch.handle,
      { callerId: 'u_b' }
    );
    return { sm, batch, unfoldPlan };
  }

  const probe = buildUnfoldReady();
  assert.ok(probe.unfoldPlan.edits.length >= 4);

  for (let k = 0; k <= probe.unfoldPlan.edits.length; k++) {
    const { sm: smCrash, batch, unfoldPlan: freshUnfold } = buildUnfoldReady();
    smCrash.appendMessage(result('u_b', 'unfold', `[unfolded ${batch.handle}]`, 6));
    smCrash.appendCustomEntry('episodes-fold-unfold-started', {
      handle: batch.handle,
      edits: freshUnfold.edits
    });
    for (let i = 0; i < k; i++) {
      const e = freshUnfold.edits[i]!;
      smCrash.appendContextEdit(e.targetId, e.replacement);
    }
    const midMsgs = smCrash.buildSessionProjection().messages;
    assertResultsHaveCallers(midMsgs, `unfold crash prefix k=${k}`);

    const repairs = completeCutShortFolds(smCrash.getBranch());
    for (const item of repairs) {
      if (item.type === 'context_edit') smCrash.appendContextEdit(item.targetId, item.replacement);
      else smCrash.appendCustomEntry(item.customType, item.data);
    }
    assert.ok(!appliedFolds(smCrash.getBranch()).has(batch.handle));
    const recoveredTxt = smCrash.buildSessionProjection().messages.map(text).join('\n');
    assert.match(recoveredTxt, /out1/);
    assert.match(recoveredTxt, /gap-out/);
    assert.match(recoveredTxt, /out2/);
  }
});
