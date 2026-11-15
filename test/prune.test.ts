import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
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
  type ToolCall
} from '@earendil-works/pi-ai';
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
  FOLD_APPLIED_TYPE,
  FOLD_STARTED_TYPE,
  FOLD_UNFOLD_STARTED_TYPE,
  FOLD_UNFOLDED_TYPE
} from '../src/archive.js';
import {
  attributeEpisodesEdits,
  completeCutShortFolds,
  ERROR_ARG_MIN_CHARS,
  ERROR_PRUNE_AFTER_TURNS,
  planFold,
  planFoldTailPruning,
  planUnfold
} from '../src/fold.js';
import episodes from '../src/index.js';
import { CAPPED_KEY, capOutput } from '../src/output.js';
import { readStamp, withStamp } from '../src/stamps.js';
import { episodeStats } from '../src/stats.js';
import { assistant, call, result, session, user } from './helpers.js';

function text(message: AgentMessage): string {
  if (!('content' in message)) return '';
  if (typeof message.content === 'string') return message.content;
  return message.content.flatMap((block) => (block.type === 'text' ? [block.text] : [])).join('\n');
}

function errorResult(
  toolCallId: string,
  toolName: string,
  errorText: string,
  stamp?: number
): Message {
  return {
    role: 'toolResult',
    toolCallId,
    toolName,
    content:
      stamp === undefined
        ? [{ type: 'text', text: errorText }]
        : withStamp([{ type: 'text', text: errorText }], stamp),
    isError: true,
    timestamp: 0
  };
}

function assertPaired(messages: readonly AgentMessage[]) {
  const llm = convertToLlm([...messages]);
  const open = new Set<string>();
  for (const message of llm) {
    if (message.role === 'assistant') {
      for (const block of message.content) {
        if (block.type === 'toolCall') open.add(block.id);
      }
    } else if (message.role === 'toolResult') {
      assert.ok(open.delete(message.toolCallId), `unpaired toolResult ${message.toolCallId}`);
    }
  }
  assert.equal(open.size, 0, `unclosed toolCalls: ${[...open].join(', ')}`);
}

test('pre-fold prefix before since is never modified by fold tail pruning', () => {
  const bigRead = 'line of code\n'.repeat(40);
  const bigWrite = 'broken payload\n'.repeat(60);
  const { manager, ids } = session([
    user('setup', 1),
    assistant(call('pre_r1', 'read', { path: 'src/a.ts' })),
    result('pre_r1', 'read', bigRead, 2),
    assistant(call('pre_err', 'write', { path: 'src/a.ts', content: bigWrite })),
    errorResult('pre_err', 'write', 'Error: EACCES', 3),
    assistant(call('pre_r2', 'read', { path: 'src/a.ts' })),
    result('pre_r2', 'read', bigRead, 4),
    user('episode', 5),
    assistant(call('s1', 'bash', { command: 'echo build' })),
    result('s1', 'bash', 'ok', 6),
    assistant(call('tail_r', 'read', { path: 'src/a.ts' })),
    result('tail_r', 'read', bigRead, 7),
    assistant(call('f1', 'fold', { since: 't5', until: 't6', note: 'Built.' }))
  ]);

  const branch = manager.getBranch();
  const tailPlan = planFoldTailPruning({
    branch,
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[9]!,
    beforeEntryId: ids[12]!,
    cwd: '/repo'
  });

  for (const prefixId of ids.slice(0, 7)) {
    assert.equal(tailPlan.assistantReplacements.has(prefixId), false);
    assert.equal(
      tailPlan.readEdits.some((edit) => edit.targetId === prefixId),
      false
    );
  }
  assert.equal(tailPlan.readEdits.length, 0);
  assert.equal(tailPlan.assistantReplacements.size, 0);
});

test('duplicate reads in surviving tail keep only newest identical read, preserve stamps, and skip mismatches', () => {
  const fileText = Array.from({ length: 50 }, (_, i) => `export const v${i} = ${i};`).join('\n');
  const changedText = `${fileText}\nexport const extra = true;`;
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('span_c', 'bash', { command: 'echo 1' })),
    result('span_c', 'bash', '1', 2),
    assistant(call('r1', 'read', { path: './src/same.ts' })),
    result('r1', 'read', fileText, 3),
    assistant(call('r2', 'read', { path: 'src/same.ts', offset: '1' })),
    result('r2', 'read', fileText, 4),
    assistant({
      type: 'toolCall',
      id: 'r3',
      name: 'read',
      arguments: { path: 'src/same.ts', offset: 1 }
    }),
    result('r3', 'read', fileText, 5),
    assistant(call('r4', 'read', { path: 'src/same.ts' })),
    result('r4', 'read', fileText, 6),
    assistant(call('c1', 'read', { path: 'src/changed.ts' })),
    result('c1', 'read', fileText, 7),
    assistant(call('c2', 'read', { path: 'src/changed.ts' })),
    result('c2', 'read', changedText, 8),
    assistant({
      type: 'toolCall',
      id: 'lim1',
      name: 'read',
      arguments: { path: 'src/slice.ts', offset: 1, limit: 100 }
    }),
    result('lim1', 'read', fileText, 9),
    assistant({
      type: 'toolCall',
      id: 'lim2',
      name: 'read',
      arguments: { path: 'src/slice.ts', offset: 1, limit: 200 }
    }),
    result('lim2', 'read', fileText, 10),
    assistant(call('tiny1', 'read', { path: 'src/tiny.ts' })),
    result('tiny1', 'read', 'x = 1', 11),
    assistant(call('tiny2', 'read', { path: 'src/tiny.ts' })),
    result('tiny2', 'read', 'x = 1', 12),
    assistant(call('img1', 'read', { path: 'src/pic.png' })),
    {
      role: 'toolResult',
      toolCallId: 'img1',
      toolName: 'read',
      content: withStamp(
        [
          { type: 'text', text: `Read image file [image/png]\n${'h'.repeat(200)}` },
          { type: 'image', data: 'AAAA', mimeType: 'image/png' }
        ],
        13
      ),
      isError: false,
      timestamp: 0
    },
    assistant(call('img2', 'read', { path: 'src/pic.png' })),
    {
      role: 'toolResult',
      toolCallId: 'img2',
      toolName: 'read',
      content: withStamp(
        [
          { type: 'text', text: `Read image file [image/png]\n${'h'.repeat(200)}` },
          { type: 'image', data: 'AAAA', mimeType: 'image/png' }
        ],
        14
      ),
      isError: false,
      timestamp: 0
    },
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
  ]);

  const branch = manager.getBranch();
  const tailPlan = planFoldTailPruning({
    branch,
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[2]!,
    beforeEntryId: ids.at(-1)!,
    cwd: '/repo'
  });

  assert.deepEqual(
    tailPlan.readEdits.map((edit) => edit.targetId),
    [ids[4], ids[8]]
  );
  for (const [idx, expectedStamp] of [
    [0, 3],
    [1, 5]
  ] as const) {
    const content = tailPlan.readEdits[idx]!.replacement?.content;
    assert.ok(Array.isArray(content));
    const msg: AgentMessage = {
      role: 'toolResult',
      toolCallId: 'x',
      toolName: 'read',
      content: content as TextContent[],
      isError: false,
      timestamp: 0
    };
    assert.equal(readStamp(msg), expectedStamp);
    assert.match(
      text(msg),
      /\[duplicate read of src\/same\.ts, \d+ chars; identical to t6 \(if t6 is later folded, check that fold note for its handle and call recall with turn "t6"\)\]/
    );
  }
});

test('capped read deduplication verifies full archived files and skips middle-line mismatches or missing files', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-prune-cap-'));
  try {
    const headLines = Array.from({ length: 40 }, (_, i) => `head line ${i}`).join('\n');
    const tailLines = Array.from({ length: 40 }, (_, i) => `tail line ${i}`).join('\n');
    const middleA = Array.from({ length: 120 }, (_, i) => `mid A line ${i}`).join('\n');
    const middleB = Array.from({ length: 120 }, (_, i) => `mid B line ${i}`).join('\n');
    const fullA = `${headLines}\n${middleA}\n${tailLines}`;
    const fullB = `${headLines}\n${middleB}\n${tailLines}`;

    const cutA1 = (await capOutput([{ type: 'text', text: fullA }], dir, 1000, 300))!;
    const cutA2 = (await capOutput([{ type: 'text', text: fullA }], dir, 1000, 300))!;
    const cutB = (await capOutput([{ type: 'text', text: fullB }], dir, 1000, 300))!;

    const cappedMsg = (
      id: string,
      cut: NonNullable<Awaited<ReturnType<typeof capOutput>>>,
      stamp: number
    ): Message => ({
      role: 'toolResult',
      toolCallId: id,
      toolName: 'read',
      content: withStamp(cut.content, stamp),
      details: { [CAPPED_KEY]: { ...cut.capped } },
      isError: false,
      timestamp: 0
    });

    const { manager, ids } = session([
      user('go', 1),
      assistant(call('s1', 'bash', { command: 'echo 1' })),
      result('s1', 'bash', '1', 2),
      assistant(call('ca1', 'read', { path: 'same.txt' })),
      cappedMsg('ca1', cutA1, 3),
      assistant(call('ca2', 'read', { path: 'same.txt' })),
      cappedMsg('ca2', cutA2, 4),
      assistant(call('cm1', 'read', { path: 'diff-mid.txt' })),
      cappedMsg('cm1', cutA1, 5),
      assistant(call('cm2', 'read', { path: 'diff-mid.txt' })),
      cappedMsg('cm2', cutB, 6),
      assistant(call('off1', 'read', { path: 'off.txt' })),
      {
        role: 'toolResult',
        toolCallId: 'off1',
        toolName: 'read',
        content: withStamp([{ type: 'text', text: 'x'.repeat(500) }], 7),
        details: { offload: { path: join(dir, 'nope.txt'), sha256: '00' } },
        isError: false,
        timestamp: 0
      },
      assistant(call('off2', 'read', { path: 'off.txt' })),
      {
        role: 'toolResult',
        toolCallId: 'off2',
        toolName: 'read',
        content: withStamp([{ type: 'text', text: 'x'.repeat(500) }], 8),
        details: { offload: { path: join(dir, 'nope.txt'), sha256: '00' } },
        isError: false,
        timestamp: 0
      },
      assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
    ]);

    const branch = manager.getBranch();
    const plan1 = planFoldTailPruning({
      branch,
      editOwners: attributeEpisodesEdits(branch),
      afterEntryId: ids[2]!,
      beforeEntryId: ids.at(-1)!,
      cwd: dir
    });
    assert.deepEqual(
      plan1.readEdits.map((edit) => edit.targetId),
      [ids[4]]
    );

    // Same-length modification in the omitted middle (preserving head and tail) must fail SHA-256 verification.
    const tamperedA = fullA.replace('mid A line 60', 'mid X line 60');
    assert.equal(tamperedA.length, fullA.length);
    await writeFile(cutA1.capped.path, tamperedA);
    const planTampered = planFoldTailPruning({
      branch,
      editOwners: attributeEpisodesEdits(branch),
      afterEntryId: ids[2]!,
      beforeEntryId: ids.at(-1)!,
      cwd: dir
    });
    assert.deepEqual(planTampered.readEdits, []);
    await rm(cutA1.capped.path, { force: true });
    const plan2 = planFoldTailPruning({
      branch,
      editOwners: attributeEpisodesEdits(branch),
      afterEntryId: ids[2]!,
      beforeEntryId: ids.at(-1)!,
      cwd: dir
    });
    assert.deepEqual(plan2.readEdits, []);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('failed tool argument pruning enforces age threshold, preserves structure/paths/siblings/grammar strings, and leaves signatures as they are', () => {
  assert.equal(ERROR_PRUNE_AFTER_TURNS, 3);
  const hugeCode = `const payload = "${'x'.repeat(ERROR_ARG_MIN_CHARS + 200)}";`;
  const hugeWrite = 'y'.repeat(ERROR_ARG_MIN_CHARS + 300);

  const oldFailedAssistant: AssistantMessage = {
    role: 'assistant',
    content: [
      {
        type: 'thinking',
        thinking: 'Try writing and running script.',
        thinkingSignature: '{"id":"rs_old","type":"reasoning"}'
      } satisfies ThinkingContent,
      {
        type: 'toolCall',
        id: 'err_write|fc_1',
        name: 'write',
        thoughtSignature: 'sig_w1',
        arguments: {
          path: 'src/target.ts',
          content: hugeWrite,
          nested: { keepShort: 'brief', bigPatch: hugeWrite }
        }
      },
      {
        type: 'toolCall',
        id: 'ok_write|fc_2',
        name: 'write',
        thoughtSignature: 'sig_w2',
        arguments: { path: 'src/ok.ts', content: hugeWrite }
      },
      {
        type: 'toolCall',
        id: 'err_cm|ctc_3',
        name: 'codemode',
        arguments: { code: hugeCode }
      }
    ],
    api: 'openai-responses',
    provider: 'openai',
    model: 'gpt-5',
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
    },
    stopReason: 'toolUse',
    timestamp: 0
  };

  const { manager, ids } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    oldFailedAssistant,
    errorResult('err_write|fc_1', 'write', 'Error: ENOENT: src/target.ts', 3),
    result('ok_write|fc_2', 'write', 'wrote src/ok.ts', 4),
    errorResult('err_cm|ctc_3', 'codemode', 'SyntaxError: unexpected token', 5),
    assistant({
      type: 'toolCall',
      id: 'recent_err',
      name: 'write',
      arguments: { path: 'src/recent.ts', content: hugeWrite }
    }),
    errorResult('recent_err', 'write', 'Error: EBUSY', 6),
    assistant(call('step2', 'bash', { command: 'echo 2' })),
    result('step2', 'bash', '2', 7),
    assistant(call('step3', 'bash', { command: 'echo 3' })),
    result('step3', 'bash', '3', 8),
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
  ]);

  const branch = manager.getBranch();
  const tailPlan = planFoldTailPruning({
    branch,
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[2]!,
    beforeEntryId: ids.at(-1)!,
    cwd: '/repo'
  });

  assert.deepEqual([...tailPlan.assistantReplacements.keys()], [ids[3]]);
  const blocks = tailPlan.assistantReplacements.get(ids[3]!)!;
  const thinking = blocks[0] as ThinkingContent;
  assert.equal(thinking.type, 'thinking');
  assert.equal(thinking.thinkingSignature, '{"id":"rs_old","type":"reasoning"}');

  const errWrite = blocks[1] as ToolCall;
  assert.equal(errWrite.thoughtSignature, 'sig_w1');
  assert.equal(errWrite.arguments.path, 'src/target.ts');
  assert.match(
    String(errWrite.arguments.content),
    /^\[\d+ characters omitted from failed write call\]$/
  );
  const nested = errWrite.arguments.nested as Record<string, unknown>;
  assert.equal(nested.keepShort, 'brief');
  assert.match(String(nested.bigPatch), /^\[\d+ characters omitted from failed write call\]$/);

  const okWrite = blocks[2] as ToolCall;
  assert.equal(okWrite.thoughtSignature, 'sig_w2');
  assert.equal(okWrite.arguments.content, hugeWrite);

  const errCm = blocks[3] as ToolCall;
  assert.equal(typeof errCm.arguments.code, 'string');
  assert.match(
    String(errCm.arguments.code),
    /^\[\d+ characters omitted from failed codemode call\]$/
  );

  const model = {
    id: 'gpt-5',
    name: 'GPT-5',
    api: 'openai-responses',
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    reasoning: true,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 16000
  } as Model<'openai-responses'>;
  const replayed = convertResponsesMessages(
    model,
    normalizeContext({
      messages: [
        user('go', 1),
        { ...oldFailedAssistant, content: blocks },
        errorResult('err_write|fc_1', 'write', 'Error: ENOENT: src/target.ts', 3),
        result('ok_write|fc_2', 'write', 'wrote src/ok.ts', 4),
        errorResult('err_cm|ctc_3', 'codemode', 'SyntaxError: unexpected token', 5)
      ]
    }),
    new Set(['openai']),
    { grammarToolInputProperties: new Map([['codemode', 'code']]) }
  );
  const customCall = replayed.find((item) => item.type === 'custom_tool_call');
  assert.ok(customCall && typeof customCall.input === 'string');
});

test('pending boundary edits on newer duplicate read prevent stale deduplication', () => {
  const fileText = 'const line = 1;\n'.repeat(30);
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('r1', 'read', { path: 'a.ts' })),
    result('r1', 'read', fileText, 3),
    assistant(call('r2', 'read', { path: 'a.ts' })),
    result('r2', 'read', fileText, 4),
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
  ]);

  const branch = manager.getBranch();
  const withPendingOnNewer = planFoldTailPruning({
    branch,
    pending: [
      {
        type: 'context_edit',
        targetId: ids[6]!,
        replacement: { content: [{ type: 'text', text: '[redacted]' }] }
      }
    ],
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[2]!,
    beforeEntryId: ids[7]!,
    cwd: '/repo'
  });
  assert.deepEqual(withPendingOnNewer.readEdits, []);
});

test('real session end-to-end: fold with until prunes duplicate reads and aged failed inputs, skips rejected folds, and unfold restores both across standard and codemode', async () => {
  for (const mode of ['off', 'only'] as const) {
    const dir = await mkdtemp(join(tmpdir(), `pi-episodes-prune-e2e-${mode}-`));
    await mkdir(join(dir, 'sessions'), { recursive: true });
    const bigFile = Array.from({ length: 80 }, (_, i) => `export const line_${i} = "${i}";`).join(
      '\n'
    );
    const oldBrokenMarker = 'OLD_BROKEN_PAYLOAD_'.repeat(40);
    const recentBrokenMarker = 'RECENT_BROKEN_PAYLOAD_'.repeat(40);
    await writeFile(join(dir, 'big.ts'), `${bigFile}\n`);
    const hugeInner = Array.from(
      { length: 700 },
      (_, i) => `inner codemode line ${i + 1} payload`
    ).join('\n');
    await writeFile(join(dir, 'huge-inner.txt'), `${hugeInner}\n`);

    let handle = '';
    const faux = fauxProvider({ models: [{ id: 'faux-prune', reasoning: true }] });
    faux.setResponses([
      () =>
        mode === 'only'
          ? fauxAssistantMessage(
              fauxToolCall('codemode', {
                code: 'const r = await tools.read({ path: "huge-inner.txt" }); const body = typeof r === "string" ? r : JSON.stringify(r); if (!body.includes("inner codemode line 350") || body.includes("left out here")) throw new Error("nested codemode read was capped"); return "ep1-ok";'
              })
            )
          : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ep1' })),
      () =>
        fauxAssistantMessage([
          fauxToolCall('read', { path: 'big.ts' }),
          fauxToolCall('bash', { command: `false # ${oldBrokenMarker}` })
        ]),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo step1' })),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'big.ts' })),
      () =>
        mode === 'off'
          ? fauxAssistantMessage([
              fauxToolCall('fold', { since: 't999', until: 't2', note: 'Bad.' }),
              fauxToolCall('bash', { command: `false # ${recentBrokenMarker}` })
            ])
          : fauxAssistantMessage([
              fauxToolCall('codemode', {
                code: 'return await tools.fold({ since: "t999", until: "t3", note: "Bad." });'
              }),
              fauxToolCall('bash', { command: `false # ${recentBrokenMarker}` })
            ]),
      (context) => {
        const msgs = context.messages;
        const reads = msgs.filter(
          (m) =>
            m.role === 'toolResult' &&
            m.toolName === 'read' &&
            text(m).includes('export const line_40')
        );
        assert.equal(reads.length, 2, `${mode}: rejected fold must leave both reads intact`);
        assert.ok(
          JSON.stringify(msgs).includes(oldBrokenMarker),
          `${mode}: rejected fold must leave failed bash payload intact`
        );
        return mode === 'off'
          ? fauxAssistantMessage(
              fauxToolCall('fold', { since: 't2', until: 't2', note: 'Closed ep1.' })
            )
          : fauxAssistantMessage(
              fauxToolCall('codemode', {
                code: 'const r = await tools.fold({ since: "t2", until: "t2", note: "Closed ep1." }); text(r);'
              })
            );
      },
      (context) => {
        const msgs = context.messages;
        assertPaired(msgs);
        const lastNote = text(msgs.at(-1)!);
        assert.match(lastNote, /ep-[0-9a-f]{8}/, `unexpected lastNote: ${lastNote}`);
        handle = /ep-[0-9a-f]{8}/.exec(lastNote)![0];

        const readResults = msgs.filter((m) => m.role === 'toolResult' && m.toolName === 'read');
        assert.equal(readResults.length, 2);
        assert.match(
          text(readResults[0]!),
          /^\[duplicate read of big\.ts, \d+ chars; identical to t\d+ \(if t\d+ is later folded, check that fold note for its handle and call recall with turn "t\d+"\)\]\n\[t3\]$/
        );
        assert.match(text(readResults[1]!), /export const line_40/);

        const allSerialized = JSON.stringify(msgs);
        assert.equal(
          allSerialized.includes(oldBrokenMarker),
          false,
          `${mode}: aged failed bash command payload should be trimmed`
        );
        assert.equal(
          allSerialized.includes(recentBrokenMarker),
          true,
          `${mode}: recent failed bash command (< 3 later turns) must stay whole`
        );
        assert.match(allSerialized, /characters omitted from failed bash call/);
        const errResults = msgs.filter((m) => m.role === 'toolResult' && m.isError);
        assert.ok(errResults.some((m) => text(m).includes('Command exited with code 1')));

        return mode === 'off'
          ? fauxAssistantMessage(fauxToolCall('unfold', { handle }))
          : fauxAssistantMessage(
              fauxToolCall('codemode', {
                code:
                  'const u = await tools.unfold({ handle: ' +
                  JSON.stringify(handle) +
                  ' }); text(u);'
              })
            );
      },
      (context) => {
        const msgs = context.messages;
        assertPaired(msgs);
        const readResults = msgs.filter((m) => m.role === 'toolResult' && m.toolName === 'read');
        assert.equal(readResults.length, 2);
        assert.match(text(readResults[0]!), /export const line_40/);
        assert.match(text(readResults[1]!), /export const line_40/);
        assert.equal(
          JSON.stringify(msgs).includes(oldBrokenMarker),
          true,
          `${mode}: unfold should restore tail-pruned failed bash command`
        );
        return fauxAssistantMessage('all good');
      }
    ]);

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
        'write',
        'fold',
        'unfold',
        'recall',
        'ask',
        'fork',
        ...(mode === 'off' ? [] : ['codemode'])
      ],
      sessionManager: SessionManager.create(dir, join(dir, 'sessions')),
      settingsManager: SettingsManager.inMemory({
        compaction: { enabled: false },
        ...{ episodes: { snapshots: false } },
        ...(mode === 'off' ? {} : { codemode: { mode } })
      })
    });
    await agentSession.bindExtensions({});

    try {
      await agentSession.prompt('run');
      const last = agentSession.messages.at(-1);
      if (last && 'errorMessage' in last && last.errorMessage)
        throw new Error(`${mode}: ${last.errorMessage}`);
      assert.equal(faux.getPendingResponseCount(), 0);
      const started = agentSession.sessionManager
        .getBranch()
        .find((entry) => entry.type === 'custom' && entry.customType === FOLD_STARTED_TYPE);
      assert.ok(started && started.type === 'custom');
      const startedDeps = (started.data as { readDeps?: unknown[] } | undefined)?.readDeps;
      assert.equal(Array.isArray(startedDeps) ? startedDeps.length : 0, 1);
    } finally {
      agentSession.dispose();
      await rm(dir, { recursive: true, force: true });
    }
  }
});

test('interrupted fold recovery and multi-fold unfold preserve external edits and restore tail edits cleanly', () => {
  const fileText = 'const item = 42;\n'.repeat(30);
  const hugeCmd = `exit 1 # ${'z'.repeat(700)}`;
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('r1', 'read', { path: 'item.ts' })),
    result('r1', 'read', fileText, 3),
    assistant(call('bad', 'bash', { command: hugeCmd })),
    errorResult('bad', 'bash', 'Command exited with code 1', 4),
    assistant(call('step1', 'bash', { command: 'echo a' })),
    result('step1', 'bash', 'a', 5),
    assistant(call('step2', 'bash', { command: 'echo b' })),
    result('step2', 'bash', 'b', 6),
    assistant(call('r2', 'read', { path: 'item.ts' })),
    result('r2', 'read', fileText, 7),
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Ep1.' }))
  ]);

  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 2 });
  const noteEntry = Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 8), {
    details: {
      kind: 'fold',
      handle: plan.handle,
      since: 't2',
      until: 't2',
      firstEntryId: plan.firstEntryId,
      lastEntryId: plan.lastEntryId,
      messages: plan.omitted.length,
      tokens: 10,
      headline: 'Ep1.',
      keptUserTurns: [],
      files: []
    }
  });
  manager.appendMessage(noteEntry);

  const branchBefore = manager.getBranch();
  const tail = planFoldTailPruning({
    branch: branchBefore,
    editOwners: attributeEpisodesEdits(branchBefore),
    afterEntryId: plan.lastEntryId,
    beforeEntryId: ids[13]!,
    cwd: '/repo'
  });

  const fullBatch = [
    ...plan.edits.map((edit) =>
      tail.assistantReplacements.has(edit.targetId)
        ? {
            type: 'context_edit' as const,
            targetId: edit.targetId,
            replacement: { content: tail.assistantReplacements.get(edit.targetId)! }
          }
        : edit
    ),
    ...[...tail.assistantReplacements.entries()]
      .filter(([id]) => !plan.edits.some((e) => e.targetId === id))
      .map(([targetId, content]) => ({
        type: 'context_edit' as const,
        targetId,
        replacement: { content }
      })),
    ...tail.readEdits
  ];

  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: fullBatch });
  manager.appendContextEdit(fullBatch[0]!.targetId, fullBatch[0]!.replacement);

  const recovery = completeCutShortFolds(manager.getBranch());
  for (const item of recovery) {
    if (item.type === 'context_edit') {
      manager.appendContextEdit(item.targetId, item.replacement);
    } else {
      manager.appendCustomEntry(item.customType, item.data);
    }
  }

  const afterRecovery = manager.buildSessionProjection().messages;
  assertPaired(afterRecovery);
  assert.ok(afterRecovery.some((m) => text(m).includes('[duplicate read of item.ts')));
  assert.equal(JSON.stringify(afterRecovery).includes('zzzzzzzzzz'), false);

  manager.appendContextEdit(ids[4]!, {
    content: withStamp([{ type: 'text', text: '[external override]' }], 3)
  });
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));
  const unfoldPlan = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    plan.handle,
    { callerId: 'u1' }
  );
  for (const edit of unfoldPlan.edits) {
    manager.appendContextEdit(edit.targetId, edit.replacement);
  }

  const afterUnfold = manager.buildSessionProjection().messages;
  assert.ok(afterUnfold.some((m) => text(m).includes('[external override]')));
  assert.equal(JSON.stringify(afterUnfold).includes('zzzzzzzzzz'), true);
});

test('multi-fold sequence, hook-rejected fold, and boundary-note-removal rejection', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-prune-multi-'));
  await mkdir(join(dir, 'sessions'), { recursive: true });
  const bigFile = Array.from({ length: 70 }, (_, i) => `export const item_${i} = ${i};`).join('\n');
  const oldBroken = 'MULTI_FOLD_BROKEN_ARG_'.repeat(40);
  await writeFile(join(dir, 'mod.ts'), `${bigFile}\n`);

  let ep1 = '';
  let ep2 = '';
  let hookRejectNextFold = true;
  let removeNextFoldAtBoundary = true;
  const faux = fauxProvider({ models: [{ id: 'faux-multi', reasoning: true }] });
  faux.setResponses([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one' })),
    () =>
      fauxAssistantMessage([
        fauxToolCall('read', { path: 'mod.ts' }),
        fauxToolCall('bash', { command: `false # ${oldBroken}` })
      ]),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo two' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo three' })),
    () => fauxAssistantMessage(fauxToolCall('read', { path: 'mod.ts' })),
    // Attempt 1: tool_result hook fails the fold with isError: true -> must NOT prune tail!
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', { since: 't2', until: 't2', note: 'Hook will fail this fold.' })
      ),
    (context) => {
      const msgs = context.messages;
      const reads = msgs.filter(
        (m) =>
          m.role === 'toolResult' &&
          m.toolName === 'read' &&
          text(m).includes('export const item_35')
      );
      assert.equal(reads.length, 2, 'hook-rejected fold must not prune duplicate reads');
      assert.equal(
        JSON.stringify(msgs).includes(oldBroken),
        true,
        'hook-rejected fold must not prune failed args'
      );
      // Attempt 2: earlier turn_end handler removes the fold note (`replacement: null`) -> stands is false, must NOT prune tail!
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: 't2', until: 't2', note: 'Boundary will omit this note.' })
      );
    },
    (context) => {
      const msgs = context.messages;
      const reads = msgs.filter(
        (m) =>
          m.role === 'toolResult' &&
          m.toolName === 'read' &&
          text(m).includes('export const item_35')
      );
      assert.equal(reads.length, 2, 'boundary-removed fold note must not prune duplicate reads');
      assert.equal(
        JSON.stringify(msgs).includes(oldBroken),
        true,
        'boundary-removed fold note must not prune failed args'
      );
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: 't2', until: 't2', note: 'Ep1 applied.' })
      );
    },
    (context) => {
      ep1 = /ep-[0-9a-f]{8}/.exec(text(context.messages.at(-1)!))![0];
      assert.equal(JSON.stringify(context.messages).includes(oldBroken), false);
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: 't3', note: 'Folded tail into ep2.' })
      );
    },
    (context) => {
      ep2 = /ep-[0-9a-f]{8}/.exec(text(context.messages.at(-1)!))![0];
      return fauxAssistantMessage(
        fauxToolCall('recall', { handle: ep2, query: 'MULTI_FOLD_BROKEN_ARG' })
      );
    },
    (context) => {
      assert.match(text(context.messages.at(-1)!), /MULTI_FOLD_BROKEN_ARG/);
      return fauxAssistantMessage(fauxToolCall('unfold', { handle: ep2 }));
    },
    (context) => {
      const msgs = context.messages;
      assertPaired(msgs);
      assert.ok(msgs.some((m) => text(m).includes('[duplicate read of mod.ts')));
      assert.equal(JSON.stringify(msgs).includes(oldBroken), false);
      return fauxAssistantMessage(fauxToolCall('unfold', { handle: ep1 }));
    },
    (context) => {
      const msgs = context.messages;
      assertPaired(msgs);
      const reads = msgs.filter(
        (m) =>
          m.role === 'toolResult' &&
          m.toolName === 'read' &&
          text(m).includes('export const item_35')
      );
      assert.equal(reads.length, 2);
      assert.equal(JSON.stringify(msgs).includes(oldBroken), true);
      return fauxAssistantMessage('done');
    }
  ]);

  const resourceLoader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: dir,
    noSkills: true,
    extensionFactories: [
      (pi) => pi.registerProvider(faux.provider),
      // Registered BEFORE episodes so its turn_end runs first and populates event.entries!
      (pi) => {
        pi.on('tool_result', (event) => {
          if (event.toolName === 'fold' && hookRejectNextFold) {
            hookRejectNextFold = false;
            return { content: [{ type: 'text', text: '[hook rejected fold]' }], isError: true };
          }
          return undefined;
        });
        pi.on('turn_end', (event, ctx) => {
          const okFold = event.toolResults.find((r) => r.toolName === 'fold' && !r.isError);
          if (!okFold || !removeNextFoldAtBoundary) return undefined;
          removeNextFoldAtBoundary = false;
          const branch = ctx.sessionManager.getBranch();
          const noteEntry = branch.find(
            (e) =>
              e.type === 'message' &&
              e.message.role === 'toolResult' &&
              e.message.toolCallId === okFold.toolCallId
          );
          const callerEntryId = event.messageEntryId;
          if (!noteEntry || !callerEntryId) return undefined;
          // Remove both the fold note and its assistant call as an atomic pair.
          return {
            entries: [
              ...event.entries,
              { type: 'context_edit' as const, targetId: noteEntry.id, replacement: null },
              { type: 'context_edit' as const, targetId: callerEntryId, replacement: null }
            ]
          };
        });
      },
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
    tools: ['bash', 'read', 'fold', 'unfold', 'recall', 'ask', 'fork'],
    sessionManager: SessionManager.create(dir, join(dir, 'sessions')),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      ...{ episodes: { snapshots: false } }
    })
  });
  await agentSession.bindExtensions({});

  try {
    await agentSession.prompt('run');
    const last = agentSession.messages.at(-1);
    if (last && 'errorMessage' in last && last.errorMessage) throw new Error(last.errorMessage);
    assert.equal(faux.getPendingResponseCount(), 0);
  } finally {
    agentSession.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

test('completeCutShortFolds revalidates newer-read dependency across before/after crash points, persisted/queued edits, and repeated recovery', () => {
  const fileText = 'const keep = true;\n'.repeat(30);

  function buildScenario() {
    const { manager, ids } = session([
      user('go', 1),
      assistant(call('s1', 'bash', { command: 'echo 1' })),
      result('s1', 'bash', '1', 2),
      assistant(call('r1', 'read', { path: 'dep.ts' })),
      result('r1', 'read', fileText, 3),
      assistant(call('r2', 'read', { path: 'dep.ts' })),
      result('r2', 'read', fileText, 4),
      assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Ep1.' }))
    ]);
    const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 2 });
    manager.appendMessage(
      Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 5), {
        details: {
          kind: 'fold',
          handle: plan.handle,
          since: 't2',
          until: 't2',
          firstEntryId: plan.firstEntryId,
          lastEntryId: plan.lastEntryId,
          messages: plan.omitted.length,
          tokens: 10,
          headline: 'Ep1.',
          keptUserTurns: [],
          files: []
        }
      })
    );
    const branch = manager.getBranch();
    const tail = planFoldTailPruning({
      branch,
      editOwners: attributeEpisodesEdits(branch),
      afterEntryId: plan.lastEntryId,
      beforeEntryId: ids[7]!,
      cwd: '/repo'
    });
    const edits = [...plan.edits, ...tail.readEdits];
    return { manager, ids, plan, tail, edits, olderId: ids[4]!, newerId: ids[6]! };
  }

  // Case 1: Interruption BEFORE older-read edit + persisted external edit on newer read + repeated recovery.
  {
    const { manager, plan, tail, edits, newerId } = buildScenario();
    manager.appendCustomEntry(FOLD_STARTED_TYPE, {
      handle: plan.handle,
      edits,
      readDeps: tail.readDeps
    });
    manager.appendContextEdit(edits[0]!.targetId, edits[0]!.replacement);
    // External extension alters the newer read before recovery.
    manager.appendContextEdit(newerId, {
      content: withStamp([{ type: 'text', text: '[newer redacted]' }], 4)
    });

    // Recovery pass 1: apply only context_edit items (simulating a second crash before FOLD_APPLIED_TYPE).
    const rec1 = completeCutShortFolds(manager.getBranch());
    for (const item of rec1) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
    }
    // Recovery pass 2 (repeated recovery): finishes cleanly without stubbing the older read.
    const rec2 = completeCutShortFolds(manager.getBranch());
    for (const item of rec2) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    const msgs = manager.buildSessionProjection().messages;
    assert.equal(
      msgs.some((m) => text(m).includes('[duplicate read of dep.ts')),
      false
    );
    assert.ok(msgs.some((m) => text(m).includes('const keep = true;')));
  }

  // Case 2: Interruption BEFORE older-read edit + queued pending boundary edit on newer read.
  {
    const { manager, plan, tail, edits, newerId } = buildScenario();
    manager.appendCustomEntry(FOLD_STARTED_TYPE, {
      handle: plan.handle,
      edits,
      readDeps: tail.readDeps
    });
    const pending = [
      {
        type: 'context_edit' as const,
        targetId: newerId,
        replacement: { content: withStamp([{ type: 'text', text: '[queued redact]' }], 4) }
      }
    ];
    const rec = completeCutShortFolds(manager.getBranch(), pending);
    for (const p of pending) manager.appendContextEdit(p.targetId, p.replacement);
    for (const item of rec) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    const msgs = manager.buildSessionProjection().messages;
    assert.equal(
      msgs.some((m) => text(m).includes('[duplicate read of dep.ts')),
      false
    );
    assert.ok(msgs.some((m) => text(m).includes('const keep = true;')));
  }

  // Case 3: Interruption AFTER older-read edit was already written + persisted edit on newer read + repeated recovery + unfold.
  {
    const { manager, plan, tail, edits, newerId } = buildScenario();
    manager.appendCustomEntry(FOLD_STARTED_TYPE, {
      handle: plan.handle,
      edits,
      readDeps: tail.readDeps
    });
    for (const edit of edits) {
      manager.appendContextEdit(edit.targetId, edit.replacement);
    }
    // Crash occurred right before FOLD_APPLIED_TYPE; external extension then alters newer read.
    manager.appendContextEdit(newerId, {
      content: withStamp([{ type: 'text', text: '[newer redacted]' }], 4)
    });

    // Recovery pass 1: rolls back the now-unjustified older-read stub; crash before marker.
    const rec1 = completeCutShortFolds(manager.getBranch());
    for (const item of rec1) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
    }
    // Recovery pass 2 (repeated recovery): idempotent, writes FOLD_APPLIED_TYPE.
    const rec2 = completeCutShortFolds(manager.getBranch());
    assert.equal(rec2.filter((item) => item.type === 'context_edit').length, 0);
    for (const item of rec2) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    const msgsAfterRec = manager.buildSessionProjection().messages;
    assert.equal(
      msgsAfterRec.some((m) => text(m).includes('[duplicate read of dep.ts')),
      false
    );
    assert.ok(msgsAfterRec.some((m) => text(m).includes('const keep = true;')));
    assert.ok(msgsAfterRec.some((m) => text(m).includes('[newer redacted]')));

    // Unfolding afterwards still works cleanly and keeps the external edit on newerId.
    manager.appendMessage(assistant(call('u_rec', 'unfold', { handle: plan.handle })));
    const unfoldPlan = planUnfold(
      manager.getBranch(),
      manager.buildSessionProjection().entries,
      plan.handle,
      { callerId: 'u_rec' }
    );
    for (const edit of unfoldPlan.edits) {
      manager.appendContextEdit(edit.targetId, edit.replacement);
    }
    const msgsAfterUnfold = manager.buildSessionProjection().messages;
    assert.ok(msgsAfterUnfold.some((m) => text(m).includes('const keep = true;')));
    assert.ok(msgsAfterUnfold.some((m) => text(m).includes('[newer redacted]')));
  }

  // Case 4: Interruption AFTER older-read edit + queued pending edit on newer read.
  {
    const { manager, plan, tail, edits, newerId } = buildScenario();
    manager.appendCustomEntry(FOLD_STARTED_TYPE, {
      handle: plan.handle,
      edits,
      readDeps: tail.readDeps
    });
    for (const edit of edits) {
      manager.appendContextEdit(edit.targetId, edit.replacement);
    }
    const pending = [
      {
        type: 'context_edit' as const,
        targetId: newerId,
        replacement: { content: withStamp([{ type: 'text', text: '[queued redact]' }], 4) }
      }
    ];
    const rec = completeCutShortFolds(manager.getBranch(), pending);
    for (const p of pending) manager.appendContextEdit(p.targetId, p.replacement);
    for (const item of rec) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    const msgs = manager.buildSessionProjection().messages;
    assert.equal(
      msgs.some((m) => text(m).includes('[duplicate read of dep.ts')),
      false
    );
    assert.ok(msgs.some((m) => text(m).includes('const keep = true;')));
  }
});

test('completeCutShortFolds revalidates newer capped-read archive file across before and after crash points', async () => {
  const dirOlder = await mkdtemp(join(tmpdir(), 'pi-episodes-prune-rec-old-'));
  const dirNewer = await mkdtemp(join(tmpdir(), 'pi-episodes-prune-rec-new-'));
  try {
    const headLines = Array.from({ length: 20 }, (_, i) => `head line ${i}`).join('\n');
    const midLines = Array.from(
      { length: 120 },
      (_, i) => `middle line ${String(i).padStart(3, '0')}`
    ).join('\n');
    const tailLines = Array.from({ length: 20 }, (_, i) => `tail line ${i}`).join('\n');
    const fullText = `${headLines}\n${midLines}\n${tailLines}`;
    const cutOlder = await capOutput([{ type: 'text', text: fullText }], dirOlder, 500, 200);
    const cutNewer = await capOutput([{ type: 'text', text: fullText }], dirNewer, 500, 200);
    assert.ok(cutOlder && cutNewer);

    const buildCappedDepScenario = () => {
      const olderMsg = {
        role: 'toolResult' as const,
        toolCallId: 'r1',
        toolName: 'read',
        content: withStamp(cutOlder.content, 3),
        details: { [CAPPED_KEY]: { ...cutOlder.capped } },
        isError: false,
        timestamp: 0
      };
      const newerMsg = {
        role: 'toolResult' as const,
        toolCallId: 'r2',
        toolName: 'read',
        content: withStamp(cutNewer.content, 4),
        details: { [CAPPED_KEY]: { ...cutNewer.capped } },
        isError: false,
        timestamp: 0
      };
      const { manager, ids } = session([
        user('go', 1),
        assistant(call('s1', 'bash', { command: 'echo 1' })),
        result('s1', 'bash', '1', 2),
        assistant(call('r1', 'read', { path: 'big.ts' })),
        olderMsg,
        assistant(call('r2', 'read', { path: 'big.ts' })),
        newerMsg,
        assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Ep1.' }))
      ]);
      const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 2 });
      manager.appendMessage(
        Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 5), {
          details: {
            kind: 'fold',
            handle: plan.handle,
            since: 't2',
            until: 't2',
            firstEntryId: plan.firstEntryId,
            lastEntryId: plan.lastEntryId,
            messages: plan.omitted.length,
            tokens: 10,
            headline: 'Ep1.',
            keptUserTurns: [],
            files: []
          }
        })
      );
      const branch = manager.getBranch();
      const tail = planFoldTailPruning({
        branch,
        editOwners: attributeEpisodesEdits(branch),
        afterEntryId: plan.lastEntryId,
        beforeEntryId: ids[7]!,
        cwd: dirNewer
      });
      assert.equal(tail.readEdits.length, 1);
      return { manager, plan, tail, edits: [...plan.edits, ...tail.readEdits] };
    };

    {
      const { manager, plan, tail, edits } = buildCappedDepScenario();
      manager.appendCustomEntry(FOLD_STARTED_TYPE, {
        handle: plan.handle,
        edits,
        readDeps: tail.readDeps
      });
      const tampered = fullText.replace('middle line 060', 'middle XXXX 060');
      assert.equal(tampered.length, fullText.length);
      await writeFile(cutNewer.capped.path, tampered);

      for (const item of completeCutShortFolds(manager.getBranch())) {
        if (item.type === 'context_edit')
          manager.appendContextEdit(item.targetId, item.replacement);
        else manager.appendCustomEntry(item.customType, item.data);
      }
      const msgs = manager.buildSessionProjection().messages;
      assert.equal(
        msgs.some((m) => text(m).includes('[duplicate read of big.ts')),
        false
      );
      assert.ok(msgs.some((m) => text(m).includes(cutOlder.capped.path)));
      await writeFile(cutNewer.capped.path, fullText);
    }

    {
      const { manager, plan, tail, edits } = buildCappedDepScenario();
      manager.appendCustomEntry(FOLD_STARTED_TYPE, {
        handle: plan.handle,
        edits,
        readDeps: tail.readDeps
      });
      for (const edit of edits) {
        manager.appendContextEdit(edit.targetId, edit.replacement);
      }
      await rm(cutNewer.capped.path, { force: true });

      for (const item of completeCutShortFolds(manager.getBranch())) {
        if (item.type === 'context_edit')
          manager.appendContextEdit(item.targetId, item.replacement);
        else manager.appendCustomEntry(item.customType, item.data);
      }
      const msgs = manager.buildSessionProjection().messages;
      assert.equal(
        msgs.some((m) => text(m).includes('[duplicate read of big.ts')),
        false
      );
      assert.ok(msgs.some((m) => text(m).includes(cutOlder.capped.path)));
    }
  } finally {
    await rm(dirOlder, { recursive: true, force: true });
    await rm(dirNewer, { recursive: true, force: true });
  }
});

test('double crash persists recovery-segment markers, restores episodes edits, and never claims identical external edits', () => {
  const dupBody = 'export const keep = 42;\n'.repeat(30);
  const hugeArg = 'BROKEN_MULTI_CRASH_'.repeat(40);
  const { manager, ids } = session([
    user('start', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('s2', 'bash', { command: 'echo 2' })),
    result('s2', 'bash', '2', 3),
    assistant(
      call('r1', 'read', { path: 'x.ts' }),
      call('e1', 'bash', { command: `false # ${hugeArg}` })
    ),
    result('r1', 'read', dupBody, 4),
    Object.assign(result('e1', 'bash', 'exit 1', 5), { isError: true }),
    assistant(call('t1', 'bash', { command: 'echo a' })),
    result('t1', 'bash', 'a', 6),
    assistant(call('t2', 'bash', { command: 'echo b' })),
    result('t2', 'bash', 'b', 7),
    assistant(call('r2', 'read', { path: 'x.ts' })),
    result('r2', 'read', dupBody, 8),
    assistant(call('f1', 'fold', { since: 't2', until: 't3', note: 'Ep1.' }))
  ]);

  const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 3 });
  manager.appendMessage(
    Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 9), {
      details: {
        kind: 'fold',
        handle: plan.handle,
        since: 't2',
        until: 't3',
        firstEntryId: plan.firstEntryId,
        lastEntryId: plan.lastEntryId,
        messages: plan.omitted.length,
        tokens: 10,
        headline: 'Ep1.',
        keptUserTurns: [],
        files: []
      }
    })
  );
  const branch0 = manager.getBranch();
  const tail = planFoldTailPruning({
    branch: branch0,
    editOwners: attributeEpisodesEdits(branch0),
    afterEntryId: plan.lastEntryId,
    beforeEntryId: ids[14]!,
    cwd: '/repo'
  });
  const planTargets = new Set(plan.edits.map((edit) => edit.targetId));
  const fullBatch = [
    ...[...tail.assistantReplacements.entries()]
      .filter(([targetId]) => !planTargets.has(targetId))
      .map(([targetId, content]) => ({
        type: 'context_edit' as const,
        targetId,
        replacement: { content }
      })),
    ...plan.edits.map((edit) =>
      tail.assistantReplacements.has(edit.targetId)
        ? {
            type: 'context_edit' as const,
            targetId: edit.targetId,
            replacement: { content: tail.assistantReplacements.get(edit.targetId)! }
          }
        : edit
    ),
    ...tail.readEdits
  ];

  // Crash 1: FOLD_STARTED_TYPE written, then crash before any context_edit!
  manager.appendCustomEntry(FOLD_STARTED_TYPE, {
    handle: plan.handle,
    edits: fullBatch,
    readDeps: tail.readDeps
  });

  // Resumed Turn A: user/assistant messages + an external edit IDENTICAL to planned removal (ids[2] -> null, ids[1] -> null),
  // PLUS an external replacement on t1 result!
  manager.appendMessage(user('resumed 1', 10));
  manager.appendMessage(assistant('ack 1'));
  manager.appendContextEdit(ids[1]!, null);
  manager.appendContextEdit(ids[2]!, null);
  manager.appendContextEdit(ids[9]!, {
    content: withStamp([{ type: 'text', text: '[external note on t6]' }], 6)
  });
  // Recovery runs and persists all entries EXCEPT the final FOLD_APPLIED_TYPE (Crash 2 right before completion marker!).
  const rec1 = completeCutShortFolds(manager.getBranch());
  assert.equal(rec1[0]?.type, 'custom');
  assert.equal((rec1[0] as { customType?: string }).customType, 'episodes-fold-recovery-started');
  for (const item of rec1.slice(0, -1)) {
    if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
    else manager.appendCustomEntry(item.customType, item.data);
  }

  // Resumed Turn B: another conversation turn, then recovery finishes FOLD_APPLIED_TYPE!
  manager.appendMessage(user('resumed 2', 11));
  manager.appendMessage(assistant('ack 2'));
  for (const item of completeCutShortFolds(manager.getBranch())) {
    if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
    else manager.appendCustomEntry(item.customType, item.data);
  }

  const foldedMsgs = manager.buildSessionProjection().messages;
  assert.ok(foldedMsgs.some((m) => text(m).includes('[duplicate read of x.ts')));
  assert.equal(JSON.stringify(foldedMsgs).includes(hugeArg), false);

  // Unfold ep-1: episodes-owned span (s2 at ids[3..4]) & tail edits ARE restored,
  // while external identical removals (ids[1..2] -> null) and external replacement on ids[9] stand!
  manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));
  const unfoldPlan = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    plan.handle,
    { callerId: 'u1' }
  );
  for (const edit of unfoldPlan.edits) {
    manager.appendContextEdit(edit.targetId, edit.replacement);
  }
  const unfoldedProj = manager.buildSessionProjection();
  const unfoldedById = new Map(unfoldedProj.entries.map((e) => [e.sourceEntry.id, e]));
  assert.equal(unfoldedById.get(ids[1]!)!.messages.length, 0);
  assert.equal(unfoldedById.get(ids[2]!)!.messages.length, 0);
  assert.equal(unfoldedById.get(ids[3]!)!.messages.length, 1);
  assert.equal(unfoldedById.get(ids[4]!)!.messages.length, 1);
  const unfoldedMsgs = unfoldedProj.messages;
  assert.equal(
    unfoldedMsgs.some((m) => text(m).includes('[duplicate read of x.ts')),
    false
  );
  assert.equal(JSON.stringify(unfoldedMsgs).includes(hugeArg), true);
  assert.ok(unfoldedMsgs.some((m) => text(m).includes('[external note on t6]')));
});

test('recovery edits no response written since the fold began and never claims external edits before an empty completion marker', () => {
  // 1. A response written between the cut-short fold and its recovery stays as it is.
  {
    const { manager } = session([
      user('start', 1),
      assistant(call('s1', 'bash', { command: 'echo 1' })),
      result('s1', 'bash', '1', 2),
      assistant(call('f1', 'fold', { since: 't2', note: 'Ep1.' }))
    ]);
    const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
    manager.appendMessage(
      Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 3), {
        details: {
          kind: 'fold',
          handle: plan.handle,
          since: 't2',
          firstEntryId: plan.firstEntryId,
          lastEntryId: plan.lastEntryId,
          messages: plan.omitted.length,
          tokens: 10,
          headline: 'Ep1.',
          keptUserTurns: [],
          files: []
        }
      })
    );
    // Crash 1 right after FOLD_STARTED_TYPE.
    manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
    // Resumed Turn A persists a SIGNED assistant response, then crashes again before recovery!
    manager.appendMessage(user('resumed A', 4));
    const signedTurnA: AssistantMessage = {
      ...assistant('reply A'),
      content: [
        { type: 'thinking', thinking: 'reasoning A', thinkingSignature: 'sig-turn-A' },
        { type: 'text', text: 'reply A', textSignature: 'txt-sig-A' }
      ]
    };
    const turnAId = manager.appendMessage(signedTurnA);
    // Resumed Turn B completes recovery.
    manager.appendMessage(user('resumed B', 5));
    const rec = completeCutShortFolds(manager.getBranch());
    const editTargets = rec.flatMap((d) => (d.type === 'context_edit' ? [d.targetId] : []));
    assert.deepEqual(
      editTargets,
      plan.edits.map((edit) => edit.targetId)
    );
    assert.equal(editTargets.includes(turnAId), false);
    for (const item of rec) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    const projStr = JSON.stringify(manager.buildSessionProjection().messages);
    assert.equal(projStr.includes('sig-turn-A'), true);
    assert.equal(projStr.includes('txt-sig-A'), true);
  }

  // 2. Empty recovery (`rest.length === 0`) immediately following external identical `targetId -> null` edits
  //    must NOT claim those external edits.
  {
    const { manager, ids } = session([
      user('start', 1),
      assistant(call('s1', 'bash', { command: 'echo 1' })),
      result('s1', 'bash', '1', 2),
      assistant(call('s2', 'bash', { command: 'echo 2' })),
      result('s2', 'bash', '2', 3),
      assistant(call('f1', 'fold', { since: 't2', note: 'Ep1.' }))
    ]);
    const plan = planFold(manager.buildSessionProjection().entries, 'f1', 2);
    manager.appendMessage(
      Object.assign(result('f1', 'fold', `[fold ${plan.handle}]\nEp1.`, 4), {
        details: {
          kind: 'fold',
          handle: plan.handle,
          since: 't2',
          firstEntryId: plan.firstEntryId,
          lastEntryId: plan.lastEntryId,
          messages: plan.omitted.length,
          tokens: 10,
          headline: 'Ep1.',
          keptUserTurns: [],
          files: []
        }
      })
    );
    manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: plan.handle, edits: plan.edits });
    // Persist all edits EXCEPT the last two (`ids[1]`, `ids[2]`), then crash.
    const remaining = new Set([ids[1]!, ids[2]!]);
    for (const edit of plan.edits) {
      if (!remaining.has(edit.targetId)) manager.appendContextEdit(edit.targetId, edit.replacement);
    }
    manager.appendMessage(user('resumed', 5));
    // External handler writes `ids[1] -> null` and `ids[2] -> null` IMMEDIATELY before empty recovery!
    manager.appendContextEdit(ids[1]!, null);
    manager.appendContextEdit(ids[2]!, null);
    const emptyRec = completeCutShortFolds(manager.getBranch());
    assert.equal(emptyRec.length, 1);
    for (const item of emptyRec) {
      if (item.type === 'context_edit') manager.appendContextEdit(item.targetId, item.replacement);
      else manager.appendCustomEntry(item.customType, item.data);
    }
    manager.appendMessage(assistant(call('u1', 'unfold', { handle: plan.handle })));
    const uPlan = planUnfold(
      manager.getBranch(),
      manager.buildSessionProjection().entries,
      plan.handle,
      { callerId: 'u1' }
    );
    for (const edit of uPlan.edits) manager.appendContextEdit(edit.targetId, edit.replacement);
    const byId = new Map(
      manager.buildSessionProjection().entries.map((e) => [e.sourceEntry.id, e])
    );
    // External removals on ids[1..2] must remain omitted, while episodes-owned ids[3..4] are restored!
    assert.equal(byId.get(ids[1]!)!.messages.length, 0);
    assert.equal(byId.get(ids[2]!)!.messages.length, 0);
    assert.equal(byId.get(ids[3]!)!.messages.length, 1);
    assert.equal(byId.get(ids[4]!)!.messages.length, 1);
  }
});

test('synthetic projection chains multiple pending and repairDrafts entries without losing earlier edits', () => {
  const dupBody = 'export const item = 99;\n'.repeat(30);
  const hugeBad = 'SYNTHETIC_CHAIN_ERR_'.repeat(40);
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('bad', 'bash', { command: `false # ${hugeBad}` })),
    Object.assign(result('bad', 'bash', 'err', 3), { isError: true }),
    assistant(call('r1', 'read', { path: 'a.ts' })),
    result('r1', 'read', dupBody, 4),
    assistant(call('t2', 'bash', { command: 'echo 2' })),
    result('t2', 'bash', '2', 5),
    assistant(call('r2', 'read', { path: 'a.ts' })),
    result('r2', 'read', dupBody, 6),
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
  ]);

  const branch = manager.getBranch();
  // 1. Multiple pending edits: first removes r2 (ids[10]), second edits t2 result (ids[8]).
  //    Without chaining parentIds, the second pending edit orphans the first!
  const withTwoPending = planFoldTailPruning({
    branch,
    pending: [
      { type: 'context_edit', targetId: ids[10]!, replacement: null },
      {
        type: 'context_edit',
        targetId: ids[8]!,
        replacement: { content: withStamp([{ type: 'text', text: 'patched' }], 5) }
      }
    ],
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[2]!,
    beforeEntryId: ids[11]!,
    cwd: '/repo'
  });
  assert.deepEqual(withTwoPending.readEdits, []);

  // 2. Multiple repairDrafts: first omits assistant t2 (ids[7]) and result (ids[8]), dropping surviving
  //    assistant turns after 'bad' (ids[3]) from 3 down to 2 -> 'bad' must NOT be error-pruned!
  const withTwoRepairs = planFoldTailPruning({
    branch,
    repairDrafts: [
      { type: 'context_edit', targetId: ids[7]!, replacement: null },
      { type: 'context_edit', targetId: ids[8]!, replacement: null }
    ],
    editOwners: attributeEpisodesEdits(branch),
    afterEntryId: ids[2]!,
    beforeEntryId: ids[11]!,
    cwd: '/repo'
  });
  assert.equal(withTwoRepairs.assistantReplacements.has(ids[3]!), false);
});

test('payload coverage rejects smaller capped preview with longer archive path, handles @path, and skips duplicate toolCallIds', async () => {
  const base = await mkdtemp(join(tmpdir(), 'pi-ep-cov-'));
  const shortDir = join(base, 's');
  const longDir = join(
    base,
    ...Array.from({ length: 8 }, () => 'very-long-archive-directory-segment')
  );
  await mkdir(shortDir, { recursive: true });
  await mkdir(longDir, { recursive: true });
  try {
    const fullText = Array.from(
      { length: 160 },
      (_, i) => `row ${String(i).padStart(3, '0')} content`
    ).join('\n');
    const cutMoreShortPath = await capOutput(
      [{ type: 'text', text: fullText }],
      shortDir,
      500,
      320
    );
    const cutLessLongPath = await capOutput([{ type: 'text', text: fullText }], longDir, 500, 180);
    assert.ok(cutMoreShortPath && cutLessLongPath);
    // Confirm the longer archive path makes the smaller-payload preview string LONGER overall!
    assert.ok(
      (cutLessLongPath.content[0] as TextContent).text.length >
        (cutMoreShortPath.content[0] as TextContent).text.length
    );

    // 1. Older has MORE payload (320 keep) in shortDir; newer has LESS payload (180 keep) in longDir -> MUST NOT stub!
    {
      const { manager, ids } = session([
        user('go', 1),
        assistant(call('s1', 'bash', { command: 'echo 1' })),
        result('s1', 'bash', '1', 2),
        assistant(call('r1', 'read', { path: 'file.ts' })),
        {
          role: 'toolResult' as const,
          toolCallId: 'r1',
          toolName: 'read',
          content: withStamp(cutMoreShortPath.content, 3),
          details: { [CAPPED_KEY]: { ...cutMoreShortPath.capped } },
          isError: false,
          timestamp: 0
        },
        assistant(call('r2', 'read', { path: 'file.ts' })),
        {
          role: 'toolResult' as const,
          toolCallId: 'r2',
          toolName: 'read',
          content: withStamp(cutLessLongPath.content, 4),
          details: { [CAPPED_KEY]: { ...cutLessLongPath.capped } },
          isError: false,
          timestamp: 0
        },
        assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
      ]);
      const branch = manager.getBranch();
      const plan = planFoldTailPruning({
        branch,
        editOwners: attributeEpisodesEdits(branch),
        afterEntryId: ids[2]!,
        beforeEntryId: ids.at(-1)!,
        cwd: base
      });
      assert.deepEqual(plan.readEdits, []);
    }

    // 2. Reverse order: older has LESS payload (180 keep, longDir) and newer has MORE payload (320 keep, shortDir) -> IS stubbed!
    //    Also verifies @path normalization and skipping repeated toolCallIds.
    {
      const hugeBad = 'REPEATED_ID_PAYLOAD_'.repeat(40);
      const { manager, ids } = session([
        user('go', 1),
        assistant(call('s1', 'bash', { command: 'echo 1' })),
        result('s1', 'bash', '1', 2),
        assistant(
          {
            type: 'toolCall',
            id: 'r_old',
            name: 'read',
            arguments: { path: '@file.ts', offset: 1 }
          },
          call('dup', 'bash', { command: `echo ok # ${hugeBad}` })
        ),
        {
          role: 'toolResult' as const,
          toolCallId: 'r_old',
          toolName: 'read',
          content: withStamp(cutLessLongPath.content, 3),
          details: { [CAPPED_KEY]: { ...cutLessLongPath.capped } },
          isError: false,
          timestamp: 0
        },
        result('dup', 'bash', 'ok', 4),
        assistant(call('dup', 'bash', { command: 'false' })),
        Object.assign(result('dup', 'bash', 'err', 5), { isError: true }),
        assistant(call('t2', 'bash', { command: 'echo 2' })),
        result('t2', 'bash', '2', 6),
        assistant(call('t3', 'bash', { command: 'echo 3' })),
        result('t3', 'bash', '3', 7),
        assistant(call('r_new', 'read', { path: 'file.ts' })),
        {
          role: 'toolResult' as const,
          toolCallId: 'r_new',
          toolName: 'read',
          content: withStamp(cutMoreShortPath.content, 8),
          details: { [CAPPED_KEY]: { ...cutMoreShortPath.capped } },
          isError: false,
          timestamp: 0
        },
        assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Done.' }))
      ]);
      const branch = manager.getBranch();
      const plan = planFoldTailPruning({
        branch,
        editOwners: attributeEpisodesEdits(branch),
        afterEntryId: ids[2]!,
        beforeEntryId: ids.at(-1)!,
        cwd: base
      });
      assert.deepEqual(
        plan.readEdits.map((e) => e.targetId),
        [ids[4]]
      );
      assert.equal(plan.assistantReplacements.has(ids[3]!), false);
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test('later fold removing only newer read preserves prefix bytes and advertises recall handle+turn that retrieves original file after disk change', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-ep-redirect-'));
  try {
    const originalFile = Array.from({ length: 40 }, (_, i) => `orig_cfg_${i} = "v1";`).join('\n');
    await writeFile(join(dir, 'config.ts'), `${originalFile}\n`);

    let advertisedHandle = '';
    let advertisedTurn = '';
    const faux = fauxProvider({ models: [{ id: 'faux-redir', reasoning: true }] });
    faux.setResponses([
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo phase1' })),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'config.ts' })),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo mid' })),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'config.ts' })),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', until: 't2', note: 'Closed phase1.' })
        ),
      async () => {
        await writeFile(join(dir, 'config.ts'), 'CHANGED_ON_DISK = true;\n');
        return fauxAssistantMessage(
          fauxToolCall('fold', { since: 't4', note: 'Folded mid and second read.' })
        );
      },
      (context) => {
        const noteText = text(context.messages.at(-1)!);
        const match = /recall\(\{ handle: "(ep-[0-9a-f]{8})", turn: "(t\d+)" \}\)/.exec(noteText);
        assert.ok(match, `expected redirect locator in fold note, got: ${noteText}`);
        advertisedHandle = match[1]!;
        advertisedTurn = match[2]!;
        return fauxAssistantMessage(
          fauxToolCall('recall', { handle: advertisedHandle, turn: advertisedTurn })
        );
      },
      (context) => {
        const recalled = text(context.messages.at(-1)!);
        assert.match(recalled, /orig_cfg_25 = "v1";/);
        assert.doesNotMatch(recalled, /CHANGED_ON_DISK/);
        return fauxAssistantMessage('verified');
      }
    ]);

    const sessionDir = join(dir, 'sessions');
    await mkdir(sessionDir, { recursive: true });
    const manager = SessionManager.create(dir, sessionDir);
    const settings = SettingsManager.inMemory();
    const loader = new DefaultResourceLoader({
      cwd: dir,
      agentDir: dir,
      noSkills: true,
      settingsManager: settings,
      extensionFactories: [(pi) => pi.registerProvider(faux.provider), episodes]
    });
    await loader.reload();
    const created = await createAgentSession({
      cwd: dir,
      agentDir: dir,
      model: faux.getModel(),
      thinkingLevel: 'off',
      modelRuntime: await ModelRuntime.create({
        modelsPath: null,
        authPath: join(dir, 'auth.json')
      }),
      resourceLoader: loader,
      tools: ['bash', 'read', 'write', 'fold', 'unfold', 'recall', 'ask', 'fork'],
      sessionManager: manager,
      settingsManager: settings
    });
    try {
      await created.session.prompt('run');
      const branch = manager.getBranch();
      const olderReadEntry = branch.find(
        (e) =>
          e.type === 'message' && e.message.role === 'toolResult' && e.message.toolName === 'read'
      )!;
      const editsOnOlder = branch.filter(
        (e) => e.type === 'context_edit' && e.targetId === olderReadEntry.id
      );
      assert.equal(editsOnOlder.length, 1);
      const lastAsst = [...branch]
        .reverse()
        .find((e) => e.type === 'message' && e.message.role === 'assistant');
      assert.equal(
        lastAsst?.type === 'message' && lastAsst.message.role === 'assistant'
          ? lastAsst.message.errorMessage
          : undefined,
        undefined
      );
      assert.equal(faux.getPendingResponseCount(), 0);
    } finally {
      await created.session.dispose();
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('same-batch laterRemovals + episodeStats replay across omit, unfold restore, re-prune, and recovery rollback', () => {
  const fileText = 'const entry = 12345;\n'.repeat(40);
  const { manager, ids } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('r1', 'read', { path: 'lib.ts' })),
    result('r1', 'read', fileText, 3),
    assistant(call('r2', 'read', { path: 'lib.ts' })),
    result('r2', 'read', fileText, 4),
    assistant(call('f1', 'fold', { since: 't2', until: 't3', note: 'FoldA.' }))
  ]);

  // 1. Fold A omits r1 (ids[4] -> replacement: null), then Unfold A restores r1 ({ content }).
  const planA = planFold(manager.buildSessionProjection().entries, 'f1', 2, { until: 3 });
  manager.appendMessage(
    Object.assign(result('f1', 'fold', `[fold ${planA.handle}]\nFoldA.`, 5), {
      details: {
        kind: 'fold',
        handle: planA.handle,
        since: 't2',
        until: 't3',
        firstEntryId: planA.firstEntryId,
        lastEntryId: planA.lastEntryId,
        messages: planA.omitted.length,
        tokens: 220,
        noteTokens: 4,
        keptTokens: 0,
        headline: 'FoldA.',
        keptUserTurns: [],
        files: []
      }
    })
  );
  manager.appendCustomEntry(FOLD_STARTED_TYPE, { handle: planA.handle, edits: planA.edits });
  for (const edit of planA.edits) manager.appendContextEdit(edit.targetId, edit.replacement);
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: planA.handle });

  manager.appendMessage(assistant(call('uA', 'unfold', { handle: planA.handle })));
  const uPlanA = planUnfold(
    manager.getBranch(),
    manager.buildSessionProjection().entries,
    planA.handle,
    { callerId: 'uA' }
  );
  manager.appendMessage(result('uA', 'unfold', '[unfolded]', 6));
  manager.appendCustomEntry(FOLD_UNFOLD_STARTED_TYPE, {
    handle: planA.handle,
    edits: uPlanA.edits
  });
  for (const edit of uPlanA.edits) manager.appendContextEdit(edit.targetId, edit.replacement);
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: planA.handle });

  // 2. Fold B folds only t2 (since: 't2', until: 't2'), tail-pruning the restored r1 (ids[4]) against r2 (ids[6])!
  manager.appendMessage(
    assistant(call('f2', 'fold', { since: 't2', until: 't2', note: 'FoldB.' }))
  );
  const f2CallerId = manager.getBranch().at(-1)!.id;
  const planB = planFold(manager.buildSessionProjection().entries, 'f2', 2, { until: 2 });
  manager.appendMessage(
    Object.assign(result('f2', 'fold', `[fold ${planB.handle}]\nFoldB.`, 7), {
      details: {
        kind: 'fold',
        handle: planB.handle,
        since: 't2',
        until: 't2',
        firstEntryId: planB.firstEntryId,
        lastEntryId: planB.lastEntryId,
        messages: planB.omitted.length,
        tokens: 8,
        noteTokens: 4,
        keptTokens: 0,
        headline: 'FoldB.',
        keptUserTurns: [],
        files: []
      }
    })
  );
  const branchB = manager.getBranch();
  const tailB = planFoldTailPruning({
    branch: branchB,
    editOwners: attributeEpisodesEdits(branchB),
    afterEntryId: planB.lastEntryId,
    beforeEntryId: f2CallerId,
    cwd: '/repo'
  });
  assert.equal(tailB.readEdits.length, 1);

  // Verify same-batch laterRemovals skips stubbing r1 when r2 (ids[6]) is removed.
  manager.appendCustomEntry(FOLD_STARTED_TYPE, {
    handle: planB.handle,
    edits: [...planB.edits, ...tailB.readEdits],
    readDeps: tailB.readDeps
  });
  const skipped = completeCutShortFolds(manager.getBranch(), [], new Set([ids[6]!]));
  assert.equal(
    skipped.some((item) => item.type === 'context_edit' && item.targetId === ids[4]),
    false
  );

  // Apply Fold B + tail stub: because r1 was previously null then restored by Unfold A,
  // episodeStats must reconstruct r1 properly and credit Fold B with ~170+ tail tokens!
  for (const edit of [...planB.edits, ...tailB.readEdits]) {
    manager.appendContextEdit(edit.targetId, edit.replacement);
  }
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: planB.handle });
  const req = assistant('next response');
  req.usage = {
    input: 500,
    output: 20,
    cacheRead: 1000,
    cacheWrite: 0,
    totalTokens: 1520,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
  };
  manager.appendMessage(req);
  const statsB = episodeStats(manager.getBranch(), '/repo');
  assert.ok(
    statsB.folds.each[1]!.net > 100,
    `expected Fold B net > 100 after omit->unfold->prune, got ${statsB.folds.each[1]!.net}`
  );

  // 3. Recovery rollback nets tail savings back to 0 without double counting.
  const { manager: mRoll, ids: idsRoll } = session([
    user('go', 1),
    assistant(call('s1', 'bash', { command: 'echo 1' })),
    result('s1', 'bash', '1', 2),
    assistant(call('r1', 'read', { path: 'lib.ts' })),
    result('r1', 'read', fileText, 3),
    assistant(call('r2', 'read', { path: 'lib.ts' })),
    result('r2', 'read', fileText, 4),
    assistant(call('f1', 'fold', { since: 't2', until: 't2', note: 'Roll.' }))
  ]);
  const pRoll = planFold(mRoll.buildSessionProjection().entries, 'f1', 2, { until: 2 });
  mRoll.appendMessage(
    Object.assign(result('f1', 'fold', `[fold ${pRoll.handle}]\nRoll.`, 5), {
      details: {
        kind: 'fold',
        handle: pRoll.handle,
        since: 't2',
        until: 't2',
        firstEntryId: pRoll.firstEntryId,
        lastEntryId: pRoll.lastEntryId,
        messages: pRoll.omitted.length,
        tokens: 8,
        noteTokens: 4,
        keptTokens: 0,
        headline: 'Roll.',
        keptUserTurns: [],
        files: []
      }
    })
  );
  const bRoll = mRoll.getBranch();
  const tRoll = planFoldTailPruning({
    branch: bRoll,
    editOwners: attributeEpisodesEdits(bRoll),
    afterEntryId: pRoll.lastEntryId,
    beforeEntryId: idsRoll[7]!,
    cwd: '/repo'
  });
  const batchRoll = [...pRoll.edits, ...tRoll.readEdits];
  mRoll.appendCustomEntry(FOLD_STARTED_TYPE, {
    handle: pRoll.handle,
    edits: batchRoll,
    readDeps: tRoll.readDeps
  });
  for (const edit of batchRoll) mRoll.appendContextEdit(edit.targetId, edit.replacement);
  // External hook removes r2 before completion -> recovery rolls back r1 stub!
  mRoll.appendContextEdit(idsRoll[6]!, null);
  for (const item of completeCutShortFolds(mRoll.getBranch())) {
    if (item.type === 'context_edit') mRoll.appendContextEdit(item.targetId, item.replacement);
    else mRoll.appendCustomEntry(item.customType, item.data);
  }
  mRoll.appendMessage(req);
  const statsRoll = episodeStats(mRoll.getBranch(), '/repo');
  assert.equal(statsRoll.folds.each[0]!.net, 4);
});
