import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  getCurrentTools,
  type Message,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import {
  createAgentSession,
  createCodemodeExtension,
  DefaultResourceLoader,
  type ExtensionFactory,
  ModelRuntime,
  SessionManager,
  SettingsManager
} from '@earendil-works/pi-coding-agent';
import {
  appliedFolds,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  threadRecords
} from '../src/archive.js';
import { cacheAnchor } from '../src/cache.js';
import episodes from '../src/index.js';
import { episodeStats } from '../src/stats.js';

type Step = (context: TranscriptContext) => AssistantMessage | Promise<AssistantMessage>;
type Mode = 'off' | 'on' | 'only';

function text(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  return message.content
    .map((block) =>
      block.type === 'text' ? block.text : block.type === 'toolCall' ? block.name : ''
    )
    .join('\n');
}

function conversation(context: TranscriptContext): Message[] {
  return context.messages.filter((message) => message.role !== 'system');
}

function declared(context: TranscriptContext): string[] {
  return getCurrentTools(context.messages)
    .map((tool) => tool.name)
    .sort();
}

/** A response that runs `code` as a codemode script. */
function script(code: string, id?: string): AssistantMessage {
  return fauxAssistantMessage(fauxToolCall('codemode', { code }, id ? { id } : undefined), {
    stopReason: 'toolUse'
  });
}

async function createSession(mode: Mode, steps: Step[], extensions: ExtensionFactory[] = []) {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-codemode-'));
  const faux = fauxProvider();
  const requests: TranscriptContext[] = [];
  const failures: unknown[] = [];
  faux.setResponses(
    steps.map((step) => (context) => {
      requests.push(context);
      try {
        return step(context);
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
      ...(mode === 'off' ? [] : [createCodemodeExtension({ models: false })]),
      episodes,
      ...extensions
    ]
  });
  await resourceLoader.reload();

  const { session } = await createAgentSession({
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
  await session.bindExtensions({});

  return {
    dir,
    session,
    requests,
    async run(...prompts: string[]) {
      for (const prompt of prompts) {
        await session.prompt(prompt);
        // A command such as /fold starts its run without waiting for it.
        await session.waitForIdle();
      }
      if (failures[0]) throw failures[0];
      assert.equal(faux.getPendingResponseCount(), 0);
    },
    async cleanup() {
      session.dispose();
      await rm(dir, { recursive: true, force: true });
    }
  };
}

test('fold and fork stay declared next to codemode; the other tools follow the mode', async () => {
  const expected: Record<Mode, string[]> = {
    off: ['ask', 'bash', 'fold', 'fork', 'read', 'recall'],
    on: ['ask', 'bash', 'codemode', 'fold', 'fork', 'read', 'recall'],
    // fold is reached from scripts; fork cannot be, so it stays declared.
    only: ['codemode', 'fork']
  };
  for (const mode of ['off', 'on', 'only'] as const) {
    const harness = await createSession(mode, [() => fauxAssistantMessage('ok')]);
    try {
      await harness.run('hi');
      assert.deepEqual(declared(harness.requests[0]!), expected[mode], mode);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a fold from a codemode script folds the span and leaves the script result as its note', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo error TS2322 in api.ts" }));'),
    () =>
      script(
        'text(await tools.fold({ since: "t1", note: "Build fails with TS2322: fixed by regenerating the client." }));'
      ),
    (context) => {
      const messages = conversation(context);
      assert.deepEqual(
        messages.map((message) => message.role),
        ['user', 'assistant', 'toolResult']
      );
      const note = text(messages[2]!);
      assert.match(note, /\[fold ep-[0-9a-f]{8}: 2 messages, t1-t2,/);
      assert.match(note, /fixed by regenerating the client/);
      // Printed by the script once; not appended a second time.
      assert.equal(note.match(/\[fold ep-/g)?.length, 1);
      assert.match(note, /\[t3\]$/);
      assert.doesNotMatch(note, /error TS2322 in api\.ts/);
      return fauxAssistantMessage('Folded.');
    },
    // recall reaches the archived span from a script too.
    () => script('text(await tools.recall({ query: "TS2322" }));'),
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /toolResult: .*error TS2322 in api\.ts/);
      return fauxAssistantMessage('Found it.');
    }
  ]);
  try {
    await harness.run('make it build', 'look back');
    const branch = harness.session.sessionManager.getBranch();
    const records = threadRecords(branch);
    assert.equal(records.length, 1);
    assert.equal(records[0]!.kind, 'fold');
    assert.ok(
      branch.some((entry) => entry.type === 'custom' && entry.customType === FOLD_APPLIED_TYPE)
    );
    const note = branch.find(
      (entry) =>
        entry.type === 'message' &&
        entry.message.role === 'toolResult' &&
        entry.message.toolName === 'codemode' &&
        (entry.message.details as Record<string, unknown> | undefined)?.[FOLD_DETAILS_KEY]
    );
    assert.ok(note, 'the script result carries the fold details');
    assert.equal(episodeStats(branch, harness.dir).folds.count, 1);
  } finally {
    await harness.cleanup();
  }
});
test('a script that does not print the fold result still leaves the note in context', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo long output" }));'),
    () =>
      script(
        'await tools.fold({ since: "t1", note: "Ran the command; nothing to keep." }); text("done");'
      ),
    (context) => {
      const note = text(conversation(context).at(-1)!);
      assert.match(note, /Output:\n\ndone\n/);
      assert.match(
        note,
        /\[fold ep-[0-9a-f]{8}: 2 messages, t1-t2,[\s\S]*Ran the command; nothing to keep\./
      );
      assert.match(note, /\[t3\]$/);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('a fold from a script that then fails does not take effect', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo kept" }));'),
    () =>
      script('await tools.fold({ since: "t1", note: "n" }); throw new Error("later step broke");'),
    (context) => {
      const messages = conversation(context);
      assert.equal(messages.length, 5);
      assert.match(text(messages[2]!), /kept/);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
    assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 0);
  } finally {
    await harness.cleanup();
  }
});

test('one fold per response, across the folds of a script', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () =>
      script(
        [
          'const a = await tools.fold({ since: "t1", note: "first" });',
          'let b; try { b = await tools.fold({ since: "t1", note: "second" }); } catch (e) { b = String(e); }',
          'text(a); text(b);'
        ].join('\n')
      ),
    (context) => {
      const note = text(conversation(context).at(-1)!);
      assert.match(note, /Only one fold per response/);
      assert.equal(note.match(/\[fold ep-/g)?.length, 1);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
    assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a later fold cannot reach behind a note a script left', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "first" }));'),
    () => fauxAssistantMessage('next'),
    () => script('text(await tools.bash({ command: "echo two" }));'),
    () =>
      script(
        'let r; try { r = await tools.fold({ since: "t1", note: "again" }); } catch (e) { r = String(e); } text(r);'
      ),
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        /A fold cannot reach behind an earlier fold note\. The earliest stamp you can fold from is t4\./
      );
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go', 'more');
  } finally {
    await harness.cleanup();
  }
});

test('fork is not callable from a script, and a fork under "only" gets the direct tools', async () => {
  const harness = await createSession('only', [
    () =>
      script(
        'let r; try { r = await tools.fork({ task: "x" }); } catch (e) { r = String(e); } text(r);'
      ),
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /tools\.fork does not exist/);
      return fauxAssistantMessage(fauxToolCall('fork', { task: 'Check it.' }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      // The child's view is the parent's transcript before Pi hides declarations.
      assert.ok(declared(context).includes('bash'));
      assert.ok(declared(context).includes('fold'));
      return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo child ok' }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /child ok/);
      return fauxAssistantMessage('Changed: nothing. Learned: it works.');
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /^\[fork fk-[0-9a-f]{8} finished/);
      return fauxAssistantMessage('done');
    }
  ]);
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('files a script read are in the fold, and reading one again cues the thread', async () => {
  let dir = '';
  const harness = await createSession('only', [
    () => script('text(await tools.read({ path: "api.ts" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "Read api.ts; it exports run." }));'),
    () => fauxAssistantMessage('noted'),
    () => script('text(await tools.read({ path: "api.ts" }));'),
    (context) => {
      const result = text(conversation(context).at(-1)!);
      assert.match(result, /\[episodes\] api\.ts was also touched in folded span ep-[0-9a-f]{8}/);
      assert.equal(result.match(/\[episodes\]/g)?.length, 1);
      return fauxAssistantMessage('ok');
    }
  ]);
  dir = harness.dir;
  await writeFile(join(dir, 'api.ts'), 'export function run() {}\n');
  try {
    await harness.run('read it', 'again');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fold');
    assert.deepEqual(record.files, [join(dir, 'api.ts')]);
    assert.deepEqual(record.readOnly, [join(dir, 'api.ts')]);
  } finally {
    await harness.cleanup();
  }
});

test('/fold under "only" asks for a fold from a script', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () => fauxAssistantMessage('done'),
    (context) => {
      const request = text(conversation(context).at(-1)!);
      assert.match(
        request,
        /Use since "t1", write the note, and make one codemode call whose script calls only fold/
      );
      return script('text(await tools.fold({ since: "t1", note: "Ran one command." }));');
    },
    // Codemode does not pass the fold's terminate on, so the run continues once.
    () => fauxAssistantMessage('Folded.')
  ]);
  try {
    await harness.run('go', '/fold t1');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fold');
    assert.equal(record.manual, 'since');
  } finally {
    await harness.cleanup();
  }
});
test('two scripts in one response: only the first fold takes effect', async () => {
  const fold = (note: string) =>
    `let r; try { r = await tools.fold({ since: "t1", note: "${note}" }); } catch (e) { r = String(e); } text(r);`;
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () =>
      fauxAssistantMessage(
        [
          fauxToolCall('codemode', { code: fold('first') }, { id: 'cm-a' }),
          fauxToolCall('codemode', { code: fold('second') }, { id: 'cm-b' })
        ],
        { stopReason: 'toolUse' }
      ),
    (context) => {
      const all = conversation(context).map(text).join('\n');
      assert.match(all, /Only one fold per response/);
      assert.equal(all.match(/\[fold ep-/g)?.length, 1);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 1);
    assert.equal(
      branch.filter((entry) => entry.type === 'custom' && entry.customType === FOLD_APPLIED_TYPE)
        .length,
      1
    );
  } finally {
    await harness.cleanup();
  }
});

test('a fold is not applied when a later hook rewrites its note in the script result', async () => {
  const harness = await createSession(
    'only',
    [
      () => script('text(await tools.bash({ command: "echo kept" }));'),
      () => script('text(await tools.fold({ since: "t1", note: "Secret body." }));'),
      (context) => {
        // The header survives the rewrite but the note does not, so the span stays and the
        // result is no anchor.
        assert.match(text(conversation(context).at(-1)!), /\[fold ep-[0-9a-f]{8}:[\s\S]*redacted/);
        assert.equal(conversation(context).length, 5);
        const applied = appliedFolds(harness.session.sessionManager.getBranch());
        assert.deepEqual(cacheAnchor(conversation(context), applied), { stamp: '[t1]' });
        return fauxAssistantMessage('ok');
      }
    ],
    [
      (pi) => {
        pi.on('tool_result', (event) =>
          event.toolName === 'codemode' && !event.parentToolCallId
            ? {
                content: event.content.map((block) =>
                  block.type === 'text'
                    ? { ...block, text: block.text.replace('Secret body.', 'redacted') }
                    : block
                )
              }
            : undefined
        );
      }
    ]
  );
  try {
    await harness.run('go');
    assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 0);
  } finally {
    await harness.cleanup();
  }
});

test('with codemode "on", a top-level fold still works', async () => {
  const harness = await createSession('on', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () =>
      fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Ran one.' }), {
        stopReason: 'toolUse'
      }),
    (context) => {
      const messages = conversation(context);
      assert.equal(messages.length, 3);
      assert.match(text(messages[2]!), /\[fold ep-[0-9a-f]{8}: 2 messages/);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('ask reaches a span folded from a script, from a script', async () => {
  let handle = '';
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo error TS2322" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "Build failed." }));'),
    (context) => {
      handle = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
      return fauxAssistantMessage('Folded.');
    },
    () =>
      script(`text(await tools.ask({ handle: "${handle}", question: "Which error was it?" }));`),
    (context) => {
      // The ask request resumes the archived thread.
      assert.match(text(conversation(context).at(-1)!), /Which error was it\?/);
      assert.ok(conversation(context).some((message) => /error TS2322/.test(text(message))));
      return fauxAssistantMessage('TS2322.');
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /answers\]\nTS2322\./);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go', 'which error?');
  } finally {
    await harness.cleanup();
  }
});

test('a fork under "only" can fold its own work with a direct fold call', async () => {
  const harness = await createSession('only', [
    () =>
      fauxAssistantMessage(fauxToolCall('fork', { task: 'Check it.' }), { stopReason: 'toolUse' }),
    () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: 'echo noisy' }), {
        stopReason: 'toolUse'
      }),
    (context) => {
      const stamp = /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!;
      return fauxAssistantMessage(fauxToolCall('fold', { since: stamp, note: 'Ran it.' }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      const last = text(conversation(context).at(-1)!);
      assert.match(last, /\[fold ep-[0-9a-f]{8}:/);
      assert.ok(!conversation(context).some((message) => /^noisy/m.test(text(message))));
      return fauxAssistantMessage('Changed: nothing. Learned: it works.');
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /^\[fork fk-[0-9a-f]{8} finished/);
      return fauxAssistantMessage('done');
    }
  ]);
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});
test('a fold that a result hook failed does not take effect, even if the script catches it', async () => {
  const harness = await createSession(
    'only',
    [
      () => script('text(await tools.bash({ command: "echo kept" }));'),
      () =>
        script(
          'let r; try { r = await tools.fold({ since: "t1", note: "n" }); } catch (e) { r = String(e); } text(r);'
        ),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /Denied by result hook/);
        assert.equal(conversation(context).length, 5);
        return fauxAssistantMessage('ok');
      }
    ],
    [
      (pi) => {
        pi.on('tool_result', (event) =>
          event.toolName === 'fold'
            ? { content: [{ type: 'text', text: 'Denied by result hook' }], isError: true }
            : undefined
        );
      }
    ]
  );
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 0);
    assert.ok(
      !branch.some((entry) => entry.type === 'custom' && entry.customType === FOLD_APPLIED_TYPE)
    );
  } finally {
    await harness.cleanup();
  }
});
