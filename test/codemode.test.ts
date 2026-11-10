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
  type ImageContent,
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
import { foldableTurns } from '../src/fold.js';
import episodes from '../src/index.js';
import { nudgeGrowth, nudgeThreshold } from '../src/nudge.js';
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

async function createSession(
  mode: Mode,
  steps: Step[],
  extensions: ExtensionFactory[] = [],
  /** The `episodes` settings. */
  settings: Record<string, unknown> = {},
  compaction: Record<string, unknown> = {},
  /** Entries to resume from, as after a restart. */
  entries?: Parameters<typeof SessionManager.inMemory>[2],
  /** Extensions whose handlers run before those of episodes. */
  before: ExtensionFactory[] = []
) {
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
      ...before,
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
    sessionManager: SessionManager.inMemory(dir, undefined, entries),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false, ...compaction },
      // Pi's settings type has no slot for extension keys.
      ...({ episodes: settings } as object),
      ...(mode === 'off' ? {} : { codemode: { mode } })
    })
  });
  await session.bindExtensions({});

  return {
    dir,
    session,
    requests,
    async run(...prompts: (string | { text: string; images: ImageContent[] })[]) {
      for (const prompt of prompts) {
        if (typeof prompt === 'string') await session.prompt(prompt);
        else await session.prompt(prompt.text, { images: prompt.images });
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
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 1);
    // The script's result stands for the first fold, not a retry of the rejected one.
    const folds = episodeStats(branch, harness.dir).folds;
    assert.equal(folds.rejected, 1);
    assert.equal(folds.retriedOk, 0);
  } finally {
    await harness.cleanup();
  }
});

test('a later fold covers a note a script left, and both spans stay recallable', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo first-output" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "first" }));'),
    () => fauxAssistantMessage('next'),
    () => script('text(await tools.bash({ command: "echo two" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "again" }));'),
    (context) => {
      const view = conversation(context);
      assert.match(
        text(view.at(-1)!),
        /covers the notes of earlier folds; .*ep-[0-9a-f]+ \(first\)/
      );
      assert.doesNotMatch(view.slice(0, -1).map(text).join('\n'), /\bfirst\b|echo two/);
      return script('text(await tools.recall({ query: "first-output" }));');
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /first-output/);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go', 'more');
    const records = threadRecords(harness.session.sessionManager.getBranch());
    assert.equal(records.length, 2);
    assert.deepEqual((records[1] as { covers?: string[] }).covers, [records[0]!.handle]);
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
    // The sibling script's fold is in the same response, not a retry.
    assert.equal(episodeStats(branch, harness.dir).folds.retriedOk, 0);
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
test('a fold that a result hook rewrote is not added again next to what the script printed', async () => {
  const harness = await createSession(
    'only',
    [
      () => script('text(await tools.bash({ command: "echo one" }));'),
      () => script('text(await tools.fold({ since: "t1", note: "Ran one." }));'),
      (context) => {
        const note = text(conversation(context).at(-1)!);
        assert.equal(conversation(context).length, 3);
        assert.match(note, /Ran one, checked\./);
        assert.equal(note.match(/\[fold ep-/g)?.length, 1);
        return fauxAssistantMessage('ok');
      }
    ],
    [
      (pi) => {
        pi.on('tool_result', (event) =>
          event.toolName === 'fold' && !event.isError
            ? {
                content: event.content.map((block) =>
                  block.type === 'text'
                    ? { ...block, text: block.text.replace('Ran one.', 'Ran one, checked.') }
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
    assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 1);
  } finally {
    await harness.cleanup();
  }
});

test('the first fold of a script holds the response even when it is rejected', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () =>
      script(
        [
          'let a; try { a = await tools.fold({ since: "t99", note: "x" }); } catch (e) { a = String(e); }',
          'let b; try { b = await tools.fold({ since: "t1", note: "y" }); } catch (e) { b = String(e); }',
          'text(a); text(b);'
        ].join('\n')
      ),
    (context) => {
      const result = text(conversation(context).at(-1)!);
      assert.match(result, /t99/);
      assert.match(result, /Only one fold per response/);
      assert.equal(conversation(context).length, 5);
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

test('a retry counts once, after rejections from scripts in a row', async () => {
  const reject =
    'let r; try { r = await tools.fold({ since: "t99", note: "x" }); } catch (e) { r = String(e); } text(r);';
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo one" }));'),
    () => script(reject),
    () => script(reject),
    () => script('text(await tools.fold({ since: "t1", note: "Ran one." }));'),
    () => fauxAssistantMessage('ok')
  ]);
  try {
    await harness.run('go');
    const folds = episodeStats(harness.session.sessionManager.getBranch(), harness.dir).folds;
    assert.equal(folds.count, 1);
    assert.equal(folds.rejected, 2);
    assert.equal(folds.retriedOk, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a fork child reading a folded file does not use up the parent cue', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.read({ path: "api.ts" }));'),
    () => script('text(await tools.fold({ since: "t1", note: "Read api.ts." }));'),
    () => fauxAssistantMessage(fauxToolCall('fork', { task: 'Look.' }), { stopReason: 'toolUse' }),
    () => fauxAssistantMessage(fauxToolCall('read', { path: 'api.ts' }), { stopReason: 'toolUse' }),
    () => fauxAssistantMessage('Changed: nothing. Learned: fine.'),
    (context) => {
      assert.doesNotMatch(text(conversation(context).at(-1)!), /\[episodes\]/);
      return script('text(await tools.read({ path: "api.ts" }));');
    },
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /\[episodes\] api\.ts was also touched/);
      return fauxAssistantMessage('ok');
    }
  ]);
  await writeFile(join(harness.dir, 'api.ts'), 'export {};\n');
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});
test('a fold whose text a result hook emptied does not take effect', async () => {
  for (const replacement of [[], [{ type: 'text' as const, text: ' ' }]]) {
    const harness = await createSession(
      'only',
      [
        () => script('text(await tools.bash({ command: "echo kept" }));'),
        () => script('text(await tools.fold({ since: "t1", note: "n" }));'),
        (context) => {
          assert.equal(conversation(context).length, 5);
          return fauxAssistantMessage('ok');
        }
      ],
      [
        (pi) => {
          pi.on('tool_result', (event) =>
            event.toolName === 'fold' ? { content: replacement } : undefined
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
  }
});
test('a nudge on the next prompt leads to a fold from the earliest stamp', async () => {
  for (const mode of ['off', 'only'] as const) {
    const big = 'x'.repeat(800);
    const call = (tool: string, args: Record<string, string>) =>
      mode === 'only'
        ? script(`text(await tools.${tool}(${JSON.stringify(args)}));`)
        : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
    const done = () => fauxAssistantMessage('done');
    const harness = await createSession(
      mode,
      [
        () => call('bash', { command: `echo ${big}` }),
        done,
        (context) => {
          const nudge = text(conversation(context).at(-1)!);
          assert.match(nudge, /tokens of work built up/, mode);
          assert.match(
            nudge,
            mode === 'only'
              ? /make one codemode call whose script calls only fold/
              : /call only fold in this response/
          );
          const since = /since "(t\d+)"/.exec(nudge)![1]!;
          assert.equal(since, 't1');
          return call('fold', { since, note: 'Printed a long line.' });
        },
        (context) => {
          // The first request is folded; the user messages and the nudge's request remain.
          assert.doesNotMatch(conversation(context).map(text).join('\n'), /xxxx/);
          return done();
        },
        done
      ],
      [],
      // About 400 tokens build up in the first request: over the threshold, under twice it.
      { foldNudgeTokens: 300 }
    );
    try {
      await harness.run('first', 'second', 'third');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(threadRecords(branch).length, 1, mode);
      // After the fold, too little built up for a second nudge.
      assert.equal(episodeStats(branch, harness.dir).nudges, 1, mode);
    } finally {
      await harness.cleanup();
    }
  }
});

test('once earlier notes take up room, the nudge names a stamp before them to merge them', async () => {
  for (const [note, since, merges] of [
    ['Printed a long line.', 't5', false],
    [`Printed a long line. ${'detail '.repeat(60)}`, 't1', true]
  ] as const) {
    const big = 'x'.repeat(800);
    const bash = () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${big}` }), {
        stopReason: 'toolUse'
      });
    const done = () => fauxAssistantMessage('done');
    const nudged = (context: Parameters<typeof conversation>[0]) => {
      const nudge = text(conversation(context).at(-1)!);
      assert.match(nudge, /tokens of work built up/);
      return nudge;
    };
    const harness = await createSession(
      'off',
      [
        bash,
        done,
        (context) => {
          nudged(context);
          return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note }), {
            stopReason: 'toolUse'
          });
        },
        done,
        bash,
        done,
        (context) => {
          const nudge = nudged(context);
          assert.equal(/since "(t\d+)"/.exec(nudge)![1], since, note);
          assert.equal(/which it merges/.test(nudge), merges, note);
          return fauxAssistantMessage(fauxToolCall('fold', { since, note: 'Both lines.' }), {
            stopReason: 'toolUse'
          });
        },
        done
      ],
      [],
      // A note of a quarter of the threshold, 75 tokens, takes up room.
      { foldNudgeTokens: 300 }
    );
    try {
      await harness.run('first', 'second', 'third', 'fourth');
      const records = threadRecords(harness.session.sessionManager.getBranch());
      assert.equal(records.length, 2, note);
      assert.equal(((records[1] as { covers?: string[] }).covers ?? []).length, merges ? 1 : 0);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a long run is nudged at a turn boundary once it grew twice the threshold', async () => {
  const big = 'x'.repeat(800);
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${big}` }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        assert.match(
          text(conversation(context).at(-1)!),
          /If the work since t1 is finished and the task/
        );
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 100 }
  );
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('the nudge threshold is at most a quarter of the context window', () => {
  assert.equal(nudgeThreshold(50_000, 64_000), 16_000);
  assert.equal(nudgeThreshold(50_000, 1_000_000), 50_000);
  assert.equal(nudgeThreshold(50_000, undefined), 50_000);
});

test('a call made next to a fold call is not offered as a start', async () => {
  const harness = await createSession('on', [
    () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one' }), {
        stopReason: 'toolUse'
      }),
    () =>
      fauxAssistantMessage(
        [
          fauxToolCall('codemode', { code: 'text(await tools.fold({ since: "t1", note: "n" }));' }),
          fauxToolCall('bash', { command: `echo sibling ${'y'.repeat(2000)}` })
        ],
        { stopReason: 'toolUse' }
      ),
    () => fauxAssistantMessage('done'),
    () => fauxAssistantMessage('ok')
  ]);
  try {
    await harness.run('go', 'next');
    const entries = harness.session.sessionManager.buildSessionProjection().entries;
    assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 1);
    // The sibling's result sits after the note, but a span from it would start before it.
    const sibling = entries
      .flatMap((entry) => entry.messages)
      .find(
        (message) =>
          message.role === 'toolResult' &&
          message.toolName === 'bash' &&
          /sibling/.test(text(message as Message))
      );
    assert.ok(sibling);
    assert.deepEqual(
      foldableTurns(entries).map((turn) => turn.label.replace(/^t\d+ +/, '')),
      ['you: next']
    );
    // Nor is its result counted as work a fold could remove.
    assert.ok(nudgeGrowth(entries) < 100);
  } finally {
    await harness.cleanup();
  }
});

test('the nudge can be turned off', async () => {
  const harness = await createSession(
    'off',
    [
      () => fauxAssistantMessage('one'),
      () => fauxAssistantMessage('two'),
      () => fauxAssistantMessage('three')
    ],
    [],
    { foldNudge: false, foldNudgeTokens: 1 }
  );
  try {
    await harness.run('first', 'second', 'third');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).nudges, 0);
  } finally {
    await harness.cleanup();
  }
});

test('a nudge the agent ignored is where growth is counted from', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('one'),
      () => fauxAssistantMessage('two'),
      () => fauxAssistantMessage('three')
    ],
    [],
    { foldNudgeTokens: 300 }
  );
  try {
    await harness.run('first', 'second', 'third');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).nudges, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a fold in a long run is not followed by a nudge in the same run', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /tokens of work built up/);
        return fauxAssistantMessage(
          fauxToolCall('fold', { since: 't1', note: 'Printed a line.' }),
          {
            stopReason: 'toolUse'
          }
        );
      },
      (context) => {
        assert.equal(conversation(context).at(-1)!.role, 'toolResult');
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 100 }
  );
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 1);
    assert.equal(episodeStats(branch, harness.dir).nudges, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a response cut off by the length limit inside a tool call is nudged too', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
          stopReason: 'length'
        }),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /tokens of work built up/);
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 50 }
  );
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('close to compaction, a long run is nudged before it grew twice the threshold', async () => {
  // About 400 tokens build up: under twice the threshold, over a quarter of it.
  for (const reserveTokens of [16_384, 127_000]) {
    const harness = await createSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(1600)}` }), {
            stopReason: 'toolUse'
          }),
        () => fauxAssistantMessage('done')
      ],
      [],
      { foldNudgeTokens: 1000 },
      { reserveTokens }
    );
    try {
      await harness.run('go');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(episodeStats(branch, harness.dir).nudges, reserveTokens > 100_000 ? 1 : 0);
    } finally {
      await harness.cleanup();
    }
  }
});

test('near compaction, the incoming prompt counts toward the room left', async () => {
  // About 2,400 tokens of foldable work: over a quarter of the threshold, under the threshold.
  // The prompt's 25,000 tokens, or its 21 images at 1,200 each, move the context past the
  // window less reserve and threshold.
  const image: ImageContent = {
    type: 'image',
    mimeType: 'image/png',
    data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
  };
  for (const [prompt, nudges] of [
    ['second', 0],
    ['y'.repeat(100_000), 1],
    [{ text: 'second', images: Array.from({ length: 21 }, () => image) }, 1]
  ] as const) {
    const harness = await createSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4800)}` }), {
            stopReason: 'toolUse'
          }),
        () => fauxAssistantMessage('one'),
        () => fauxAssistantMessage('two')
      ],
      [],
      { foldNudgeTokens: 4000 },
      { reserveTokens: 104_000 }
    );
    try {
      await harness.run('first', prompt);
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(episodeStats(branch, harness.dir).nudges, nudges);
    } finally {
      await harness.cleanup();
    }
  }
});

test('the start of a user message, which a fold keeps, is not counted as growth', async () => {
  const prompts = Array.from({ length: 10 }, () => 'p'.repeat(3900));
  const harness = await createSession(
    'off',
    prompts.map(() => () => fauxAssistantMessage('ok'))
  );
  try {
    await harness.run(...prompts);
    const entries = harness.session.sessionManager.buildSessionProjection().entries;
    assert.ok(nudgeGrowth(entries) < 100);
  } finally {
    await harness.cleanup();
  }
});

test('a fold cut short partway through its edits is completed at the next turn end', async () => {
  const needle = `needle-${'x'.repeat(400)}`;
  const first = await createSession('off', [
    () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${needle}` }), {
        stopReason: 'toolUse'
      }),
    () =>
      fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Printed a line.' }), {
        stopReason: 'toolUse'
      }),
    () => fauxAssistantMessage('done')
  ]);
  let saved: NonNullable<Parameters<typeof SessionManager.inMemory>[2]>;
  try {
    await first.run('go');
    const manager = first.session.sessionManager;
    const all = manager.getEntries();
    const edits = all.flatMap((entry, index) => (entry.type === 'context_edit' ? [index] : []));
    assert.ok(edits.length > 1);
    // A crash after the first edit was written: the rest of the batch and the marker are lost.
    saved = [manager.getHeader()!, ...all.slice(0, edits[0]! + 1)];
  } finally {
    await first.cleanup();
  }

  const resumed = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo next' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        assert.doesNotMatch(conversation(context).map(text).join('\n'), /needle-x/);
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'needle' }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /needle-x/);
        return fauxAssistantMessage('done');
      }
    ],
    [],
    {},
    {},
    saved
  );
  try {
    await resumed.run('next');
    assert.equal(threadRecords(resumed.session.sessionManager.getBranch()).length, 1);
  } finally {
    await resumed.cleanup();
  }
});

test('a fold whose note an earlier turn-end handler removes does not take effect', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo kept-output' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'n' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        assert.match(conversation(context).map(text).join('\n'), /kept-output/);
        return fauxAssistantMessage('done');
      }
    ],
    [],
    {},
    {},
    undefined,
    [
      (pi) => {
        pi.on('turn_end', (event, ctx) => {
          const note = ctx.sessionManager
            .getBranch()
            .find(
              (entry) =>
                entry.type === 'message' &&
                entry.message.role === 'toolResult' &&
                entry.message.toolName === 'fold'
            );
          if (!note) return undefined;
          return {
            entries: [
              ...event.entries,
              { type: 'context_edit' as const, targetId: note.id, replacement: null }
            ]
          };
        });
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
