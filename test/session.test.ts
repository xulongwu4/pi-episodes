import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, rmSync, writeFileSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  getSystemMessageText,
  type Message,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager
} from '@earendil-works/pi-coding-agent';
import { EVENT_TYPE, META_TYPE } from '../src/events.js';
import episodes from '../src/index.js';
import { readStamp } from '../src/stamps.js';
import { episodeStats, statsLines } from '../src/stats.js';

type Step = (context: TranscriptContext) => AssistantMessage | Promise<AssistantMessage>;

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

async function createSession(
  steps: Step[],
  options: {
    git?: boolean;
    persist?: boolean;
    keepRecentTokens?: number;
    episodes?: Record<string, unknown>;
  } = {}
) {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-session-'));
  if (options.git) execFileSync('git', ['init', '-q'], { cwd: dir });
  const faux = fauxProvider();
  const requests: TranscriptContext[] = [];
  // Assertions inside a step surface as provider errors; keep them to rethrow.
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
    extensionFactories: [(pi) => pi.registerProvider(faux.provider), episodes]
  });
  await resourceLoader.reload();

  const { session } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model: faux.getModel(),
    thinkingLevel: 'off',
    modelRuntime: await ModelRuntime.create({ modelsPath: null, authPath: join(dir, 'auth.json') }),
    resourceLoader,
    tools: ['bash', 'read', 'fold', 'unfold', 'recall', 'ask', 'fork'],
    sessionManager: options.persist
      ? SessionManager.create(dir, join(dir, 'sessions'))
      : SessionManager.inMemory(dir),
    settingsManager: SettingsManager.inMemory({
      compaction: {
        enabled: false,
        ...(options.keepRecentTokens === undefined
          ? {}
          : { keepRecentTokens: options.keepRecentTokens })
      },
      ...(options.episodes ? { episodes: options.episodes } : {})
    })
  });
  await session.bindExtensions({});

  return {
    dir,
    session,
    requests,
    pending: () => faux.getPendingResponseCount(),
    failures,
    async cleanup() {
      session.dispose();
      await rm(dir, { recursive: true, force: true });
    }
  };
}

test('fold, recall, ask and fork in a live session', async () => {
  let handle = '';
  const harness = await createSession([
    // Episode: a build loop that fails once, then passes.
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "error TS2322 in api.ts"' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo build passed' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', { since: 't1', note: 'Build passes after regenerating the client.' }),
        { stopReason: 'toolUse' }
      ),
    (context) => {
      // The next request already runs on the folded view.
      const roles = conversation(context).map((message) => message.role);
      assert.deepEqual(roles, ['user', 'assistant', 'toolResult']);
      const note = text(conversation(context)[2]!);
      assert.match(note, /^\[fold ep-[0-9a-f]{8}: 4 messages, t1-t3,/);
      assert.match(note, /Build passes after regenerating the client\./);
      assert.doesNotMatch(note, /TS2322/);
      handle = /ep-[0-9a-f]{8}/.exec(note)![0];
      return fauxAssistantMessage('Folded.');
    },
    // Going back.
    () =>
      fauxAssistantMessage(fauxToolCall('recall', { query: 'ts2322' }), { stopReason: 'toolUse' }),
    (context) => {
      const result = text(conversation(context).at(-1)!);
      assert.match(result, new RegExp(`${handle} t\\d+ toolResult: error TS2322 in api.ts`));
      return fauxAssistantMessage(
        fauxToolCall('ask', { handle, question: 'Why did the first build fail?' }),
        { stopReason: 'toolUse' }
      );
    },
    (context) => {
      // The ask request resumes the archived thread: it still holds the failing output.
      const messages = conversation(context);
      assert.match(text(messages.at(-1)!), /Why did the first build fail\?/);
      assert.ok(messages.some((message) => /error TS2322/.test(text(message))));
      return fauxAssistantMessage('The generated client was stale.');
    },
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        /answers\]\nThe generated client was stale\./
      );
      return fauxAssistantMessage(fauxToolCall('fork', { task: 'Check the tests.' }), {
        stopReason: 'toolUse'
      });
    },
    (context) => {
      // The child starts from the parent's view, with the task as the fork result.
      const messages = conversation(context);
      assert.ok(messages.some((message) => /Build passes after regenerating/.test(text(message))));
      assert.match(text(messages.at(-1)!), /Task: Check the tests\./);
      return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo tests ok' }), {
        stopReason: 'toolUse'
      });
    },
    () => fauxAssistantMessage([fauxText('Changed: nothing. Learned: tests pass.')]),
    (context) => {
      const result = text(conversation(context).at(-1)!);
      assert.match(result, /^\[fork fk-[0-9a-f]{8} finished after 2 turns\]/);
      assert.match(result, /Learned: tests pass\./);
      assert.doesNotMatch(result, /tests ok/);
      return fauxAssistantMessage('All done.');
    }
  ]);

  try {
    await harness.session.prompt('make it build');
    await harness.session.prompt('look back at it');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);

    // Stamps are persisted on user messages and tool results, never on responses.
    const stamped = harness.session.messages.filter((message) => readStamp(message) !== undefined);
    assert.ok(stamped.every((message) => message.role === 'user' || message.role === 'toolResult'));
    // Main-thread and fork requests after the fold all start with the same folded prefix, so
    // they stay cacheable; only the ask request (index 6) resumes the archived thread instead.
    const prefix = (context: TranscriptContext) => JSON.stringify(context.messages.slice(0, 4));
    const afterFold = harness.requests.slice(3).filter((_, index) => index !== 3);
    assert.equal(new Set(afterFold.map(prefix)).size, 1);
    assert.notEqual(prefix(harness.requests[6]!), prefix(harness.requests[3]!));
    // The prompt section is declared once and never re-sent as a change.
    const systems = harness.requests
      .at(-1)!
      .messages.filter((message) => message.role === 'system');
    assert.equal(systems.length, 1);
    assert.ok(systems[0]?.role === 'system');
    assert.match(
      getSystemMessageText(systems[0]),
      /<episodes>\nEvery user message and tool result ends with a turn stamp/
    );
  } finally {
    await harness.cleanup();
  }
});

test('a fold that reaches behind the previous note merges it', async () => {
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one' })),
    () => fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'first' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo two' })),
    () => fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'second' })),
    (context) => {
      const view = conversation(context);
      // The earlier note stays, as written, under the new one.
      assert.match(
        text(view.at(-1)!),
        /second\n[\s\S]*This fold covers the note of an earlier fold, which follows this note as written[\s\S]*\n\n\[note of ep-[0-9a-f]{8}, t1-t2, 5 characters\]\nfirst\n/
      );
      assert.equal(view.filter((message) => message.role === 'toolResult').length, 1);
      return fauxAssistantMessage('ok');
    }
  ]);

  try {
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a fold with until removes old turns and keeps the work after them', async () => {
  const harness = await createSession(
    [
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one > a.txt' })),
      // Longer than one turn is read back whole, shorter than a range is.
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'seq -f "two %g" 1 1500' })),
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo three > c.txt; echo 3' })),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't2', until: 't3', note: 'one, two' })),
      (context) => {
        const view = conversation(context);
        const results = view.filter((message) => message.role === 'toolResult');
        assert.equal(results.length, 2);
        assert.match(text(results[0]!), /^3\n/);
        const note = text(results[1]!);
        assert.match(note, /^\[fold ep-[0-9a-f]{8}: 4 messages, t2-t3, /);
        assert.match(note, /The turns from t4 on were not folded and stay above\./);
        // The disk changes are the span's, not those of the turns kept after it.
        assert.match(note, /A a\.txt/);
        assert.doesNotMatch(note, /c\.txt/);
        return fauxAssistantMessage(fauxToolCall('recall', { query: '"two 1500"' }));
      },
      (context) => {
        const found = text(conversation(context).at(-1)!);
        assert.match(found, /ep-[0-9a-f]{8} .*two 1500/);
        const handle = /ep-[0-9a-f]{8}/.exec(found)![0];
        return fauxAssistantMessage(fauxToolCall('recall', { handle, turn: ' t2-t3 ' }));
      },
      (context) => {
        // Both turns of the span, each message under its own turn, and all of them.
        const back = text(conversation(context).at(-1)!);
        assert.match(back, /^ep-[0-9a-f]{8} t2 toolResult \(bash\), \d+ chars\n/);
        assert.match(back, /\n\nep-[0-9a-f]{8} t3 toolResult \(bash\), \d+ chars\ntwo 1\n/);
        assert.match(back, /\ntwo 1500\n/);
        assert.doesNotMatch(back, /more characters/);
        const handle = /ep-[0-9a-f]{8}/.exec(back)![0];
        return fauxAssistantMessage(fauxToolCall('recall', { handle, turn: 't3' }));
      },
      (context) => {
        const back = text(conversation(context).at(-1)!);
        assert.match(back, /more characters\. All of t3 is in /);
        const handle = /ep-[0-9a-f]{8}/.exec(back)![0];
        // A turn of blanks is no turn: the query runs.
        return fauxAssistantMessage(
          fauxToolCall('recall', { handle, turn: ' ', query: '"two 1499"' })
        );
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /ep-[0-9a-f]{8} t3 toolResult.*two 1499/);
        return fauxAssistantMessage('ok');
      }
    ],
    { git: true }
  );

  try {
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
    const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
    assert.equal(stats.folds.bounded, 1);
    assert.ok(statsLines(stats).some((line) => /^folds 1, .*1 ending at until/.test(line)));
  } finally {
    await harness.cleanup();
  }
});

test('/fold asks for a fold and ends the run on it', async () => {
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo build passed' })),
    () => fauxAssistantMessage('It builds.'),
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /Use since "t1"/);
      return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'It builds.' }));
    },
    (context) => {
      // The request message and the folded response are gone; only the note follows t1.
      const roles = conversation(context).map((message) => message.role);
      assert.deepEqual(roles, ['user', 'assistant', 'toolResult', 'user']);
      return fauxAssistantMessage('Next.');
    }
  ]);

  try {
    await harness.session.prompt('make it build');
    await harness.session.prompt('/fold t1');
    await harness.session.waitForIdle();
    // The fold ended the run: no follow-up response was requested.
    assert.equal(harness.pending(), 1);
    await harness.session.prompt('next');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('points back to a folded span when its files are touched or changed again', async () => {
  let handle = '';
  const harness = await createSession(
    [
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'printf one > a.txt' })),
      () => fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'Wrote a.txt.' })),
      (context) => {
        // The shell wrote the file; the working tree diff still lists it.
        const note = text(conversation(context).at(-1)!);
        assert.match(note, /Changed on disk \(working tree diff\):\nA a\.txt/);
        handle = /ep-[0-9a-f]{8}/.exec(note)![0];
        rmSync(join(harness.dir, 'a.txt'));
        return fauxAssistantMessage(fauxToolCall('read', { path: 'a.txt' }));
      },
      (context) => {
        // The read failed, so it touched nothing and the cue is still to come.
        assert.doesNotMatch(text(conversation(context).at(-1)!), /was also touched/);
        writeFileSync(join(harness.dir, 'a.txt'), 'one');
        return fauxAssistantMessage(fauxToolCall('read', { path: 'a.txt' }));
      },
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, new RegExp(`a\\.txt was also touched in folded span ${handle}`));
        return fauxAssistantMessage(fauxToolCall('read', { path: 'a.txt' }));
      },
      (context) => {
        // Pointed at once per file, not on every touch.
        assert.doesNotMatch(text(conversation(context).at(-1)!), /was also touched/);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'printf two > a.txt' }));
      },
      // Quoted: an exact substring, as the command was written.
      () => fauxAssistantMessage(fauxToolCall('recall', { query: '"printf one >"' })),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, new RegExp(`${handle} predates changes to: a\\.txt`));
        assert.match(result, /assistant: bash\(\{"command":"printf one > a\.txt"\}\)/);
        return fauxAssistantMessage('done');
      }
    ],
    { git: true }
  );

  try {
    await harness.session.prompt('write a file');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('with snapshots off, a note lists the files that file tools used and no diff', async () => {
  const harness = await createSession(
    [
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'printf one > a.txt; printf two > b.txt' })
        ),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'b.txt' })),
      () => fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'Wrote two files.' })),
      (context) => {
        // In a repository, where a snapshot would have listed the file the shell wrote too.
        const note = text(conversation(context).at(-1)!);
        assert.match(
          note,
          /Files used by file tools \(no working tree diff is available[^\n]*\nb\.txt\n/
        );
        assert.doesNotMatch(note, /a\.txt|working tree diff\):/);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'printf three > b.txt' }));
      },
      () => fauxAssistantMessage(fauxToolCall('recall', { query: '"printf one"' })),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, /ep-[0-9a-f]{8} /);
        assert.doesNotMatch(result, /predates changes/);
        return fauxAssistantMessage('done');
      }
    ],
    { git: true, episodes: { snapshots: false } }
  );

  try {
    await harness.session.prompt('write two files');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
    // No private index or object store was made for the session.
    const id = harness.session.sessionManager.getSessionId();
    assert.equal(existsSync(join(tmpdir(), `pi-episodes-${id}-${process.pid}`)), false);
  } finally {
    await harness.cleanup();
  }
});

test('parallel forks see each other as running, and each transcript is archived as it grows', async () => {
  // Read by a step after the session exists; steps only run once prompted.
  const harness = await createSession(
    [
      () =>
        fauxAssistantMessage([
          fauxToolCall('fork', { task: 'Run the unit tests.' }, { id: 'fork-a' }),
          fauxToolCall('fork', { task: 'Run the lint.' }, { id: 'fork-b' })
        ]),
      // Both children start from the same parent turn; whichever asks first gets this.
      (context) => {
        const messages = conversation(context);
        const siblings = messages.filter((message) =>
          /running this in parallel/.test(text(message))
        );
        assert.equal(siblings.length, 1);
        assert.match(text(messages.at(-1)!), /Task: Run the (unit tests|lint)\./);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ok' }));
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /Task: Run the (unit tests|lint)\./);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo ok' }));
      },
      async () => {
        // The first turn of each child reaches disk before the child finishes; the write is
        // queued at turn end, so give it a moment.
        const dir = harness.session.sessionManager
          .getSessionFile()!
          .replace(/\.jsonl$/, '.episodes');
        let files: string[] = [];
        for (let attempt = 0; attempt < 20 && files.length < 2; attempt++) {
          await new Promise((resolve) => setTimeout(resolve, 25));
          files = (await readdir(dir).catch(() => [])).filter((name) => name.startsWith('fk-'));
        }
        assert.equal(files.length, 2);
        return fauxAssistantMessage('Green.');
      },
      () => fauxAssistantMessage('Green.'),
      (context) => {
        const results = conversation(context).slice(-2);
        for (const message of results) {
          assert.match(text(message), /\[fork fk-[0-9a-f]{8} finished after 2 turns\]/);
        }
        return fauxAssistantMessage('Both done.');
      }
    ],
    { persist: true }
  );

  try {
    await harness.session.prompt('check everything');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);

    const sessionFile = harness.session.sessionManager.getSessionFile()!;
    const archives = await readdir(sessionFile.replace(/\.jsonl$/, '.episodes'));
    assert.equal(archives.filter((name) => name.startsWith('fk-')).length, 2);
    const archive = JSON.parse(
      await readFile(join(sessionFile.replace(/\.jsonl$/, '.episodes'), archives[0]!), 'utf8')
    );
    assert.equal(
      archive.messages.filter((message: Message) => message.role === 'assistant').length,
      2
    );
  } finally {
    await harness.cleanup();
  }
});

test('reminds once of open items after a fold, on the next user prompt', async () => {
  const reminder = /Open items from work folded since the last reminder/;
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo build passed' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          since: 't2',
          note: 'Build passes after regenerating the client.\n- Open: remove the debug flag in vite.config.ts.'
        })
      ),
    (context) => {
      assert.ok(!conversation(context).some((message) => reminder.test(text(message))));
      return fauxAssistantMessage('Done.');
    },
    (context) => {
      const messages = conversation(context);
      const reminded = messages.filter((message) => reminder.test(text(message)));
      assert.equal(reminded.length, 1);
      assert.match(
        text(reminded[0]!),
        /ep-[0-9a-f]{8} \(Build passes[^)]*\): remove the debug flag/
      );
      return fauxAssistantMessage('Removing it.');
    },
    (context) => {
      // Not sent again while nothing new was folded.
      const reminded = conversation(context).filter((message) => reminder.test(text(message)));
      assert.equal(reminded.length, 1);
      return fauxAssistantMessage('ok');
    }
  ]);

  try {
    await harness.session.prompt('make it build');
    await harness.session.prompt('next task');
    await harness.session.prompt('and another');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('ask on a span folded by /fold does not end with the request to fold', async () => {
  let handle = '';
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo build passed' })),
    () => fauxAssistantMessage('It builds.'),
    () => fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'It builds.' })),
    (context) => {
      const note = conversation(context).find((message) => /^\[fold ep-/.test(text(message)));
      handle = /ep-[0-9a-f]{8}/.exec(text(note!))![0];
      return fauxAssistantMessage(fauxToolCall('ask', { handle, question: 'Did it build?' }));
    },
    (context) => {
      const messages = conversation(context);
      assert.match(text(messages.at(-1)!), /Did it build\?/);
      assert.ok(!messages.some((message) => /asked you to fold now/.test(text(message))));
      return fauxAssistantMessage('Yes.');
    },
    () => fauxAssistantMessage('It did.')
  ]);

  try {
    await harness.session.prompt('make it build');
    await harness.session.prompt('/fold t1');
    await harness.session.waitForIdle();
    await harness.session.prompt('did it build?');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('after a compaction, recall finds the compacted turns and fold names what is gone', async () => {
  let handle = '';
  const harness = await createSession(
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "error TS2322 in api.ts"' })),
      () => fauxAssistantMessage('Fixed by regenerating the client.'),
      () => fauxAssistantMessage('ok'),
      // pi's summarizer, once for the history and once for the split turn: whatever it
      // writes, stamps in it are only text.
      () =>
        fauxAssistantMessage('## Goal\nMake it build. At t2 the build failed with a type error.'),
      () => fauxAssistantMessage('The user said next.'),
      (context) => {
        const notice = conversation(context).find((message) =>
          /archived as cp-[0-9a-f]{8}/.test(text(message))
        );
        assert.ok(notice);
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'TS2322' }));
      },
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, /cp-[0-9a-f]{8} t\d+ toolResult: error TS2322 in api\.ts/);
        handle = /cp-[0-9a-f]{8}/.exec(result)![0];
        return fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'x' }));
      },
      (context) => {
        assert.match(
          text(conversation(context).at(-1)!),
          /t2 is no longer in your context: it was compacted or already folded\./
        );
        return fauxAssistantMessage(fauxToolCall('ask', { handle, question: 'why?' }));
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /too large to ask\. Use recall/);
        return fauxAssistantMessage('done');
      }
    ],
    { keepRecentTokens: 1 }
  );

  try {
    await harness.session.prompt('make it build');
    await harness.session.prompt('next');
    await harness.session.compact();
    await harness.session.prompt('what failed earlier?');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a fork folds its own work, and recall and ask reach the fold inside it', async () => {
  let stamp = '';
  let nested = '';
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('fork', { task: 'Make the build pass.' })),
    (context) => {
      assert.match(text(conversation(context).at(-1)!), /Fold your own finished episodes/);
      return fauxAssistantMessage(
        fauxToolCall('bash', { command: 'echo "error TS9999 deep in the loop"' })
      );
    },
    (context) => {
      stamp = /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!;
      // The parent's own turns are not the fork's to fold.
      return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'x' }));
    },
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        /cannot reach behind the start of this fork; a fork folds only its own work\. The earliest stamp you can fold from is /
      );
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: stamp, note: 'Build loop: fixed the type error.' })
      );
    },
    (context) => {
      const messages = conversation(context);
      assert.ok(!messages.some((message) => /TS9999/.test(text(message))));
      const note = text(messages.at(-1)!);
      assert.match(note, /^\[fold ep-[0-9a-f]{8}: /);
      nested = /ep-[0-9a-f]{8}/.exec(note)![0];
      return fauxAssistantMessage('Build passes.\n- Open: none.');
    },
    (context) => {
      const result = text(conversation(context).at(-1)!);
      assert.match(result, new RegExp(`Folded inside the fork:\\n${nested}: Build loop`));
      return fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' }));
    },
    (context) => {
      const result = text(conversation(context).at(-1)!);
      // Found under the nested fold, not twice under the fork too.
      assert.match(result, new RegExp(`${nested} t\\d+ toolResult: error TS9999`));
      assert.doesNotMatch(result, /fk-[0-9a-f]{8} t\d+ toolResult/);
      return fauxAssistantMessage(
        fauxToolCall('ask', { handle: nested, question: 'What failed?' })
      );
    },
    (context) => {
      const messages = conversation(context);
      assert.match(text(messages.at(-1)!), /What failed\?/);
      assert.ok(messages.some((message) => /error TS9999/.test(text(message))));
      return fauxAssistantMessage('A type error, TS9999.');
    },
    () => fauxAssistantMessage('done')
  ]);

  try {
    await harness.session.prompt('make it build in a fork');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a running fork reaches the folds it made with recall and ask', async () => {
  let stamp = '';
  let nested = '';
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('fork', { task: 'Make the build pass.' })),
    () =>
      fauxAssistantMessage(
        fauxToolCall('bash', { command: 'echo "error TS9999 deep in the loop"' })
      ),
    (context) => {
      stamp = /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!;
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: stamp, note: 'Build loop: fixed the type error.' })
      );
    },
    (context) => {
      const messages = conversation(context);
      assert.ok(!messages.some((message) => /TS9999/.test(text(message))));
      nested = /ep-[0-9a-f]{8}/.exec(text(messages.at(-1)!))![0];
      return fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' }));
    },
    (context) => {
      const found = text(conversation(context).at(-1)!);
      assert.match(found, new RegExp(`${nested} ${stamp} toolResult: error TS9999`));
      // The fork's own thread is what the child still sees; it is not searched.
      assert.doesNotMatch(found, /fk-[0-9a-f]{8} t\d+/);
      return fauxAssistantMessage(fauxToolCall('recall', { handle: nested, turn: stamp }));
    },
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        new RegExp(`^${nested} ${stamp} toolResult \\(bash\\), \\d+ chars\\nerror TS9999`)
      );
      return fauxAssistantMessage(
        fauxToolCall('ask', { handle: nested, question: 'What failed?' })
      );
    },
    // The folded thread answers from what the child saw when it folded.
    (context) => {
      const messages = conversation(context);
      assert.match(text(messages.at(-1)!), /What failed\?/);
      assert.ok(messages.some((message) => /error TS9999/.test(text(message))));
      assert.ok(messages.some((message) => /Task: Make the build pass\./.test(text(message))));
      return fauxAssistantMessage('A type error, TS9999.');
    },
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        new RegExp(`^\\[${nested} answers\\]\\nA type error, TS9999\\.`)
      );
      return fauxAssistantMessage('Build passes.\n- Open: none.');
    },
    () => fauxAssistantMessage('done')
  ]);

  try {
    await harness.session.prompt('make it build in a fork');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('ask rebuilds what a fork started from, also a result that came before it in its response', async () => {
  let nested = '';
  // What the child saw: the note of the fold that ran before the fork in the same response.
  const sawNote = (context: TranscriptContext) => {
    const messages = conversation(context);
    assert.ok(messages.some((message) => /Parent note: ran one\./.test(text(message))));
    assert.ok(messages.some((message) => /Task: Go on\./.test(text(message))));
    assert.ok(messages.some((message) => /row two/.test(text(message))));
  };
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "row one"' })),
    (context) =>
      // A fold makes the calls of its response run one after the other.
      fauxAssistantMessage([
        fauxToolCall('fold', {
          since: /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!,
          note: 'Parent note: ran one.'
        }),
        fauxToolCall('fork', { task: 'Go on.' })
      ]),
    (context) => {
      assert.ok(
        conversation(context).some((message) => /Parent note: ran one\./.test(text(message)))
      );
      return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "row two"' }));
    },
    (context) =>
      fauxAssistantMessage(
        fauxToolCall('fold', {
          since: /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!,
          note: 'Ran two.'
        })
      ),
    (context) => {
      nested = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
      return fauxAssistantMessage(fauxToolCall('ask', { handle: nested, question: 'Rows?' }));
    },
    // While the fork runs.
    (context) => {
      sawNote(context);
      return fauxAssistantMessage('One and two.');
    },
    () => fauxAssistantMessage('Both rows printed.\n- Open: none.'),
    () => fauxAssistantMessage(fauxToolCall('ask', { handle: nested, question: 'Rows?' })),
    // And after it has ended.
    (context) => {
      sawNote(context);
      return fauxAssistantMessage('One and two.');
    },
    () => fauxAssistantMessage('done')
  ]);

  try {
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a fork running next to another does not reach the folds the other made', async () => {
  let folded = () => {};
  const alphaFolded = new Promise<void>((resolve) => {
    folded = resolve;
  });
  let searched = () => {};
  const betaSearched = new Promise<void>((resolve) => {
    searched = resolve;
  });
  const done = { alpha: 0, beta: 0 };
  // The children take their replies in whatever order they ask; each reply is picked by the
  // task of the child that asks and by how far that child is.
  const child = async (context: TranscriptContext) => {
    const messages = conversation(context);
    const last = text(messages.at(-1)!);
    if (messages.some((message) => /Task: alpha\./.test(text(message)))) {
      switch (done.alpha++) {
        case 0:
          return fauxAssistantMessage(
            fauxToolCall('bash', { command: 'echo "error TS9999 deep in the loop"' })
          );
        case 1:
          return fauxAssistantMessage(
            fauxToolCall('fold', {
              since: /\[(t\d+)\]$/.exec(last)![1]!,
              note: 'Build loop: fixed the type error.'
            })
          );
        case 2:
          // Stays running until beta has searched.
          folded();
          await betaSearched;
          return fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' }));
        default:
          assert.match(last, /^ep-[0-9a-f]{8} t\d+ toolResult: error TS9999/);
          return fauxAssistantMessage('Alpha done.');
      }
    }
    switch (done.beta++) {
      case 0:
        await alphaFolded;
        return fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' }));
      default:
        // Alpha goes on whatever beta found, so that a failure here ends the test.
        searched();
        assert.match(last, /^No archived thread is on this branch/);
        return fauxAssistantMessage('Beta done.');
    }
  };
  const harness = await createSession([
    () =>
      fauxAssistantMessage([
        fauxToolCall('fork', { task: 'alpha.' }, { id: 'fork-a' }),
        fauxToolCall('fork', { task: 'beta.' }, { id: 'fork-b' })
      ]),
    child,
    child,
    child,
    child,
    child,
    child,
    () => fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' })),
    (context) => {
      // Once both have ended, the main agent finds the span, once.
      const found = text(conversation(context).at(-1)!);
      assert.equal(found.match(/^ep-[0-9a-f]{8} t\d+ toolResult: error TS9999/gm)?.length, 1);
      return fauxAssistantMessage('done');
    }
  ]);

  try {
    await harness.session.prompt('two forks');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
    assert.deepEqual(done, { alpha: 4, beta: 2 });
  } finally {
    await harness.cleanup();
  }
});

test('a fork folds with until and keeps its later work', async () => {
  let stamp = '';
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('fork', { task: 'Make the build pass.' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "old bulk"' })),
    (context) => {
      stamp = /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!;
      return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo "live work"' }));
    },
    () => fauxAssistantMessage(fauxToolCall('fold', { since: stamp, until: stamp, note: 'old' })),
    (context) => {
      const messages = conversation(context);
      assert.ok(!messages.some((message) => /^old bulk/.test(text(message))));
      assert.ok(messages.some((message) => /^live work/.test(text(message))));
      assert.match(
        text(messages.at(-1)!),
        /The turns from t\d+ on were not folded and stay above\./
      );
      return fauxAssistantMessage('Build passes.\n- Open: none.');
    },
    () => fauxAssistantMessage('done')
  ]);

  try {
    await harness.session.prompt('make it build in a fork');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a fold in a fork keeps the note it covers under its own, and the next one keeps both', async () => {
  const last = (context: Parameters<typeof conversation>[0]) => text(conversation(context).at(-1)!);
  const stamp = (context: Parameters<typeof conversation>[0]) =>
    /\[(t\d+)\]$/.exec(last(context))![1]!;
  let first = '';
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('fork', { task: 'Make the build pass.' })),
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one' })),
    (context) => {
      first = stamp(context);
      return fauxAssistantMessage(
        fauxToolCall('fold', { since: first, note: 'Ran one.\nLearned: alpha is 7.' })
      );
    },
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo two' })),
    (context) => {
      // From the stamp of the first note, two before this result's.
      const since = `t${Number(stamp(context).slice(1)) - 1}`;
      return fauxAssistantMessage(fauxToolCall('fold', { since, note: 'second' }));
    },
    (context) => {
      const messages = conversation(context);
      assert.match(
        last(context),
        new RegExp(
          `second\n[\\s\\S]*\n\n\\[note of ep-[0-9a-f]{8}, ${first}, 29 characters\\]\nRan one\\.\nLearned: alpha is 7\\.\n`
        )
      );
      assert.equal(messages.map(text).join('\n').split('alpha is 7').length - 1, 1);
      return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo three' }));
    },
    (context) => {
      const since = `t${Number(stamp(context).slice(1)) - 1}`;
      return fauxAssistantMessage(fauxToolCall('fold', { since, note: 'third' }));
    },
    (context) => {
      // The first note is read from under the second, as the fork's fold recorded it.
      assert.match(
        last(context),
        new RegExp(
          `third\n[\\s\\S]*\n\n\\[note of ep-[0-9a-f]{8}, ${first}, 29 characters\\]\nRan one\\.\nLearned: alpha is 7\\.\n\n\\[note of ep-[0-9a-f]{8}, t\\d+-t\\d+, 6 characters\\]\nsecond\n\\[t\\d+\\]$`
        )
      );
      return fauxAssistantMessage('Build passes.\n- Open: none.');
    },
    () => fauxAssistantMessage('done')
  ]);

  try {
    await harness.session.prompt('make it build in a fork');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});

test('a note names the skill files read inside its span', async () => {
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('read', { path: 'skills/tdd/SKILL.md' })),
    () => fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'Loaded the skill.' })),
    (context) => {
      const note = text(conversation(context).at(-1)!);
      assert.ok(
        note.includes(
          `Skills read inside the span, whose instructions left your context with it: ${join(harness.dir, 'skills/tdd/SKILL.md')}. Read a skill again before relying on it.`
        )
      );
      return fauxAssistantMessage('done');
    }
  ]);

  try {
    await mkdir(join(harness.dir, 'skills/tdd'), { recursive: true });
    await writeFile(join(harness.dir, 'skills/tdd/SKILL.md'), 'Write the test first.\n');
    await harness.session.prompt('use the skill');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});
test('a large tool result arrives cut, and recall searches and reads back all of it', async () => {
  let handle = '';
  const harness = await createSession(
    [
      () =>
        fauxAssistantMessage(
          fauxToolCall('bash', { command: 'seq -f "row %g of the table" 1 600' })
        ),
      (context) => {
        const output = text(conversation(context).at(-1)!);
        assert.match(output, /^row 1 of the table\n/);
        assert.match(
          output,
          /\[episodes\] \d+ of \d+ characters left out here\. All of this output is in /
        );
        assert.match(output, /row 600 of the table\n+\[t2\]$/);
        assert.doesNotMatch(output, /row 300 of/);
        assert.ok(output.length < 1000);
        return fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', note: 'Printed the table.' })
        );
      },
      (context) => {
        handle = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
        return fauxAssistantMessage(fauxToolCall('recall', { query: '"row 300 of the table"' }));
      },
      (context) => {
        // The line was never in context; it is found in what the result was cut from.
        assert.equal(
          text(conversation(context).at(-1)!).split('\n')[0],
          `${handle} t2 toolResult: row 300 of the table`
        );
        return fauxAssistantMessage(fauxToolCall('recall', { handle, turn: 't2' }));
      },
      (context) => {
        const back = text(conversation(context).at(-1)!);
        assert.match(
          back,
          new RegExp(
            `^${handle} t2 toolResult \\(bash\\), \\d+ chars\nrow 1 of the table\nrow 2 of`
          )
        );
        assert.match(back, /row 300 of the table\n/);
        assert.doesNotMatch(back, /row 600 of/);
        assert.match(back, new RegExp(`more characters\\. All of t2 is in .*${handle}-t2\\.txt`));
        return fauxAssistantMessage(fauxToolCall('recall', { handle, turn: 't9' }));
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), new RegExp(`No turn t9 in ${handle} `));
        return fauxAssistantMessage('done');
      }
    ],
    { persist: true, episodes: { outputCapChars: 2000, outputCapKeepChars: 600 } }
  );

  try {
    await harness.session.prompt('print the table');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);

    const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
    assert.equal(stats.outputs.capped, 1);
    assert.ok(stats.outputs.tokens > 2000);
    assert.equal(stats.recall.count, 2);
    assert.equal(stats.recall.readBack, 1);
    const lines = statsLines(stats);
    assert.ok(lines.some((line) => /^outputs capped 1, ~\dk kept out of context$/.test(line)));
    assert.ok(lines.includes('recall 2 (1 read back)'));

    const sessions = join(harness.dir, 'sessions');
    const archive = (await readdir(sessions)).find((name) => name.endsWith('.episodes'))!;
    const files = await readdir(join(sessions, archive));
    assert.ok(files.some((name) => /^output-[0-9a-f]{16}\.txt$/.test(name)));
    const whole = await readFile(join(sessions, archive, `${handle}-t2.txt`), 'utf8');
    assert.match(whole, /row 600 of the table/);
  } finally {
    await harness.cleanup();
  }
});

test('an output cap that is off leaves large results whole', async () => {
  const harness = await createSession(
    [
      () => fauxAssistantMessage(fauxToolCall('bash', { command: 'seq 1 3000' })),
      (context) => {
        const output = text(conversation(context).at(-1)!);
        assert.match(output, /\n1500\n/);
        assert.doesNotMatch(output, /left out here/);
        return fauxAssistantMessage('done');
      }
    ],
    { persist: true, episodes: { outputCap: false, outputCapChars: 2000, outputCapKeepChars: 600 } }
  );

  try {
    await harness.session.prompt('count');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});
test('recall with nothing archived points to the context and makes no claim about it', async () => {
  const harness = await createSession([
    () => fauxAssistantMessage(fauxToolCall('bash', { command: 'echo release-ea81b665' })),
    () => fauxAssistantMessage(fauxToolCall('recall', { query: 'release' })),
    (context) => {
      assert.match(
        text(conversation(context).at(-1)!),
        /^No archived thread is on this branch, so recall has nothing to search\. Look in your context instead; for a tool result that arrived cut, follow any pointer its preview gives to the rest\./
      );
      return fauxAssistantMessage('done');
    }
  ]);

  try {
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);
  } finally {
    await harness.cleanup();
  }
});
test('stats read the session: folds, rejections, cues followed, and the drop after a fold', async () => {
  let handle = '';
  const harness = await createSession(
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'printf one > a.txt; seq 1 3000' })),
      () => fauxAssistantMessage(fauxToolCall('read', { path: 'b.txt' })),
      () => fauxAssistantMessage(fauxToolCall('fold', { since: 't99', note: 'x' })),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', note: 'Wrote a.txt, read b.txt.' })
        ),
      (context) => {
        handle = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
        return fauxAssistantMessage(fauxToolCall('read', { path: 'b.txt' }));
      },
      () => fauxAssistantMessage(fauxToolCall('recall', { query: 'printf' })),
      // Folds a span holding the cue's entry, which sits between a call and its result.
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't6', note: 'Looked at b.txt again.' })),
      (context) => {
        const messages = conversation(context);
        assert.ok(!messages.some((message) => /was also touched/.test(text(message))));
        assert.match(text(messages.at(-1)!), /Looked at b\.txt again\./);
        return fauxAssistantMessage('done');
      }
    ],
    { git: true }
  );

  try {
    await writeFile(join(harness.dir, 'b.txt'), 'two\n');
    await harness.session.prompt('go');
    if (harness.failures[0]) throw harness.failures[0];
    assert.equal(harness.pending(), 0);

    const branch = harness.session.sessionManager.getBranch();
    const events = branch.flatMap((entry) =>
      entry.type === 'custom' && entry.customType === EVENT_TYPE ? [entry.data] : []
    );
    assert.deepEqual(
      events.map(
        (event) =>
          (event as { kind: string; reason?: string }).reason ?? (event as { kind: string }).kind
      ),
      ['no-such-stamp', 'cue']
    );
    const meta = branch.find((entry) => entry.type === 'custom' && entry.customType === META_TYPE);
    assert.match(
      String((meta as { data?: { promptHash?: string } })?.data?.promptHash),
      /^[0-9a-f]{12}$/
    );

    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.folds.count, 2);
    assert.equal(stats.folds.rejected, 1);
    assert.equal(stats.folds.retriedOk, 1);
    assert.equal(stats.folds.notAlone, 0);
    // The second fold came right before the final reply.
    assert.equal(stats.folds.beforeFinal, 1);
    assert.deepEqual(stats.cues, { shown: 1, followed: 1, readOnly: 1, capped: 0 });
    assert.equal(stats.recall.count, 1);
    assert.equal(stats.requests.length, 8);
    // The request after the first fold is the one that drops.
    const after = stats.requests.findIndex((point) => point.after === 'fold');
    assert.equal(after, 4);
    assert.ok(stats.requests[after]!.tokens < stats.requests[after - 1]!.tokens);
    assert.ok(stats.folds.saved > 0);
    // Each fold with what it cost on its first request and how long it has been in effect.
    assert.deepEqual(
      stats.folds.each.map((fold) => [fold.handle === handle, fold.requests]),
      [
        [true, 4],
        [false, 1]
      ]
    );
    assert.deepEqual(
      stats.folds.each.map((fold) => fold.uncachedAfter),
      stats.folds.uncachedAfter
    );
    assert.ok(stats.folds.each.every((fold) => fold.net > 0));
    // Every request's uncached input is put down to one cause; a fold's is its first request.
    const split = stats.uncached;
    assert.equal(
      split.afterFold,
      stats.folds.uncachedAfter.reduce((sum, tokens) => sum + tokens, 0)
    );
    assert.equal(
      split.fresh + split.afterFold + split.afterCompaction + split.missed,
      stats.requests.reduce((sum, request) => sum + request.uncached, 0)
    );
    assert.ok(stats.overhead.stamps > 0);
    const lines = statsLines(stats);
    assert.match(lines[0]!, /^context per request {2}[▁▂▃▄▅▆▇█]{8}$/);
    assert.ok(lines.some((line) => /^folds 2, .*1 right before a final reply/.test(line)));
    assert.ok(lines.includes('fold errors 1, next fold succeeded after 1'));
    assert.ok(
      lines.some((line) =>
        /^fold cost break-even after median \d+ request\(s\), [0-2]\/2 folds past it \(cached input counted at 10%\)$/.test(
          line
        )
      )
    );
    assert.ok(lines.includes('cues 1, followed 1 (1 read-only)'));
    assert.ok(handle);
  } finally {
    await harness.cleanup();
  }
});
