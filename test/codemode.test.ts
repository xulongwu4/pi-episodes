import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  getCurrentTools,
  getSystemMessageText,
  type ImageContent,
  type Message,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import {
  createAgentSession,
  createCodemodeExtension,
  DefaultResourceLoader,
  type ExtensionFactory,
  type ExtensionUIContext,
  ModelRuntime,
  type SessionEntry,
  SessionManager,
  SettingsManager
} from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import {
  appliedFolds,
  callingFork,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_STARTED_TYPE,
  ThreadIndex,
  threadRecords
} from '../src/archive.js';
import { cacheAnchor } from '../src/cache.js';
import { foldableTurns, foldStart, USER_TEXT_CAP } from '../src/fold.js';
import episodes from '../src/index.js';
import { FILES_KEY } from '../src/messages.js';
import { AFTER_RUN_END_FOLD, nudgeGrowth, nudgeThreshold } from '../src/nudge.js';
import { cappedOutput } from '../src/output.js';
import { episodeStats, statsLines } from '../src/stats.js';

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
  before: ExtensionFactory[] = [],
  /** Tools of `extensions` to make active. */
  tools: string[] = [],
  /** A session with a file, next to which large outputs are kept. */
  persist = false,
  /** A host with a UI, as the terminal and RPC mode are; false is print and JSON mode. */
  ui: 'tui' | 'rpc' | false = false
) {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-codemode-'));
  const faux = fauxProvider();
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
      'unfold',
      'recall',
      'ask',
      'fork',
      ...tools,
      ...(mode === 'off' ? [] : ['codemode'])
    ],
    sessionManager: persist
      ? SessionManager.create(dir, join(dir, 'sessions'))
      : SessionManager.inMemory(dir, undefined, entries),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false, ...compaction },
      // Pi's settings type has no slot for extension keys.
      ...({ episodes: settings } as object),
      ...(mode === 'off' ? {} : { codemode: { mode } })
    })
  });
  // Pi runs without a UI in print mode, which is what a session bound to nothing reports.
  // Own properties: pi copies the object, which a proxy with nothing of its own does not survive.
  const nothing = () => undefined;
  const anyUi = {
    notify: nothing,
    setStatus: nothing,
    setWidget: nothing,
    setWorkingMessage: nothing,
    requestRender: nothing,
    select: nothing,
    custom: nothing,
    theme: { fg: (_color: string, shown: string) => shown, bold: (shown: string) => shown }
  } as unknown as ExtensionUIContext;
  await session.bindExtensions(ui ? { uiContext: anyUi, mode: ui } : {});

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

/** The edits that take a request for a fold at the end of a run out of the context. */
function askRemovals(branch: readonly SessionEntry[]) {
  const asks = new Set(
    branch
      .filter(
        (entry) => entry.type === 'custom_message' && entry.customType === 'episodes-fold-run-end'
      )
      .map((entry) => entry.id)
  );
  return branch.filter(
    (entry) => entry.type === 'context_edit' && asks.has(entry.targetId) && !entry.replacement
  ) as Extract<SessionEntry, { type: 'context_edit' }>[];
}

test('fold and fork stay declared next to codemode; the other tools follow the mode', async () => {
  const expected: Record<Mode, string[]> = {
    off: ['ask', 'bash', 'fold', 'fork', 'read', 'recall', 'unfold'],
    on: ['ask', 'bash', 'codemode', 'fold', 'fork', 'read', 'recall', 'unfold'],
    // fold and unfold are reached from scripts; fork cannot be, so it stays declared.
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
      // The marker of the fold stands where the script that ran the build was.
      assert.deepEqual(
        messages.map((message) => message.role),
        ['user', 'assistant', 'assistant', 'toolResult']
      );
      const note = text(messages[3]!);
      assert.match(note, /\[fold ep-[0-9a-f]{8}: 2 messages, t1-t2,/);
      assert.ok(
        text(messages[1]!).startsWith(
          `[Earlier responses of mine stood here, with the tool calls I made (t1-t2) and the outputs they returned. I read those outputs, and then I folded all of it out of this conversation as ${/ep-[0-9a-f]{8}/.exec(note)![0]}: `
        )
      );
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

test('a script that prints the fold result as JSON leaves the note in context once', async () => {
  const harness = await createSession('only', [
    () => script('text(await tools.bash({ command: "echo long output" }));'),
    () =>
      script(
        'const r = await tools.fold({ since: "t1", note: "Ran \\"it\\"; nothing to keep." }); text(JSON.stringify({ r }));'
      ),
    (context) => {
      const messages = conversation(context);
      assert.deepEqual(
        messages.map((message) => message.role),
        ['user', 'assistant', 'assistant', 'toolResult']
      );
      const note = text(messages[3]!);
      // As the script printed it: the line breaks and quotes of the note escaped.
      assert.match(
        note,
        /\{"r":"\[fold ep-[0-9a-f]{8}: 2 messages, t1-t2,[^\n]*\\n\\nRan \\"it\\"; nothing to keep\./
      );
      assert.equal(note.match(/\[fold ep-/g)?.length, 1);
      assert.equal(note.match(/nothing to keep/g)?.length, 1);
      assert.match(note, /\[t3\]$/);
      return fauxAssistantMessage('ok');
    }
  ]);
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).folds.count, 1);
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

test('a fold keeps the notes it covers under its own as written, also through a second such fold', async () => {
  const first = 'Ran "it".\nLearned: alpha is 7.';
  const second = 'Ran more.\nLearned: beta is 8.';
  const literal = (words: string) => words.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  for (const mode of ['off', 'only'] as const) {
    const call = (tool: string, args: Record<string, string>) => () =>
      mode === 'only'
        ? script(`text(JSON.stringify(await tools.${tool}(${JSON.stringify(args)})));`)
        : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
    const harness = await createSession(mode, [
      call('bash', { command: 'echo one; seq 1 400' }),
      call('fold', { since: 't2', note: first }),
      call('bash', { command: 'echo two; seq 1 400' }),
      // From the stamp of the first note, which it covers.
      call('fold', { since: 't3', note: second }),
      call('bash', { command: 'echo three; seq 1 400' }),
      call('fold', { since: 't5', note: 'Ran the last.' }),
      (context) => {
        const view = conversation(context);
        const note = text(view.at(-1)!);
        assert.match(
          note,
          /covers the notes of 2 earlier folds, which follow this note as written/
        );
        // A script printed its own note as JSON; the notes it covers are added as they were.
        assert.match(
          note,
          new RegExp(
            `Notes of the earlier folds covered here, oldest first, as they were written:\n\n\\[note of ep-[0-9a-f]{8}, t2, ${first.length} characters\\]\n${literal(first)}\n\n\\[note of ep-[0-9a-f]{8}, t3-t4, ${second.length} characters\\]\n${literal(second)}\n\\[t7\\]$`
          ),
          mode
        );
        const all = view.map(text).join('\n');
        for (const fact of ['alpha is 7', 'beta is 8', 'Ran the last']) {
          assert.equal(all.split(fact).length - 1, 1, `${mode} ${fact}`);
        }
        assert.doesNotMatch(all, /echo (one|two|three)/, mode);
        return fauxAssistantMessage('ok');
      }
    ]);
    try {
      await harness.run('go');
      const branch = harness.session.sessionManager.getBranch();
      const records = threadRecords(branch) as { handle: string; covers?: string[] }[];
      assert.equal(records.length, 3, mode);
      assert.deepEqual(records[1]!.covers, [records[0]!.handle], mode);
      assert.deepEqual(records[2]!.covers, [records[0]!.handle, records[1]!.handle], mode);
      // The agent wrote each note once, and a fold did not free the notes it kept.
      const tokens = (words: string) => Math.ceil(words.length / 4);
      const folds = episodeStats(branch, harness.dir).folds;
      assert.equal(folds.noteTokens, tokens(first) + tokens(second) + 4, mode);
      const spans = records as unknown as { tokens: number; lineTokens?: number }[];
      // Nor is the line a fold leaves free: the first fold pays for one, and a fold that
      // takes the line of the fold it covers and leaves its own pays next to nothing.
      const line = spans.map((span) => span.lineTokens ?? 0);
      assert.ok(line[0]! > 90 && line[0]! < 120, `${mode} ${line[0]}`);
      assert.ok(Math.abs(line[1]!) < 5 && Math.abs(line[2]!) < 5, `${mode} ${line}`);
      assert.deepEqual(
        folds.each.map((fold) => [fold.note, fold.net]),
        [
          [tokens(first), spans[0]!.tokens - tokens(first) - line[0]!],
          [tokens(second), spans[1]!.tokens - tokens(second) - tokens(first) - line[1]!],
          [4, spans[2]!.tokens - 4 - tokens(first) - tokens(second) - line[2]!]
        ],
        mode
      );
      assert.ok(
        folds.each.every((fold) => fold.net > 0),
        mode
      );
    } finally {
      await harness.cleanup();
    }
  }
});

const dropCovered: ExtensionFactory = (pi) => {
  pi.on('tool_result', (event) => {
    const title = 'Notes of the earlier folds covered here';
    const content: typeof event.content = [];
    for (const block of event.content) {
      const kept = block.type === 'text' ? block.text.split(`\n\n${title}`)[0]! : undefined;
      if (block.type !== 'text' || kept === undefined) content.push(block);
      else if (!kept.startsWith(title)) content.push({ ...block, text: kept });
    }
    return { content };
  });
};

test('a fold whose result lost the notes it covers to a later handler does not take effect', async () => {
  for (const mode of ['off', 'only'] as const) {
    const call = (tool: string, args: Record<string, string>) => () =>
      mode === 'only'
        ? script(`text(await tools.${tool}(${JSON.stringify(args)}));`)
        : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
    const harness = await createSession(
      mode,
      [
        call('bash', { command: 'echo one' }),
        call('fold', { since: 't2', note: 'Learned: alpha is 7.' }),
        call('bash', { command: 'echo two' }),
        call('fold', { since: 't3', note: 'second' }),
        (context) => {
          // The first note still stands, with the work after it.
          const all = conversation(context).map(text).join('\n');
          assert.match(all, /\[fold ep-[0-9a-f]{8}: [^\n]*\n\nLearned: alpha is 7\./, mode);
          assert.match(all, /two\n/, mode);
          return fauxAssistantMessage('ok');
        }
      ],
      [dropCovered]
    );
    try {
      await harness.run('go');
      assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 1, mode);
    } finally {
      await harness.cleanup();
    }
  }
});

test('an image in the result of a note stays in context when a fold covers the note, or the fold does not take effect', async () => {
  const image = { type: 'image' as const, data: 'AAAA', mimeType: 'image/png' };
  // Another extension puts an image into the result of the first note.
  const addImage: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const words = event.content.map((block) => (block.type === 'text' ? block.text : ''));
      if (
        !/alpha is 7/.test(words.join('\n')) ||
        /Notes of the earlier folds/.test(words.join('\n'))
      )
        return undefined;
      return { content: [...event.content, image] };
    });
  };
  // And one takes the images out of the result of the second.
  const dropImage: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const words = event.content.map((block) => (block.type === 'text' ? block.text : ''));
      // Not "seconds", which a script's result says, and not the third note's result.
      if (!/second(?!s)/.test(words.join('\n')) || /third/.test(words.join('\n'))) return undefined;
      return { content: event.content.filter((block) => block.type !== 'image') };
    });
  };
  for (const mode of ['off', 'only'] as const) {
    for (const dropped of [false, true]) {
      const call = (tool: string, args: Record<string, string>) => () =>
        mode === 'only'
          ? script(`text(JSON.stringify(await tools.${tool}(${JSON.stringify(args)})));`)
          : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
      const harness = await createSession(
        mode,
        [
          call('bash', { command: 'echo one' }),
          call('fold', { since: 't2', note: 'Learned: alpha is 7.' }),
          call('bash', { command: 'echo two' }),
          call('fold', { since: 't3', note: 'second' }),
          call('bash', { command: 'echo three' }),
          call('fold', { since: 't5', note: 'third' }),
          (context) => {
            const view = conversation(context);
            const images = view.flatMap((message) =>
              typeof message.content === 'string'
                ? []
                : message.content.filter((block) => block.type === 'image')
            );
            // Once, whichever note's result it stands in by now.
            assert.deepEqual(images, [image], `${mode} ${dropped}`);
            const all = view.map(text).join('\n');
            assert.equal(all.split('alpha is 7').length - 1, 1, `${mode} ${dropped}`);
            return fauxAssistantMessage('ok');
          }
        ],
        dropped ? [addImage, dropImage] : [addImage]
      );
      try {
        await harness.run('go');
        const records = threadRecords(harness.session.sessionManager.getBranch());
        // Without the image in its result the second fold left the first note standing, and
        // the third, from the stamp of the work after it, covered nothing.
        assert.deepEqual(
          (records as { covers?: string[] }[]).map((record) => record.covers?.length ?? 0),
          dropped ? [0, 0] : [0, 1, 2],
          `${mode} ${dropped}`
        );
      } finally {
        await harness.cleanup();
      }
    }
  }
});

test('a fold whose result an earlier handler of its boundary leaves without the notes it covers does not take effect', async () => {
  const preview = { content: [{ type: 'text' as const, text: 'A preview of the result.' }] };
  // What the handler queues for the result, in order: the last edit is the one that stands.
  const cases = {
    cut: { edits: (_whole: object) => [preview], folds: 1 },
    'whole, then cut': { edits: (whole: object) => [whole, preview], folds: 1 },
    'removed, then whole again': { edits: (whole: object) => [null, whole], folds: 2 },
    // Pi takes a string for content, and makes a text block of it.
    'cut, then whole again as one string': {
      edits: (whole: object) => [
        preview,
        {
          content: (whole as { content: { type: string; text?: string }[] }).content
            .map((block) => block.text ?? '')
            .join('\n')
        }
      ],
      folds: 2
    }
  };
  for (const [name, { edits, folds }] of Object.entries(cases)) {
    const editAtBoundary: ExtensionFactory = (pi) => {
      pi.on('turn_end', (event, ctx) => {
        const kept = ctx.sessionManager
          .getBranch()
          .findLast(
            (entry) =>
              entry.type === 'message' &&
              entry.message.role === 'toolResult' &&
              /Notes of the earlier folds covered here/.test(JSON.stringify(entry.message.content))
          );
        if (kept?.type !== 'message' || kept.message.role !== 'toolResult') return undefined;
        return {
          entries: [
            ...event.entries,
            ...edits({ content: kept.message.content }).map((replacement) => ({
              type: 'context_edit' as const,
              targetId: kept.id,
              replacement: replacement as typeof preview | { content: string } | null
            }))
          ]
        };
      });
    };
    for (const mode of ['off', 'only'] as const) {
      const call = (tool: string, args: Record<string, string>) => () =>
        mode === 'only'
          ? script(`text(await tools.${tool}(${JSON.stringify(args)}));`)
          : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
      const harness = await createSession(
        mode,
        [
          call('bash', { command: 'echo one' }),
          call('fold', { since: 't2', note: 'Learned: alpha is 7.' }),
          call('bash', { command: 'echo two' }),
          call('fold', { since: 't3', note: 'second' }),
          (context) => {
            const all = conversation(context).map(text).join('\n');
            const label = `${name} ${mode}`;
            assert.equal(all.split('alpha is 7').length - 1, 1, label);
            if (folds === 1) {
              // The preview is what the second fold left; the first note stands, with the work.
              assert.match(all, /A preview of the result\./, label);
              assert.match(all, /\[fold ep-[0-9a-f]{8}: [^\n]*\n\nLearned: alpha is 7\./, label);
              assert.match(all, /two/, label);
            } else {
              assert.match(
                all,
                /\[note of ep-[0-9a-f]{8}, t2, 20 characters\]\nLearned: alpha is 7\./
              );
              assert.doesNotMatch(all, /two/, label);
            }
            return fauxAssistantMessage('ok');
          }
        ],
        [],
        {},
        {},
        undefined,
        [editAtBoundary]
      );
      try {
        await harness.run('go');
        const records = threadRecords(harness.session.sessionManager.getBranch());
        assert.equal(records.length, folds, `${name} ${mode}`);
      } finally {
        await harness.cleanup();
      }
    }
  }
});

test('a note that a handler rewrote is kept as the handler left it', async () => {
  const redact: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) =>
      event.toolName === 'fold' && !event.isError
        ? {
            content: event.content.map((block) =>
              block.type === 'text'
                ? { ...block, text: block.text.replace('The key is 42.', 'The key is [redacted].') }
                : block
            )
          }
        : undefined
    );
  };
  for (const mode of ['off', 'only'] as const) {
    const call = (tool: string, args: Record<string, string>) => () =>
      mode === 'only'
        ? script(`text(await tools.${tool}(${JSON.stringify(args)}));`)
        : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
    const harness = await createSession(
      mode,
      [
        call('bash', { command: 'echo one' }),
        call('fold', { since: 't2', note: 'The key is 42.' }),
        call('bash', { command: 'echo two' }),
        call('fold', { since: 't3', note: 'second' }),
        (context) => {
          // What the agent wrote does not come back over the handler's text.
          const all = conversation(context).map(text).join('\n');
          assert.equal(all.split('The key is [redacted].').length - 1, 1, mode);
          assert.doesNotMatch(all, /The key is 42/, mode);
          return fauxAssistantMessage('ok');
        }
      ],
      [redact]
    );
    try {
      await harness.run('go');
      assert.equal(threadRecords(harness.session.sessionManager.getBranch()).length, 2, mode);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a script that printed several things gets each cut by itself under a line that says so, and recall reads all of it', async () => {
  const lines = (name: string) =>
    `Array.from({ length: 300 }, (_, i) => "${name} " + i + " of the report").join("\\n")`;
  const harness = await createSession(
    'only',
    [
      () =>
        script(
          `text(${lines('event')} + "\\nCOUNT 342 58"); text("TOP 17"); text(${lines('source')});`
        ),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(
          result,
          /^Script completed\nWall time [\d.]+ seconds\nOutput:\n\n?\[episodes\] Only part of this output is shown: \d+ of \d+ characters, with the rest left out at the 2 places marked below\. /
        );
        // The end of the first thing, the short one whole, the start of the third.
        assert.match(
          result,
          /\nCOUNT 342 58\n+==> text 2\/3 <==\nTOP 17\n+==> text 3\/3 <==\nsource 0 of the report\n/
        );
        assert.doesNotMatch(result, /event 150 of|source 150 of/);
        assert.equal(result.split(' characters left out here.').length - 1, 2);
        assert.ok(result.length < 3200, `${result.length}`);
        return script('text(await tools.fold({ since: "t2", note: "Printed the report." }));');
      },
      () => script('text(await tools.recall({ query: "\\"source 150 of the report\\"" }));'),
      (context) => {
        const found = text(conversation(context).at(-1)!);
        assert.match(found, /ep-[0-9a-f]{8} t2 toolResult: source 150 of the report/);
        assert.doesNotMatch(found, /Only part of this output is shown/);
        return fauxAssistantMessage('ok');
      }
    ],
    [],
    { outputCapChars: 4000, outputCapKeepChars: 1800 },
    {},
    undefined,
    [],
    [],
    true
  );
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.outputs.capped, 1);
    // The line the preview opens with is no cue: nothing here touched a file of a thread.
    assert.equal(stats.overhead.cues, 0);
  } finally {
    await harness.cleanup();
  }
});

test('a script whose output codemode cut to its own budget gets the cut made from all of it, and recall reads the middle', async () => {
  const lines = (name: string) =>
    `Array.from({ length: 1500 }, (_, i) => "${name} " + i + " of the report").join("\\n")`;
  const harness = await createSession(
    'only',
    [
      () => script(`text(${lines('event')}); text("TOP 17"); text(${lines('source')});`),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        // Over 40,000 characters were printed: codemode kept the start and the end, and the
        // second thing was in neither. The preview is cut from the file codemode kept.
        assert.match(
          result,
          /^Script completed\nWall time [\d.]+ seconds\nOutput:\n\n?\[episodes\] Only part of this output is shown: \d+ of 7\d{4} characters, with the rest left out at the 2 places marked below\./
        );
        assert.match(
          result,
          /\nevent 1499 of the report\n+==> text 2\/3 <==\nTOP 17\n+==> text 3\/3 <==\nsource 0 of the report\n/
        );
        assert.doesNotMatch(result, /Warning: truncated output|tokens truncated|pi-codemode/);
        assert.ok(result.length < 3200, `${result.length}`);
        return script(`// @options: {"max_output_tokens": 500}\ntext(${lines('event')});`);
      },
      (context) => {
        // A script that asked for less than the limit gets what codemode made of it.
        const result = text(conversation(context).at(-1)!);
        assert.match(
          result,
          /^Script completed\nWall time [\d.]+ seconds\nOutput:\n\n?Warning: truncated output \(original token count: \d+\)/
        );
        assert.doesNotMatch(result, /\[episodes\] Only part/);
        assert.ok(result.length < 2500, `${result.length}`);
        return script('text(await tools.fold({ since: "t2", note: "Printed the report." }));');
      },
      () => script('text(await tools.recall({ query: "\\"event 1400 of the report\\"" }));'),
      (context) => {
        const found = text(conversation(context).at(-1)!);
        assert.match(found, /ep-[0-9a-f]{8} t2 toolResult: event 1400 of the report/);
        return fauxAssistantMessage('ok');
      }
    ],
    [],
    { outputCapChars: 4000, outputCapKeepChars: 1800 },
    {},
    undefined,
    [],
    [],
    true
  );
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    const record = branch.flatMap((entry) => {
      const capped =
        entry.type === 'message' && entry.message.role === 'toolResult'
          ? cappedOutput(entry.message.details)
          : undefined;
      return capped ? [capped] : [];
    });
    assert.equal(record.length, 1);
    const { chars, arrived, preview, path } = record[0]!;
    // What arrived was codemode's 40,000 characters and its two lines; the file has all.
    assert.ok(arrived !== undefined && arrived > 40_000 && arrived < 40_400, `${arrived}`);
    assert.ok(chars > 70_000 && preview !== undefined);
    assert.equal((await readFile(path, 'utf8')).length, chars);
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.outputs.capped, 1);
    assert.equal(stats.outputs.tokens, Math.ceil((arrived - preview) / 4));
  } finally {
    await harness.cleanup();
  }
});

test('the output cap does not cut the note of a script that folded, only what it printed next to it', async () => {
  const note = `Learned a lot. ${'detail '.repeat(600)}end.`;
  for (const beside of [0, 3000]) {
    const harness = await createSession(
      'only',
      [
        () => script('text(await tools.bash({ command: "echo one" }));'),
        () =>
          script(
            `text(await tools.fold({ since: "t2", note: ${JSON.stringify(note)} })); text("row ".repeat(${beside}));`
          ),
        (context) => {
          const result = text(conversation(context).at(-1)!);
          // The whole note, once: cut with the output around it, it is added after the cut.
          assert.equal(result.split('detail end.').length - 1, 1, `${beside}`);
          assert.equal(result.split('characters left out here').length - 1, beside ? 1 : 0);
          assert.ok(result.length < (beside ? 6000 : 5000), `${result.length}`);
          return script('text(await tools.bash({ command: "echo two" }));');
        },
        // A fold that covers that note keeps the note, not the cut output around it.
        () => script('text(await tools.fold({ since: "t3", note: "second" }));'),
        (context) => {
          const result = text(conversation(context).at(-1)!);
          assert.ok(
            result.includes(`, ${note.length} characters]\n${note}\n[t5]`),
            `${beside} ${result.slice(-300)}`
          );
          assert.doesNotMatch(result, /characters left out here|row row/, `${beside}`);
          return fauxAssistantMessage('ok');
        }
      ],
      [],
      { outputCapChars: 2000, outputCapKeepChars: 600 },
      {},
      undefined,
      [],
      [],
      true
    );
    try {
      await harness.run('go');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(threadRecords(branch).length, 2, `${beside}`);
      assert.equal(episodeStats(branch, harness.dir).outputs.capped, beside ? 1 : 0);
    } finally {
      await harness.cleanup();
    }
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
      // The script printed its own note; the note it covers is added once, as written.
      const note = text(view.at(-1)!);
      assert.match(note, /covers the note of an earlier fold, which follows this note as written/);
      assert.equal(
        note.match(/\[note of ep-[0-9a-f]+, t1-t2, 5 characters\]\nfirst\n/g)?.length,
        1
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
    // The covered span is archived under its own handle only, as the covering fold's ask sees it.
    const threads = await new ThreadIndex(harness.session.sessionManager, new Map()).onBranch();
    const holding = threads.filter((thread) =>
      thread.messages.some((message) => /first-output/.test(text(message as Message)))
    );
    assert.deepEqual(
      holding.map((thread) => thread.handle),
      [records[0]!.handle]
    );
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

/**
 * Tools shaped like those of pi-hashline-edit-pro: an edit takes line anchors and names its file
 * only in the structured result, and a search takes the path it searched. An edit between the
 * `SAME` anchors changes nothing, and says so; with `STALE` anchors it reports a failure.
 */
const anchorTools: ExtensionFactory = (pi) => {
  pi.registerTool({
    name: 'replace',
    label: 'replace',
    description: 'Replace lines by anchor.',
    parameters: Type.Object({ remove_from: Type.String(), remove_to: Type.String() }),
    async execute(_id, params) {
      const classification = params.remove_from === SAME.remove_from ? 'noop' : 'applied';
      if (params.remove_from === STALE.remove_from) {
        return {
          content: [{ type: 'text', text: 'Stale anchor.' }],
          details: undefined,
          structuredContent: { ok: false, kind: 'error', path: 'api.ts' }
        };
      }
      return {
        content: [{ type: 'text', text: 'Replaced 1 line.' }],
        details: { diff: '+x' },
        structuredContent: { ok: true, kind: 'edit', classification, path: 'api.ts' }
      };
    }
  });
  pi.registerTool({
    name: 'grep',
    label: 'grep',
    description: 'Search files.',
    parameters: Type.Object({ pattern: Type.String(), path: Type.String() }),
    async execute() {
      return { content: [{ type: 'text', text: 'no hits' }], details: undefined };
    }
  });
};
const ANCHORS = { remove_from: 'abcde', remove_to: 'abcde' };
const SAME = { remove_from: 'fghij', remove_to: 'fghij' };
const STALE = { remove_from: 'klmno', remove_to: 'klmno' };

function toolResults(harness: Awaited<ReturnType<typeof createSession>>, toolName: string) {
  return harness.session.sessionManager
    .getBranch()
    .flatMap((entry) => (entry.type === 'message' ? [entry.message] : []))
    .flatMap((message) =>
      message.role === 'toolResult' && message.toolName === toolName ? [message] : []
    );
}

test('an edit by line anchors counts for the file its result names, and a search does not', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(
          [
            fauxToolCall('read', { path: 'api.ts' }),
            fauxToolCall('replace', ANCHORS),
            fauxToolCall('grep', { pattern: 'x', path: 'lib.ts' })
          ],
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Edited the api.' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        const note = text(conversation(context).at(-1)!);
        assert.match(note, /api\.ts/);
        assert.doesNotMatch(note, /lib\.ts/);
        return fauxAssistantMessage('noted');
      },
      () => fauxAssistantMessage(fauxToolCall('replace', ANCHORS), { stopReason: 'toolUse' }),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, /\[episodes\] api\.ts was also touched in folded span ep-[0-9a-f]{8}/);
        return fauxAssistantMessage('ok');
      }
    ],
    [anchorTools],
    {},
    {},
    undefined,
    [],
    ['replace', 'grep']
  );
  try {
    await writeFile(join(harness.dir, 'api.ts'), 'x\n');
    await harness.run('edit it', 'again');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fold');
    assert.deepEqual(record.files, [join(harness.dir, 'api.ts')]);
    // The span read the file too: the edit is what makes it more than read.
    assert.equal(record.readOnly, undefined);
    // The result keeps the tool's own details next to the file.
    assert.deepEqual(toolResults(harness, 'replace')[0]?.details, {
      diff: '+x',
      [FILES_KEY]: [{ name: 'replace', path: 'api.ts' }]
    });
  } finally {
    await harness.cleanup();
  }
});

test('an edit by line anchors from a script is in the fold, and the next one cues the thread', async () => {
  const call = `await tools.replace(${JSON.stringify(ANCHORS)});`;
  const edit = `${call} ${call} text("edited");`;
  const harness = await createSession(
    'only',
    [
      () => script(edit),
      () => script('text(await tools.fold({ since: "t1", note: "Edited the api." }));'),
      () => fauxAssistantMessage('noted'),
      () => script(edit),
      (context) => {
        const result = text(conversation(context).at(-1)!);
        assert.match(result, /\[episodes\] api\.ts was also touched in folded span ep-[0-9a-f]{8}/);
        return fauxAssistantMessage('ok');
      }
    ],
    [anchorTools],
    {},
    {},
    undefined,
    [],
    ['replace', 'grep']
  );
  try {
    await harness.run('edit it', 'again');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fold');
    assert.deepEqual(record.files, [join(harness.dir, 'api.ts')]);
    assert.equal(record.readOnly, undefined);
    const details = toolResults(harness, 'codemode')[0]?.details as Record<string, unknown>;
    assert.deepEqual(details[FILES_KEY], [{ name: 'replace', path: 'api.ts' }]);
  } finally {
    await harness.cleanup();
  }
});

/** As pi-hashline-edit-pro does, replaces the result of an edit with its diff. */
/** A write that touches no disk: a compaction records the path as the call gave it. */
const fakeWrite: ExtensionFactory = (pi) => {
  pi.registerTool({
    name: 'write',
    label: 'write',
    description: 'Write a file.',
    parameters: Type.Object({ path: Type.String(), content: Type.String() }),
    async execute() {
      return { content: [{ type: 'text', text: 'Wrote the file.' }], details: undefined };
    }
  });
};

test('a file under ~ that compacted turns changed cues their thread when it is touched again', async () => {
  const write = () =>
    fauxAssistantMessage(fauxToolCall('write', { path: '~/notes.md', content: 'x' }), {
      stopReason: 'toolUse'
    });
  const harness = await createSession(
    'off',
    [
      write,
      () => fauxAssistantMessage('saved'),
      () => fauxAssistantMessage('ok'),
      // pi's summarizer, for the history and for the split turn.
      () => fauxAssistantMessage('## Goal\nKeep notes.'),
      () => fauxAssistantMessage('The user said next.'),
      write,
      (context) => {
        assert.match(
          text(conversation(context).at(-1)!),
          /\[episodes\] \S*notes\.md was also touched in compacted turns cp-[0-9a-f]{8}/
        );
        return fauxAssistantMessage('ok');
      }
    ],
    [fakeWrite],
    {},
    { keepRecentTokens: 1 },
    undefined,
    [],
    ['write']
  );
  try {
    await harness.session.prompt('note it');
    await harness.session.prompt('next');
    await harness.session.compact();
    await harness.run('again');
  } finally {
    await harness.cleanup();
  }
});

const listing: ExtensionFactory = (pi) => {
  pi.registerTool({
    name: 'listing',
    label: 'listing',
    description: 'List a directory.',
    parameters: Type.Object({ path: Type.String() }),
    async execute() {
      const entries = Array.from({ length: 300 }, (_, i) => `entry ${i} of the listing`);
      return { content: [{ type: 'text', text: entries.join('\n') }], details: undefined };
    }
  });
};

test('only a read is sent back to the path it was given: the output of another tool with a path has its copy named', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('listing', { path: 'reports' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        const output = text(conversation(context).at(-1)!);
        assert.match(output, /: grep \S+output-[0-9a-f]{16}\.txt, which holds all of it, /);
        assert.match(
          output,
          /left out here\. All of this output is in \S+output-[0-9a-f]{16}\.txt; /
        );
        assert.doesNotMatch(output, /the file that was read/);
        return fauxAssistantMessage('ok');
      }
    ],
    [listing],
    { outputCapChars: 2000, outputCapKeepChars: 600 },
    {},
    undefined,
    [],
    ['listing'],
    true
  );
  try {
    await harness.run('list it');
  } finally {
    await harness.cleanup();
  }
});

const diffResult: ExtensionFactory = (pi) => {
  pi.on('tool_result', (event) => {
    if (event.toolName !== 'replace' || event.structuredContent === undefined) return undefined;
    return {
      content: [{ type: 'text' as const, text: '+x' }],
      structuredContent: event.structuredContent
    };
  });
};

for (const order of ['before', 'after'] as const) {
  test(`a cue stays on an edit whose result a handler ${order} episodes replaces`, async () => {
    const harness = await createSession(
      'off',
      [
        () => fauxAssistantMessage(fauxToolCall('replace', ANCHORS), { stopReason: 'toolUse' }),
        () =>
          fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Edited the api.' }), {
            stopReason: 'toolUse'
          }),
        () => fauxAssistantMessage('noted'),
        () => fauxAssistantMessage(fauxToolCall('replace', ANCHORS), { stopReason: 'toolUse' }),
        (context) => {
          const result = text(conversation(context).at(-1)!);
          assert.match(result, /^\+x\n\[episodes\] api\.ts was also touched in folded span/);
          return fauxAssistantMessage('ok');
        }
      ],
      order === 'after' ? [anchorTools, diffResult] : [anchorTools],
      {},
      {},
      undefined,
      order === 'before' ? [diffResult] : [],
      ['replace', 'grep']
    );
    try {
      await harness.run('edit it', 'again');
    } finally {
      await harness.cleanup();
    }
  });
}

test('an edit by line anchors that changed nothing or failed leaves the file as only read', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(
          [
            fauxToolCall('read', { path: 'api.ts' }),
            fauxToolCall('replace', SAME),
            fauxToolCall('replace', STALE)
          ],
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Looked at the api.' }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('noted')
    ],
    [anchorTools],
    {},
    {},
    undefined,
    [],
    ['replace', 'grep']
  );
  try {
    await writeFile(join(harness.dir, 'api.ts'), 'x\n');
    await harness.run('look at it');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fold');
    assert.deepEqual(record.readOnly, [join(harness.dir, 'api.ts')]);
    assert.deepEqual(
      toolResults(harness, 'replace').map((message) => message.details),
      [{ diff: '+x' }, undefined]
    );
  } finally {
    await harness.cleanup();
  }
});

test('an edit by line anchors in a fork is among the files of the fork', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('fork', { task: 'Edit.' }), { stopReason: 'toolUse' }),
      () => fauxAssistantMessage(fauxToolCall('replace', ANCHORS), { stopReason: 'toolUse' }),
      () => fauxAssistantMessage('Changed: api.ts. Learned: nothing.'),
      () => fauxAssistantMessage('ok')
    ],
    [anchorTools],
    {},
    {},
    undefined,
    [],
    ['replace', 'grep']
  );
  try {
    await harness.run('go');
    const [record] = threadRecords(harness.session.sessionManager.getBranch());
    assert.ok(record?.kind === 'fork');
    assert.deepEqual(record.files, [join(harness.dir, 'api.ts')]);
    // The files of the child are its own, not the fork call's.
    const details = toolResults(harness, 'fork')[0]?.details as Record<string, unknown>;
    assert.equal(details[FILES_KEY], undefined);
  } finally {
    await harness.cleanup();
  }
});

test('what a call from a fork left running finds no running fork once the call has returned', async () => {
  const seen: { during?: string | undefined; after?: Promise<string | undefined> } = {};
  const observer: ExtensionFactory = (pi) => {
    pi.on('tool_call', (event, ctx) => {
      if (event.toolName !== 'recall') return;
      seen.during = callingFork(ctx.sessionManager)?.handle;
      seen.after = new Promise((resolve) => {
        setTimeout(() => resolve(callingFork(ctx.sessionManager)?.handle), 50);
      });
    });
  };
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('fork', { task: 'Build.' }), { stopReason: 'toolUse' }),
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo TS9999' }), {
          stopReason: 'toolUse'
        }),
      (context) =>
        fauxAssistantMessage(
          fauxToolCall('fold', {
            since: /\[(t\d+)\]$/.exec(text(conversation(context).at(-1)!))![1]!,
            note: 'Built.'
          }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(fauxToolCall('recall', { query: 'TS9999' }), {
          stopReason: 'toolUse'
        }),
      async (context) => {
        assert.match(
          text(conversation(context).at(-1)!),
          /^ep-[0-9a-f]{8} t\d+ toolResult: TS9999/
        );
        // The fork is still running, and the recall call has returned.
        assert.match(String(seen.during), /^fk-[0-9a-f]{8}$/);
        assert.equal(await seen.after, undefined);
        return fauxAssistantMessage('Built.\n- Open: none.');
      },
      () => fauxAssistantMessage('ok')
    ],
    [observer]
  );
  try {
    await harness.run('go');
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
      assert.equal(messages.length, 4);
      assert.match(text(messages[3]!), /\[fold ep-[0-9a-f]{8}: 2 messages/);
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
        assert.equal(conversation(context).length, 4);
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

test('once notes take up room, the nudge names the earliest of them to merge them, close to compaction or not', async () => {
  const long = `Printed a long line. ${'detail '.repeat(400)}`;
  // A note of 300 tokens: with its call it is over a quarter of the threshold, and a fold
  // that covers it keeps it and frees less than that.
  const medium = `Printed a long line. ${'detail '.repeat(170)}`;
  // Far from compaction a run is nudged at twice the threshold. The note is at t3.
  for (const [note, since, merges, reserveTokens, size] of [
    ['Printed a long line.', 't4', false, 127_000, 4000],
    [medium, 't4', false, 127_000, 4000],
    [long, 't3', true, 127_000, 4000],
    ['Printed a long line.', 't4', false, 16_384, 17_000],
    [medium, 't4', false, 16_384, 17_000],
    [long, 't3', true, 16_384, 17_000]
  ] as const) {
    const bash = () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(size)}` }), {
        stopReason: 'toolUse'
      });
    const nudge = (context: Parameters<typeof conversation>[0]) => {
      const last = text(conversation(context).at(-1)!);
      assert.match(last, /tokens of work built up/);
      return last;
    };
    const harness = await createSession(
      'off',
      [
        bash,
        (context) => {
          nudge(context);
          return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note }), {
            stopReason: 'toolUse'
          });
        },
        () => fauxAssistantMessage('done'),
        bash,
        (context) => {
          const last = nudge(context);
          assert.equal(/since "(t\d+)"/.exec(last)![1], since, note);
          assert.equal(/under the new one as you wrote it/.test(last), merges, note);
          // It says whose stamp that is: the note is something the agent can find.
          assert.equal(
            /since "t3", the stamp at the end of your note ep-[0-9a-f]{8}, early enough to take in all your notes/.test(
              last
            ),
            merges,
            note
          );
          // Why it pays: Pi would compact, or the calls are sent with every request.
          const pressed = reserveTokens > 100_000;
          assert.equal(/pi would compact soon otherwise/.test(last), merges && pressed, note);
          assert.equal(
            /tokens, sent again with every request\)/.test(last),
            merges && !pressed,
            note
          );
          return fauxAssistantMessage(fauxToolCall('fold', { since, note: 'Both lines.' }), {
            stopReason: 'toolUse'
          });
        },
        (context) => {
          // The covered note is still in context, whole, under the note of the fold asked for.
          const view = conversation(context).map(text).join('\n');
          if (merges) assert.equal(view.split('Both lines.')[1]?.split(note.trim()).length, 2);
          return fauxAssistantMessage('done');
        }
      ],
      [],
      // A quarter of the threshold, 500 tokens, that a fold of the notes would free is enough
      // to ask for it: here the call that wrote the long note, whose argument is the note again.
      // Under pressure it is enough to nudge as well.
      { foldNudgeTokens: 2000 },
      { reserveTokens }
    );
    try {
      await harness.run('first', 'second');
      const records = threadRecords(harness.session.sessionManager.getBranch());
      assert.equal(records.length, 2, note);
      assert.equal(((records[1] as { covers?: string[] }).covers ?? []).length, merges ? 1 : 0);
    } finally {
      await harness.cleanup();
    }
  }
});

test('close to compaction, work a fold left behind its note is nudged to merge with nothing after the note', async () => {
  const harness = await createSession(
    'off',
    [
      // 1000 tokens with its output: under the threshold of 2000, over the quarter that is
      // enough close to compaction.
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(2000)}` }), {
          stopReason: 'toolUse'
        }),
      // The nudge asks to fold from t1; the agent runs one more command, then folds only that.
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo hi' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't3', note: `Said hi. ${'detail '.repeat(400)}` }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        // It names where that work starts, not the first stamp.
        const last = text(conversation(context).at(-1)!);
        assert.match(last, /About 1k tokens of work that later folds did not cover/);
        assert.match(
          last,
          /since "t2" and .*under the new one as you wrote it.*; pi would compact soon otherwise\)/
        );
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 2000 },
    { reserveTokens: 127_000 }
  );
  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('far from compaction, work a fold left behind its note is nudged from where it starts, until it is folded', async () => {
  const bash = (chars: number) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(chars)}` }), {
      stopReason: 'toolUse'
    });
  // The agent folds only its last command and leaves two behind it, of 1100 or 500 tokens
  // each with their output; the threshold is 2000 tokens, and a run is nudged at twice that.
  // Asked right after its fold, it folds from where they start or carries on.
  for (const [size, folds, nudges] of [
    [2200, true, 1],
    [2200, false, 2],
    [1000, false, 0]
  ] as const) {
    const harness = await createSession(
      'off',
      [
        bash(size),
        bash(size),
        bash(2),
        () =>
          fauxAssistantMessage(fauxToolCall('fold', { since: 't4', note: 'Said xx.' }), {
            stopReason: 'toolUse'
          }),
        (context) => {
          const last = text(conversation(context).at(-1)!);
          if (nudges === 0) assert.doesNotMatch(last, /tokens of work/);
          else {
            assert.match(last, /About 2k tokens of work that later folds did not cover/);
            assert.match(
              last,
              /with since "t2" and a note .*under the new one as you wrote it.* tokens\); then/
            );
            assert.doesNotMatch(last, /pi would compact/);
          }
          return folds
            ? fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'All three.' }), {
                stopReason: 'toolUse'
              })
            : fauxAssistantMessage('one');
        },
        ...(folds ? [() => fauxAssistantMessage('one')] : []),
        () => fauxAssistantMessage('two'),
        // A quarter of the threshold later, the work is still there.
        bash(2500),
        () => fauxAssistantMessage('three')
      ],
      [],
      { foldNudgeTokens: 2000 }
    );
    try {
      await harness.run('first', 'second', 'third');
      const branch = harness.session.sessionManager.getBranch();
      const label = `${size} ${folds}`;
      assert.equal(episodeStats(branch, harness.dir).nudges, nudges, label);
      const records = threadRecords(branch);
      assert.equal(records.length, folds ? 2 : 1, label);
      if (folds) assert.equal((records[1] as { covers?: string[] }).covers?.length, 1, label);
    } finally {
      await harness.cleanup();
    }
  }
});

test('the nudge for work left behind a note counts what a fold from its stamp removes', async () => {
  const bash = (chars: number) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(chars)}` }), {
      stopReason: 'toolUse'
    });
  const fold = (since: string, note: string) => () =>
    fauxAssistantMessage(fauxToolCall('fold', { since, note }), { stopReason: 'toolUse' });
  const harness = await createSession(
    'off',
    [
      // A first request, folded whole into a note of 700 tokens.
      bash(2200),
      fold('t2', `Printed a long line. ${'detail '.repeat(400)}`),
      () => fauxAssistantMessage('one'),
      // A second, of which only the last command is folded.
      bash(2200),
      bash(2200),
      bash(2),
      fold('t7', 'Said xx.'),
      (context) => {
        const last = text(conversation(context).at(-1)!);
        // The first note and its call, 1400 tokens, lie before t5 and are not counted.
        assert.match(last, /About 2k tokens of work that later folds did not cover .* from t5 on/);
        assert.match(last, /since "t5" and .*what they left behind \(about 2k tokens\); then/);
        return fauxAssistantMessage('two');
      }
    ],
    [],
    { foldNudgeTokens: 2000 }
  );
  try {
    await harness.run('first', 'second');
  } finally {
    await harness.cleanup();
  }
});

test('a fold that answers a nudge and leaves its work behind is followed by a nudge for that work', async () => {
  const harness = await createSession(
    'off',
    [
      // 2200 tokens with its output, over the threshold of 2000.
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4400)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('one'),
      // Asked to fold from t1, the agent runs one more command and folds only that.
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /tokens of work built up/);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo hi' }), {
          stopReason: 'toolUse'
        });
      },
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't4', note: 'Said hi.' }), {
          stopReason: 'toolUse'
        }),
      // Nothing was added since that nudge, but the fold left the work it asked about.
      (context) => {
        const last = text(conversation(context).at(-1)!);
        assert.match(last, /About 2k tokens of work that later folds did not cover .* from t2 on/);
        return fauxAssistantMessage('two');
      },
      // Declined, it does not come back without more work.
      (context) => {
        assert.equal(conversation(context).at(-1)!.role, 'user');
        return fauxAssistantMessage('three');
      }
    ],
    [],
    { foldNudgeTokens: 2000 }
  );
  try {
    await harness.run('first', 'second', 'third');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).nudges, 2);
  } finally {
    await harness.cleanup();
  }
});

test('turns a fold kept with until are asked about at the next prompt, or once the run has grown', async () => {
  const bash = (chars: number) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(chars)}` }), {
      stopReason: 'toolUse'
    });
  const asked = /About 2k tokens of work that later folds did not cover .* from t3 on/;
  // The agent folds its first command and keeps the second, of 2200 tokens with its output,
  // for its next step: over the threshold of 2000. That step adds nothing, or 600 tokens,
  // which is over a quarter of the threshold.
  for (const more of [false, true]) {
    const harness = await createSession(
      'off',
      [
        bash(2),
        bash(4400),
        () =>
          fauxAssistantMessage(
            fauxToolCall('fold', { since: 't2', until: 't2', note: 'Said xx.' }),
            { stopReason: 'toolUse' }
          ),
        // Right after the fold the agent is not asked to fold what it just chose to keep.
        (context) => {
          assert.equal(conversation(context).at(-1)!.role, 'toolResult');
          return more ? bash(1200)() : fauxAssistantMessage('one');
        },
        ...(more
          ? [
              (context: TranscriptContext) => {
                assert.match(text(conversation(context).at(-1)!), asked);
                return fauxAssistantMessage('one');
              }
            ]
          : []),
        (context) => {
          const last = text(conversation(context).at(-1)!);
          if (more) assert.doesNotMatch(last, /tokens of work/);
          else assert.match(last, asked);
          return fauxAssistantMessage('two');
        }
      ],
      [],
      { foldNudgeTokens: 2000 }
    );
    try {
      await harness.run('first', 'second');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(episodeStats(branch, harness.dir).nudges, 1, String(more));
    } finally {
      await harness.cleanup();
    }
  }
});

test('a fold that only merges notes does not bring the nudge for left-behind work back', async () => {
  const bash = (chars: number) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(chars)}` }), {
      stopReason: 'toolUse'
    });
  const fold = (args: Record<string, string>) =>
    fauxAssistantMessage(fauxToolCall('fold', { note: 'Said xx.', ...args }), {
      stopReason: 'toolUse'
    });
  const asked = /tokens of work that later folds did not cover/;
  const merged = `Said xx. ${'detail '.repeat(290)}`;
  const quiet = (context: TranscriptContext) =>
    assert.equal(conversation(context).at(-1)!.role, 'toolResult');
  // 2200 tokens of work stay behind a note, over the threshold of 2000: left there by a fold
  // from a later stamp and asked about at once, or kept with until and not asked about in
  // the run. Either way the agent then folds twice from the stamp of the newest note, which
  // covers that note and nothing else. Its call and its note, of 500 tokens each, are each a
  // quarter of the threshold, and are not work.
  for (const until of [false, true]) {
    const harness = await createSession(
      'off',
      [
        ...(until ? [bash(2), bash(4400)] : [bash(4400), bash(2)]),
        () => fold(until ? { since: 't2', until: 't2' } : { since: 't3' }),
        (context) => {
          if (until) quiet(context);
          else assert.match(text(conversation(context).at(-1)!), asked);
          return fold({ since: 't4', note: merged });
        },
        (context) => {
          quiet(context);
          return fold({ since: 't5', note: merged });
        },
        (context) => {
          quiet(context);
          return fauxAssistantMessage('one');
        },
        // The next prompt asks about turns that were kept and never asked about.
        (context) => {
          const last = text(conversation(context).at(-1)!);
          if (until) assert.match(last, asked);
          else assert.doesNotMatch(last, asked);
          return fauxAssistantMessage('two');
        }
      ],
      [],
      { foldNudgeTokens: 2000 }
    );
    try {
      await harness.run('first', 'second');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(episodeStats(branch, harness.dir).nudges, 1, String(until));
      assert.equal(threadRecords(branch).length, 3, String(until));
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
    const next = foldableTurns(entries).find((turn) => turn.label.endsWith('you: next'));
    assert.equal(foldStart(entries)?.stamp, next?.stamp);
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

test('a long note is not work: the fold that wrote it is not followed by a nudge', async () => {
  // Over twice the threshold with the call that wrote it, and enough to ask for a merge.
  const note = `Printed a line. ${'detail '.repeat(200)}`;
  const fold = (since: string) => (context: Parameters<typeof conversation>[0]) => {
    // Only the first fold answers a nudge.
    const last = conversation(context).at(-1)!;
    assert.equal(/tokens of work built up/.test(text(last)), since === 't1', since);
    return fauxAssistantMessage(fauxToolCall('fold', { since, note }), { stopReason: 'toolUse' });
  };
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
          stopReason: 'toolUse'
        }),
      fold('t1'),
      // The note is at t3: folding from it merges it, and adds no work either.
      fold('t3'),
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
    assert.equal(threadRecords(branch).length, 2);
    assert.equal(episodeStats(branch, harness.dir).nudges, 1);
  } finally {
    await harness.cleanup();
  }
});

test('what the agent wrote after its folds is asked about with the notes it follows', async () => {
  // One long reply after the only note, or two shorter ones after two notes: over the threshold.
  for (const [size, rounds] of [
    [9000, 1],
    [6000, 2]
  ] as const) {
    const quiet = (context: Parameters<typeof conversation>[0]) =>
      assert.doesNotMatch(text(conversation(context).at(-1)!), /tokens of work built up/);
    const steps = Array.from({ length: rounds }, (_, round) => [
      (context: Parameters<typeof conversation>[0]) => {
        quiet(context);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo small' }), {
          stopReason: 'toolUse'
        });
      },
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: `t${3 * round + 1}`, note: 'Ran it.' }),
          { stopReason: 'toolUse' }
        ),
      () => fauxAssistantMessage('r'.repeat(size))
    ]).flat();
    const harness = await createSession(
      'off',
      [
        ...steps,
        (context) => {
          const nudge = text(conversation(context).at(-1)!);
          // Some of it was written before the last fold.
          assert.match(nudge, /tokens of work built up that no fold has removed, since/);
          // No stamp after the first note reaches the reply that followed it.
          assert.match(
            nudge,
            /since "t3", the stamp at the end of your note ep-[0-9a-f]{8}, early enough to take in all your notes/
          );
          return fauxAssistantMessage(fauxToolCall('fold', { since: 't3', note: 'All of it.' }), {
            stopReason: 'toolUse'
          });
        },
        (context) => {
          assert.doesNotMatch(conversation(context).map(text).join('\n'), /rrrr/);
          return fauxAssistantMessage('done');
        }
      ],
      [],
      { foldNudgeTokens: 2000 }
    );
    try {
      await harness.run(...Array.from({ length: rounds + 1 }, (_, round) => `prompt ${round}`));
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(threadRecords(branch).length, rounds + 1);
      assert.equal(episodeStats(branch, harness.dir).nudges, 1);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a fold of replies written since a nudge for left-behind work brings that nudge back', async () => {
  const bash = (output: string) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${output}` }), {
      stopReason: 'toolUse'
    });
  const fold = (since: string) => () =>
    fauxAssistantMessage(fauxToolCall('fold', { since, note: 'Saved it.' }), {
      stopReason: 'toolUse'
    });
  const asked = (context: Parameters<typeof conversation>[0]) => {
    assert.match(text(conversation(context).at(-1)!), /work that later folds did not cover/);
    return fauxAssistantMessage('done');
  };
  const harness = await createSession(
    'off',
    [
      // The first command stays behind the note of a fold that took only the second.
      bash('x'.repeat(4800)),
      bash('tiny'),
      fold('t3'),
      asked,
      // No tool ran since: the work is a reply, and the fold of it leaves the command again.
      () => fauxAssistantMessage('r'.repeat(500)),
      fold('t5'),
      asked
    ],
    [],
    { foldNudgeTokens: 2000 }
  );
  try {
    await harness.run('old task', 'explain', 'save that explanation');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).nudges, 2);
  } finally {
    await harness.cleanup();
  }
});

test('a reply merged away with its note brings the nudge for left-behind work back once', async () => {
  const asks = /work that later folds did not cover/;
  const bash = (output: string) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${output}` }), {
      stopReason: 'toolUse'
    });
  // A fold from the stamp of the latest note: it merges that note and what follows it.
  const merge = (asked: boolean) => (context: Parameters<typeof conversation>[0]) => {
    const messages = conversation(context);
    assert.equal(asks.test(text(messages.at(-1)!)), asked);
    const note = messages.findLast(
      (message) => message.role === 'toolResult' && text(message).includes('[fold ep-')
    );
    const since = `t${/\[t(\d+)\]$/.exec(text(note!))![1]}`;
    return fauxAssistantMessage(fauxToolCall('fold', { since, note: 'Saved it.' }), {
      stopReason: 'toolUse'
    });
  };
  const done = (asked: boolean) => (context: Parameters<typeof conversation>[0]) => {
    assert.equal(asks.test(text(conversation(context).at(-1)!)), asked);
    return fauxAssistantMessage('done');
  };
  const harness = await createSession(
    'off',
    [
      // The first command stays behind the note of a fold that took only the second.
      bash('x'.repeat(4800)),
      bash('tiny'),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't3', note: 'Saved it.' }), {
          stopReason: 'toolUse'
        }),
      done(true),
      // The merge removes the reply written since the nudge: asked once more. The merges
      // that answer remove notes only, and are not.
      merge(false),
      merge(true),
      merge(false),
      done(false),
      done(false)
    ],
    [],
    { foldNudgeTokens: 2000 }
  );
  try {
    await harness.run('old task', 'merge', 'nothing');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(episodeStats(branch, harness.dir).nudges, 2);
    assert.equal(threadRecords(branch).length, 4);
  } finally {
    await harness.cleanup();
  }
});

test('what a script printed next to its fold counts as room its note takes, and leaves with a merge', async () => {
  const bash = (size: number) => () =>
    script(`text(await tools.bash({ command: "echo ${'y'.repeat(size)}" }));`);
  const harness = await createSession(
    'only',
    [
      bash(5),
      // The result of this script is the note, and 3,000 tokens of output besides.
      () =>
        script(
          'text(await tools.fold({ since: "t1", note: "Ran it." })); text("x".repeat(12000));'
        ),
      () => fauxAssistantMessage('done'),
      // Not new work: the next prompt is not nudged for it.
      (context) => {
        assert.doesNotMatch(text(conversation(context).at(-1)!), /tokens of work built up/);
        return bash(5000)();
      },
      () => fauxAssistantMessage('done'),
      // Once new work is due for a fold, the nudge asks for one that takes in the note.
      (context) => {
        const nudge = text(conversation(context).at(-1)!);
        assert.match(
          nudge,
          /since "t3", the stamp at the end of your note ep-[0-9a-f]{8}, early enough to take in all your notes/
        );
        assert.match(
          nudge,
          /what they left behind \(about 3k tokens, sent again with every request\)/
        );
        return script('text(await tools.fold({ since: "t3", note: "Ran it, and printed." }));');
      },
      (context) => {
        assert.doesNotMatch(conversation(context).map(text).join('\n'), /xxxx|yyyy/);
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 2000 }
  );
  try {
    await harness.run('first', 'second', 'third');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 2);
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

test('a fold is refused while an earlier one is cut short, and works after the turn end', async () => {
  const first = await createSession('off', [
    () =>
      fauxAssistantMessage(fauxToolCall('bash', { command: 'echo first-output' }), {
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
    saved = [manager.getHeader()!, ...all.slice(0, edits[0]! + 1)];
  } finally {
    await first.cleanup();
  }

  const resumed = await createSession(
    'off',
    [
      // The rest of the first span is still in context, and so is its call of fold.
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Again.' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /^ep-[0-9a-f]{8} was interrupted/);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo second-output' }), {
          stopReason: 'toolUse'
        });
      },
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't4', note: 'Printed another.' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        const sent = conversation(context).map(text).join('\n');
        assert.doesNotMatch(sent, /first-output|second-output/);
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
    const branch = resumed.session.sessionManager.getBranch();
    const records = threadRecords(branch);
    assert.equal(records.length, 2);
    // Each entry is in one thread only.
    const threads = await new ThreadIndex(resumed.session.sessionManager, new Map()).onBranch();
    const hits = threads.filter((thread) => /first-output/.test(JSON.stringify(thread.messages)));
    assert.equal(hits.length, 1);
  } finally {
    await resumed.cleanup();
  }
});

test('a fold leaves a user message that an earlier turn-end handler edits at its boundary', async () => {
  const paste = 'p'.repeat(USER_TEXT_CAP * 2);
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo out' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'n' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        const sent = conversation(context).map(text).join('\n');
        assert.match(sent, /redacted/);
        assert.doesNotMatch(sent, /pppp/);
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
          const branch = ctx.sessionManager.getBranch();
          const folded = branch.some(
            (entry) =>
              entry.type === 'message' &&
              entry.message.role === 'toolResult' &&
              entry.message.toolName === 'fold'
          );
          const pasted = branch.find(
            (entry) => entry.type === 'message' && entry.message.role === 'user'
          );
          if (!folded || !pasted) return undefined;
          return {
            entries: [
              ...event.entries,
              {
                type: 'context_edit' as const,
                targetId: pasted.id,
                replacement: { content: [{ type: 'text' as const, text: 'redacted' }] }
              }
            ]
          };
        });
      }
    ]
  );
  try {
    await harness.run(paste);
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 1);
    // The start entry holds what the fold wrote, so completing it would not write the cut either.
    const started = branch.find(
      (entry) => entry.type === 'custom' && entry.customType === FOLD_STARTED_TYPE
    );
    const pasted = branch.find(
      (entry) => entry.type === 'message' && entry.message.role === 'user'
    );
    const planned = (started as { data: { edits: { targetId: string }[] } }).data.edits;
    assert.ok(planned.length > 0 && planned.every((edit) => edit.targetId !== pasted?.id));
  } finally {
    await harness.cleanup();
  }
});

test('the line of a fold is no turn that followed a failed call: its arguments are cut after three real ones', async () => {
  const words = Array.from({ length: 90 }, (_, index) => `word${index}`).join(' ');
  const bash = (command: string) => () =>
    fauxAssistantMessage(fauxToolCall('bash', { command }), { stopReason: 'toolUse' });
  const fold = (args: Parameters<typeof fauxToolCall>[1]) => () =>
    fauxAssistantMessage(fauxToolCall('fold', args), { stopReason: 'toolUse' });
  // Between the failed call and the fold of the first command stand the kept responses and a
  // line: of the second span of the same call, or of an earlier fold, whose call is a turn too.
  const cases = [
    { earlier: false, kept: 2, turns: 2 },
    { earlier: false, kept: 3, turns: 3 },
    { earlier: true, kept: 1, turns: 2 },
    { earlier: true, kept: 2, turns: 3 }
  ];
  for (const { earlier, kept, turns } of cases) {
    const last = `t${kept + 4}`;
    const first = { since: 't2', until: 't2', note: 'first' };
    const harness = await createSession('off', [
      bash('echo one'),
      bash(`false # ${words}`),
      ...Array.from({ length: kept }, (_, index) => bash(`echo kept ${index}`)),
      bash('echo last'),
      ...(earlier
        ? [fold({ since: last, note: 'second' }), fold(first)]
        : [fold({ spans: [first, { since: last, note: 'second' }] })]),
      (context) => {
        const sent = JSON.stringify(conversation(context));
        const label = `${earlier ? 'an earlier fold' : 'a second span'}, ${kept} kept`;
        assert.doesNotMatch(sent, /echo (one|last)/, label);
        assert.equal(sent.includes(words), turns < 3, label);
        assert.equal(/characters omitted from failed bash call/.test(sent), turns >= 3, label);
        return fauxAssistantMessage('done');
      }
    ]);
    try {
      await harness.run('go');
    } finally {
      await harness.cleanup();
    }
  }
});

test('a fold puts its line over a response that an earlier turn-end handler edits at its boundary', async () => {
  let edited = '';
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo out' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'n' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        // The response leaves with its span whatever the handler made of it: the line stands
        // there, and no call is left without its result.
        const sent = conversation(context).map(text);
        assert.match(sent[1]!, /^\[Earlier responses of mine stood here/);
        assert.doesNotMatch(sent.join('\n'), /redacted response/);
        assert.doesNotMatch(sent.join('\n'), /^out$/m);
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
          const branch = ctx.sessionManager.getBranch();
          const folded = branch.some(
            (entry) =>
              entry.type === 'message' &&
              entry.message.role === 'toolResult' &&
              entry.message.toolName === 'fold'
          );
          const response = branch.find(
            (entry) => entry.type === 'message' && entry.message.role === 'assistant'
          );
          if (!folded || !response) return undefined;
          edited = response.id;
          return {
            entries: [
              ...event.entries,
              {
                type: 'context_edit' as const,
                targetId: response.id,
                replacement: { content: [{ type: 'text' as const, text: 'redacted response' }] }
              }
            ]
          };
        });
      }
    ]
  );
  try {
    await harness.run('go');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(threadRecords(branch).length, 1);
    const started = branch.find(
      (entry) => entry.type === 'custom' && entry.customType === FOLD_STARTED_TYPE
    );
    const data = (started as { data: { edits: { targetId: string }[]; lines?: string[] } }).data;
    const planned = data.edits;
    assert.ok(edited !== '' && planned.some((edit) => edit.targetId === edited));
    // The start entry says where the line stands.
    assert.deepEqual(data.lines, [edited]);
  } finally {
    await harness.cleanup();
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

test('under codemode "only" the prompt gives the calls of fold, recall and ask', async () => {
  for (const mode of ['on', 'only'] as const) {
    let prompt = '';
    const harness = await createSession(mode, [
      (context) => {
        prompt = context.messages
          .filter((message) => message.role === 'system')
          .map(getSystemMessageText)
          .join('\n');
        return fauxAssistantMessage('ok');
      }
    ]);
    try {
      await harness.run('go');
      assert.match(prompt, /<episodes>/);
      assert.equal(/tools\.recall\(\{ query, handle\?/.test(prompt), mode === 'only', mode);
      assert.equal(/tools\.unfold\(\{ handle:/.test(prompt), mode === 'only', mode);
    } finally {
      await harness.cleanup();
    }
  }
});

for (const foldLine of [true, false]) {
  test(`a fold with spans from a script leaves ${foldLine ? 'a line for each span' : 'no line'} as set`, async () => {
    const harness = await createSession(
      'only',
      [
        () => script('text(await tools.bash({ command: "echo one" }));'),
        () => script('text(await tools.bash({ command: "echo two" }));'),
        () => script('text(await tools.bash({ command: "echo three" }));'),
        () =>
          script(
            'text(await tools.fold({ spans: [{ since: "t2", until: "t2", note: "one" }, { since: "t4", note: "three" }] }));'
          ),
        (context) => {
          const shown = conversation(context).map(text);
          assert.equal(
            shown.filter((message) => /Earlier responses of mine stood here/.test(message)).length,
            foldLine ? 2 : 0
          );
          assert.ok(shown.some((message) => /two/.test(message)));
          return fauxAssistantMessage('ok');
        }
      ],
      [],
      { foldLine }
    );
    try {
      await harness.run('go');
    } finally {
      await harness.cleanup();
    }
  });
}

for (const mode of ['off', 'only'] as const) {
  const call = (tool: string, args: Record<string, string>) =>
    mode === 'only'
      ? script(`text(await tools.${tool}(${JSON.stringify(args)}));`)
      : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });
  const answer = (said: string) => fauxAssistantMessage(said);
  const how =
    mode === 'only'
      ? /make one codemode call whose script calls only fold and prints its result/
      : /call only fold in this response/;
  const ending = mode === 'only' ? /reply with the one word "Folded\."/ : /the fold ends it/;
  // Where a script folds, the run goes on for one more response; a direct fold ends it.
  const reply = (check: (shown: string) => void): Step[] =>
    mode === 'only'
      ? [
          (context) => {
            check(conversation(context).map(text).join('\n'));
            return answer('Folded.');
          }
        ]
      : [];

  test(`a run that ends over the threshold is asked to fold all but its last request, and the next such fold takes that in (codemode ${mode})`, async () => {
    const [x, y, z, w] = ['x', 'y', 'z', 'w'].map((letter, index) =>
      letter.repeat(index % 2 === 0 ? 2800 : 600)
    ) as [string, string, string, string];
    const harness = await createSession(
      mode,
      [
        () => call('bash', { command: `echo ${x}` }),
        () => answer('first answer'),
        // About 1400 tokens so far: under the threshold. The second request takes it over.
        () => call('bash', { command: `echo ${y}` }),
        () => answer('second answer'),
        (context) => {
          const shown = conversation(context);
          const asked = text(shown.at(-1)!);
          assert.match(asked, /^The request above is answered\. About 1k tokens of finished work/);
          assert.match(asked, how);
          assert.match(asked, ending);
          assert.equal(text(shown.at(-2)!), 'second answer');
          // The count is of what the fold removes, without the request it keeps.
          // From the first stamp to the last one before the request just answered.
          assert.match(asked, /with since "t1" and until "t2" \(the turns after it stay in view/);
          assert.match(asked, /a note covering everything between them\. /);
          return call('fold', { since: 't1', until: 't2', note: 'Printed a long line of x.' });
        },
        ...reply((shown) => {
          assert.doesNotMatch(shown, /xxxx/);
          assert.match(shown, /yyyy/);
          assert.match(shown, /The request above is answered/);
        }),
        (context) => {
          const shown = conversation(context);
          // No nudge comes with the prompt: the fold is made.
          assert.match(text(shown.at(-1)!), /^third/);
          const all = shown.map(text).join('\n');
          assert.doesNotMatch(all, /xxxx/);
          assert.match(all, /yyyy/);
          assert.match(all, /second answer/);
          assert.match(all, /Printed a long line of x\./);
          return call('bash', { command: `echo ${z}` });
        },
        () => answer('third answer'),
        () => call('bash', { command: `echo ${w}` }),
        () => answer('fourth answer'),
        (context) => {
          const asked = text(conversation(context).at(-1)!);
          // From the request the first fold kept, behind its note, to before the last request.
          assert.match(asked, /with since "t4" and until "t7" \(the turns after it stay in view/);
          assert.match(
            asked,
            /a note covering everything between them that your notes do not say already: the fold keeps each note of earlier folds there under the new one as you wrote it\. /
          );
          return call('fold', { since: 't4', until: 't7', note: 'Printed a long line of z.' });
        },
        ...reply((shown) => {
          assert.doesNotMatch(shown, /yyyy|zzzz/);
          assert.match(shown, /wwww/);
        }),
        (context) => {
          const all = conversation(context).map(text).join('\n');
          assert.doesNotMatch(all, /xxxx|yyyy|zzzz|second answer|third answer/);
          assert.match(all, /wwww/);
          assert.match(all, /fourth answer/);
          // The first note stands once, under the second.
          assert.equal(all.split('Printed a long line of x.').length - 1, 1);
          assert.equal(all.split('Printed a long line of z.').length - 1, 1);
          return answer('fifth answer');
        }
      ],
      [],
      // With the terminal UI the fold is asked for by default, and applied with that UI bound.
      mode === 'off'
        ? { foldNudgeTokens: 1600 }
        : { foldNudgeTokens: 1600, foldAfterRun: 'always' },
      {},
      undefined,
      [],
      [],
      false,
      mode === 'off' ? 'tui' : false
    );
    try {
      await harness.run('first', 'second', 'third', 'fourth', 'fifth');
      const branch = harness.session.sessionManager.getBranch();
      const records = threadRecords(branch) as {
        handle: string;
        manual?: string;
        keptFrom?: string;
        covers?: string[];
      }[];
      assert.deepEqual(
        records.map((record) => [record.manual, record.keptFrom, record.covers ?? []]),
        [
          ['run-end', 't3', []],
          ['run-end', 't8', [records[0]!.handle]]
        ]
      );
      const stats = episodeStats(branch, harness.dir);
      assert.equal(stats.runEndAsks, 2);
      assert.equal(stats.nudges, 0);
      // The one word a script's fold is followed by is no final reply the agent chose to give.
      assert.equal(stats.folds.beforeFinal, 0);
      for (const type of ['episodes-nudge', 'episodes-fold-run-end']) {
        assert.ok(harness.session.extensionRunner.getMessageRenderer(type), type);
      }
    } finally {
      await harness.cleanup();
    }
  });

  test(`a last request too large to keep is folded with the rest, and the request for the fold stays in view (codemode ${mode})`, async () => {
    // One request over the threshold; a second one that is over a quarter of it; and, close
    // to compaction, where a quarter is enough, a second one larger than what stands before it:
    // its long answer takes the run over, with no nudge inside the run before it.
    for (const sizes of [[2200], [1800, 800], [300, 100]]) {
      const prints = sizes.map((size, index) => 'xy'[index]!.repeat(size));
      const pressed = sizes[0]! < 1000;
      // The answer that ended the run stays after the line, so that the request before it is not
      // shown answered by the line alone; the answers before it are folded.
      const assertOnlyLastAnswer = (shown: string) => {
        assert.match(shown, new RegExp(`answer ${sizes.length}`));
        for (let index = 1; index < sizes.length; index++) {
          assert.doesNotMatch(shown, new RegExp(`answer ${index}\\b`));
        }
      };
      const harness = await createSession(
        mode,
        [
          ...prints.flatMap((print, index): Step[] => [
            () => call('bash', { command: `echo ${print}` }),
            () => answer(`answer ${index + 1}${pressed && index === 1 ? ' z'.repeat(500) : ''}`)
          ]),
          (context) => {
            const asked = text(conversation(context).at(-1)!);
            assert.match(asked, /with since "t1", and a note covering everything from there\. /);
            return call('fold', { since: 't1', note: 'Printed long lines.' });
          },
          ...reply((shown) => {
            assert.doesNotMatch(shown, /xxxx|yyyy/);
            assertOnlyLastAnswer(shown);
            assert.match(shown, /The request above is answered/);
          }),
          (context) => {
            const all = conversation(context).map(text).join('\n');
            assert.doesNotMatch(all, /xxxx|yyyy/);
            assertOnlyLastAnswer(all);
            // What asked for the fold is still there, and so is every request of the user.
            assert.match(all, /The request above is answered/);
            for (const index of sizes.keys()) assert.match(all, new RegExp(`request ${index + 1}`));
            return answer('last answer');
          }
        ],
        [],
        { foldNudgeTokens: pressed ? 1600 : 1000, foldAfterRun: 'always' },
        pressed ? { reserveTokens: 127_000 } : {}
      );
      try {
        await harness.run(...sizes.map((_, index) => `request ${index + 1}`), 'last');
        const records = threadRecords(harness.session.sessionManager.getBranch()) as {
          keptFrom?: string;
        }[];
        assert.equal(records.length, 1, String(sizes));
        assert.equal(records[0]!.keptFrom, undefined);
      } finally {
        await harness.cleanup();
      }
    }
  });
}

test('the fold at the end of a run is asked for where there is a UI or the setting says always, once, and the nudge at the next prompt is what is left elsewhere', async () => {
  const cases: [settings: Record<string, unknown>, ui: 'tui' | 'rpc' | false, asked: boolean][] = [
    [{}, 'tui', true],
    // An RPC host has a UI too, and may still take the last response of a run for its answer.
    [{}, 'rpc', false],
    [{}, false, false],
    [{ foldAfterRun: 'always' }, false, true],
    [{ foldAfterRun: 'always' }, 'rpc', true],
    [{ foldAfterRun: false }, 'tui', false],
    [{ foldAfterRun: 'no' }, 'tui', true],
    [{ foldNudge: false }, 'tui', false]
  ];
  for (const [settings, ui, asked] of cases) {
    const label = JSON.stringify([settings, ui]);
    const nudged = !asked && settings.foldNudge !== false;
    const harness = await createSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
            stopReason: 'toolUse'
          }),
        () => fauxAssistantMessage('done'),
        // Asked, the agent does not fold: it is not asked again when this response ends the run.
        ...(asked
          ? [
              (context: TranscriptContext) => {
                assert.match(
                  text(conversation(context).at(-1)!),
                  /^The request above is answered/,
                  label
                );
                return fauxAssistantMessage('Not now.');
              }
            ]
          : []),
        (context: TranscriptContext) => {
          const last = text(conversation(context).at(-1)!);
          if (nudged) {
            assert.match(last, /tokens of work built up since the last fold/, label);
            assert.match(last, /the work before it would stay in your context\.$/, label);
          } else {
            // Nor with the next prompt: growth is counted from where it was asked.
            assert.match(last, /^second/, label);
          }
          return fauxAssistantMessage('done');
        }
      ],
      [],
      { foldNudgeTokens: 300, ...settings },
      {},
      undefined,
      [],
      [],
      false,
      ui
    );
    try {
      await harness.run('first', 'second');
      const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
      assert.equal(stats.runEndAsks, asked ? 1 : 0, label);
      assert.equal(stats.nudges, nudged ? 1 : 0, label);
      // Either message is counted with what the extension adds to the context.
      assert.equal(stats.overhead.notices > 0, asked || nudged, label);
      assert.equal(
        statsLines(stats).includes('folds asked for at the end of a run 1'),
        asked,
        label
      );
    } finally {
      await harness.cleanup();
    }
  }
});

test('after the fold a script made at the end of a run, no call runs until the next message of the user', async () => {
  const x = 'x'.repeat(4000);
  let ran = 0;
  let ended = 0;
  const counts: ExtensionFactory = (pi) => {
    pi.on('tool_execution_start', (event) => {
      if (event.toolName === 'bash') ran++;
    });
    pi.on('agent_before_settle', () => {
      ended++;
      return undefined;
    });
  };
  const harness = await createSession(
    'only',
    [
      () => script(`text(JSON.stringify(await tools.bash({ command: "echo ${x}" })));`),
      () => fauxAssistantMessage('first answer'),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /^The request above is answered/);
        // A call made next to the fold, before it is applied, runs as it did.
        return script(
          `text(await tools.fold({ since: "t1", note: "Printed x." })); await tools.bash({ command: "echo beside" });`
        );
      },
      // The agent goes on instead of ending the run: on the request it kept, or on one to come.
      // Its calls are not run, and the run ends there, with no further response.
      () =>
        fauxAssistantMessage(
          [
            fauxText('Folded.'),
            fauxToolCall('codemode', {
              code: 'text(await tools.bash({ command: "echo ahead" }));'
            }),
            fauxToolCall('codemode', { code: 'text(await tools.bash({ command: "echo more" }));' })
          ],
          { stopReason: 'toolUse' }
        ),
      // The next message of the user ends that: its calls run.
      (context) => {
        const shown = conversation(context).map(text);
        assert.match(shown.at(-1)!, /^second/);
        assert.ok(shown.at(-2)!.startsWith(AFTER_RUN_END_FOLD), shown.at(-2));
        assert.ok(shown.at(-3)!.startsWith(AFTER_RUN_END_FOLD), shown.at(-3));
        assert.equal(ran, 2);
        return script('text(JSON.stringify(await tools.bash({ command: "echo next" })));');
      },
      (context) => {
        assert.match(conversation(context).map(text).join('\n'), /next/);
        assert.equal(ran, 3);
        return fauxAssistantMessage('second answer');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' },
    {},
    undefined,
    [counts]
  );
  try {
    await harness.run('first', 'second');
    const branch = harness.session.sessionManager.getBranch();
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.runEndAsks, 1);
    assert.equal(stats.folds.count, 1);
    // Pi ended the run by itself, as it ends any run: once after the answer, where the fold
    // was asked for, once after the calls that were not run, and once for the second request.
    // A run that is stopped does not come by there.
    assert.equal(ended, 3);
  } finally {
    await harness.cleanup();
  }
});

test('a fold asked for at the end of a run ends it also after a try that was refused', async () => {
  const fold = (since: string) =>
    fauxAssistantMessage(fauxToolCall('fold', { since, note: 'Printed x.' }), {
      stopReason: 'toolUse'
    });
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      () => fold('t9'),
      (context) => {
        assert.doesNotMatch(text(conversation(context).at(-1)!), /^\[fold /);
        return fold('t1');
      },
      // That fold ended the run: this is the answer to the next prompt. The request and the
      // try that was refused are still in view.
      (context) => {
        const shown = conversation(context).map(text);
        assert.match(shown.at(-1)!, /^second/);
        assert.match(shown.join('\n'), /The request above is answered/);
        assert.doesNotMatch(shown.join('\n'), /xxxx/);
        return fauxAssistantMessage('second answer');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first', 'second');
    const records = threadRecords(harness.session.sessionManager.getBranch()) as {
      manual?: string;
    }[];
    assert.deepEqual(
      records.map((record) => record.manual),
      ['run-end']
    );
  } finally {
    await harness.cleanup();
  }
});

test('a message that arrives while the fold at the end of a run is asked for gets its answer', async () => {
  // Sent from the same boundary, by an extension loaded before or after this one: pi has not
  // queued it yet when the request for the fold is made.
  for (const where of ['before', 'after'] as const) {
    for (const kind of ['steer', 'followUp'] as const) {
      const label = `${where} ${kind}`;
      let sent = false;
      const sends: ExtensionFactory = (pi) => {
        pi.on('agent_before_settle', () => {
          if (!sent) pi.sendUserMessage('steered', { deliverAs: kind });
          sent = true;
          return undefined;
        });
      };
      const harness = await createSession(
        'off',
        [
          () =>
            fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
              stopReason: 'toolUse'
            }),
          () => fauxAssistantMessage('first answer'),
          (context) => {
            const shown = conversation(context).map(text);
            // A steering message is delivered before this response, a follow-up after it.
            assert.equal(shown.at(-1)!.startsWith('steered'), kind === 'steer', label);
            // With the message behind it the request is not sent: it would stand in front of
            // new work and ask for a fold first.
            assert.equal(
              shown.some((said) => said.startsWith('The request above is answered')),
              kind === 'followUp',
              label
            );
            return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Printed x.' }), {
              stopReason: 'toolUse'
            });
          },
          // The run goes on to the message, whether the fold came before it or after.
          (context) => {
            const shown = conversation(context).map(text).join('\n');
            assert.match(shown, /steered/, label);
            assert.match(shown, /Printed x\./, label);
            return fauxAssistantMessage('steered answer');
          }
        ],
        where === 'after' ? [sends] : [],
        { foldNudgeTokens: 1600, foldAfterRun: 'always' },
        {},
        undefined,
        where === 'before' ? [sends] : []
      );
      try {
        await harness.run('first');
        const records = threadRecords(harness.session.sessionManager.getBranch()) as {
          manual?: string;
        }[];
        // With the message in front of it, the fold no longer answers the request alone.
        assert.deepEqual(
          records.map((record) => record.manual),
          [kind === 'steer' ? undefined : 'run-end'],
          label
        );
        // That fold removed the request with its span, once; the fold that answered the
        // request left it in view.
        assert.equal(
          askRemovals(harness.session.sessionManager.getBranch()).length,
          kind === 'steer' ? 1 : 0,
          label
        );
      } finally {
        await harness.cleanup();
      }
    }
  }
});

test('a message another extension adds behind the request for the fold is what the run goes on for, and the fold is asked for after that', async () => {
  let added = false;
  const adds: ExtensionFactory = (pi) => {
    pi.on('agent_before_settle', (event) => {
      const asked = event.entries.some(
        (entry) => entry.type === 'custom_message' && entry.customType === 'episodes-fold-run-end'
      );
      if (added || !asked) return undefined;
      added = true;
      const more = { type: 'custom_message' as const, customType: 'other', content: 'Check it.' };
      return { entries: [...event.entries, { ...more, display: true }], continue: true };
    });
  };
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      (context) => {
        const shown = conversation(context).map(text);
        // The request is not sent with the other message behind it.
        assert.equal(shown.at(-1), 'Check it.');
        assert.equal(shown.at(-2), 'first answer');
        return fauxAssistantMessage('checked');
      },
      // That request counts for nothing: with the other message answered, the fold is asked for.
      (context) => {
        const shown = conversation(context).map(text);
        assert.match(shown.at(-1)!, /^The request above is answered/);
        assert.equal(shown.at(-2), 'checked');
        assert.equal(shown.filter((said) => said.startsWith('The request above')).length, 1);
        return fauxAssistantMessage(fauxToolCall('fold', { since: 't1', note: 'Printed x.' }), {
          stopReason: 'toolUse'
        });
      }
    ],
    [adds],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first');
    const branch = harness.session.sessionManager.getBranch();
    const records = threadRecords(branch) as { manual?: string }[];
    assert.deepEqual(
      records.map((record) => record.manual),
      ['run-end']
    );
    assert.equal(episodeStats(branch, harness.dir).runEndAsks, 2);
    // The first request left the session when the response to the other message ended, so
    // nothing built from the session later holds it; the second was answered and stays.
    const asks = branch.filter(
      (entry) => entry.type === 'custom_message' && entry.customType === 'episodes-fold-run-end'
    );
    assert.deepEqual(
      askRemovals(branch).map((edit) => edit.targetId),
      [asks[0]!.id]
    );
  } finally {
    await harness.cleanup();
  }
});

test('work that an extension continues the run with after the fold at its end is not refused, nor what follows a fold made in it', async () => {
  let ran = 0;
  let continued = false;
  const continues: ExtensionFactory = (pi) => {
    pi.on('tool_execution_start', (event) => {
      if (event.toolName === 'bash') ran++;
    });
    // Once the fold is made and answered, the same run goes on for a message of this extension.
    pi.on('agent_before_settle', (event) => {
      const folded = event.context.contextEntries.some(
        ({ sourceEntry: source }) =>
          source.type === 'custom' && source.customType === FOLD_APPLIED_TYPE
      );
      if (continued || !folded) return undefined;
      continued = true;
      const more = { type: 'custom_message' as const, customType: 'other', content: 'Check it.' };
      return { entries: [...event.entries, { ...more, display: true }], continue: true };
    });
  };
  const harness = await createSession(
    'only',
    [
      () =>
        script(`text(JSON.stringify(await tools.bash({ command: "echo ${'x'.repeat(4000)}" })));`),
      () => fauxAssistantMessage('first answer'),
      () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
      () => fauxAssistantMessage('Folded.'),
      () => script('text(JSON.stringify(await tools.bash({ command: "echo check" })));'),
      // A fold the agent makes by itself in that work is not one asked for at the end of a run.
      (context) => {
        const stamp = /\[(t\d+)\]\s*$/.exec(text(conversation(context).at(-1)!))![1];
        return script(`text(await tools.fold({ since: "${stamp}", note: "Checked." }));`);
      },
      () => script('text(JSON.stringify(await tools.bash({ command: "echo goes-on" })));'),
      (context) => {
        assert.match(conversation(context).map(text).join('\n'), /goes-on/);
        return fauxAssistantMessage('checked');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' },
    {},
    undefined,
    [continues]
  );
  try {
    await harness.run('first');
    assert.equal(ran, 3);
    const records = threadRecords(harness.session.sessionManager.getBranch()) as {
      manual?: string;
    }[];
    assert.deepEqual(
      records.map((record) => record.manual),
      ['run-end', undefined]
    );
  } finally {
    await harness.cleanup();
  }
});

test('turns a fold kept that the agent was asked about at the end of the run and left are not asked about again with the next prompt', async () => {
  const bash = (print: string) =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${print}` }), {
      stopReason: 'toolUse'
    });
  const harness = await createSession(
    'off',
    [
      () => bash('y'.repeat(200)),
      () => bash('z'.repeat(3400)),
      // The agent keeps the second command in view; it comes to more than the threshold.
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't1', until: 't2', note: 'Printed y.' }),
          { stopReason: 'toolUse' }
        ),
      () => fauxAssistantMessage('first answer'),
      (context) => {
        const asked = text(conversation(context).at(-1)!);
        assert.match(asked, /^The request above is answered\. About 2k tokens of finished work/);
        assert.match(asked, /with since "t3", and a note covering everything from there that/);
        return fauxAssistantMessage('Not now.');
      },
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /^second/);
        return fauxAssistantMessage('second answer');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first', 'second');
    const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
    assert.equal(stats.runEndAsks, 1);
    assert.equal(stats.nudges, 0);
  } finally {
    await harness.cleanup();
  }
});

test('a fold at the end of a run starts at what the last one kept, and at an earlier note once what stands around the notes it would leave takes up room', async () => {
  const bash = (print: string) =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${print}` }), {
      stopReason: 'toolUse'
    });
  const fold = (args: Record<string, string>) =>
    fauxAssistantMessage(fauxToolCall('fold', args), { stopReason: 'toolUse' });
  const done = () => fauxAssistantMessage('done');
  // A fold the user asked for comes first, with a note whose call takes up room at once,
  // none to speak of, or enough together with what the next fold keeps. The two requests at
  // the end of a run that follow name an earlier note or do not. The requests themselves
  // take up room behind the notes too, about 150 tokens each.
  for (const [detail, named] of [
    [600, [true, false]],
    [0, [false, false]],
    [400, [false, true]]
  ] as const) {
    const asked: string[] = [];
    const answer = (context: TranscriptContext) => {
      asked.push(text(conversation(context).at(-1)!));
      const [, since, until] = /with since "(t\d+)"[^"]* and until "(t\d+)"/.exec(asked.at(-1)!)!;
      return fold({ since: since!, until: until!, note: `Printed more, ${asked.length}.` });
    };
    const harness = await createSession(
      'off',
      [
        () => bash('x'.repeat(7200)),
        done,
        () => fold({ since: 't1', note: `Printed x. ${'detail '.repeat(detail)}` }),
        () => bash('y'.repeat(7400)),
        done,
        () => bash('z'.repeat(800)),
        done,
        answer,
        () => bash('v'.repeat(7400)),
        done,
        () => bash('w'.repeat(800)),
        done,
        answer
      ],
      [],
      { foldNudgeTokens: 4000, foldAfterRun: 'always' }
    );
    try {
      await harness.run('one', '/fold t1', 'two', 'three', 'four', 'five');
      const records = threadRecords(harness.session.sessionManager.getBranch()) as {
        handle: string;
        covers?: string[];
      }[];
      const [first, second, third] = records.map((record) => record.handle) as [
        string,
        string,
        string
      ];
      const label = String(detail);
      assert.equal(records.length, 3, label);
      for (const [index, said] of asked.entries()) {
        // No nudge would come to ask for the merge: each of these requests counts as one.
        const note = new RegExp(
          `with since "t\\d+", the stamp at the end of your note ${first} and until "t\\d+"`
        );
        if (named[index]) assert.match(said, note, label);
        else assert.doesNotMatch(said, /the stamp at the end of your note/, label);
      }
      assert.deepEqual(
        records.slice(1).map((record) => record.covers ?? []),
        // The second such fold takes in the note of the first and what that kept, whether or
        // not it reaches back to the note before them. A note kept under one it covers counts.
        [named[0] ? [first] : [], named[0] || named[1] ? [first, second] : [second]],
        label
      );
      assert.equal(/keeps each note of earlier folds there/.test(asked[0]!), named[0], label);
      assert.match(asked[1]!, /keeps each note of earlier folds there/, label);
      if (named[0]) {
        // What the fold frees, without the note of 1k tokens that it keeps.
        assert.match(asked[0]!, /^The request above is answered\. About 5k tokens of finished/);
      }
      void third;
    } finally {
      await harness.cleanup();
    }
  }
});

test('a run that did not end in an answer is not asked to fold, and one that another extension continues is asked when that is done', async () => {
  const bash = () =>
    fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(800)}` }), {
      stopReason: 'toolUse'
    });
  // Cut off by the length limit: nothing is answered yet.
  const cut = await createSession(
    'off',
    [bash, () => fauxAssistantMessage('partial', { stopReason: 'length' })],
    [],
    { foldNudgeTokens: 300, foldAfterRun: 'always' }
  );
  try {
    await cut.run('first');
    assert.equal(episodeStats(cut.session.sessionManager.getBranch(), cut.dir).runEndAsks, 0);
  } finally {
    await cut.cleanup();
  }

  // Another extension continues the run with a message of its own.
  {
    let continued = false;
    const continues: ExtensionFactory = (pi) => {
      pi.on('agent_before_settle', (event) => {
        if (continued) return undefined;
        continued = true;
        const more = { type: 'custom_message' as const, customType: 'other', content: 'queued' };
        return { entries: [...event.entries, { ...more, display: true }], continue: true };
      });
    };
    const harness = await createSession(
      'off',
      [
        bash,
        () => fauxAssistantMessage('done'),
        (context) => {
          const shown = conversation(context).map(text);
          assert.doesNotMatch(shown.join('\n'), /The request above is answered/);
          assert.match(shown.at(-1)!, /^queued/);
          return fauxAssistantMessage('more');
        },
        (context) => {
          assert.match(text(conversation(context).at(-1)!), /^The request above is answered/);
          return fauxAssistantMessage('Not now.');
        }
      ],
      [],
      { foldNudgeTokens: 300, foldAfterRun: 'always' },
      {},
      undefined,
      [continues]
    );
    try {
      await harness.run('first');
      const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
      assert.equal(stats.runEndAsks, 1);
    } finally {
      await harness.cleanup();
    }
  }

  // Another extension writes an entry the agent does not see: the answer is still the last
  // thing shown, and the entry is written with the request.
  {
    const marks: ExtensionFactory = (pi) => {
      pi.on('agent_before_settle', (event) => ({
        entries: [...event.entries, { type: 'custom' as const, customType: 'mark', data: {} }]
      }));
    };
    const harness = await createSession(
      'off',
      [bash, () => fauxAssistantMessage('done'), () => fauxAssistantMessage('Not now.')],
      [],
      { foldNudgeTokens: 300, foldAfterRun: 'always' },
      {},
      undefined,
      [marks]
    );
    try {
      await harness.run('first');
      const branch = harness.session.sessionManager.getBranch();
      assert.equal(episodeStats(branch, harness.dir).runEndAsks, 1);
      const types = branch.map((entry) =>
        entry.type === 'custom' || entry.type === 'custom_message' ? entry.customType : entry.type
      );
      assert.deepEqual(types.slice(-4), ['mark', 'episodes-fold-run-end', 'message', 'mark']);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a request for the fold that the provider failed on leaves the session when its run ends, so a fork does not start from it', async () => {
  let child = '';
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      () => fauxAssistantMessage('', { stopReason: 'error', errorMessage: 'boom' }),
      () =>
        fauxAssistantMessage(fauxToolCall('fork', { task: 'Report what was printed.' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        child = conversation(context).map(text).join('\n');
        return fauxAssistantMessage('child report');
      },
      () => fauxAssistantMessage('second answer')
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.session.prompt('first');
    await harness.session.waitForIdle();
    // Removed once, when the run that failed on it ended, and not again later.
    assert.equal(askRemovals(harness.session.sessionManager.getBranch()).length, 1);
    await harness.run('second');
    assert.match(child, /xxxx/);
    assert.doesNotMatch(child, /The request above is answered/);
    assert.equal(askRemovals(harness.session.sessionManager.getBranch()).length, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a session that ended on the request for the fold sends it to no one when it goes on: not to the agent, a fork or ask', async () => {
  const settings = { foldNudgeTokens: 1600, foldAfterRun: 'always' };
  const source = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      () => fauxAssistantMessage('', { stopReason: 'error', errorMessage: 'boom' })
    ],
    [],
    settings
  );
  try {
    await source.run('first');
    // As the file stood when the process ended: the request is its last entry.
    const all = source.session.sessionManager.getEntries();
    const cut = all.slice(
      0,
      all.findLastIndex(
        (entry) => entry.type === 'custom_message' && entry.customType === 'episodes-fold-run-end'
      ) + 1
    );
    const sent: string[] = [];
    const see = (context: TranscriptContext) => {
      sent.push(conversation(context).map(text).join('\n'));
    };
    const call = (tool: string, args: Record<string, string>) =>
      fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });

    // The first response starts a fork: the removal is not written yet.
    const forks = await createSession(
      'off',
      [
        (context) => {
          see(context);
          return call('fork', { task: 'Report what was printed.' });
        },
        (context) => {
          see(context);
          return fauxAssistantMessage('child report');
        },
        (context) => {
          see(context);
          // Written when the first response ended, not only when the run does.
          assert.equal(askRemovals(forks.session.sessionManager.getBranch()).length, 1);
          return fauxAssistantMessage('second answer');
        }
      ],
      [],
      settings,
      {},
      cut
    );
    try {
      await forks.run('second');
      // The nudge is the question about that work now.
      assert.match(sent[0]!, /tokens of work built up since the last fold/);
      assert.match(sent[1]!, /xxxx/);
      assert.equal(askRemovals(forks.session.sessionManager.getBranch()).length, 1);
    } finally {
      await forks.cleanup();
    }

    // The first response folds up to the last stamp before the request, so the thread of that
    // fold ends on the request, and the second asks the thread. No nudge here: what the nudge
    // adds would stand behind the request in the thread.
    const asks = await createSession(
      'off',
      [
        (context) => {
          see(context);
          return call('fold', { since: 't1', until: 't2', note: 'Printed x.' });
        },
        (context) => {
          see(context);
          const handle = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
          return call('ask', { handle, question: 'What was printed?' });
        },
        (context) => {
          see(context);
          assert.match(sent.at(-1)!, /What was printed\?/);
          assert.match(sent.at(-1)!, /xxxx/);
          return fauxAssistantMessage('A line of x.');
        },
        (context) => {
          see(context);
          return fauxAssistantMessage('second answer');
        }
      ],
      [],
      { ...settings, foldNudge: false },
      {},
      cut
    );
    try {
      await asks.run('second');
      // The fold removed the request with its span, and nothing removed it a second time.
      assert.equal(askRemovals(asks.session.sessionManager.getBranch()).length, 1);
    } finally {
      await asks.cleanup();
    }
    assert.equal(sent.length, 7);
    for (const shown of sent) assert.doesNotMatch(shown, /The request above is answered/);
  } finally {
    await source.cleanup();
  }
});

test('a call after the fold at the end of a run is not run, also next to calls that pi refused, and a message that came in gets its answer', async () => {
  const more = { code: 'text(await tools.bash({ command: "echo more" }));' };
  const calls = (...args: Record<string, string>[]) =>
    fauxAssistantMessage(
      args.map((arg) => fauxToolCall('codemode', arg)),
      { stopReason: 'toolUse' }
    );
  // A call whose arguments do not validate never comes by the extension: pi answers it and
  // sends the next request. Nothing is run in between, and the run ends with the first
  // response that makes no such call.
  const cases = [
    { name: 'next to one', after: [() => calls(more, {}), () => fauxAssistantMessage('Folded.')] },
    { name: 'after one', after: [() => calls({}), () => calls(more)] },
    { name: 'alone', after: [() => calls(more)] }
  ];
  for (const { name, after } of cases) {
    const harness = await createSession(
      'only',
      [
        () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
        () => fauxAssistantMessage('first answer'),
        () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
        ...after
      ],
      [],
      { foldNudgeTokens: 1600, foldAfterRun: 'always' }
    );
    try {
      await harness.run('first');
      assert.equal(harness.requests.length, 3 + after.length, name);
      const results = harness.session.messages.filter((message) => message.role === 'toolResult');
      assert.equal(
        results.filter((result) => text(result).startsWith(AFTER_RUN_END_FOLD)).length,
        1,
        name
      );
      assert.doesNotMatch(JSON.stringify(harness.session.messages), /"more\\n/, name);
      // No response was cut off.
      for (const entry of harness.session.sessionManager.getBranch()) {
        if (entry.type !== 'message' || entry.message.role !== 'assistant') continue;
        assert.match(entry.message.stopReason, /^(stop|toolUse)$/, name);
      }
    } finally {
      await harness.cleanup();
    }
  }

  // A message that arrives during that response is answered, sent either way.
  for (const deliverAs of ['steer', 'followUp'] as const) {
    let api: Parameters<ExtensionFactory>[0] | undefined;
    const answers = (context: TranscriptContext) =>
      fauxAssistantMessage(
        conversation(context).some((message) => text(message).includes('one more thing'))
          ? 'answered that'
          : 'Folded.'
      );
    const harness = await createSession(
      'only',
      [
        () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
        () => fauxAssistantMessage('first answer'),
        () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
        () => {
          api!.sendUserMessage('one more thing', { deliverAs });
          return calls(more, {});
        },
        answers,
        ...(deliverAs === 'followUp' ? [answers] : [])
      ],
      [
        (pi) => {
          api = pi;
        }
      ],
      { foldNudgeTokens: 1600, foldAfterRun: 'always' }
    );
    try {
      await harness.run('first');
      assert.equal(text(harness.session.messages.at(-1) as Message), 'answered that', deliverAs);
      assert.equal(harness.session.pendingMessageCount, 0, deliverAs);
    } finally {
      await harness.cleanup();
    }
  }
});

test('a fold at the end of a run that did not take effect is not taken for made: the agent can try again', async () => {
  // A later handler takes the note out of the result of the first try, so that fold is not
  // applied, though its record says what it was asked for. Another fold was applied before.
  let redacted = false;
  const redact: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const holds = event.content.some(
        (block) => block.type === 'text' && block.text.includes('Secret note.')
      );
      if (redacted || event.toolName !== 'codemode' || event.parentToolCallId || !holds) {
        return undefined;
      }
      redacted = true;
      return {
        content: event.content.map((block) =>
          block.type === 'text'
            ? { ...block, text: block.text.replace('Secret note.', 'redacted') }
            : block
        )
      };
    });
  };
  const harness = await createSession(
    'only',
    [
      () => script('text(await tools.bash({ command: "echo old" }));'),
      () => script('text(await tools.fold({ since: "t1", note: "An earlier fold." }));'),
      () => fauxAssistantMessage('old answer'),
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () => script('text(await tools.fold({ since: "t4", note: "Secret note." }));'),
      (context) => {
        assert.equal(appliedFolds(harness.session.sessionManager.getBranch()).size, 1);
        assert.match(conversation(context).map(text).join('\n'), /xxxx/);
        return script('text(await tools.fold({ since: "t4", note: "Printed x." }));');
      },
      () => fauxAssistantMessage('Folded.')
    ],
    [redact],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('old', 'first');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(appliedFolds(branch).size, 2);
    assert.doesNotMatch(JSON.stringify(harness.session.messages), /Not run: the fold/);
    assert.equal(
      (threadRecords(branch).at(-1) as { manual?: string } | undefined)?.manual,
      'run-end'
    );
  } finally {
    await harness.cleanup();
  }
});

test('work that an extension continues the run with behind the fold at its end gets the nudge of a long run', async () => {
  const direct = (name: string, args: Record<string, string>) =>
    fauxAssistantMessage(fauxToolCall(name, args), { stopReason: 'toolUse' });
  let continued = false;
  const more: ExtensionFactory = (pi) => {
    pi.on('agent_before_settle', (event, ctx) => {
      if (continued || appliedFolds(ctx.sessionManager.getBranch()).size === 0) return undefined;
      continued = true;
      return {
        entries: [
          ...event.entries,
          {
            type: 'custom_message' as const,
            customType: 'more-work',
            content: 'Do the next check.',
            display: true
          }
        ],
        continue: true
      };
    });
  };
  const harness = await createSession(
    'off',
    [
      () => direct('bash', { command: `echo ${'x'.repeat(4000)}` }),
      () => fauxAssistantMessage('first answer'),
      // The request for the fold stays in view behind the fold that answers it.
      () => direct('fold', { since: 't1', note: 'Printed x.' }),
      () => direct('bash', { command: `echo ${'z'.repeat(8000)}` }),
      (context) => {
        const shown = conversation(context).map(text);
        assert.match(shown.join('\n'), /The request above is answered/);
        assert.match(shown.at(-1)!, /tokens of work built up/);
        return fauxAssistantMessage('checked');
      }
    ],
    [more],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first');
    assert.equal(episodeStats(harness.session.sessionManager.getBranch(), harness.dir).nudges, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a request for the fold whose answer another handler removes when the run ends leaves the session there, with nothing to ask', async () => {
  const seed = SessionManager.inMemory();
  seed.appendMessage({ role: 'user', content: [{ type: 'text', text: 'old' }], timestamp: 0 });
  seed.appendMessage(fauxAssistantMessage('old answer'));
  const ask = seed.appendCustomMessageEntry('episodes-fold-run-end', 'Fold that work now.', true);
  const denial = seed.appendMessage(fauxAssistantMessage('Not now.'));
  // Registered before this extension: its edit is in the preview this one sees.
  const trims: ExtensionFactory = (pi) => {
    pi.on('agent_before_settle', (event) => ({
      entries: [
        ...event.entries,
        { type: 'context_edit' as const, targetId: denial, replacement: null }
      ]
    }));
  };
  const harness = await createSession(
    'off',
    [() => fauxAssistantMessage('new answer')],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always', foldNudge: false },
    {},
    seed.getEntries(),
    [trims]
  );
  try {
    await harness.run('new');
    const branch = harness.session.sessionManager.getBranch();
    assert.deepEqual(
      askRemovals(branch).map((edit) => edit.type === 'context_edit' && edit.targetId),
      [ask]
    );
  } finally {
    await harness.cleanup();
  }
});

test('a call next to the direct fold at the end of a run runs, and what comes after the fold is applied does not', async () => {
  const bash = (print: string) => fauxToolCall('bash', { command: `echo ${print}` });
  const fold = fauxToolCall('fold', { since: 't1', note: 'Printed x.' });
  // Asked to call only fold, the agent adds a call, behind the fold or in front of it. The
  // fold is not applied before the response ends, so that call runs and the run goes on; the
  // call of the next response is not run, and the run ends.
  for (const first of ['fold', 'bash'] as const) {
    const harness = await createSession(
      'off',
      [
        () => fauxAssistantMessage(bash('x'.repeat(4000)), { stopReason: 'toolUse' }),
        () => fauxAssistantMessage('first answer'),
        () =>
          fauxAssistantMessage(first === 'fold' ? [fold, bash('beside')] : [bash('beside'), fold], {
            stopReason: 'toolUse'
          }),
        () => fauxAssistantMessage(bash('after'), { stopReason: 'toolUse' })
      ],
      [],
      { foldNudgeTokens: 1600, foldAfterRun: 'always' }
    );
    try {
      await harness.run('first');
      // The command of the first request went with the fold.
      const results = (harness.session.messages as Message[])
        .filter((message) => message.role === 'toolResult' && message.toolName === 'bash')
        .map(text);
      assert.equal(results.length, 2, first);
      assert.match(results[0]!, /^beside\n/, first);
      assert.ok(results[1]!.startsWith(AFTER_RUN_END_FOLD), first);
      assert.equal(harness.requests.length, 4, first);
    } finally {
      await harness.cleanup();
    }
  }
});

test('no nudge comes between the request for the fold at the end of a run and the fold, however large a try that was refused', async () => {
  const fold = (since: string, note: string) =>
    fauxAssistantMessage(fauxToolCall('fold', { since, note }), { stopReason: 'toolUse' });
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      // Refused, and by itself twice the threshold: what a nudge inside a run waits for.
      () => fold('t99', 'word '.repeat(4000)),
      (context) => {
        assert.doesNotMatch(conversation(context).map(text).join('\n'), /tokens of work built up/);
        return fold('t1', 'Printed x.');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first');
    const branch = harness.session.sessionManager.getBranch();
    // The fold that was made answers the request, so it ended the run.
    assert.equal((threadRecords(branch).at(-1) as { manual?: string }).manual, 'run-end');
    assert.equal(harness.requests.length, 4);
    const stats = episodeStats(branch, harness.dir);
    assert.equal(stats.nudges, 0);
    assert.equal(stats.runEndAsks, 1);
  } finally {
    await harness.cleanup();
  }
});

test('a request for the fold stays for the try that pi makes again after a failure of the provider', async () => {
  const harness = await createSession(
    'only',
    [
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () =>
        fauxAssistantMessage('', { stopReason: 'error', errorMessage: '429 rate limit exceeded' }),
      (context) => {
        assert.match(conversation(context).map(text).join('\n'), /The request above is answered/);
        return script('text(await tools.fold({ since: "t1", note: "Printed x." }));');
      },
      () => fauxAssistantMessage('Folded.')
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  harness.session.settingsManager.applyOverrides({
    retry: { enabled: true, maxRetries: 1, baseDelayMs: 1 }
  });
  try {
    await harness.run('first');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(askRemovals(branch).length, 0);
    assert.equal((threadRecords(branch)[0] as { manual?: string }).manual, 'run-end');
  } finally {
    await harness.cleanup();
  }
});

test('a request for the fold that got no answer leaves the session also when the setting was turned off since, with what an earlier handler queued', async () => {
  const settings = { foldNudgeTokens: 1600, foldAfterRun: 'always' };
  const mark: ExtensionFactory = (pi) => {
    pi.on('agent_before_settle', (event) => ({
      entries: [...event.entries, { type: 'custom' as const, customType: 'mark', data: {} }]
    }));
  };
  const harness = await createSession(
    'only',
    [
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () => {
        harness.session.settingsManager.applyOverrides({
          episodes: { ...settings, foldAfterRun: false }
        } as object);
        return fauxAssistantMessage('', { stopReason: 'error', errorMessage: 'boom' });
      }
    ],
    [],
    settings,
    {},
    undefined,
    [mark]
  );
  try {
    await harness.run('first');
    const branch = harness.session.sessionManager.getBranch();
    assert.equal(askRemovals(branch).length, 1);
    // One mark for the end of the answer, one for the end of the response that failed.
    const marks = branch.filter((entry) => entry.type === 'custom' && entry.customType === 'mark');
    assert.equal(marks.length, 2);
  } finally {
    await harness.cleanup();
  }
});

test('the summary of a branch behind the fold at the end of a run is something new: calls run again', async () => {
  const harness = await createSession(
    'only',
    [
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
      () => fauxAssistantMessage('Folded.'),
      () => script('text(await tools.bash({ command: "echo went-on" }));'),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /went-on/);
        return fauxAssistantMessage('done');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.session.prompt('first');
    await harness.session.waitForIdle();
    const manager = harness.session.sessionManager;
    manager.branchWithSummary(manager.getLeafId(), 'The other branch asked for one more check.');
    harness.session.agent.state.messages = manager.buildSessionProjection().messages;
    await harness.session.agent.continue();
    await harness.run();
    assert.equal(harness.requests.length, 6);
  } finally {
    await harness.cleanup();
  }
});

test('a fold that covered the request of one made at the end of a run gives it back when it is unfolded', async () => {
  let covering = '';
  const shown = (context: TranscriptContext) => conversation(context).map(text).join('\n');
  const harness = await createSession(
    'only',
    [
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
      () => fauxAssistantMessage('Folded.'),
      (context) => {
        const note = conversation(context).find(
          (message) => message.role === 'toolResult' && /\[fold ep-/.test(text(message))
        )!;
        const stamp = /\[(t\d+)\]\s*$/.exec(text(note))![1];
        return script(`text(await tools.fold({ since: "${stamp}", note: "Covering note." }));`);
      },
      (context) => {
        covering = /ep-[0-9a-f]{8}/.exec(text(conversation(context).at(-1)!))![0];
        assert.doesNotMatch(shown(context), /The request above is answered/);
        return fauxAssistantMessage('covered');
      },
      () => script(`text(await tools.unfold({ handle: "${covering}" }));`),
      (context) => {
        assert.match(shown(context), /The request above is answered/);
        return fauxAssistantMessage('restored');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first', 'cover the old note', 'unfold the covering note');
  } finally {
    await harness.cleanup();
  }
});

test('after a new request the block is over for good: a fold the agent makes in it does not bring it back', async () => {
  const harness = await createSession(
    'only',
    [
      () => script(`text(await tools.bash({ command: "echo ${'x'.repeat(4000)}" }));`),
      () => fauxAssistantMessage('first answer'),
      () => script('text(await tools.fold({ since: "t1", note: "Printed x." }));'),
      () => fauxAssistantMessage('Folded.'),
      () => script('text(await tools.bash({ command: "echo second" }));'),
      () => script('text(await tools.fold({ since: "t4", note: "Second task, first step." }));'),
      () => script('text(await tools.bash({ command: "echo goes-on" }));'),
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /goes-on/);
        return fauxAssistantMessage('second answer');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first', 'second');
  } finally {
    await harness.cleanup();
  }
});

test('a request for the fold that got no answer is not sent again, and the nudge comes with the next prompt', async () => {
  const harness = await createSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: `echo ${'x'.repeat(4000)}` }), {
          stopReason: 'toolUse'
        }),
      () => fauxAssistantMessage('first answer'),
      // The provider fails on the request for the fold.
      (context) => {
        assert.match(text(conversation(context).at(-1)!), /^The request above is answered/);
        return fauxAssistantMessage('', { stopReason: 'error', errorMessage: 'boom' });
      },
      (context) => {
        const shown = conversation(context).map(text);
        assert.doesNotMatch(shown.join('\n'), /The request above is answered/);
        // No decision was made about that work: the nudge that is the fallback is due.
        assert.match(shown.join('\n'), /tokens of work built up since the last fold/);
        assert.match(shown.join('\n'), /second/);
        return fauxAssistantMessage('second answer');
      },
      // The nudge was the question about that work: nothing is asked when its run ends, nor
      // with the next prompt.
      (context) => {
        const shown = conversation(context).map(text);
        assert.match(shown.at(-1)!, /^third/);
        assert.doesNotMatch(shown.join('\n'), /The request above is answered/);
        assert.equal(shown.filter((said) => /tokens of work built up/.test(said)).length, 1);
        return fauxAssistantMessage('third answer');
      }
    ],
    [],
    { foldNudgeTokens: 1600, foldAfterRun: 'always' }
  );
  try {
    await harness.run('first', 'second', 'third');
    const stats = episodeStats(harness.session.sessionManager.getBranch(), harness.dir);
    assert.equal(stats.runEndAsks, 1);
    assert.equal(stats.nudges, 1);
  } finally {
    await harness.cleanup();
  }
});
