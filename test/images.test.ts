import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { deflateSync } from 'node:zlib';
import {
  type AssistantMessage,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  type ImageContent,
  type JsonObject,
  type Message,
  type TextContent,
  type TranscriptContext
} from '@earendil-works/pi-ai';
import {
  createAgentSession,
  createCodemodeExtension,
  DefaultResourceLoader,
  type ExtensionFactory,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  type Theme
} from '@earendil-works/pi-coding-agent';
import {
  archiveDir,
  callingFork,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_STARTED_TYPE,
  FORK_TOOL,
  type FoldDetails,
  type ForkArchive,
  type ForkDetails,
  runningFork,
  ThreadIndex,
  threadRecords,
  writeForkArchive
} from '../src/archive.js';
import { planFold, planUnfold, writtenMark } from '../src/fold.js';
import { ForkFolds } from '../src/fork-fold.js';
import {
  archiveImage,
  decodeImageBlock,
  formatImageRef,
  holdsCarriedMedia,
  imageDimensions,
  matchArchivedRefBlocks,
  prepareCarriedMedia,
  readArchivedImage,
  sniffImageMimeType,
  verifyArchivedImageSync
} from '../src/images.js';
import episodes from '../src/index.js';
import { EpisodeState } from '../src/state.js';
import { episodeStats } from '../src/stats.js';
import { recallRenderers } from '../src/ui.js';
import { assistant, call, result, session, user } from './helpers.js';

function crc32(buf: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of buf) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Uint8Array): Buffer {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 4, 'ascii');
  Buffer.from(data).copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

/** Valid RGB PNG of `width` x `height` filled with `rgb`. */
function makePng(
  width: number,
  height: number,
  rgb: [number, number, number] = [20, 120, 220],
  extraChunkBeforeIdat?: { type: string; data: Uint8Array }
): Buffer {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const rowLen = 1 + width * 3;
  const raw = Buffer.alloc(rowLen * height);
  for (let y = 0; y < height; y++) {
    const rowStart = y * rowLen;
    raw[rowStart] = 0;
    for (let x = 0; x < width; x++) {
      raw[rowStart + 1 + x * 3] = rgb[0];
      raw[rowStart + 1 + x * 3 + 1] = rgb[1];
      raw[rowStart + 1 + x * 3 + 2] = rgb[2];
    }
  }
  const chunks = [pngChunk('IHDR', ihdr)];
  if (extraChunkBeforeIdat) {
    chunks.push(pngChunk(extraChunkBeforeIdat.type, extraChunkBeforeIdat.data));
  }
  chunks.push(pngChunk('IDAT', deflateSync(raw)), pngChunk('IEND', new Uint8Array(0)));
  return Buffer.concat([sig, ...chunks]);
}

/** Minimal valid GIF89a header + trailer with dimensions. */
function makeGif(width: number, height: number): Buffer {
  const buf = Buffer.alloc(14);
  buf.write('GIF89a', 0, 'ascii');
  buf.writeUInt16LE(width, 6);
  buf.writeUInt16LE(height, 8);
  buf[10] = 0x00;
  buf[13] = 0x3b;
  return buf;
}

/** Minimal JPEG header with optional EXIF orientation tag and SOF0 dimensions. */
function makeJpeg(width: number, height: number, orientation?: number): Buffer {
  const parts: Buffer[] = [Buffer.from([0xff, 0xd8])];
  if (orientation !== undefined) {
    // 6-byte Exif header + 8-byte TIFF header + 2-byte count + 12-byte IFD entry + 4-byte next IFD = 32 bytes.
    const tiff = Buffer.alloc(32);
    Buffer.from([0x45, 0x78, 0x69, 0x66, 0x00, 0x00]).copy(tiff, 0);
    tiff.write('II', 6, 'ascii');
    tiff.writeUInt16LE(0x002a, 8);
    tiff.writeUInt32LE(8, 10);
    tiff.writeUInt16LE(1, 14);
    tiff.writeUInt16LE(0x0112, 16);
    tiff.writeUInt16LE(3, 18);
    tiff.writeUInt32LE(1, 20);
    tiff.writeUInt16LE(orientation, 24);
    const app1 = Buffer.alloc(4 + tiff.length);
    app1[0] = 0xff;
    app1[1] = 0xe1;
    app1.writeUInt16BE(2 + tiff.length, 2);
    tiff.copy(app1, 4);
    parts.push(app1);
  }
  const sof0 = Buffer.alloc(11);
  sof0[0] = 0xff;
  sof0[1] = 0xc0;
  sof0.writeUInt16BE(9, 2);
  sof0[4] = 8;
  sof0.writeUInt16BE(height, 5);
  sof0.writeUInt16BE(width, 7);
  sof0[9] = 1;
  parts.push(sof0, Buffer.from([0xff, 0xd9]));
  return Buffer.concat(parts);
}

/** Minimal WebP VP8X container with 24-bit canvas width-1 and height-1. */
function makeWebpVp8x(width: number, height: number): Buffer {
  const payload = Buffer.alloc(10);
  payload.writeUIntLE(width - 1, 4, 3);
  payload.writeUIntLE(height - 1, 7, 3);
  const out = Buffer.alloc(12 + 8 + 10);
  out.write('RIFF', 0, 'ascii');
  out.writeUInt32LE(out.length - 8, 4);
  out.write('WEBP', 8, 'ascii');
  out.write('VP8X', 12, 'ascii');
  out.writeUInt32LE(10, 16);
  payload.copy(out, 20);
  return out;
}

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

function script(code: string): AssistantMessage {
  return fauxAssistantMessage(fauxToolCall('codemode', { code }), { stopReason: 'toolUse' });
}

async function createPersistedSession(
  mode: Mode,
  steps: Step[],
  extensions: ExtensionFactory[] = [],
  options: {
    sessionManager?: SessionManager;
    dir?: string;
    keepDir?: boolean;
    preExtensions?: ExtensionFactory[];
  } = {}
) {
  const dir = options.dir ?? (await mkdtemp(join(tmpdir(), 'pi-episodes-images-')));
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
      ...(options.preExtensions ?? []),
      episodes,
      ...extensions
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
      ...(mode === 'off' ? [] : ['codemode'])
    ],
    sessionManager: options.sessionManager ?? SessionManager.create(dir, join(dir, 'sessions')),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      ...(mode === 'off' ? {} : { codemode: { mode } })
    })
  });
  await agentSession.bindExtensions({});

  return {
    dir,
    session: agentSession,
    requests,
    failures,
    async run(prompt: string) {
      await agentSession.prompt(prompt);
      if (failures[0]) throw failures[0];
    },
    async cleanup() {
      agentSession.dispose();
      if (!options.keepDir) {
        await rm(dir, { recursive: true, force: true });
      }
    }
  };
}

test('sniffImageMimeType, imageDimensions, archiveImage and readArchivedImage verify magic headers, dimensions and SHA-256', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-img-unit-'));
  try {
    const png = makePng(4, 7);
    const gif = makeGif(12, 9);
    const jpegNormal = makeJpeg(64, 32, 1);
    const jpegRotated = makeJpeg(64, 32, 6);
    const webp = makeWebpVp8x(19, 11);
    const apng = makePng(4, 7, [1, 2, 3], { type: 'acTL', data: new Uint8Array(8) });

    assert.equal(sniffImageMimeType(png), 'image/png');
    assert.equal(sniffImageMimeType(gif), 'image/gif');
    assert.equal(sniffImageMimeType(jpegNormal), 'image/jpeg');
    assert.equal(sniffImageMimeType(webp), 'image/webp');
    assert.equal(sniffImageMimeType(apng), undefined);

    assert.deepEqual(imageDimensions(png, 'image/png'), { width: 4, height: 7 });
    assert.deepEqual(imageDimensions(gif, 'image/gif'), { width: 12, height: 9 });
    assert.deepEqual(imageDimensions(jpegNormal, 'image/jpeg'), { width: 64, height: 32 });
    assert.deepEqual(imageDimensions(jpegRotated, 'image/jpeg'), { width: 32, height: 64 });
    assert.deepEqual(imageDimensions(webp, 'image/webp'), { width: 19, height: 11 });

    assert.equal(
      decodeImageBlock({ type: 'image', data: 'AAAA', mimeType: 'image/png' }),
      undefined
    );
    assert.equal(
      decodeImageBlock({ type: 'image', data: png.toString('base64'), mimeType: 'image/jpeg' }),
      undefined
    );
    assert.equal(
      decodeImageBlock({ type: 'image', data: apng.toString('base64'), mimeType: 'image/png' }),
      undefined
    );

    const block: ImageContent = {
      type: 'image',
      data: png.toString('base64'),
      mimeType: 'image/png'
    };
    const first = await archiveImage(dir, 'ep-11111111', 1, block);
    const second = await archiveImage(dir, 'ep-22222222', 1, block);
    assert.ok(first && second);
    assert.equal(first.path, second.path);
    assert.equal(first.width, 4);
    assert.equal(first.height, 7);
    assert.equal(verifyArchivedImageSync(first), true);
    assert.match(
      formatImageRef(first),
      /^\[ep-11111111 image #1: image\/png, 4x7, \d+ B, sha256 [0-9a-f]{16} — .*recall\(\{ handle: "ep-11111111", image: 1 \}\) or read path\]$/
    );

    const roundTrip = await readArchivedImage(first);
    assert.deepEqual(roundTrip, block);

    const corrupted = Buffer.from(png);
    const lastIdx = corrupted.length - 5;
    corrupted[lastIdx] = (corrupted[lastIdx] ?? 0) ^ 0xff;
    await writeFile(first.path, corrupted);
    assert.equal(verifyArchivedImageSync(first), false);
    await assert.rejects(() => readArchivedImage(first), /failed SHA-256 verification/);

    await rm(first.path);
    assert.equal(verifyArchivedImageSync(first), false);
    await assert.rejects(() => readArchivedImage(first), /is missing at/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('covering folds offload covered-note images to disk and recall reads them back in direct and codemode sessions', async () => {
  const pngBytes = makePng(2, 3, [200, 40, 90]);
  const pngBlock: ImageContent = {
    type: 'image',
    data: pngBytes.toString('base64'),
    mimeType: 'image/png'
  };

  const attachOnFirstFold: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (
        !/Learned: header spacing is 12px/.test(joined) ||
        /Notes of the earlier folds/.test(joined)
      ) {
        return undefined;
      }
      return { content: [...event.content, pngBlock] };
    });
  };

  for (const mode of ['off', 'only'] as const) {
    let ep2 = '';
    let archivedPath = '';
    const invoke = (tool: string, args: JsonObject) => () =>
      mode === 'only'
        ? script(`text(JSON.stringify(await tools.${tool}(${JSON.stringify(args)})));`)
        : fauxAssistantMessage(fauxToolCall(tool, args), { stopReason: 'toolUse' });

    const harness = await createPersistedSession(
      mode,
      [
        invoke('bash', { command: 'echo inspect-ui' }),
        mode === 'only'
          ? () =>
              script(
                `image(${JSON.stringify(pngBlock)}); text(await tools.fold({ since: "t2", note: "Learned: header spacing is 12px." }));`
              )
          : invoke('fold', { since: 't2', note: 'Learned: header spacing is 12px.' }),
        (context) => {
          const view = conversation(context);
          const inlineImages = view.flatMap((msg) =>
            typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
          );
          assert.deepEqual(inlineImages, [pngBlock], mode);
          const foldText = view.map(text).join('\n');
          assert.match(foldText, /ep-[0-9a-f]{8}/);
          return mode === 'only'
            ? script('text(await tools.bash({ command: "echo step-two" }));')
            : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo step-two' }), {
                stopReason: 'toolUse'
              });
        },
        // Covering fold #2 starts at t3 (the first fold's note), offloading ep1's inline image.
        invoke('fold', { since: 't3', note: 'Second episode complete.' }),
        (context) => {
          const view = conversation(context);
          const inlineImages = view.flatMap((msg) =>
            typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
          );
          assert.equal(
            inlineImages.length,
            0,
            `${mode}: covering fold should offload inline image`
          );
          const latest = text(view.at(-1)!);
          ep2 = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
          const match = new RegExp(
            `\\[${ep2} image #1: image/png, 2x3, \\d+ B, sha256 [0-9a-f]{16} — ([^;]+); recall\\(\\{ handle: "${ep2}", image: 1 \\}\\) or read path\\]`
          ).exec(latest);
          assert.ok(match, `${mode}: missing offloaded image reference in:\n${latest}`);
          archivedPath = match[1]!;
          return mode === 'only'
            ? script('text(await tools.bash({ command: "echo step-three" }));')
            : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo step-three' }), {
                stopReason: 'toolUse'
              });
        },
        // Covering fold #3 starts at t5 (second fold's note): carries the reference forward cleanly.
        invoke('fold', { since: 't5', note: 'Third episode complete.' }),
        (context) => {
          const view = conversation(context);
          const latest = text(view.at(-1)!);
          assert.equal(
            latest.split(`[${ep2} image #1:`).length - 1,
            1,
            `${mode}: reference should appear exactly once on third fold`
          );
          return mode === 'only'
            ? script(
                `const out = await tools.recall({ handle: ${JSON.stringify(ep2)}, image: 1 }); image(out); text(out.note);`
              )
            : fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
                stopReason: 'toolUse'
              });
        },
        (context) => {
          const last = conversation(context).at(-1)!;
          assert.ok(typeof last.content !== 'string');
          const returnedImages = last.content.filter((b) => b.type === 'image');
          assert.deepEqual(returnedImages, [pngBlock], `${mode}: recall should return exact PNG`);
          assert.match(text(last), new RegExp(`\\[${ep2} image #1: image/png, 2x3,`));
          return mode === 'only'
            ? script(`image(await tools.read({ path: ${JSON.stringify(archivedPath)} }));`)
            : fauxAssistantMessage(fauxToolCall('read', { path: archivedPath }), {
                stopReason: 'toolUse'
              });
        },
        (context) => {
          const last = conversation(context).at(-1)!;
          assert.ok(typeof last.content !== 'string');
          const readImages = last.content.filter((b) => b.type === 'image');
          assert.equal(readImages.length, 1, `${mode}: read(path) should return image`);
          assert.equal(readImages[0]!.mimeType, 'image/png');
          return fauxAssistantMessage('All verified.');
        }
      ],
      mode === 'off' ? [attachOnFirstFold] : []
    );

    try {
      await harness.run('start');
      const branch = harness.session.sessionManager.getBranch();
      const records = threadRecords(branch);
      assert.equal(records.length, 3, mode);
      if (mode === 'off') {
        const stats = episodeStats(branch, harness.dir);
        assert.equal(stats.recall.count, 1);
        assert.equal(stats.recall.readBack, 1);
      }
    } finally {
      await harness.cleanup();
    }
  }
});

test('stripping an offloaded image reference from a covering fold aborts the fold, and unfolding restores the covered note without orphan markers', async () => {
  const pngBytes = makePng(3, 2, [10, 180, 60]);
  const pngBlock: ImageContent = {
    type: 'image',
    data: pngBytes.toString('base64'),
    mimeType: 'image/png'
  };

  const attachOnFirstFold: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!/First note with screenshot/.test(joined) || /Notes of the earlier folds/.test(joined)) {
        return undefined;
      }
      return { content: [...event.content, pngBlock] };
    });
  };

  const stripRefOnBlockedFold: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!/Blocked covering fold/.test(joined)) return undefined;
      return {
        content: event.content.filter(
          (b) => !(b.type === 'text' && /image #1: image\/png/.test(b.text))
        )
      };
    });
  };

  let ep2 = '';
  const harness = await createPersistedSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo one' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', note: 'First note with screenshot.' }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        assert.match(conversation(context).map(text).join('\n'), /ep-[0-9a-f]{8}/);
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo two' }), {
          stopReason: 'toolUse'
        });
      },
      // Covering fold from t3 whose reference block gets stripped -> must NOT apply!
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't3', note: 'Blocked covering fold.' }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const view = conversation(context);
        const inlineImages = view.flatMap((msg) =>
          typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
        );
        assert.deepEqual(inlineImages, [pngBlock]);
        return fauxAssistantMessage(
          fauxToolCall('fold', { since: 't3', note: 'Valid covering fold.' }),
          { stopReason: 'toolUse' }
        );
      },
      (context) => {
        const latest = text(conversation(context).at(-1)!);
        ep2 = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
        assert.match(latest, new RegExp(`\\[${ep2} image #1: image/png`));
        return fauxAssistantMessage(fauxToolCall('unfold', { handle: ep2 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const view = conversation(context);
        const inlineImages = view.flatMap((msg) =>
          typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
        );
        assert.deepEqual(inlineImages, [pngBlock]);
        const joined = view.map(text).join('\n');
        assert.doesNotMatch(joined, /image #1: image\/png/);
        return fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.equal(last.role, 'toolResult');
        assert.equal(last.isError, true);
        assert.match(text(last), new RegExp(`No archived thread ${ep2}`));
        return fauxAssistantMessage('done');
      }
    ],
    [attachOnFirstFold, stripRefOnBlockedFold]
  );

  try {
    await harness.run('go');
  } finally {
    await harness.cleanup();
  }
});

test('ForkFolds offloads covered child-note images with running-fork isolation, nested unfold retirement, and parent archive visibility', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-fork-img-'));
  try {
    const pngBytes = makePng(5, 4, [70, 80, 90]);
    const pngBlock: ImageContent = {
      type: 'image',
      data: pngBytes.toString('base64'),
      mimeType: 'image/png'
    };

    const { manager, ids } = session([
      user('parent start', 1),
      assistant(call('fk_call', FORK_TOOL, { task: 'child work' }))
    ]);

    const state = new EpisodeState();
    state.nextStamp = 10;
    const proj = manager.buildSessionProjection();
    const taskMsg = result('fk_call', FORK_TOOL, 'Task: child work', 2);
    const childMsgs = [...proj.messages, taskMsg];
    const folds = new ForkFolds({
      forkHandle: 'fk-img00001',
      state,
      parentEntries: proj.entries,
      prefixLength: proj.messages.length,
      siblings: 0,
      archiveDir: dir
    });
    const foldTool = folds.tool(
      { name: 'fold', description: '', parameters: {} as never },
      () => childMsgs
    );
    const unfoldTool = folds.tool(
      { name: 'unfold', description: '', parameters: {} as never },
      () => childMsgs
    );

    childMsgs.push(assistant(call('cb1', 'bash', { command: 'echo step1' })));
    const cr1 = result('cb1', 'bash', 'step1');
    folds.stamp(cr1);
    childMsgs.push(cr1);

    childMsgs.push(assistant(call('cf1', 'fold', { since: 't10', note: 'child first' })));
    const cf1Res = await foldTool.execute('cf1', { since: 't10', note: 'child first' });
    const cf1Msg: Message = {
      role: 'toolResult',
      toolCallId: 'cf1',
      toolName: 'fold',
      content: [{ type: 'text', text: (cf1Res.content[0] as TextContent).text }, pngBlock],
      details: cf1Res.details,
      isError: false,
      timestamp: 0
    };
    folds.stamp(cf1Msg);
    childMsgs.push(cf1Msg);

    childMsgs.push(assistant(call('cb2', 'bash', { command: 'echo step2' })));
    const cr2 = result('cb2', 'bash', 'step2');
    folds.stamp(cr2);
    childMsgs.push(cr2);

    // Covering fold inside the fork offloads cf1's PNG to dir.
    childMsgs.push(assistant(call('cf2', 'fold', { since: 't11', note: 'child second' })));
    const cf2Res = await foldTool.execute('cf2', { since: 't11', note: 'child second' });
    const cf2Handle = (cf2Res.details as FoldDetails).handle;
    assert.equal(cf2Res.content.filter((b) => b.type === 'image').length, 0);
    assert.ok(
      cf2Res.content.some(
        (b) => b.type === 'text' && b.text.includes(`[${cf2Handle} image #1: image/png, 5x4,`)
      )
    );
    const cf2Msg: Message = {
      role: 'toolResult',
      toolCallId: 'cf2',
      toolName: 'fold',
      content: [...cf2Res.content],
      details: cf2Res.details,
      isError: false,
      timestamp: 0
    };
    folds.stamp(cf2Msg);
    childMsgs.push(cf2Msg);

    const snapshot = (): ForkArchive => ({
      handle: 'fk-img00001',
      task: 'child work',
      parentEntryId: ids[1]!,
      messages: childMsgs.slice(proj.messages.length),
      folds: folds.folds.map((f) => ({ ...f }))
    });

    // Running child sees cf2's offloaded image via cf2's own record; outside / sibling fork does not.
    const outsideThread = await new ThreadIndex(manager, state.forkArchives).find(cf2Handle);
    assert.equal(outsideThread, undefined);
    const runningThread = await runningFork.run(
      { sessionId: manager.getSessionId(), transcript: snapshot },
      () => new ThreadIndex(manager, state.forkArchives, callingFork(manager)).find(cf2Handle)
    );
    assert.ok(runningThread && runningThread.record.kind === 'fold');
    const runningHit = runningThread.record.archivedImages?.find(
      (item) => item.handle === cf2Handle && item.index === 1
    );
    assert.ok(runningHit);
    assert.deepEqual(await readArchivedImage(runningHit), pngBlock);

    // Unfolding cf2 inside the fork retires cf2 and its offloaded manifest.
    await unfoldTool.execute('cu1', { handle: cf2Handle });
    const afterUnfold = await runningFork.run(
      { sessionId: manager.getSessionId(), transcript: snapshot },
      () => new ThreadIndex(manager, state.forkArchives, callingFork(manager)).find(cf2Handle)
    );
    assert.equal(afterUnfold, undefined);

    // Re-folding mints a fresh cf3 identity and finishing the fork makes cf3 reachable from parent.
    childMsgs.push(assistant(call('cf3', 'fold', { since: 't11', note: 'child re-covered' })));
    const cf3Res = await foldTool.execute('cf3', { since: 't11', note: 'child re-covered' });
    const cf3Handle = (cf3Res.details as FoldDetails).handle;
    assert.notEqual(cf3Handle, cf2Handle);
    const cf3Msg: Message = {
      role: 'toolResult',
      toolCallId: 'cf3',
      toolName: 'fold',
      content: [...cf3Res.content],
      details: cf3Res.details,
      isError: false,
      timestamp: 0
    };
    folds.stamp(cf3Msg);
    childMsgs.push(cf3Msg);

    const archivePath = await writeForkArchive(dir, snapshot());
    manager.appendMessage(
      Object.assign(result('fk_call', FORK_TOOL, 'Child finished', 15), {
        details: {
          kind: 'fork',
          handle: 'fk-img00001',
          task: 'child work',
          parentEntryId: ids[1]!,
          turns: 4,
          headline: 'Child finished',
          files: [],
          folds: 2,
          ...(archivePath ? { archive: archivePath } : {})
        } satisfies ForkDetails as never
      })
    );
    const parentThread = await new ThreadIndex(manager, state.forkArchives).find(cf3Handle);
    assert.ok(parentThread && parentThread.record.kind === 'fold');
    const parentHit = parentThread.record.archivedImages?.find(
      (item) => item.handle === cf3Handle && item.index === 1
    );
    assert.ok(parentHit);
    assert.deepEqual(await readArchivedImage(parentHit), pngBlock);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('recallRenderers renders image readback call and result summaries', () => {
  const theme = {
    fg: (_color: string, value: string) => value,
    bold: (value: string) => value
  } as unknown as Theme;
  const callText = recallRenderers
    .renderCall({ handle: 'ep-12345678', image: 2 }, theme)
    .render(120)
    .join('\n');
  assert.match(callText, /recall.*image #2.*in.*ep-12345678/);

  const resText = recallRenderers
    .renderResult(
      {
        content: [{ type: 'text', text: '[ep-12345678 image #2: image/png, 10x20, 99 bytes]' }],
        details: {
          query: '',
          threads: 1,
          shown: 1,
          total: 1,
          image: { index: 2, mimeType: 'image/png', bytes: 99, width: 10, height: 20 }
        }
      },
      { expanded: false },
      theme
    )
    .render(120)
    .join('\n');
  assert.match(resText, /image #2 read back, image\/png, 10x20, 99 bytes/);
});

test('reference matching is exact standalone 1-to-1 across verification, reordering, modification, and manifest stripping', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-ref-match-'));
  try {
    const png = makePng(3, 3, [12, 34, 56]);
    const pngBlock: ImageContent = {
      type: 'image',
      data: png.toString('base64'),
      mimeType: 'image/png'
    };
    const prepared = await prepareCarriedMedia(
      'ep-22222222',
      [{ kind: 'image', image: pngBlock }],
      dir
    );
    assert.equal(prepared[0]?.kind, 'ref');
    const ref = prepared[0].ref;
    assert.equal(ref.handle, 'ep-22222222');
    const refText = formatImageRef(ref);

    // Substring embeds must NOT satisfy reference verification.
    assert.equal(
      holdsCarriedMedia([{ type: 'text', text: `Wrapped: ${refText}` }], [{ kind: 'ref', ref }]),
      false
    );
    assert.equal(
      holdsCarriedMedia([{ type: 'text', text: refText }], [{ kind: 'ref', ref }]),
      true
    );

    // Reordered standalone ref block BEFORE written.blocks is still carried as media and omitted from note text.
    const noteBody = '[ep-22222222: folded t1 (2 messages, ~10 tokens)]\n\nClean covering note.';
    const detailsWithManifest: FoldDetails = {
      kind: 'fold',
      handle: 'ep-22222222',
      since: 't1',
      firstEntryId: 'x',
      lastEntryId: 'x',
      messages: 2,
      tokens: 10,
      headline: 'Clean covering note.',
      keptUserTurns: [],
      files: [],
      note: 'Clean covering note.',
      written: writtenMark([{ type: 'text', text: noteBody }]),
      archivedImages: [ref]
    };

    const reorderedSession = session([
      user('start', 1),
      assistant(call('f2', 'fold', { since: 't1', note: 'Clean covering note.' })),
      Object.assign({
        role: 'toolResult' as const,
        toolCallId: 'f2',
        toolName: 'fold',
        content: [
          { type: 'text' as const, text: refText },
          { type: 'text' as const, text: noteBody },
          { type: 'text' as const, text: '[t2]' }
        ],
        details: detailsWithManifest,
        isError: false,
        timestamp: 0
      }),
      assistant(call('f3', 'fold', { since: 't2', note: 'Third fold' }))
    ]);
    const reorderedPlan = planFold(
      reorderedSession.manager.buildSessionProjection().entries,
      'f3',
      2,
      { latestStamp: 2, applied: new Set(['ep-22222222']) }
    );
    assert.equal(reorderedPlan.media.length, 1);
    assert.equal(reorderedPlan.media[0]?.kind, 'ref');
    assert.equal(reorderedPlan.covers[0]?.text, 'Clean covering note.');

    // Modified reference text is NOT dropped; it stays verbatim in CoveredNote.text.
    const modifiedSession = session([
      user('start', 1),
      assistant(call('f2', 'fold', { since: 't1', note: 'Clean covering note.' })),
      Object.assign({
        role: 'toolResult' as const,
        toolCallId: 'f2',
        toolName: 'fold',
        content: [
          { type: 'text' as const, text: noteBody },
          { type: 'text' as const, text: `Edited: ${refText}` },
          { type: 'text' as const, text: '[t2]' }
        ],
        details: detailsWithManifest,
        isError: false,
        timestamp: 0
      }),
      assistant(call('f3', 'fold', { since: 't2', note: 'Third fold' }))
    ]);
    const modifiedPlan = planFold(
      modifiedSession.manager.buildSessionProjection().entries,
      'f3',
      2,
      { latestStamp: 2, applied: new Set(['ep-22222222']) }
    );
    assert.equal(modifiedPlan.media.length, 0);
    assert.match(modifiedPlan.covers[0]?.text ?? '', /Edited:/);

    // Stripped archivedImages manifest preserves the text block verbatim in CoveredNote.text.
    const strippedDetails: FoldDetails = { ...detailsWithManifest };
    delete strippedDetails.archivedImages;
    const strippedSession = session([
      user('start', 1),
      assistant(call('f2', 'fold', { since: 't1', note: 'Clean covering note.' })),
      Object.assign({
        role: 'toolResult' as const,
        toolCallId: 'f2',
        toolName: 'fold',
        content: [
          { type: 'text' as const, text: noteBody },
          { type: 'text' as const, text: refText },
          { type: 'text' as const, text: '[t2]' }
        ],
        details: strippedDetails,
        isError: false,
        timestamp: 0
      }),
      assistant(call('f3', 'fold', { since: 't2', note: 'Third fold' }))
    ]);
    const strippedPlan = planFold(
      strippedSession.manager.buildSessionProjection().entries,
      'f3',
      2,
      { latestStamp: 2, applied: new Set(['ep-22222222']) }
    );
    assert.equal(strippedPlan.media.length, 0);
    assert.ok((strippedPlan.covers[0]?.text ?? '').includes(refText));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('offloaded images survive session restart and post-compaction readback', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-restart-compact-'));
  const pngBlock: ImageContent = {
    type: 'image',
    data: makePng(4, 3, [90, 20, 180]).toString('base64'),
    mimeType: 'image/png'
  };
  const attachFirst: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!/Initial screenshot note/.test(joined) || /Notes of the earlier folds/.test(joined)) {
        return undefined;
      }
      return { content: [...event.content, pngBlock] };
    });
  };

  try {
    let ep2 = '';
    const first = await createPersistedSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: 'echo init' }), {
            stopReason: 'toolUse'
          }),
        () =>
          fauxAssistantMessage(
            fauxToolCall('fold', { since: 't2', note: 'Initial screenshot note.' }),
            { stopReason: 'toolUse' }
          ),
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: 'echo next' }), {
            stopReason: 'toolUse'
          }),
        () =>
          fauxAssistantMessage(
            fauxToolCall('fold', { since: 't3', note: 'Covering note before restart.' }),
            { stopReason: 'toolUse' }
          ),
        (context) => {
          const latest = text(conversation(context).at(-1)!);
          ep2 = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
          return fauxAssistantMessage('done before restart');
        }
      ],
      [attachFirst],
      { dir, keepDir: true }
    );
    await first.run('start');
    const sessionFile = first.session.sessionManager.getSessionFile()!;
    assert.ok(sessionFile);
    await first.cleanup();

    // Reopen persisted session from disk, verify recall across restart, then compact and recall again.
    const reopenedManager = SessionManager.open(sessionFile);
    const second = await createPersistedSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
            stopReason: 'toolUse'
          }),
        (context) => {
          const last = conversation(context).at(-1)!;
          assert.ok(typeof last.content !== 'string');
          assert.deepEqual(
            last.content.filter((b) => b.type === 'image'),
            [pngBlock],
            'recall after restart'
          );
          const branch = reopenedManager.getBranch();
          reopenedManager.appendCompaction('Compacted summary after ep2.', branch.at(-1)!.id, 500);
          return fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
            stopReason: 'toolUse'
          });
        },
        (context) => {
          const last = conversation(context).at(-1)!;
          assert.ok(typeof last.content !== 'string');
          assert.deepEqual(
            last.content.filter((b) => b.type === 'image'),
            [pngBlock],
            'recall after compaction'
          );
          return fauxAssistantMessage('done after restart + compaction');
        }
      ],
      [],
      { dir, keepDir: true, sessionManager: reopenedManager }
    );
    await second.run('continue');
    await second.cleanup();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('valid images fall back to inline ImageContent in inMemory sessions and on archive storage failure', async () => {
  const pngBlock: ImageContent = {
    type: 'image',
    data: makePng(2, 2, [40, 120, 200]).toString('base64'),
    mimeType: 'image/png'
  };
  const attachFirst: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!/First valid image note/.test(joined) || /Notes of the earlier folds/.test(joined)) {
        return undefined;
      }
      return { content: [...event.content, pngBlock] };
    });
  };

  // 1) SessionManager.inMemory() -> archiveDir is undefined -> valid PNG stays inline.
  const memHarness = await createPersistedSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo mem1' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', note: 'First valid image note.' }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo mem2' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't3', note: 'Covering note in memory.' }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const view = conversation(context);
        const inlineImages = view.flatMap((msg) =>
          typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
        );
        assert.deepEqual(inlineImages, [pngBlock]);
        return fauxAssistantMessage('done mem');
      }
    ],
    [attachFirst],
    { sessionManager: SessionManager.inMemory() }
  );
  try {
    await memHarness.run('go');
    const records = threadRecords(memHarness.session.sessionManager.getBranch());
    const covering = records.at(-1);
    assert.ok(covering && covering.kind === 'fold');
    assert.equal(covering.carriedImages, 1);
    assert.equal(covering.archivedImages, undefined);
  } finally {
    await memHarness.cleanup();
  }

  // 2) Storage failure: occupy <session>.episodes with a regular file so mkdir fails.
  const diskFailHarness = await createPersistedSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo d1' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't2', note: 'First valid image note.' }),
          { stopReason: 'toolUse' }
        ),
      async () => {
        const epDir = archiveDir(diskFailHarness.session.sessionManager)!;
        await writeFile(epDir, 'not-a-directory');
        return fauxAssistantMessage(fauxToolCall('bash', { command: 'echo d2' }), {
          stopReason: 'toolUse'
        });
      },
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't3', note: 'Covering note on storage error.' }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const view = conversation(context);
        const inlineImages = view.flatMap((msg) =>
          typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
        );
        assert.deepEqual(inlineImages, [pngBlock]);
        return fauxAssistantMessage('done disk fail');
      }
    ],
    [attachFirst]
  );
  try {
    await diskFailHarness.run('go');
    const records = threadRecords(diskFailHarness.session.sessionManager.getBranch());
    const covering = records.at(-1);
    assert.ok(covering && covering.kind === 'fold');
    assert.equal(covering.carriedImages, 1);
    assert.equal(covering.archivedImages, undefined);
  } finally {
    await diskFailHarness.cleanup();
  }
});

test('codemode unfold drops offloaded reference blocks while preserving script-owned images and script output', async () => {
  const coveredPng: ImageContent = {
    type: 'image',
    data: makePng(2, 3, [200, 10, 10]).toString('base64'),
    mimeType: 'image/png'
  };
  const scriptOwnPng: ImageContent = {
    type: 'image',
    data: makePng(4, 2, [10, 200, 30]).toString('base64'),
    mimeType: 'image/png'
  };

  let ep2 = '';
  const harness = await createPersistedSession('only', [
    () => script('text(await tools.bash({ command: "echo step1" }));'),
    () =>
      script(
        `image(${JSON.stringify(coveredPng)}); text(await tools.fold({ since: "t2", note: "Covered screenshot note." }));`
      ),
    () => script('text(await tools.bash({ command: "echo step2" }));'),
    // Script emits its own image AND text AND folds from t3 (offloading coveredPng to [ep2 image #1]).
    () =>
      script(
        `image(${JSON.stringify(scriptOwnPng)}); text("script-owned-banner"); text(await tools.fold({ since: "t3", note: "Codemode covering note." }));`
      ),
    (context) => {
      const last = conversation(context).at(-1)!;
      assert.ok(typeof last.content !== 'string');
      assert.deepEqual(
        last.content.filter((b) => b.type === 'image'),
        [scriptOwnPng],
        'only scriptOwnPng stays inline after covering fold'
      );
      const latest = text(last);
      ep2 = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
      assert.match(latest, new RegExp(`\\[${ep2} image #1: image/png`));
      return script(`text(await tools.unfold({ handle: ${JSON.stringify(ep2)} }));`);
    },
    (context) => {
      const view = conversation(context);
      const joined = view.map(text).join('\n');
      assert.match(joined, /script-owned-banner/);
      assert.match(joined, new RegExp(`\\[unfolded ${ep2}\\]`));
      assert.doesNotMatch(joined, new RegExp(`\\[${ep2} image #1:`));
      const inlineImages = view.flatMap((msg) =>
        typeof msg.content === 'string' ? [] : msg.content.filter((b) => b.type === 'image')
      );
      assert.deepEqual(inlineImages, [coveredPng, scriptOwnPng]);
      return fauxAssistantMessage('done codemode unfold');
    }
  ]);
  try {
    await harness.run('start');
  } finally {
    await harness.cleanup();
  }
});

test('multiple images, branch/unfold/refold immutable identities, interrupted recovery, and ForkFolds cancellation safety', async () => {
  const pngA: ImageContent = {
    type: 'image',
    data: makePng(2, 2, [250, 0, 0]).toString('base64'),
    mimeType: 'image/png'
  };
  const pngB: ImageContent = {
    type: 'image',
    data: makePng(3, 2, [0, 250, 0]).toString('base64'),
    mimeType: 'image/png'
  };
  const pngC: ImageContent = {
    type: 'image',
    data: makePng(2, 4, [0, 0, 250]).toString('base64'),
    mimeType: 'image/png'
  };

  // 1-to-1 duplicate check in holdsCarriedMedia: two identical items require two distinct blocks.
  const dupDir = await mkdtemp(join(tmpdir(), 'pi-episodes-dup-'));
  try {
    const dupPrepared = await prepareCarriedMedia(
      'ep-dup00001',
      [
        { kind: 'image', image: pngA },
        { kind: 'image', image: pngA }
      ],
      dupDir
    );
    assert.equal(dupPrepared.length, 2);
    assert.equal(dupPrepared[0]?.kind, 'ref');
    assert.equal(dupPrepared[1]?.kind, 'ref');
    assert.equal(dupPrepared[0].ref.index, 1);
    assert.equal(dupPrepared[1].ref.index, 2);
    assert.equal(
      holdsCarriedMedia([{ type: 'text', text: formatImageRef(dupPrepared[0].ref) }], dupPrepared),
      false
    );
    assert.equal(
      holdsCarriedMedia(
        [
          { type: 'text', text: formatImageRef(dupPrepared[0].ref) },
          { type: 'text', text: formatImageRef(dupPrepared[1].ref) }
        ],
        dupPrepared
      ),
      true
    );

    // ForkFolds cancellation safety: aborted signal leaves omitted/retired/applied/folds untouched.
    const { manager } = session([
      user('parent', 1),
      assistant(call('fk_abort', FORK_TOOL, { task: 'child' }))
    ]);
    const st = new EpisodeState();
    st.nextStamp = 10;
    const proj = manager.buildSessionProjection();
    const childMsgs = [...proj.messages, result('fk_abort', FORK_TOOL, 'Task: child', 2)];
    const forkFolds = new ForkFolds({
      forkHandle: 'fk-abort001',
      state: st,
      parentEntries: proj.entries,
      prefixLength: proj.messages.length,
      siblings: 0,
      archiveDir: dupDir
    });
    const fTool = forkFolds.tool(
      { name: 'fold', description: '', parameters: {} as never },
      () => childMsgs
    );
    childMsgs.push(assistant(call('b1', 'bash', { command: 'echo x' })));
    const r1 = result('b1', 'bash', 'x');
    forkFolds.stamp(r1);
    childMsgs.push(r1);
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(
      () => fTool.execute('cf_aborted', { since: 't10', note: 'aborted note' }, ac.signal),
      /Fold aborted/
    );
    assert.equal(forkFolds.folds.length, 0);
    assert.equal(forkFolds.view(childMsgs).length, childMsgs.length);

    // Also verify cancellation AFTER async prepareCarriedMedia completes leaves ForkFolds untouched.
    childMsgs.push(assistant(call('cf_ok1', 'fold', { since: 't10', note: 'first ok' })));
    const ok1 = await fTool.execute('cf_ok1', { since: 't10', note: 'first ok' });
    const ok1Msg: Message = {
      role: 'toolResult',
      toolCallId: 'cf_ok1',
      toolName: 'fold',
      content: [{ type: 'text', text: (ok1.content[0] as TextContent).text }, pngA],
      details: ok1.details,
      isError: false,
      timestamp: 0
    };
    forkFolds.stamp(ok1Msg);
    childMsgs.push(ok1Msg);
    childMsgs.push(assistant(call('b2', 'bash', { command: 'echo y' })));
    const r2 = result('b2', 'bash', 'y');
    forkFolds.stamp(r2);
    childMsgs.push(r2);
    childMsgs.push(assistant(call('cf_mid_abort', 'fold', { since: 't11', note: 'mid abort' })));
    const viewLenBeforeMidAbort = forkFolds.view(childMsgs).length;
    let abortChecks = 0;
    const midAbortSignal = {
      get aborted() {
        abortChecks++;
        return abortChecks >= 2;
      }
    } as AbortSignal;
    await assert.rejects(
      () => fTool.execute('cf_mid_abort', { since: 't11', note: 'mid abort' }, midAbortSignal),
      /Fold aborted/
    );
    assert.equal(abortChecks, 2);
    assert.equal(forkFolds.folds.length, 1);
    assert.equal(forkFolds.view(childMsgs).length, viewLenBeforeMidAbort);
  } finally {
    await rm(dupDir, { recursive: true, force: true });
  }

  // Multiple images (#1, #2) + branch / unfold / refold immutable (handle, index) identities.
  const attachImages: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (/Note with image A/.test(joined) && !/Notes of the earlier folds/.test(joined)) {
        return { content: [...event.content, pngA] };
      }
      if (/Note with image B/.test(joined) && !/Notes of the earlier folds/.test(joined)) {
        return { content: [...event.content, pngB] };
      }
      if (
        /Replacement note with image C/.test(joined) &&
        !/Notes of the earlier folds/.test(joined)
      ) {
        return { content: [...event.content, pngC] };
      }
      return undefined;
    });
  };

  let epCoverAB = '';
  let epCoverBC = '';
  let branchBeforeUnfold = '';
  const branchHarness = await createPersistedSession(
    'off',
    [
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo a' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't2', note: 'Note with image A.' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo b' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't4', note: 'Note with image B.' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo c' }), {
          stopReason: 'toolUse'
        }),
      // Cover both Note A and Note B -> mints (epCoverAB, 1) = pngA and (epCoverAB, 2) = pngB.
      () =>
        fauxAssistantMessage(fauxToolCall('fold', { since: 't3', note: 'Covered A and B.' }), {
          stopReason: 'toolUse'
        }),
      (context) => {
        const latest = text(conversation(context).at(-1)!);
        epCoverAB = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
        assert.match(latest, new RegExp(`\\[${epCoverAB} image #1:`));
        assert.match(latest, new RegExp(`\\[${epCoverAB} image #2:`));
        branchBeforeUnfold = branchHarness.session.sessionManager.getLeafId()!;
        return fauxAssistantMessage(fauxToolCall('unfold', { handle: epCoverAB }), {
          stopReason: 'toolUse'
        });
      },
      // After unfolding epCoverAB, create a new fold with image C and cover from Note B onward -> mints epCoverBC.
      () =>
        fauxAssistantMessage(fauxToolCall('bash', { command: 'echo c2' }), {
          stopReason: 'toolUse'
        }),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't8', note: 'Replacement note with image C.' }),
          { stopReason: 'toolUse' }
        ),
      () =>
        fauxAssistantMessage(
          fauxToolCall('fold', { since: 't5', note: 'Covered B and C on unfolded branch.' }),
          { stopReason: 'toolUse' }
        ),
      (context) => {
        const latest = text(conversation(context).at(-1)!);
        epCoverBC = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
        assert.notEqual(epCoverBC, epCoverAB);
        return fauxAssistantMessage(fauxToolCall('recall', { handle: epCoverBC, image: 2 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.ok(typeof last.content !== 'string');
        assert.deepEqual(
          last.content.filter((b) => b.type === 'image'),
          [pngC]
        );
        // Switch back to the branch BEFORE unfold: (epCoverAB, 1)=pngA and (epCoverAB, 2)=pngB are intact,
        // AND (epCoverBC, 2)=pngC on the other branch still resolves without collision!
        branchHarness.session.sessionManager.branch(branchBeforeUnfold);
        return fauxAssistantMessage(fauxToolCall('recall', { handle: epCoverAB, image: 1 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.ok(typeof last.content !== 'string');
        assert.deepEqual(
          last.content.filter((b) => b.type === 'image'),
          [pngA]
        );
        return fauxAssistantMessage(fauxToolCall('recall', { handle: epCoverAB, image: 2 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.ok(typeof last.content !== 'string');
        assert.deepEqual(
          last.content.filter((b) => b.type === 'image'),
          [pngB]
        );
        return fauxAssistantMessage(fauxToolCall('recall', { handle: epCoverBC, image: 2 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.ok(typeof last.content !== 'string');
        assert.deepEqual(
          last.content.filter((b) => b.type === 'image'),
          [pngC]
        );
        return fauxAssistantMessage('done branch identity');
      }
    ],
    [attachImages]
  );
  try {
    await branchHarness.run('start');
  } finally {
    await branchHarness.cleanup();
  }
});

test('turn_end declines direct and codemode covering folds when a hook removes or alters persisted archivedImages or substitutes inline images for refs', async () => {
  const pngBlock: ImageContent = {
    type: 'image',
    data: makePng(2, 2, [77, 88, 99]).toString('base64'),
    mimeType: 'image/png'
  };

  for (const mode of ['off', 'only'] as const) {
    const tamperHooks: ExtensionFactory = (pi) => {
      pi.on('tool_result', (event) => {
        const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
        if (
          mode === 'off' &&
          /Seed screenshot note/.test(joined) &&
          !/Notes of the earlier folds/.test(joined)
        ) {
          return { content: [...event.content, pngBlock] };
        }
        if (/Removed-manifest fold/.test(joined)) {
          if (mode === 'off') {
            const d = { ...(event.details as Record<string, unknown>) };
            delete d.archivedImages;
            return { details: d };
          }
          const outer = { ...(event.details as Record<string, unknown>) };
          const inner = { ...(outer[FOLD_DETAILS_KEY] as Record<string, unknown>) };
          delete inner.archivedImages;
          outer[FOLD_DETAILS_KEY] = inner;
          return { details: outer };
        }
        if (/Altered-manifest fold/.test(joined)) {
          if (mode === 'off') {
            const d = { ...(event.details as Record<string, unknown>) };
            const list = [...((d.archivedImages as Record<string, unknown>[]) ?? [])];
            list[0] = { ...list[0], sha256: '0000000000000000' };
            d.archivedImages = list;
            return { details: d };
          }
          const outer = { ...(event.details as Record<string, unknown>) };
          const inner = { ...(outer[FOLD_DETAILS_KEY] as Record<string, unknown>) };
          const list = [...((inner.archivedImages as Record<string, unknown>[]) ?? [])];
          list[0] = { ...list[0], sha256: '0000000000000000' };
          inner.archivedImages = list;
          outer[FOLD_DETAILS_KEY] = inner;
          return { details: outer };
        }
        if (/Substituted-image fold/.test(joined)) {
          return {
            content: event.content.map((b) =>
              b.type === 'text' && /image #1: image\/png/.test(b.text) ? pngBlock : b
            )
          };
        }
        return undefined;
      });
    };

    const harness = await createPersistedSession(
      mode,
      [
        () =>
          mode === 'only'
            ? script('text(await tools.bash({ command: "echo s1" }));')
            : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo s1' }), {
                stopReason: 'toolUse'
              }),
        () =>
          mode === 'only'
            ? script(
                `image(${JSON.stringify(pngBlock)}); text(await tools.fold({ since: "t2", note: "Seed screenshot note." }));`
              )
            : fauxAssistantMessage(
                fauxToolCall('fold', { since: 't2', note: 'Seed screenshot note.' }),
                { stopReason: 'toolUse' }
              ),
        () =>
          mode === 'only'
            ? script('text(await tools.bash({ command: "echo s2" }));')
            : fauxAssistantMessage(fauxToolCall('bash', { command: 'echo s2' }), {
                stopReason: 'toolUse'
              }),
        // 1) Removed manifest -> must be declined (Seed screenshot note + inline pngBlock remain in context).
        () =>
          mode === 'only'
            ? script('text(await tools.fold({ since: "t3", note: "Removed-manifest fold." }));')
            : fauxAssistantMessage(
                fauxToolCall('fold', { since: 't3', note: 'Removed-manifest fold.' }),
                { stopReason: 'toolUse' }
              ),
        (context) => {
          const view = conversation(context);
          const inline = view.flatMap((m) =>
            typeof m.content === 'string' ? [] : m.content.filter((b) => b.type === 'image')
          );
          assert.deepEqual(
            inline,
            [pngBlock],
            `${mode}: removed manifest must decline covering fold`
          );
          return mode === 'only'
            ? script('text(await tools.fold({ since: "t3", note: "Altered-manifest fold." }));')
            : fauxAssistantMessage(
                fauxToolCall('fold', { since: 't3', note: 'Altered-manifest fold.' }),
                { stopReason: 'toolUse' }
              );
        },
        // 2) Altered manifest -> must also be declined.
        (context) => {
          const view = conversation(context);
          const inline = view.flatMap((m) =>
            typeof m.content === 'string' ? [] : m.content.filter((b) => b.type === 'image')
          );
          assert.deepEqual(
            inline,
            [pngBlock],
            `${mode}: altered manifest must decline covering fold`
          );
          return mode === 'only'
            ? script('text(await tools.fold({ since: "t3", note: "Substituted-image fold." }));')
            : fauxAssistantMessage(
                fauxToolCall('fold', { since: 't3', note: 'Substituted-image fold.' }),
                { stopReason: 'toolUse' }
              );
        },
        // 3) Substituting inline image in place of [ep-... image #1] -> must also be declined.
        () => {
          const branch = harness.session.sessionManager.getBranch();
          assert.equal(
            threadRecords(branch).length,
            1,
            `${mode}: only seed fold should be applied`
          );
          return fauxAssistantMessage('done tamper checks');
        }
      ],
      [tamperHooks]
    );
    try {
      await harness.run('start');
    } finally {
      await harness.cleanup();
    }
  }
});

test('written prefix protects script-printed inherited image references on unfold and fails conservatively when ambiguous', async () => {
  const pngBlock: ImageContent = {
    type: 'image',
    data: makePng(3, 2, [11, 22, 33]).toString('base64'),
    mimeType: 'image/png'
  };
  let inheritedRefLine = '';
  let ep3 = '';

  // Pre-episodes hook injects a standalone inheritedRefLine block at index 0 of Ep3's codemode result
  // BEFORE episodes records written.blocks and appends fold's generated copy after written.blocks.
  const injectPrePrefixCopy: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!inheritedRefLine || !/script-echo-marker/.test(joined)) return undefined;
      return {
        content: [{ type: 'text', text: inheritedRefLine }, ...event.content]
      };
    });
  };

  const harness = await createPersistedSession(
    'only',
    [
      () => script('text(await tools.bash({ command: "echo p1" }));'),
      () =>
        script(
          `image(${JSON.stringify(pngBlock)}); text(await tools.fold({ since: "t2", note: "Ep1 with image." }));`
        ),
      () => script('text(await tools.bash({ command: "echo p2" }));'),
      // Ep2 offloads Ep1's image to [ep2 image #1].
      () => script('text(await tools.fold({ since: "t3", note: "Ep2 offloads image." }));'),
      (context) => {
        const latest = text(conversation(context).at(-1)!);
        const m = /^\[ep-[0-9a-f]{8} image #1:.*\]$/m.exec(latest);
        assert.ok(m);
        inheritedRefLine = m[0];
        return script('text(await tools.bash({ command: "echo p3" }));');
      },
      // Ep3 runs with standalone inheritedRefLine at index 0 (inside written.blocks) AND fold's copy appended after written.blocks.
      () =>
        script(
          'text("script-echo-marker"); text(await tools.fold({ since: "t5", note: "Ep3 echoes ref." }));'
        ),
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.ok(typeof last.content !== 'string');
        const exactBlocks = last.content.filter(
          (b) => b.type === 'text' && b.text === inheritedRefLine
        );
        assert.equal(exactBlocks.length, 2, 'script copy + fold-appended copy');
        ep3 = /\[fold (ep-[0-9a-f]{8}):/.exec(text(last))![1]!;
        return script(`text(await tools.unfold({ handle: ${JSON.stringify(ep3)} }));`);
      },
      (context) => {
        const view = conversation(context);
        const unfoldedEp3 = view.find(
          (m) => m.role === 'toolResult' && text(m).includes(`[unfolded ${ep3}]`)
        );
        assert.ok(unfoldedEp3 && typeof unfoldedEp3.content !== 'string');
        // Pre-prefix standalone copy at index 0 MUST survive unfold, while fold's appended copy MUST be dropped!
        assert.equal(unfoldedEp3.content[0]?.type, 'text');
        assert.equal((unfoldedEp3.content[0] as TextContent).text, inheritedRefLine);
        const remainingCopies = unfoldedEp3.content.filter(
          (b) => b.type === 'text' && b.text === inheritedRefLine
        );
        assert.equal(remainingCopies.length, 1, 'fold copy dropped, script copy preserved');
        return fauxAssistantMessage('done prefix protection');
      }
    ],
    [],
    { preExtensions: [injectPrePrefixCopy] }
  );
  try {
    await harness.run('start');
  } finally {
    await harness.cleanup();
  }

  // Also verify conservative failure when written prefix is absent/broken and C > K (ambiguous duplicates).
  const ambDir = await mkdtemp(join(tmpdir(), 'pi-episodes-ambig-'));
  try {
    const ambRef = (await archiveImage(ambDir, 'ep-ambig001', 1, pngBlock))!;
    const ambText = formatImageRef(ambRef);
    const ambBlocks: (TextContent | ImageContent)[] = [
      { type: 'text', text: ambText },
      { type: 'text', text: 'mutated note body' },
      { type: 'text', text: ambText }
    ];
    // Without intact protectedPrefixLen, 2 copies vs 1 manifest entry is ambiguous -> matches 0 blocks.
    assert.equal(matchArchivedRefBlocks(ambBlocks, [ambRef], undefined).size, 0);
    // With intact protectedPrefixLen = 2, only index >= 2 is matched -> matches index 2 unambiguously.
    assert.deepEqual([...matchArchivedRefBlocks(ambBlocks, [ambRef], 2).keys()], [2]);
    // holdsCarriedMedia also declines ambiguous broken-prefix duplicates (undefined prefix, 2 copies vs 1 item).
    assert.equal(holdsCarriedMedia(ambBlocks, [{ kind: 'ref', ref: ambRef }], undefined), false);
    assert.equal(holdsCarriedMedia(ambBlocks, [{ kind: 'ref', ref: ambRef }], 2), true);
  } finally {
    await rm(ambDir, { recursive: true, force: true });
  }
});

test('truncated real session journal recovers interrupted covering fold and recall reads exact bytes with archived path', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-trunc-recovery-'));
  const pngBytes = makePng(5, 3, [15, 125, 235]);
  const pngBlock: ImageContent = {
    type: 'image',
    data: pngBytes.toString('base64'),
    mimeType: 'image/png'
  };
  const attachFirst: ExtensionFactory = (pi) => {
    pi.on('tool_result', (event) => {
      const joined = event.content.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      if (!/First real journal note/.test(joined) || /Notes of the earlier folds/.test(joined)) {
        return undefined;
      }
      return { content: [...event.content, pngBlock] };
    });
  };

  try {
    let ep2 = '';
    let expectedPath = '';
    const first = await createPersistedSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: 'echo r1' }), {
            stopReason: 'toolUse'
          }),
        () =>
          fauxAssistantMessage(
            fauxToolCall('fold', { since: 't2', note: 'First real journal note.' }),
            { stopReason: 'toolUse' }
          ),
        () =>
          fauxAssistantMessage(fauxToolCall('bash', { command: 'echo r2' }), {
            stopReason: 'toolUse'
          }),
        () =>
          fauxAssistantMessage(
            fauxToolCall('fold', { since: 't3', note: 'Covering real journal note.' }),
            { stopReason: 'toolUse' }
          ),
        (context) => {
          const latest = text(conversation(context).at(-1)!);
          ep2 = /\[fold (ep-[0-9a-f]{8}):/.exec(latest)![1]!;
          expectedPath = /— ([^;]+); recall\(/.exec(latest)![1]!;
          return fauxAssistantMessage('done initial run');
        }
      ],
      [attachFirst],
      { dir, keepDir: true }
    );
    await first.run('start');
    const sessionFile = first.session.sessionManager.getSessionFile()!;
    await first.cleanup();

    // Truncate the real JSONL journal mid-batch right after ep2's FOLD_STARTED_TYPE + 1st context_edit
    // (before remaining context_edits and before FOLD_APPLIED_TYPE).
    const rawLines = (await readFile(sessionFile, 'utf8')).trim().split('\n');
    const startedIdx = rawLines.findIndex((line) => {
      const parsed = JSON.parse(line) as {
        type?: string;
        customType?: string;
        data?: { handle?: string };
      };
      return (
        parsed.type === 'custom' &&
        parsed.customType === FOLD_STARTED_TYPE &&
        parsed.data?.handle === ep2
      );
    });
    assert.ok(startedIdx >= 0, 'found ep2 FOLD_STARTED_TYPE in real session file');
    const truncated = `${rawLines.slice(0, startedIdx + 2).join('\n')}\n`;
    await writeFile(sessionFile, truncated);

    const reopened = SessionManager.open(sessionFile);
    // Before recovery completes, recall cannot reach the cut-short fold.
    const recoveredSession = await createPersistedSession(
      'off',
      [
        () =>
          fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
            stopReason: 'toolUse'
          }),
        (context) => {
          const errRes = conversation(context).at(-1)!;
          assert.equal(errRes.role, 'toolResult');
          assert.equal(errRes.isError, true);
          assert.match(text(errRes), new RegExp(`No archived thread ${ep2}`));
          // turn_end of this turn runs completeCutShortFolds and completes ep2!
          return fauxAssistantMessage(fauxToolCall('recall', { handle: ep2, image: 1 }), {
            stopReason: 'toolUse'
          });
        },
        (context) => {
          const okRes = conversation(context).at(-1)!;
          assert.equal(okRes.role, 'toolResult');
          assert.equal(okRes.isError, false);
          assert.deepEqual(
            okRes.content.filter((b) => b.type === 'image'),
            [pngBlock],
            'recovered ep2 recalls exact PNG bytes'
          );
          assert.ok(
            text(okRes).includes(expectedPath),
            `recall note must include archived.path (${expectedPath}), got: ${text(okRes)}`
          );
          return fauxAssistantMessage('done recovery');
        }
      ],
      [],
      { dir, keepDir: true, sessionManager: reopened }
    );
    await recoveredSession.run('recover');
    await recoveredSession.cleanup();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('readArchivedImage and verifyArchivedImageSync confine paths to archiveDir, reject symlinks, and unfold protects inline script images when written.sha256 is broken', async () => {
  const outerDir = await mkdtemp(join(tmpdir(), 'pi-episodes-confine-outer-'));
  const archiveSubdir = join(outerDir, 'session.episodes');
  try {
    const png = makePng(2, 2, [50, 100, 150]);
    const pngBlock: ImageContent = {
      type: 'image',
      data: png.toString('base64'),
      mimeType: 'image/png'
    };
    const validRef = (await archiveImage(archiveSubdir, 'ep-confine01', 1, pngBlock))!;
    assert.equal(verifyArchivedImageSync(validRef, archiveSubdir), true);
    assert.deepEqual(await readArchivedImage(validRef, archiveSubdir), pngBlock);

    // 1) Outside file with valid PNG bytes and matching sha256/size MUST be rejected when confined to archiveSubdir.
    const outsideFile = join(outerDir, 'image-0123456789abcdef.png');
    await writeFile(outsideFile, png);
    const outsideRef = { ...validRef, path: outsideFile };
    assert.equal(verifyArchivedImageSync(outsideRef, archiveSubdir), false);
    await assert.rejects(
      () => readArchivedImage(outsideRef, archiveSubdir),
      /outside the session archive directory/
    );

    // 2) Traversal path (session.episodes/../image-...) MUST be rejected.
    const traversalRef = {
      ...validRef,
      path: join(archiveSubdir, '..', 'image-0123456789abcdef.png')
    };
    assert.equal(verifyArchivedImageSync(traversalRef, archiveSubdir), false);
    await assert.rejects(
      () => readArchivedImage(traversalRef, archiveSubdir),
      /outside the session archive directory/
    );

    // 3) Arbitrary non-archive basename inside archiveSubdir MUST be rejected.
    const badNameFile = join(archiveSubdir, 'secret.png');
    await writeFile(badNameFile, png);
    const badNameRef = { ...validRef, path: badNameFile };
    assert.equal(verifyArchivedImageSync(badNameRef, archiveSubdir), false);
    await assert.rejects(
      () => readArchivedImage(badNameRef, archiveSubdir),
      /outside the session archive directory/
    );

    // 4) Symlink inside archiveSubdir pointing to outsideFile MUST be rejected.
    const symlinkPath = join(archiveSubdir, 'image-fedcba9876543210.png');
    await symlink(outsideFile, symlinkPath);
    const symlinkRef = { ...validRef, path: symlinkPath };
    assert.equal(verifyArchivedImageSync(symlinkRef, archiveSubdir), false);
    await assert.rejects(() => readArchivedImage(symlinkRef, archiveSubdir), /not a regular file/);

    // 5) When written.sha256 WAS recorded on a codemode script result and a hook prepended a block
    // (breaking written.sha256), unfold drops carriedImg from the end rather than deleting scriptImg at stale written.blocks.
    const scriptImg: ImageContent = {
      type: 'image',
      mimeType: 'image/png',
      data: 'U0NSSVBUX0lNRw=='
    };
    const carriedImg: ImageContent = {
      type: 'image',
      mimeType: 'image/png',
      data: 'Q0FSUklFRF9JTUc='
    };
    const rendered = '[fold ep-cmbroken: t2 (2 messages, ~10 tokens)]\n\nBroken prefix note.';
    const originalWritten: (TextContent | ImageContent)[] = [
      { type: 'text', text: `SCRIPT_OUT\n${rendered}` },
      scriptImg
    ];
    const sBroken = session([
      user('start', 1),
      assistant(call('b1', 'bash', { command: 'echo ok' })),
      result('b1', 'bash', 'ok', 2),
      assistant(call('cm_brk', 'codemode', { code: 'fold()' }))
    ]);
    const p = planFold(sBroken.manager.buildSessionProjection().entries, 'cm_brk', 2);
    const brkDetails: FoldDetails = {
      kind: 'fold',
      handle: 'ep-cmbroken',
      since: 't2',
      firstEntryId: p.firstEntryId,
      lastEntryId: p.lastEntryId,
      messages: p.omitted.length,
      tokens: 10,
      headline: 'Broken prefix note.',
      keptUserTurns: [],
      files: [],
      note: 'Broken prefix note.',
      renderedNote: rendered,
      carriedImages: 1,
      written: writtenMark(originalWritten)
    };
    // Hook prepended { text: 'HOOK_CUE' } at index 0 -> scriptImg is now at index 2 (=== written.blocks)!
    sBroken.manager.appendMessage(
      Object.assign({
        role: 'toolResult' as const,
        toolCallId: 'cm_brk',
        toolName: 'codemode',
        content: [
          { type: 'text' as const, text: 'HOOK_CUE' },
          ...originalWritten,
          carriedImg,
          { type: 'text' as const, text: '[t3]' }
        ],
        details: { [FOLD_DETAILS_KEY]: brkDetails },
        isError: false,
        timestamp: 0
      })
    );
    for (const ed of p.edits) sBroken.manager.appendContextEdit(ed.targetId, ed.replacement);
    sBroken.manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-cmbroken' });
    const unf = planUnfold(
      sBroken.manager.getBranch(),
      sBroken.manager.buildSessionProjection().entries,
      'ep-cmbroken'
    );
    for (const ed of unf.edits) sBroken.manager.appendContextEdit(ed.targetId, ed.replacement);
    const afterMsgs = sBroken.manager.buildSessionProjection().messages;
    const unfMsg = afterMsgs.at(-1)!;
    assert.equal(unfMsg.role, 'toolResult');
    assert.deepEqual(
      unfMsg.content.filter((b) => b.type === 'image'),
      [scriptImg],
      'scriptImg preserved and carriedImg dropped even when prepended block shifted indices'
    );
  } finally {
    await rm(outerDir, { recursive: true, force: true });
  }
});
