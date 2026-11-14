import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
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
  callingFork,
  FORK_TOOL,
  type FoldDetails,
  type ForkArchive,
  type ForkDetails,
  runningFork,
  ThreadIndex,
  threadRecords,
  writeForkArchive
} from '../src/archive.js';
import { ForkFolds } from '../src/fork-fold.js';
import {
  archiveImage,
  decodeImageBlock,
  formatImageRef,
  imageDimensions,
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
  extensions: ExtensionFactory[] = []
) {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-images-'));
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
    sessionManager: SessionManager.create(dir, join(dir, 'sessions')),
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
      await rm(dir, { recursive: true, force: true });
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
    let ep1 = '';
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
          ep1 = /ep-[0-9a-f]{8}/.exec(foldText)![0];
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
          const match = new RegExp(
            `\\[${ep1} image #1: image/png, 2x3, \\d+ B, sha256 [0-9a-f]{16} — ([^;]+); recall\\(\\{ handle: "${ep1}", image: 1 \\}\\) or read path\\]`
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
            latest.split(`[${ep1} image #1:`).length - 1,
            1,
            `${mode}: reference should appear exactly once on third fold`
          );
          return mode === 'only'
            ? script(
                `const out = await tools.recall({ handle: ${JSON.stringify(ep1)}, image: 1 }); image(out); text(out.note);`
              )
            : fauxAssistantMessage(fauxToolCall('recall', { handle: ep1, image: 1 }), {
                stopReason: 'toolUse'
              });
        },
        (context) => {
          const last = conversation(context).at(-1)!;
          assert.ok(typeof last.content !== 'string');
          const returnedImages = last.content.filter((b) => b.type === 'image');
          assert.deepEqual(returnedImages, [pngBlock], `${mode}: recall should return exact PNG`);
          assert.match(text(last), new RegExp(`\\[${ep1} image #1: image/png, 2x3,`));
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

  let ep1 = '';
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
        ep1 = /ep-[0-9a-f]{8}/.exec(conversation(context).map(text).join('\n'))![0];
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
        assert.match(latest, new RegExp(`\\[${ep1} image #1: image/png`));
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
        return fauxAssistantMessage(fauxToolCall('recall', { handle: ep1, image: 1 }), {
          stopReason: 'toolUse'
        });
      },
      (context) => {
        const last = conversation(context).at(-1)!;
        assert.equal(last.role, 'toolResult');
        assert.equal(last.isError, true);
        assert.match(text(last), new RegExp(`No archived image #1 in ${ep1}`));
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
    const cf1Handle = (cf1Res.details as FoldDetails).handle;
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
        (b) => b.type === 'text' && b.text.includes(`[${cf1Handle} image #1: image/png, 5x4,`)
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

    // Running child sees its offloaded image; outside / sibling fork does not.
    const outsideHit = await new ThreadIndex(manager, state.forkArchives).findImage(cf1Handle, 1);
    assert.equal(outsideHit, undefined);
    const runningHit = await runningFork.run(
      { sessionId: manager.getSessionId(), transcript: snapshot },
      () =>
        new ThreadIndex(manager, state.forkArchives, callingFork(manager)).findImage(cf1Handle, 1)
    );
    assert.ok(runningHit);
    assert.deepEqual(await readArchivedImage(runningHit), pngBlock);

    // Unfolding cf2 inside the fork retires cf2's offloaded manifest.
    await unfoldTool.execute('cu1', { handle: cf2Handle });
    const afterUnfold = await runningFork.run(
      { sessionId: manager.getSessionId(), transcript: snapshot },
      () =>
        new ThreadIndex(manager, state.forkArchives, callingFork(manager)).findImage(cf1Handle, 1)
    );
    assert.equal(afterUnfold, undefined);

    // Re-folding and finishing the fork makes the archived image reachable from the parent session.
    childMsgs.push(assistant(call('cf3', 'fold', { since: 't11', note: 'child re-covered' })));
    const cf3Res = await foldTool.execute('cf3', { since: 't11', note: 'child re-covered' });
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
    const parentHit = await new ThreadIndex(manager, state.forkArchives).findImage(cf1Handle, 1);
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
