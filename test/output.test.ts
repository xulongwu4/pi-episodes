import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { CAPPED_KEY, capOutput, cappedOutput, withFullOutput } from '../src/output.js';
import { readStamp, withStamp } from '../src/stamps.js';
import { result } from './helpers.js';

const long = Array.from({ length: 400 }, (_, index) => `line ${index} of the build log`).join('\n');

test('a result under the limit is left alone', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    assert.equal(await capOutput([{ type: 'text', text: 'short' }], dir, 1000, 300), undefined);
    // A preview half the size of the original is not worth a file.
    assert.equal(
      await capOutput([{ type: 'text', text: 'x'.repeat(500) }], dir, 400, 300),
      undefined
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a large result keeps its start and end and names the file with all of it', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await capOutput([{ type: 'text', text: long }], dir, 1000, 300);
    assert.ok(cut);
    const block = cut.content[0]!;
    assert.ok(block.type === 'text');
    assert.match(block.text, /^line 0 of the build log\n/);
    assert.match(block.text, /\nline 399 of the build log$/);
    assert.doesNotMatch(block.text, /line 200 of/);
    assert.ok(
      block.text.includes(`characters left out here. All of this output is in ${cut.capped.path}`)
    );
    // Only whole lines are shown.
    assert.ok(block.text.split('\n').every((line) => !/^line \d+ of the build$/.test(line)));
    assert.ok(block.text.length < 700);
    assert.deepEqual(cut.capped, { path: cut.capped.path, chars: long.length, block: 0 });
    assert.equal(await readFile(cut.capped.path, 'utf8'), long);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a short first block and images stay as they are', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const image = { type: 'image' as const, data: 'AAAA', mimeType: 'image/png' };
    const cut = await capOutput(
      [{ type: 'text', text: 'Script completed\nOutput:\n' }, { type: 'text', text: long }, image],
      dir,
      1000,
      300
    );
    assert.ok(cut);
    assert.deepEqual(cut.content[0], { type: 'text', text: 'Script completed\nOutput:\n' });
    assert.equal(cut.capped.block, 1);
    assert.deepEqual(cut.content[2], image);
    assert.equal(await readFile(cut.capped.path, 'utf8'), long);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('the full text is put back for search, under the stamp', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await capOutput([{ type: 'text', text: long }], dir, 1000, 300);
    assert.ok(cut);
    const message: Extract<AgentMessage, { role: 'toolResult' }> = {
      role: 'toolResult',
      toolCallId: 'c1',
      toolName: 'bash',
      content: withStamp(cut.content, 7),
      details: { exitCode: 0, [CAPPED_KEY]: { ...cut.capped } },
      isError: false,
      timestamp: 0
    };
    assert.deepEqual(cappedOutput(message.details), cut.capped);

    const full = await withFullOutput(message);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content[0], { type: 'text', text: long });
    assert.equal(readStamp(full), 7);

    // A file that is gone leaves the preview, which is what the model saw.
    await rm(cut.capped.path);
    assert.equal(await withFullOutput(message), message);
    const plain = result('c2', 'bash', 'ok', 8);
    assert.equal(await withFullOutput(plain), plain);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
