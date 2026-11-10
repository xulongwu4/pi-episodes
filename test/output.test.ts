import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
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

test('the header of a script result and images stay where they are', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const image = { type: 'image' as const, data: 'AAAA', mimeType: 'image/png' };
    const header = {
      type: 'text' as const,
      text: 'Script completed\nWall time 0.2 seconds\nOutput:\n'
    };
    const cut = await capOutput(
      [header, image, { type: 'text', text: long }, image],
      dir,
      1000,
      300
    );
    assert.ok(cut);
    assert.deepEqual(
      cut.content.map((block) => (block.type === 'text' ? block.text.slice(0, 6) : 'image')),
      ['Script', 'image', 'line 0', 'image']
    );
    assert.equal(cut.capped.block, 2);
    assert.equal(await readFile(cut.capped.path, 'utf8'), long);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a short first block of another tool is part of the output and of its file', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await capOutput(
      [
        { type: 'text', text: 'Found 400 lines:' },
        { type: 'text', text: long }
      ],
      dir,
      1000,
      300
    );
    assert.ok(cut);
    assert.equal(cut.content.length, 1);
    assert.equal(cut.capped.block, 0);
    assert.equal(await readFile(cut.capped.path, 'utf8'), `Found 400 lines:\n${long}`);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a result with an image between its text blocks is left whole', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const image = { type: 'image' as const, data: 'AAAA', mimeType: 'image/png' };
    const text = { type: 'text' as const, text: long };
    assert.equal(await capOutput([text, image, text], dir, 1000, 300), undefined);
    assert.deepEqual(await readdir(dir), []);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a file that cannot be put in place leaves no temporary file', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const name = `output-${createHash('sha256').update(long).digest('hex').slice(0, 16)}.txt`;
    await mkdir(join(dir, name));
    await assert.rejects(capOutput([{ type: 'text', text: long }], dir, 1000, 300));
    assert.deepEqual(await readdir(dir), [name]);
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

    // A handler that ran after the cut put a block first: the preview is found by its file.
    const notice = { type: 'text' as const, text: 'a notice' };
    const moved = await withFullOutput({ ...message, content: [notice, ...message.content] });
    assert.ok(moved.role === 'toolResult');
    assert.deepEqual(moved.content.slice(0, 2), [notice, { type: 'text', text: long }]);
    // A cue added when the result was final sits between the preview and the stamp.
    const cue = { type: 'text' as const, text: '[episodes] a cue' };
    const cued = await withFullOutput({ ...message, content: withStamp([...cut.content, cue], 7) });
    assert.ok(cued.role === 'toolResult');
    assert.deepEqual(cued.content.slice(0, 2), [{ type: 'text', text: long }, cue]);
    assert.equal(readStamp(cued), 7);
    // Without the preview nothing is put back.
    const replaced = { ...message, content: withStamp([notice], 7) };
    assert.equal(await withFullOutput(replaced), replaced);

    // A file that is gone leaves the preview, which is what the model saw.
    await rm(cut.capped.path);
    assert.equal(await withFullOutput(message), message);
    const plain = result('c2', 'bash', 'ok', 8);
    assert.equal(await withFullOutput(plain), plain);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('the text another extension offloaded is put back when its file matches the record', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    // As tool-result-offload leaves a script's result: the status block, then one preview.
    const sha256 = createHash('sha256').update(long).digest('hex');
    const path = join(dir, `${sha256}.txt`);
    await writeFile(path, long);
    const header = {
      type: 'text' as const,
      text: 'Script completed\nWall time 0.1 seconds\nOutput:\n'
    };
    const preview = `line 0\n\n[tool-result-offload: 9 of 10 chars omitted. This result in full (sha256 ${sha256.slice(0, 12)}): ${path} — grep it.]\n\nline 9`;
    const message: Extract<AgentMessage, { role: 'toolResult' }> = {
      role: 'toolResult',
      toolCallId: 'c1',
      toolName: 'codemode',
      content: withStamp([header, { type: 'text', text: preview }], 7),
      details: { offload: { path, sha256, chars: long.length } },
      isError: false,
      timestamp: 0
    };
    const full = await withFullOutput(message);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content.slice(0, 2), [header, { type: 'text', text: long }]);
    assert.equal(readStamp(full), 7);

    // A record without the hash, or a result that no longer names the file, is left alone.
    const unhashed = { ...message, details: { offload: { path } } };
    assert.equal(await withFullOutput(unhashed), unhashed);
    const replaced = { ...message, content: withStamp([header], 7) };
    assert.equal(await withFullOutput(replaced), replaced);
    // A file that is not the recorded text is not the output.
    await writeFile(path, 'something else');
    assert.equal(await withFullOutput(message), message);
    await rm(path);
    assert.equal(await withFullOutput(message), message);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

/** A preview as tool-result-offload writes it, with the file it keeps the text in. */
async function offloaded(dir: string, text: string, keep: number) {
  const sha256 = createHash('sha256').update(text).digest('hex');
  const path = join(dir, `${sha256}.txt`);
  await writeFile(path, text);
  const head = text.slice(0, Math.floor((keep * 2) / 3));
  const tail = text.slice(text.length - Math.floor(keep / 3));
  const preview = `${head}\n\n[tool-result-offload: some chars omitted. This result in full (sha256 ${sha256.slice(0, 12)}): ${path} — grep it.]\n\n${tail}`;
  return { preview, details: { path, sha256, chars: text.length } };
}

type ToolResult = Extract<AgentMessage, { role: 'toolResult' }>;

function toolResult(
  content: ToolResult['content'],
  details: NonNullable<ToolResult['details']>
): ToolResult {
  return {
    role: 'toolResult',
    toolCallId: 'c1',
    toolName: 'codemode',
    content,
    details,
    isError: false,
    timestamp: 0
  };
}

test('a result that both extensions cut gets all of its text back, whichever cut first', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    // The other extension first: this one cut its preview, and its words are out of sight.
    const first = await offloaded(dir, long, 3000);
    const second = await capOutput([{ type: 'text', text: first.preview }], dir, 1000, 300);
    assert.ok(second);
    const twice = toolResult(withStamp(second.content, 7), {
      offload: first.details,
      [CAPPED_KEY]: { ...second.capped }
    });
    assert.doesNotMatch(JSON.stringify(twice.content), /tool-result-offload/);
    const full = await withFullOutput(twice);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content[0], { type: 'text', text: long });
    assert.equal(readStamp(full), 7);

    // This extension first: the other one cut its preview.
    const cut = await capOutput([{ type: 'text', text: long }], dir, 4000, 3000);
    assert.ok(cut?.content[0]?.type === 'text');
    const outer = await offloaded(dir, cut.content[0].text, 600);
    assert.doesNotMatch(outer.preview, /\[episodes\]/);
    const reversed = toolResult(withStamp([{ type: 'text', text: outer.preview }], 7), {
      [CAPPED_KEY]: { ...cut.capped },
      offload: outer.details
    });
    const whole = await withFullOutput(reversed);
    assert.ok(whole.role === 'toolResult');
    assert.deepEqual(whole.content[0], { type: 'text', text: long });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a block that only mentions the file of a cut is not taken for its preview', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await offloaded(dir, long, 600);
    const note = { type: 'text' as const, text: `Saved a copy at ${cut.details.path}.` };
    const preview = { type: 'text' as const, text: cut.preview };
    const message = toolResult(withStamp([note, preview], 7), { offload: cut.details });
    const full = await withFullOutput(message);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content.slice(0, 2), [note, { type: 'text', text: long }]);

    // Two blocks that read as the preview are not told apart: neither is replaced.
    const doubled = toolResult(withStamp([preview, preview], 7), { offload: cut.details });
    assert.equal(await withFullOutput(doubled), doubled);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
