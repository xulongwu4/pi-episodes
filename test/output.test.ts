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
    assert.deepEqual(cut.capped, {
      path: cut.capped.path,
      chars: long.length,
      block: 0,
      preview: block.text.length,
      head: block.text.indexOf('\n\n[episodes] ')
    });
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

/**
 * A preview as tool-result-offload writes it, with the file it keeps the text in: the start
 * and the end, cut at a line break where one is near, around a line with the counts.
 */
async function offloaded(dir: string, text: string, keep: number) {
  const sha256 = createHash('sha256').update(text).digest('hex');
  const path = join(dir, `${sha256}.txt`);
  await writeFile(path, text);
  const starts = Math.floor((keep * 2) / 3);
  const ends = Math.floor(keep / 3);
  const cut = text.lastIndexOf('\n', starts);
  const head = text.slice(0, cut > starts / 2 ? cut : starts);
  const from = text.indexOf('\n', text.length - ends);
  const tail = text.slice(
    from >= 0 && from < text.length - ends / 2 ? from + 1 : text.length - ends
  );
  const omitted = text.length - head.length - tail.length;
  const preview = `${head}\n\n[tool-result-offload: ${omitted} of ${text.length} chars omitted. This result in full (sha256 ${sha256.slice(0, 12)}): ${path} — grep it.]\n\n${tail}`;
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

test('the text another extension offloaded is put back when its file matches the record', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    // As tool-result-offload leaves a script's result: the status block, then one preview.
    const { preview, details } = await offloaded(dir, long, 600);
    const { path } = details;
    const header = {
      type: 'text' as const,
      text: 'Script completed\nWall time 0.1 seconds\nOutput:\n'
    };
    const message = toolResult(withStamp([header, { type: 'text', text: preview }], 7), {
      offload: details
    });
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

test('what a handler added between two cuts stays when both are put back', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const warning = 'WARNING: output came from staging, not production.';
    // This extension cut the result, a handler added a block, and the other extension cut
    // both as one text: the preview made here comes back inside a block that holds more.
    const cut = await capOutput([{ type: 'text', text: long }], dir, 4000, 3000);
    assert.ok(cut?.content[0]?.type === 'text');
    const outer = await offloaded(dir, `${cut.content[0].text}\n${warning}`, 600);
    assert.doesNotMatch(outer.preview, /\[episodes\]/);
    const message = toolResult(withStamp([{ type: 'text', text: outer.preview }], 7), {
      [CAPPED_KEY]: { ...cut.capped },
      offload: outer.details
    });
    const full = await withFullOutput(message);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content[0], { type: 'text', text: `${long}\n${warning}` });
    assert.equal(readStamp(full), 7);

    // The other way round: the other extension cut first, and this one cut both blocks.
    const first = await offloaded(dir, long, 3000);
    const second = await capOutput(
      [
        { type: 'text', text: first.preview },
        { type: 'text', text: warning }
      ],
      dir,
      1000,
      300
    );
    assert.ok(second);
    assert.doesNotMatch(JSON.stringify(second.content), /tool-result-offload/);
    const reversed = toolResult(withStamp(second.content, 7), {
      offload: first.details,
      [CAPPED_KEY]: { ...second.capped }
    });
    const whole = await withFullOutput(reversed);
    assert.ok(whole.role === 'toolResult');
    assert.deepEqual(whole.content[0], { type: 'text', text: `${long}\n${warning}` });

    // A block added before the first preview: that preview closes the block it comes back in.
    const led = await capOutput(
      [
        { type: 'text', text: 'a notice' },
        { type: 'text', text: first.preview }
      ],
      dir,
      1000,
      300
    );
    assert.ok(led);
    const after = await withFullOutput(
      toolResult(led.content, { offload: first.details, [CAPPED_KEY]: { ...led.capped } })
    );
    assert.ok(after.role === 'toolResult');
    assert.deepEqual(after.content, [{ type: 'text', text: `a notice\n${long}` }]);

    // With blocks added on both sides, the preview neither opens nor closes the block: the
    // record says where it is, and without that the preview stays as the model saw it.
    const around = `a notice\n${cut.content[0].text}\n${warning}`;
    const wrapped = await offloaded(dir, around, 600);
    const details = { [CAPPED_KEY]: { ...cut.capped }, offload: wrapped.details };
    const inside = await withFullOutput(
      toolResult([{ type: 'text', text: wrapped.preview }], details)
    );
    assert.ok(inside.role === 'toolResult');
    assert.deepEqual(inside.content, [{ type: 'text', text: `a notice\n${long}\n${warning}` }]);
    const { head: _, ...unplaced } = cut.capped;
    const kept = await withFullOutput(
      toolResult([{ type: 'text', text: wrapped.preview }], { ...details, [CAPPED_KEY]: unplaced })
    );
    assert.ok(kept.role === 'toolResult');
    assert.deepEqual(kept.content, [{ type: 'text', text: around }]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a preview of repeated lines is put back where it stands', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    // The same row before and after the one that matters, and a handler adds that row again:
    // the start and the end of the output fit around the line in more than one way.
    const rows = (count: number) => Array.from({ length: count }, () => 'PASS test').join('\n');
    const log = `${rows(300)}\nFAIL the one that matters\n${rows(300)}`;
    const first = await offloaded(dir, log, 600);
    const second = await capOutput(
      [
        { type: 'text', text: first.preview },
        { type: 'text', text: 'PASS test' }
      ],
      dir,
      600,
      300
    );
    assert.ok(second);
    const full = await withFullOutput(
      toolResult(second.content, { offload: first.details, [CAPPED_KEY]: { ...second.capped } })
    );
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content, [{ type: 'text', text: `${log}\nPASS test` }]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a preview cut inside a character still gets its text back', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const faces = '😀'.repeat(50);
    const cut = await capOutput([{ type: 'text', text: faces }], dir, 10, 5);
    assert.ok(cut?.content[0]?.type === 'text');
    // Half a character was kept at each end, and reads as the replacement character.
    assert.match(cut.content[0].text, /^😀\uFFFD\n\n\[episodes\] [^\n]*\n\n\uFFFD$/);
    const full = await withFullOutput(toolResult(cut.content, { [CAPPED_KEY]: { ...cut.capped } }));
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content, [{ type: 'text', text: faces }]);
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
    // Text around the line that is not the start and the end of the output is no preview.
    assert.match(cut.preview, /^line 0 of the build log\nline 1 of/);
    assert.match(cut.preview, /\nline 398 of the build log\nline 399 of the build log$/);
    for (const [was, now] of [
      ['line 1 of', 'line 9 of'],
      ['line 398 of', 'line 000 of']
    ] as const) {
      const other = { type: 'text' as const, text: cut.preview.replace(was, now) };
      const altered = toolResult(withStamp([other], 7), { offload: cut.details });
      assert.equal(await withFullOutput(altered), altered);
    }
    // Of two previews in one block, one is put back, and no character is lost.
    const joined = { type: 'text' as const, text: `${cut.preview}\n${cut.preview}` };
    const twice = await withFullOutput(
      toolResult(withStamp([joined], 7), { offload: cut.details })
    );
    assert.ok(twice.role === 'toolResult');
    assert.deepEqual(twice.content[0], { type: 'text', text: `${long}\n${cut.preview}` });
    // The line of a preview without the start and the end of the text around it is not one.
    const line = cut.preview.slice(
      cut.preview.indexOf('\n\n['),
      cut.preview.lastIndexOf('\n\n') + 2
    );
    const quoted = { type: 'text' as const, text: `The file said:${line}and no more.` };
    const mention = toolResult(withStamp([quoted, preview], 7), { offload: cut.details });
    const told = await withFullOutput(mention);
    assert.ok(told.role === 'toolResult');
    assert.deepEqual(told.content.slice(0, 2), [quoted, { type: 'text', text: long }]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
