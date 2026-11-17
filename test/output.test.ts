import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { test } from 'node:test';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import {
  CAPPED_KEY,
  capOutput,
  cappedOutput,
  verifyCompleteRead,
  withFullOutput
} from '../src/output.js';
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
    const marker = block.text.indexOf('\n\n[episodes] ');
    const omitted = Number(/\n\n\[episodes\] (\d+) of /.exec(block.text)?.[1]);
    // It says first that it is a part, how much of the text, and where all of it is.
    const line = `[episodes] Only part of this output is shown: ${long.length - omitted} of ${long.length} characters, its start and its end, with the middle left out at the place marked below. If what you need is not in the part shown, do not answer without it, and do not conclude that it is not there: grep ${cut.capped.path}, which holds all of it, or read a slice of that file.\n\n`;
    assert.ok(block.text.startsWith(`${line}line 0 of the build log\n`));
    assert.match(block.text, /\nline 399 of the build log$/);
    assert.doesNotMatch(block.text, /line 200 of/);
    assert.ok(
      block.text.includes(`characters left out here. All of this output is in ${cut.capped.path}`)
    );
    // Only whole lines are shown.
    assert.ok(block.text.split('\n').every((line) => !/^line \d+ of the build$/.test(line)));
    assert.ok(block.text.length < 700 + line.length);
    assert.deepEqual(cut.capped, {
      path: cut.capped.path,
      chars: long.length,
      block: 0,
      preview: block.text.length,
      head: marker - line.length,
      top: true
    });
    assert.equal(await readFile(cut.capped.path, 'utf8'), long);
    assert.deepEqual(await readdir(dir), [basename(cut.capped.path)]);

    // A preview that would not be shorter than the text is not made, and no file is written.
    const short = long.slice(0, 1050);
    await rm(cut.capped.path);
    assert.equal(await capOutput([{ type: 'text', text: short }], dir, 1000, 500), undefined);
    assert.deepEqual(await readdir(dir), []);
    assert.ok(await capOutput([{ type: 'text', text: long.slice(0, 1400) }], dir, 1000, 500));
    // Nor one as long as the text: with no line break to cut at, some length is the preview's.
    const sizes: boolean[] = [];
    for (let length = 1000; length < 1400; length++) {
      const made = await capOutput([{ type: 'text', text: 'x'.repeat(length) }], dir, 1000, 500);
      const block = made?.content[0];
      assert.ok(!block || (block.type === 'text' && block.text.length < length), `${length}`);
      sizes.push(made !== undefined);
    }
    assert.deepEqual([sizes[0], sizes.at(-1)], [false, true]);
    assert.equal(sizes.lastIndexOf(false) + 1, sizes.indexOf(true));
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
      ['Script', 'image', '[episo', 'image']
    );
    assert.equal(cut.capped.block, 2);
    assert.equal(await readFile(cut.capped.path, 'utf8'), long);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

const scriptHeader = {
  type: 'text' as const,
  text: 'Script completed\nWall time 0.1 seconds\nOutput:\n'
};
const events = Array.from(
  { length: 60 },
  (_, index) => `event=0${200 + index} latency_ms=${100 + index} trace=${'ab'.repeat(32)}`
).join('\n');
const source = Array.from(
  { length: 200 },
  (_, index) => `  const value${index} = compute(${index});`
).join('\n');
// What a script that printed three things returns: the lines asked for with the counts after
// them, one short line, and a source file.
const printed = [
  `==> text 1/3 <==\n${events}\n\n--COUNTS--\nok 342 retry 58`,
  '==> text 2/3 <==\nTOP event=0249 329',
  `==> text 3/3 <==\n${source}`
].join('\n');

test('the output of a script is cut thing by thing it printed, under a line that says so', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await capOutput([scriptHeader, { type: 'text', text: printed }], dir, 4000, 1800);
    assert.ok(cut);
    assert.deepEqual(cut.content[0], scriptHeader);
    const block = cut.content[1]!;
    assert.ok(block.type === 'text' && cut.content.length === 2);
    const recorded = { [CAPPED_KEY]: { ...cut.capped } };
    const kept = cut.capped.shown!.reduce((sum, [from, to]) => sum + to - from, 0);
    // Said first, where a reader starts: that this is a part, and where all of it is.
    assert.ok(
      block.text.startsWith(
        `[episodes] Only part of this output is shown: ${kept} of ${printed.length} characters, with the rest left out at the 2 places marked below. If what you need is not in the part shown, do not answer without it: print less of it (filter in the script), or grep or read a slice of ${cut.capped.path}, which holds all of it. recall searches it once it is folded.\n\n==> text 1/3 <==\nevent=0200 `
      ),
      block.text.slice(0, 500)
    );
    // The end of the first thing, the short one whole, and the start and end of the third.
    assert.ok(
      block.text.includes(
        '\n\n--COUNTS--\nok 342 retry 58\n==> text 2/3 <==\nTOP event=0249 329\n==> text 3/3 <==\n  const value0 = compute(0);\n'
      )
    );
    assert.ok(block.text.endsWith('\n  const value199 = compute(199);'));
    assert.doesNotMatch(block.text, /event=0230 |value100 /);
    const gaps = [...block.text.matchAll(/\n\n\[episodes\] (\d+) characters left out here\.\n\n/g)];
    assert.equal(gaps.length, 2);
    assert.equal(
      gaps.reduce((sum, gap) => sum + Number(gap[1]), 0),
      printed.length - kept
    );
    assert.ok(kept <= 1800 && kept > 1500, `${kept}`);
    // The two long things share what the short one leaves, and each stretch is whole lines.
    const [start, middle, end] = cut.capped.shown!;
    assert.ok(start![1] > 450 && end![1] - end![0] > 200, JSON.stringify(cut.capped.shown));
    assert.deepEqual(
      [printed[start![1]], printed[middle![0] - 1], printed[middle![1]], printed[end![0] - 1]],
      ['\n', '\n', '\n', '\n']
    );
    assert.deepEqual(cut.capped, {
      path: cut.capped.path,
      chars: printed.length,
      block: 1,
      preview: block.text.length,
      shown: cut.capped.shown
    });
    assert.equal(cut.capped.shown!.length, 3);
    assert.equal(await readFile(cut.capped.path, 'utf8'), printed);

    const full = await withFullOutput(toolResult(withStamp(cut.content, 7), recorded));
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content.slice(0, 2), [scriptHeader, { type: 'text', text: printed }]);
    assert.equal(readStamp(full), 7);

    // Printed as blocks of their own, the things come back as those blocks.
    const blocks = printed.split(/\n(?===> text)/).map((text) => ({ type: 'text' as const, text }));
    const several = await capOutput([scriptHeader, ...blocks], dir, 4000, 1800);
    assert.ok(several && several.content.length === 2);
    assert.deepEqual(several.content[1], block);
    assert.deepEqual(
      several.capped.parts,
      blocks.map((part) => part.text.length)
    );
    const back = await withFullOutput(
      toolResult(several.content, { [CAPPED_KEY]: { ...several.capped } })
    );
    assert.ok(back.role === 'toolResult');
    assert.deepEqual(back.content, [scriptHeader, ...blocks]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a script that printed one thing, or more things than can each keep a share, has its output cut as one', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const many = Array.from(
      { length: 30 },
      (_, index) => `==> text ${index + 1}/30 <==\n${`row ${index} `.repeat(60)}`
    ).join('\n');
    for (const text of [long, many]) {
      const cut = await capOutput([scriptHeader, { type: 'text', text }], dir, 4000, 1800);
      assert.ok(cut?.content[1]?.type === 'text');
      const preview = cut.content[1].text;
      assert.match(
        preview,
        /^\[episodes\] Only part of this output is shown: \d+ of \d+ characters, with the rest left out at the place marked below\. /
      );
      assert.equal(preview.split(' characters left out here.').length - 1, 1);
      assert.deepEqual(
        cut.capped.shown!.map(([from, to]) => [from === 0, to === text.length]),
        [
          [true, false],
          [false, true]
        ]
      );
      const full = await withFullOutput(
        toolResult(cut.content, { [CAPPED_KEY]: { ...cut.capped } })
      );
      assert.ok(full.role === 'toolResult');
      assert.deepEqual(full.content[1], { type: 'text', text });
    }
    // A line that says the output is cut, longer than what it would leave out, is not worth it.
    assert.equal(
      await capOutput([scriptHeader, { type: 'text', text: 'x'.repeat(700) }], dir, 600, 300),
      undefined
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('only the lines codemode numbers in step start a thing printed, and what stands before the first is one too', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const places = (preview: string) => preview.split(' characters left out here.').length - 1;
    const roundTrip = async (text: string, limit: number, keep: number) => {
      const cut = await capOutput([scriptHeader, { type: 'text', text }], dir, limit, keep);
      assert.ok(cut?.content[1]?.type === 'text');
      const full = await withFullOutput(
        toolResult(cut.content, { [CAPPED_KEY]: { ...cut.capped } })
      );
      assert.ok(full.role === 'toolResult');
      assert.deepEqual(full.content[1], { type: 'text', text });
      return cut.content[1].text;
    };

    // A log that holds lines like codemode's own, out of step with the three things printed.
    // One of them has the next number and another count, in the middle of the first thing.
    const fake = Array.from(
      { length: 10 },
      (_, index) =>
        `==> text ${index === 5 ? '2/4' : '99/999'} <==\n${`noise ${index} `.repeat(120)}`
    ).join('\n');
    const noisy = [
      `==> text 1/3 <==\n${fake}\n==> text 1/3 <==\n==> text 3/3 <==\nend of the log`,
      '==> text 2/3 <==\nCOUNT 342',
      `==> text 3/3 <==\n${source}`
    ].join('\n');
    const shown = await roundTrip(noisy, 16_000, 6000);
    assert.ok(shown.includes('\nend of the log\n==> text 2/3 <==\nCOUNT 342\n==> text 3/3 <==\n'));
    assert.equal(places(shown), 2);
    // The same with the right count on every such line and the wrong number: as things printed
    // they would be too many to keep a share each, and the count would go with the cut as one.
    const counted = await roundTrip(noisy.replaceAll(/\b(?:99\/999|2\/4)\b/g, '3/3'), 16_000, 6000);
    assert.ok(
      counted.includes('\nend of the log\n==> text 2/3 <==\nCOUNT 342\n==> text 3/3 <==\n')
    );
    assert.equal(places(counted), 2);

    // Codemode cut the output itself and says so before the first thing.
    const warned = `Warning: truncated output (original token count: 15455)\nTotal output lines: 4\n\n${printed}`;
    const after = await roundTrip(warned, 4000, 1800);
    assert.match(
      after,
      /marked below\. [^\n]*\n\nWarning: truncated output \(original token count: 15455\)\nTotal output lines: 4\n\n==> text 1\/3 <==\nevent=0200 /
    );
    assert.equal(places(after), 2);

    // A thing that is cut keeps 600 characters of its own, or the output is cut as one.
    const halves = [`==> text 1/2 <==\n${events}`, `==> text 2/2 <==\n${source}`].join('\n');
    assert.equal(places(await roundTrip(halves, 4000, 1200)), 2);
    assert.equal(places(await roundTrip(halves, 4000, 1198)), 1);

    // A cut inside a character: the preview is well formed, and found again all the same.
    const faces = `==> text 1/2 <==\n${'\u{1F600}'.repeat(2001)}\n==> text 2/2 <==\n${'\u{1F680}'.repeat(2000)}`;
    for (const keep of [1800, 1801, 1802, 1803]) {
      const preview = await roundTrip(faces, 4000, keep);
      assert.ok(preview.isWellFormed(), `${keep}`);
      assert.equal(places(preview), 2);
    }

    // With next to nothing to keep, what is left out before the first or after the last
    // stretch has its line too, and the count is of the lines there are.
    for (const keep of [1, 2, 3, 40]) {
      const preview = await roundTrip(`${long}\n`, 4000, keep);
      const said = /left out at the (?:(\d+) places|place) marked below/.exec(preview)!;
      assert.equal(Number(said[1] ?? 1), places(preview), `${keep}: ${preview.slice(0, 120)}`);
      assert.ok(places(preview) >= 1);
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('the preview of a script output is put back only where it stands as it was written, once', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const cut = await capOutput([scriptHeader, { type: 'text', text: printed }], dir, 4000, 1800);
    assert.ok(cut?.content[1]?.type === 'text');
    const preview = cut.content[1].text;
    const record = { ...cut.capped };
    const back = async (content: ToolResult['content'], capped: object = record) => {
      const message = toolResult(content, { [CAPPED_KEY]: capped as never });
      const full = await withFullOutput(message);
      return full === message || full.role !== 'toolResult' ? undefined : full.content;
    };

    // Inside a block that holds more, as after another cut that joined blocks was undone.
    assert.deepEqual(
      await back([scriptHeader, { type: 'text', text: `note\n${preview}\nWARNING` }]),
      [scriptHeader, { type: 'text', text: `note\n${printed}\nWARNING` }]
    );
    // In another block than the one on record, when it is the only one.
    assert.deepEqual(await back([{ type: 'text', text: preview }]), [
      { type: 'text', text: printed }
    ]);
    // A block that names the file and is not the preview; a preview that was changed.
    assert.equal(
      await back([scriptHeader, { type: 'text', text: `see ${cut.capped.path}` }]),
      undefined
    );
    assert.equal(
      await back([scriptHeader, { type: 'text', text: preview.replace('ok 342', 'ok 343') }]),
      undefined
    );
    assert.equal(
      await back([
        scriptHeader,
        {
          type: 'text',
          text: preview.replace(/\d+ characters left out here/, '1 characters left out here')
        }
      ]),
      undefined
    );
    // Twice in one block, or in two blocks neither of which is the one on record.
    assert.equal(
      await back([scriptHeader, { type: 'text', text: `${preview}\n${preview}` }]),
      undefined
    );
    assert.equal(
      await back([
        { type: 'text', text: preview },
        { type: 'text', text: 'between' },
        { type: 'text', text: preview }
      ]),
      undefined
    );
    // A record whose stretches are not those of a preview.
    const [first, second, third] = cut.capped.shown!;
    for (const shown of [
      [first, third, second],
      [first],
      [first, [second![0], third![1] + 1]],
      [[1, first![1]], second, third],
      [first, [first![1], second![1]], third],
      [first, [second![1], second![0]], third]
    ]) {
      assert.equal(
        await back([scriptHeader, { type: 'text', text: preview }], { ...record, shown }),
        undefined,
        JSON.stringify(shown)
      );
    }
    for (const shown of [[[0, 'x']], [[0, 1, 2]], [0, 1], 'all', [[0, 1.5]]]) {
      assert.equal(
        cappedOutput({ [CAPPED_KEY]: { ...record, shown } }),
        undefined,
        JSON.stringify(shown)
      );
    }
    assert.ok(cappedOutput({ [CAPPED_KEY]: record }));

    // The other extension cut the preview: its cut is undone first, then this one.
    const outer = await offloaded(dir, preview, 600);
    assert.ok(!outer.preview.includes('left out here'));
    const both = toolResult(withStamp([scriptHeader, { type: 'text', text: outer.preview }], 7), {
      [CAPPED_KEY]: { ...record },
      offload: outer.details
    });
    const whole = await withFullOutput(both);
    assert.ok(whole.role === 'toolResult');
    assert.deepEqual(whole.content.slice(0, 2), [scriptHeader, { type: 'text', text: printed }]);
    // The other extension first: this one cut its preview.
    const inner = await offloaded(dir, printed, 6000);
    const after = await capOutput(
      [scriptHeader, { type: 'text', text: inner.preview }],
      dir,
      4000,
      1800
    );
    assert.ok(after);
    const reversed = await withFullOutput(
      toolResult(after.content, { offload: inner.details, [CAPPED_KEY]: { ...after.capped } })
    );
    assert.ok(reversed.role === 'toolResult');
    assert.deepEqual(reversed.content, [scriptHeader, { type: 'text', text: printed }]);

    // The file gone: nothing to put back.
    await rm(cut.capped.path);
    assert.equal(await back([scriptHeader, { type: 'text', text: preview }]), undefined);
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
    // A record is read without the numbers added later, and not with something else there.
    const { path, chars, block } = cut.capped;
    assert.deepEqual(cappedOutput({ [CAPPED_KEY]: { path, chars, block } }), {
      path,
      chars,
      block
    });
    for (const wrong of [
      { preview: {} },
      { head: '3' },
      { top: false },
      { top: 1 },
      { parts: 3 },
      { parts: ['3'] }
    ]) {
      assert.equal(cappedOutput({ [CAPPED_KEY]: { ...cut.capped, ...wrong } }), undefined);
    }

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

    // The line that opens the preview goes with it only where it stands as it was written:
    // a handler's words before it in the same block stay, and a preview whose line was
    // changed, or is missing, is not taken for the one on record.
    const preview = cut.content[0];
    assert.ok(preview?.type === 'text' && cut.capped.head !== undefined);
    const line = preview.text.slice(0, preview.text.indexOf('\n\n') + 2);
    const shown = (text: string, capped = cut.capped) => ({
      ...message,
      content: withStamp([{ type: 'text', text }], 7),
      details: { exitCode: 0, [CAPPED_KEY]: { ...capped } }
    });
    const led = await withFullOutput(shown(`From staging.\n${preview.text}`));
    assert.ok(led.role === 'toolResult');
    assert.deepEqual(led.content[0], { type: 'text', text: `From staging.\n${long}` });
    for (const other of [
      preview.text.replace('do not answer without it', 'answer without it'),
      preview.text.replace('do not answer', 'do now answer'),
      // The line of another file, or of a text of another length, is not this preview's.
      preview.text.replace('/output-', '/outpux-'),
      preview.text.replace(
        ` of ${long.length} characters, its`,
        ` of ${long.length + 1} characters, its`
      ),
      `From staging.\n${preview.text.replace('grep', 'find')}`,
      preview.text.replace(/shown: \d+ of/, 'shown: 1 of'),
      preview.text.slice(line.length),
      preview.text.slice(1)
    ]) {
      const changed = shown(other);
      assert.equal(await withFullOutput(changed), changed);
      assert.equal(verifyCompleteRead([{ type: 'text', text: other }], changed.details), undefined);
    }
    // The line in place of the middle can stand in a block twice: the preview is where the
    // opening line stands before the start, not at the first one.
    const unled = preview.text.slice(line.length);
    const second = await withFullOutput(shown(`${unled}\n${preview.text}`));
    assert.ok(second.role === 'toolResult');
    assert.deepEqual(second.content[0], { type: 'text', text: `${unled}\n${long}` });
    // A preview made before that line was written has none, and none on record.
    const { top: _, ...before } = cut.capped;
    const old = await withFullOutput(shown(preview.text.slice(line.length), before));
    assert.ok(old.role === 'toolResult');
    assert.deepEqual(old.content[0], { type: 'text', text: long });
    assert.deepEqual(
      verifyCompleteRead([{ type: 'text', text: preview.text.slice(line.length) }], old.details)
        ?.blocks,
      [{ type: 'text', text: long }]
    );
    // With a line that the record does not have, the line is not part of the preview.
    const kept = await withFullOutput(shown(preview.text, before));
    assert.ok(kept.role === 'toolResult');
    assert.deepEqual(kept.content[0], { type: 'text', text: `${line}${long}` });
    assert.equal(verifyCompleteRead(cut.content, old.details), undefined);
    // A file that was read is known whole again from its preview, with what the preview kept.
    assert.deepEqual(verifyCompleteRead(cut.content, message.details), {
      blocks: [{ type: 'text', text: long }],
      coverage: {
        uncapped: false,
        block: 0,
        head: cut.capped.head,
        tail: preview.text.length - preview.text.lastIndexOf('\n\n') - 2
      }
    });

    // A preview that kept nothing of the start or the end is the two lines, and is found.
    const bare = await capOutput([{ type: 'text', text: long }], dir, 1000, 1);
    assert.ok(bare?.content[0]?.type === 'text');
    assert.equal(bare.capped.head, 0);
    assert.match(
      bare.content[0].text,
      /^\[episodes\] Only part [^\n]*\n\n\n\n\[episodes\] [^\n]*\n\n$/
    );
    const none = shown(bare.content[0].text, bare.capped);
    const all = await withFullOutput(none);
    assert.ok(all.role === 'toolResult');
    assert.deepEqual(all.content[0], { type: 'text', text: long });
    assert.deepEqual(verifyCompleteRead(bare.content, none.details)?.coverage, {
      uncapped: false,
      block: 0,
      head: 0,
      tail: 0
    });

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
    assert.doesNotMatch(outer.preview, /characters left out here/);
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
    assert.doesNotMatch(outer.preview, /characters left out here/);
    const message = toolResult(withStamp([{ type: 'text', text: outer.preview }], 7), {
      [CAPPED_KEY]: { ...cut.capped },
      offload: outer.details
    });
    const full = await withFullOutput(message);
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content[0], { type: 'text', text: `${long}\n${warning}` });
    assert.equal(readStamp(full), 7);

    // The other way round: the other extension cut first, and this one cut both blocks. Its
    // record has their lengths, so they come back as the blocks they were.
    const first = await offloaded(dir, long, 3000);
    const added = [
      { type: 'text' as const, text: first.preview },
      { type: 'text' as const, text: warning }
    ];
    const second = await capOutput(added, dir, 1000, 300);
    assert.ok(second);
    assert.deepEqual(second.capped.parts, [first.preview.length, warning.length]);
    assert.doesNotMatch(JSON.stringify(second.content), /tool-result-offload/);
    const reversed = toolResult(withStamp(second.content, 7), {
      offload: first.details,
      [CAPPED_KEY]: { ...second.capped }
    });
    const whole = await withFullOutput(reversed);
    assert.ok(whole.role === 'toolResult');
    assert.deepEqual(whole.content.slice(0, -1), [{ type: 'text', text: long }, added[1]]);
    assert.equal(readStamp(whole), 7);
    // Lengths that are not those of the text, or that do not meet at a line break, leave it
    // in one block, with the other extension's preview in it as the model saw it.
    const { parts: _parts, ...unsplit } = second.capped;
    for (const record of [
      unsplit,
      { ...unsplit, parts: [first.preview.length, 1] },
      { ...unsplit, parts: [first.preview.length - 1, warning.length + 1] }
    ]) {
      const one = await withFullOutput(toolResult(second.content, { [CAPPED_KEY]: record }));
      assert.ok(one.role === 'toolResult');
      assert.deepEqual(one.content, [{ type: 'text', text: `${first.preview}\n${warning}` }]);
    }

    // The later cut can leave the earlier preview whole and in sight, here before the start
    // of a long block a handler added: the later cut is still the one to undo first.
    const lines = Array.from({ length: 900 }, (_, index) => `added line ${index}`).join('\n');
    const small = await capOutput([{ type: 'text', text: long }], dir, 1000, 300);
    assert.ok(small?.content[0]?.type === 'text');
    const both = await offloaded(dir, `${small.content[0].text}\n${lines}`, 6000);
    assert.match(both.preview, /\[episodes\] [\s\S]*\[tool-result-offload: /);
    assert.doesNotMatch(both.preview, /added line 450\n/);
    const seen = await withFullOutput(
      toolResult([{ type: 'text', text: both.preview }], {
        [CAPPED_KEY]: { ...small.capped },
        offload: both.details
      })
    );
    assert.ok(seen.role === 'toolResult');
    assert.deepEqual(seen.content, [{ type: 'text', text: `${long}\n${lines}` }]);

    // This extension cut first, and blocks were added on both sides before the other cut all
    // three as one text: the record says where the preview starts, and without that it stays
    // as the model saw it.
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
    // The same row many times before the one that matters and another after it, and a
    // handler adds a block with the first row before this extension cuts both blocks: the
    // start and the end of the output would fit around the line one row further up too.
    const rows = (row: string) => Array.from({ length: 300 }, () => row).join('\n');
    const log = `${rows('PASS head')}\nFAIL the one that matters\n${rows('PASS tail')}`;
    const first = await offloaded(dir, log, 3000);
    const added = { type: 'text' as const, text: 'PASS head' };
    const second = await capOutput([added, { type: 'text', text: first.preview }], dir, 600, 300);
    assert.ok(second);
    const full = await withFullOutput(
      toolResult(second.content, { offload: first.details, [CAPPED_KEY]: { ...second.capped } })
    );
    assert.ok(full.role === 'toolResult');
    assert.deepEqual(full.content, [added, { type: 'text', text: log }]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a preview cut inside a character still gets its text back', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-output-'));
  try {
    const faces = '😀'.repeat(600);
    const cut = await capOutput([{ type: 'text', text: faces }], dir, 10, 5);
    assert.ok(cut?.content[0]?.type === 'text');
    // Half a character was kept at each end, and reads as the replacement character.
    assert.match(
      cut.content[0].text,
      /^\[episodes\] Only part [^\n]*\n\n😀\uFFFD\n\n\[episodes\] [^\n]*\n\n\uFFFD$/
    );
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
    // Nor is a preview that is not all of its block, when no record says where it stands.
    const joined = { type: 'text' as const, text: `${cut.preview}\n${cut.preview}` };
    const twice = toolResult(withStamp([joined], 7), { offload: cut.details });
    assert.equal(await withFullOutput(twice), twice);
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
