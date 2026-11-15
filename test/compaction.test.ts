import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { SessionManager } from '@earendil-works/pi-coding-agent';
import {
  archiveDir,
  FOLD_APPLIED_TYPE,
  FOLD_DETAILS_KEY,
  FOLD_UNFOLDED_TYPE,
  FORK_TOOL,
  type FoldDetails,
  type ForkArchive,
  type ForkDetails,
  ThreadIndex
} from '../src/archive.js';
import {
  compactedRecords,
  compactedThreads,
  compactionNotice,
  renderCompactionNotice
} from '../src/compaction.js';
import { assistant, call, result, session, user } from './helpers.js';

function foldRecord(handle: string, headline: string): FoldDetails {
  return {
    kind: 'fold',
    handle,
    since: 't1',
    firstEntryId: 'a',
    lastEntryId: 'b',
    messages: 2,
    tokens: 500,
    headline,
    keptUserTurns: [],
    files: []
  };
}

function forkRecord(
  handle: string,
  headline: string,
  task = 'Inspect build logs',
  extra: Partial<ForkDetails> = {}
): ForkDetails {
  return {
    kind: 'fork',
    handle,
    task,
    parentEntryId: 'p',
    turns: 3,
    headline,
    files: [],
    ...extra
  };
}

const mockTheme = {
  fg: (color: string, value: string) => (value ? `<${color}>${value}</${color}>` : '')
};

test('with no earlier folds or forks, the compaction notice is unchanged', () => {
  const plain =
    'The turns the summary above replaced are archived as cp-1a2b3c4d. recall searches them for exact text the summary left out, such as errors, paths and command output; they are too large for ask.';
  assert.equal(compactionNotice('cp-1a2b3c4d'), plain);
  assert.equal(compactionNotice('cp-1a2b3c4d', []), plain);
});

test('lists compacted folds and forks newest first and enforces the character budget including trailer', () => {
  const text = compactionNotice('cp-1a2b3c4d', [
    foldRecord('ep-11111111', 'First episode headline'),
    { ...foldRecord('ep-nested01', 'Inner episode headline'), inside: 'fk-22222222' },
    forkRecord('fk-22222222', '', '  # Fallback task line\nMore task details'),
    foldRecord('ep-33333333', 'Newest episode headline')
  ]);
  const lines = text.split('\n');
  assert.match(lines[2]!, /recall searches them.*ask\("<handle>", "<question>"\)/);
  assert.equal(lines[3], '- ep-33333333: Newest episode headline');
  assert.equal(lines[4], '- fk-22222222: Fallback task line');
  assert.equal(lines[5], '- ep-nested01 (inside fk-22222222): Inner episode headline');
  assert.equal(lines[6], '- ep-11111111: First episode headline');

  // Whole-item cut reserves trailer space so the manifest stays strictly <= maxTokens * 4.
  const tight = compactionNotice(
    'cp-1a2b3c4d',
    [
      foldRecord('ep-11111111', 'x'.repeat(110)),
      foldRecord('ep-22222222', 'y'.repeat(110)),
      foldRecord('ep-33333333', 'z'.repeat(110))
    ],
    95
  );
  const tightManifest = tight.split('\n\n')[1]!;
  assert.ok(tightManifest.length <= 95 * 4, `manifest length ${tightManifest.length} > 380`);
  assert.ok(tightManifest.includes('- ep-33333333:'));
  assert.ok(!tightManifest.includes('- ep-22222222:'));
  assert.match(tightManifest, /\(2 older thread\(s\); recall searches them\)$/);

  // Single-item partial truncation (lines.length === 0) also reserves trailer space and fits budget.
  const partial = compactionNotice(
    'cp-1a2b3c4d',
    [foldRecord('ep-11111111', 'x'.repeat(110)), foldRecord('ep-22222222', 'z'.repeat(110))],
    60
  );
  const partialManifest = partial.split('\n\n')[1]!;
  assert.ok(partialManifest.length <= 60 * 4, `manifest length ${partialManifest.length} > 240`);
  assert.match(partialManifest, /- ep-22222222: z+\.\.\./);
  assert.match(partialManifest, /\(1 older thread\(s\); recall searches them\)$/);

  // Pathologically tiny budget preserves the handle cleanly without trailing ':...'.
  const tiny = compactionNotice('cp-1a2b3c4d', [foldRecord('ep-22222222', 'z'.repeat(110))], 5);
  assert.ok(tiny.includes('- ep-22222222...'), tiny);
  const bareTiny = compactionNotice('cp-1a2b3c4d', [foldRecord('ep-22222222', '')], 5);
  assert.ok(bareTiny.includes('- ep-22222222') && !bareTiny.includes('...'), bareTiny);
});

test('compactedRecords collects applied folds, codemode folds, and forks across compactions', () => {
  const { manager } = session([user('start', 1), assistant(call('c1', 'fold'))]);

  const appendFold = (id: string, handle: string, headline: string) =>
    manager.appendMessage({
      ...result(id, 'fold', 'note', 2),
      details: { ...foldRecord(handle, headline) }
    } as unknown as Parameters<typeof manager.appendMessage>[0]);

  const appendCodemodeFold = (id: string, handle: string, headline: string) =>
    manager.appendMessage({
      ...result(id, 'codemode', 'script done', 2),
      details: { [FOLD_DETAILS_KEY]: foldRecord(handle, headline) }
    } as unknown as Parameters<typeof manager.appendMessage>[0]);

  const appendFork = (id: string, handle: string, headline: string) =>
    manager.appendMessage({
      ...result(id, FORK_TOOL, 'fork done', 3),
      details: { ...forkRecord(handle, headline) }
    } as unknown as Parameters<typeof manager.appendMessage>[0]);

  appendFold('c1', 'ep-early', 'Early fold');
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-early' });
  appendCodemodeFold('c-cm', 'ep-codemode', 'Scripted fold');
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-codemode' });
  appendFold('c2', 'ep-unfolded', 'Unfolded later');
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-unfolded' });
  appendFork('c3', 'fk-early', 'Early fork');

  // First compaction keeps nothing before it; missing firstKeptEntryId falls back to full prefix.
  const cp1Id = manager.appendCompaction('first summary', 'missing-entry-id', 10_000);
  const branch1 = manager.getBranch();
  const cp1 = branch1.find((e) => e.id === cp1Id && e.type === 'compaction')!;
  assert.equal(cp1.type, 'compaction');
  assert.deepEqual(
    compactedRecords(branch1, cp1).map((r) => r.handle),
    ['ep-early', 'ep-codemode', 'ep-unfolded', 'fk-early']
  );

  // Post-cp1 work: ep-split-marker has its result before keptStart and its applied marker inside
  // the kept window; ep-unfolded gets unfolded inside the kept window; ep-kept and fk-kept stay.
  appendFold('c4', 'ep-split-marker', 'Applied marker after cut');
  const keptStartId = manager.appendMessage(user('kept prompt', 4));
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-split-marker' });
  manager.appendCustomEntry(FOLD_UNFOLDED_TYPE, { handle: 'ep-unfolded' });
  appendFold('c5', 'ep-kept', 'Still in context');
  manager.appendCustomEntry(FOLD_APPLIED_TYPE, { handle: 'ep-kept' });
  appendFork('c6', 'fk-kept', 'Fork still in context');

  const cp2Id = manager.appendCompaction('second summary', keptStartId, 20_000);
  const branch2 = manager.getBranch();
  const cp2 = branch2.find((e) => e.id === cp2Id && e.type === 'compaction')!;
  assert.equal(cp2.type, 'compaction');
  assert.deepEqual(
    compactedRecords(branch2, cp2).map((r) => r.handle),
    ['ep-early', 'ep-codemode', 'fk-early', 'ep-split-marker']
  );
});

test('compactedThreads and ThreadIndex resolve memory, disk, and relocated fork archives identically', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'pi-episodes-compaction-'));
  try {
    const diskArchivePath = join(dir, 'fk-disk0001.json');
    const diskArchive: ForkArchive = {
      handle: 'fk-disk0001',
      task: 'Fix lint',
      parentEntryId: 'p',
      messages: [user('t', 1), user('t2', 2), user('t3', 3)],
      folds: [
        {
          record: foldRecord('ep-disk-unfolded', 'Temporary fold later unfolded'),
          first: 0,
          last: 1,
          omit: [0],
          unfolded: true,
          unfoldedAt: 2
        },
        {
          record: foldRecord('ep-disk-active', 'Lint rules: fixed unused import'),
          first: 1,
          last: 2,
          omit: [1]
        }
      ]
    };
    await writeFile(diskArchivePath, JSON.stringify(diskArchive), 'utf8');

    const memArchives = new Map<string, ForkArchive>([
      [
        'fk-mem00001',
        {
          handle: 'fk-mem00001',
          task: 'Debug test',
          parentEntryId: 'p',
          messages: [user('m1', 1), user('m2', 2)],
          folds: [
            {
              record: foldRecord('ep-mem-active', 'Test setup: isolated fixture'),
              first: 0,
              last: 1,
              omit: [0]
            }
          ]
        }
      ]
    ]);

    const manager = SessionManager.create(dir, join(dir, 'sessions'));
    manager.appendMessage(user('start', 1));
    const epDir = archiveDir(manager)!;
    assert.ok(epDir);
    await mkdir(epDir, { recursive: true });
    // Relocated fork archive: ForkDetails.archive points to a stale path, while .episodes/ next to
    // the session file holds fk-reloc001.json.
    const relocArchive: ForkArchive = {
      handle: 'fk-reloc001',
      task: 'Relocated session task',
      parentEntryId: 'p',
      messages: [user('r1', 1), user('r2', 2)],
      folds: [
        {
          record: foldRecord('ep-reloc-active', 'Relocated fold: preserved across move'),
          first: 0,
          last: 1,
          omit: [0]
        }
      ]
    };
    await writeFile(join(epDir, 'fk-reloc001.json'), JSON.stringify(relocArchive), 'utf8');

    const appendFork = (id: string, details: ForkDetails) =>
      manager.appendMessage({
        ...result(id, FORK_TOOL, 'fork done', 2),
        details
      } as unknown as Parameters<typeof manager.appendMessage>[0]);

    appendFork('f1', forkRecord('fk-mem00001', 'Tests passing', 'Debug test', { folds: 1 }));
    appendFork(
      'f2',
      forkRecord('fk-disk0001', 'Lint clean', 'Fix lint', { archive: diskArchivePath, folds: 1 })
    );
    appendFork(
      'f3',
      forkRecord('fk-reloc001', 'Relocated clean', 'Relocated session task', {
        archive: join(dir, 'stale-moved-dir', 'fk-reloc001.json'),
        folds: 1
      })
    );
    appendFork(
      'f4',
      forkRecord('fk-missing1', 'Deleted sidecar', 'Run check', {
        archive: join(dir, 'nope.json'),
        folds: 2
      })
    );
    const cpId = manager.appendCompaction('summary', null, 5000);
    const branch = manager.getBranch();
    const cp = branch.find((e) => e.id === cpId && e.type === 'compaction')!;
    assert.equal(cp.type, 'compaction');

    const threads = await compactedThreads(branch, cp, memArchives, manager);
    assert.deepEqual(
      threads.map((t) => (t.kind === 'fold' && t.inside ? `${t.handle}@${t.inside}` : t.handle)),
      [
        'ep-mem-active@fk-mem00001',
        'fk-mem00001',
        'ep-disk-active@fk-disk0001',
        'fk-disk0001',
        'ep-reloc-active@fk-reloc001',
        'fk-reloc001'
      ]
    );

    // ThreadIndex resolves the exact same handles (including relocated archives) and omits missing sidecars.
    const index = new ThreadIndex(manager, memArchives);
    assert.ok(await index.find('fk-reloc001'));
    assert.ok(await index.find('ep-reloc-active'));
    assert.equal(await index.find('fk-missing1'), undefined);
    assert.equal(await index.find('ep-disk-unfolded'), undefined);

    const notice = compactionNotice('cp-1a2b3c4d', threads);
    assert.ok(!notice.includes('fk-missing1'), notice);
    assert.ok(notice.includes('- fk-reloc001: Relocated clean'), notice);
    assert.ok(
      notice.includes(
        '- ep-reloc-active (inside fk-reloc001): Relocated fold: preserved across move'
      ),
      notice
    );
    assert.ok(notice.includes('- fk-disk0001: Lint clean'), notice);
    assert.ok(
      notice.includes('- ep-disk-active (inside fk-disk0001): Lint rules: fixed unused import'),
      notice
    );
    assert.ok(!notice.includes('ep-disk-unfolded'), notice);
    assert.ok(notice.includes('- fk-mem00001: Tests passing'), notice);
    assert.ok(
      notice.includes('- ep-mem-active (inside fk-mem00001): Test setup: isolated fixture'),
      notice
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('renderCompactionNotice respects expanded for listed threads and keeps empty notices compact', () => {
  const plain = compactionNotice('cp-1a2b3c4d');
  assert.equal(
    renderCompactionNotice(plain, false, mockTheme).render(200).join('\n').trim(),
    '<dim>episodes: compacted turns archived as cp-1a2b3c4d, recall searches them</dim>'
  );
  assert.equal(
    renderCompactionNotice(plain, true, mockTheme).render(200).join('\n').trim(),
    '<dim>episodes: compacted turns archived as cp-1a2b3c4d, recall searches them</dim>'
  );

  const listed = compactionNotice(
    'cp-1a2b3c4d',
    [
      foldRecord('ep-11111111', 'x'.repeat(110)),
      foldRecord('ep-22222222', 'y'.repeat(110)),
      foldRecord('ep-33333333', 'Newest fold')
    ],
    95
  );
  const collapsed = renderCompactionNotice(listed, false, mockTheme).render(200).join('\n').trim();
  assert.equal(
    collapsed,
    '<dim>episodes: compacted turns archived as cp-1a2b3c4d, 2 earlier thread(s) listed, ctrl+o to show</dim>'
  );
  const expanded = renderCompactionNotice(listed, true, mockTheme).render(200).join('\n').trim();
  assert.ok(
    expanded.includes(
      '<dim>episodes: compacted turns archived as cp-1a2b3c4d, 2 earlier thread(s) listed</dim>'
    ),
    expanded
  );
  assert.ok(expanded.includes('- ep-33333333: Newest fold'), expanded);
  assert.ok(expanded.includes('(1 older thread(s); recall searches them)'), expanded);
});
