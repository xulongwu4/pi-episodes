import type {
  CompactionEntry,
  ExtensionAPI,
  SessionEntry,
  Theme
} from '@earendil-works/pi-coding-agent';
import { Text } from '@earendil-works/pi-tui';
import {
  activeNestedFoldRecords,
  appliedFolds,
  compactionHandle,
  type FoldDetails,
  type ForkArchive,
  type ForkDetails,
  readForkArchive,
  type SessionView,
  threadRecords
} from './archive.js';
import { headline } from './messages.js';
import { episodesSettings } from './settings.js';
import type { EpisodeState } from './state.js';

export const COMPACTION_NOTICE_TYPE = 'episodes-compacted';

const MAX_MANIFEST_TOKENS = 1000;

export type ManifestRecord = (FoldDetails & { inside?: string }) | ForkDetails;

/**
 * Folds and forks whose records were swept out of context by `compactionEntry`, oldest first.
 * Includes those from before earlier compactions, since their previous notice was compacted too.
 */
export function compactedRecords(
  branch: readonly SessionEntry[],
  compactionEntry: Pick<CompactionEntry, 'id' | 'firstKeptEntryId'>
): (FoldDetails | ForkDetails)[] {
  const compactionIdx = branch.findIndex((entry) => entry.id === compactionEntry.id);
  const before = compactionIdx < 0 ? branch : branch.slice(0, compactionIdx);
  const atCompaction = compactionIdx < 0 ? branch : branch.slice(0, compactionIdx + 1);
  const keptIdx = before.findIndex((entry) => entry.id === compactionEntry.firstKeptEntryId);
  const cut = keptIdx < 0 ? before.length : keptIdx;
  return threadRecords(before.slice(0, cut), appliedFolds(atCompaction)).filter(
    (record): record is FoldDetails | ForkDetails => record.kind !== 'compaction'
  );
}

/**
 * Compacted branch folds and forks with active nested folds inside each fork expanded right
 * before their parent fork, so newest-first rendering groups nested folds under their fork.
 */
export async function compactedThreads(
  branch: readonly SessionEntry[],
  compactionEntry: Pick<CompactionEntry, 'id' | 'firstKeptEntryId'>,
  forkArchives: ReadonlyMap<string, ForkArchive> = new Map(),
  sessions?: SessionView
): Promise<ManifestRecord[]> {
  const out: ManifestRecord[] = [];
  for (const record of compactedRecords(branch, compactionEntry)) {
    if (record.kind === 'fold') {
      out.push(record);
      continue;
    }
    const archive = await readForkArchive(record, forkArchives, sessions);
    if (!archive) continue;
    for (const nested of activeNestedFoldRecords(archive)) {
      out.push({ ...nested, inside: record.handle });
    }
    out.push(record);
  }
  return out;
}

function formatManifestLine(record: ManifestRecord): { prefix: string; line: string } {
  if (record.kind === 'fold') {
    const prefix = record.inside
      ? `- ${record.handle} (inside ${record.inside})`
      : `- ${record.handle}`;
    return { prefix, line: record.headline ? `${prefix}: ${record.headline}` : prefix };
  }
  const prefix = `- ${record.handle}`;
  const desc = record.headline || headline(record.task);
  return { prefix, line: desc ? `${prefix}: ${desc}` : prefix };
}

/**
 * Notice appended after a compaction. When `records` is non-empty, appends a newest-first
 * manifest whose own character length stays within `maxTokens * 4` (about four characters per
 * token, separate from the base `cp-` sentence above it, including any omitted-thread trailer;
 * budgets smaller than the header plus `- <handle>...` and trailer still preserve `- <handle>...`).
 */
export function compactionNotice(
  handle: string,
  records: readonly ManifestRecord[] = [],
  maxTokens = MAX_MANIFEST_TOKENS
): string {
  const base = `The turns the summary above replaced are archived as ${handle}. recall searches them for exact text the summary left out, such as errors, paths and command output; they are too large for ask.`;
  if (records.length === 0) return base;

  const header =
    'Earlier folds and forks archived before this compaction (recall searches them, ask("<handle>", "<question>") resumes one):';
  const budget = maxTokens * 4;
  const total = records.length;
  const trailerOf = (dropped: number) => `(${dropped} older thread(s); recall searches them)`;
  const lines: string[] = [];
  let used = header.length;
  for (const record of [...records].reverse()) {
    const { prefix, line } = formatManifestLine(record);
    const remainingAfter = total - (lines.length + 1);
    const reserve = remainingAfter > 0 ? 1 + trailerOf(remainingAfter).length : 0;
    if (used + 1 + line.length + reserve <= budget) {
      lines.push(line);
      used += 1 + line.length;
      continue;
    }
    if (lines.length === 0) {
      if (line === prefix) {
        lines.push(prefix);
      } else {
        const minCut = `${prefix}...`;
        const avail = budget - used - 1 - reserve;
        const trimmed =
          avail >= minCut.length ? line.slice(0, avail - 3).replace(/[:\s]+$/, '') : '';
        lines.push(`${trimmed || prefix}...`);
      }
    }
    break;
  }
  const dropped = total - lines.length;
  if (dropped > 0) lines.push(trailerOf(dropped));
  return `${base}\n\n${[header, ...lines].join('\n')}`;
}

export function renderCompactionNotice(
  content: unknown,
  expanded: boolean,
  theme: Pick<Theme, 'fg'>
): Text {
  const text = typeof content === 'string' ? content : '';
  const handle = /cp-[0-9a-z-]+/.exec(text)?.[0];
  const lines = text.split('\n');
  const count = lines.filter((line) => line.startsWith('- ')).length;
  if (count === 0) {
    return new Text(
      theme.fg(
        'dim',
        `episodes: compacted turns archived as ${handle ?? 'cp-?'}, recall searches them`
      ),
      0,
      0
    );
  }
  const items = lines.filter(
    (line) => line.startsWith('- ') || /^\(\d+ older thread\(s\);/.test(line)
  );
  const body = expanded ? `\n${items.join('\n')}` : '';
  return new Text(
    theme.fg(
      'dim',
      `episodes: compacted turns archived as ${handle ?? 'cp-?'}, ${count} earlier thread(s) listed${expanded ? '' : ', ctrl+o to show'}`
    ) + theme.fg('muted', body),
    0,
    0
  );
}

/**
 * A fold's note names its archive; pi's compaction summary is pi's own text and cannot. This
 * is how the agent learns the compacted turns and any earlier folds or forks can still be
 * reached. It is appended once after the summary, which just rewrote the whole prompt anyway,
 * so it costs no cache.
 */
export function registerCompactionNotice(pi: ExtensionAPI, state?: EpisodeState): void {
  pi.on('session_compact', async (event, ctx) => {
    if (!episodesSettings(pi.getSettings()).compactionNotice) return;
    const records = await compactedThreads(
      ctx.sessionManager.getBranch(),
      event.compactionEntry,
      state?.forkArchives,
      ctx.sessionManager
    );
    // Mid-run, steering puts it before the next request; when idle it is appended right away.
    pi.sendMessage(
      {
        customType: COMPACTION_NOTICE_TYPE,
        content: compactionNotice(compactionHandle(event.compactionEntry.id), records),
        display: true
      },
      { deliverAs: 'steer' }
    );
  });

  pi.registerMessageRenderer(COMPACTION_NOTICE_TYPE, (message, { expanded }, theme) =>
    renderCompactionNotice(message.content, expanded, theme)
  );
}
