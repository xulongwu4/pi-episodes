<p align="center">
   <b>Using this package?</b> Please consider <a href="https://github.com/sponsors/arthurfiorette" target="_blank">donating</a> to support my open source work ❤️
  <br />
  <sup>
   Help pi-episodes grow! Star and share this amazing repository with your friends and co-workers!
  </sup>
</p>

<br />

<p align="center" title="My take on subagents, forks and collapsing context">
  <a href="https://arthur.place/subagents-forks-collapsing-context">
    <img src="https://raw.githubusercontent.com/arthurfiorette/pi-episodes/main/assets/cover.jpg" width="640" alt="A red thread winds through a landscape of papers, maps, and filing drawers." />
  </a>
</p>

# Pi Episodes

**Fold finished work out of the context instead of compacting it.** A [pi](https://github.com/earendil-works/pi) package that lets the agent close an episode, such as a build loop, into a short note while the full span stays archived, searchable and askable.

> [!NOTE]
> This package implements the post **[My take on subagents, forks and collapsing context](https://arthur.place/subagents-forks-collapsing-context)**. The post makes the argument, the research behind it and the open questions; this README covers how the package behaves.

Compaction summarizes the whole session at once, when the window fills. `pi-episodes` asks the agent to summarize one piece of work when that work ends: "the build passes, where was I?". The span leaves the context, a note takes its place, and the agent can still search it or question it later. The same archive backs `fork`, which runs a copy of the agent with its current context instead of a subagent that starts from an empty prompt. Pi's own compaction stays as the fallback.

The terms, as in the post: an **episode** is one span of work around a local question. A **fold** replaces a finished episode with a short **note** plus a **handle** that points back to the archived span. A **fork** runs a copy of the agent with the parent's current context. **recall** searches the archive without a model call, and **ask** puts one question to an archived thread.

**Requirements:** Node.js 24.16+ and pi 1.1+. [The post](https://arthur.place/subagents-forks-collapsing-context) · [Get started](#get-started) · [How it works](#how-it-works) · [Tools](#tools) · [Prompt caching](#prompt-caching) · [Limitations](#limitations) · [Measurements](MEASUREMENTS.md)

<br />

## Get started

1. Install the package:

   ```sh
   pi install npm:pi-episodes
   ```

2. Start pi and work as usual. The agent sees four new tools and a short system prompt section that explains when to use them.

3. To try the note format before trusting the model to notice boundaries on its own, fold by hand after a finished loop. `/fold` lists the turns that can still be folded, newest first with a preview; you pick where the span starts and the agent writes the note. `/fold t13` skips the list. The run ends on the fold. Without a dialog to pick in (print mode), the agent picks the span:

   ```text
   /fold
   /fold t13
   ```

   `/episodes` lists what is archived on the current branch, and `/episodes stats` shows what the folds did to it (see [Numbers](#numbers)).

4. Then give it a task that depends on something inside the fold, and watch what it does with the note: carry on from it, go back with `recall` or `ask`, or guess. That is the smallest experiment the post suggests, and [MEASUREMENTS.md](MEASUREMENTS.md) builds from there.

If you try it, [open an issue](https://github.com/arthurfiorette/pi-episodes/issues) with what the agent did with its notes. How it feels after a week of real work says more than any benchmark so far.

<br />

## How it works

Every user message and tool result gets a **turn stamp** such as `[t12]`, appended as its own text block when the message is saved. Stamps are written once and never edited, so they never change bytes that are already in the prompt cache. They are for the model, which names them in `since`; the transcript hides them.

When an episode is done, the agent calls `fold(since: "t13", note: "...")`, where `t13` is the first stamped message inside the episode. If that is a tool result, the span starts at the response that made the call, so a tool call never loses its result. From there up to the fold call, the span leaves the model's context:

| In the span | After the fold |
| --- | --- |
| Assistant responses and tool results | Removed from context, kept in the session file |
| User messages | Kept verbatim with their stamps, so a mid-episode correction still reads as one; pastes over 4,000 characters are cut, with a pointer to the archive |
| Extension messages | Removed |
| The fold call and its result | Kept; the result is the note |

The harness extends the note with what changed on disk during the span, the stamps of the user messages that stayed, and a handle such as `ep-1a2b3c4d` that points at the archive. Disk changes come from a working tree diff, so files a shell command generated are listed too; files ignored by git are not. Skill files (`SKILL.md`) read inside the span are named too, since a note says what was learned and not how the skill said to work.

A fold normally runs up to its own call, because that is the cheapest place to cut: nothing after the span has to be sent again. When a finished episode lies behind work the agent still needs verbatim, it adds `until`: `fold(since: "t13", until: "t20", note: "...")` removes the turns t13 to t20 only. The span then ends where the turn after `until` starts, so parallel tool results stay with their call, and the note says from which stamp on the turns were kept. The disk changes the note lists are those up to the end of the span. The turns kept after the span are sent uncached once, which the fold cost line in `/episodes` counts; the prompt tells the model to do this only when the span is larger than what stays after it. The note still lands at the fold call, after the kept turns, and the fold nudge for new work counts only what follows the newest note: the kept turns come up again once enough unfolded work sits behind the newest note (see the nudge below), or when the agent folds them itself.
Nothing is deleted. A fold appends `context_edit` entries to the session, which only change what the model sees from then on; `/tree`, exports and the session file still hold the full span. A fold can start before earlier notes. It then removes them with the rest of its span and keeps what they say: the harness puts each covered note under the new one in the fold's result, as the agent wrote it and oldest first, so the new note needs to say only what no note said yet. A note that itself kept earlier notes passes them on, each once. If another extension cuts that result so that the kept notes are no longer all there, in its handler for the result or by an edit it queued at the same turn end before this extension's handler ran, the fold does not take effect. An edit of the result made after that, at the same turn end by a handler that runs later or at any later time, is not seen: it takes the kept notes out of context like anything else it removes from a result, and they stay only in the archive. The kept notes are read from each note's result as it stands in context: when a handler changed a note after the agent wrote it, or a version that did not record the agent's text made the fold, all of that result is kept, not what the agent wrote. Each kept note stands under a line that gives its length and is read back by that length, and the fold records the length and a hash of all the kept notes as it wrote them. A later fold takes them apart only if that text still stands unchanged at the end of the result, so nothing a note says, a quoted line or title or a whole made-up list of kept notes, is taken for them. Kept notes that something changed since are not dropped: they stay as they stand, as part of the note they were under. `recall` and `ask` still reach each span by its own handle; the covering fold's thread holds their notes, not their spans again. The `/fold` picker offers every stamp since the last compaction. A fold cannot reach back past a compaction summary. A summary can still mention old stamps; folding from one is refused with the earliest stamp that can be folded from, so the retry is right the first time.

On screen, the folded messages stay where they were. What marks the fold is its own block (`▸ ep-1a2b3c4d t13-t27  4 messages, ~41k tokens archived`, with the note behind ctrl+o), a notification, and the footer status. The range is the stamps the span covered, so you can see which turns left the model's context. See [Limitations](#limitations) for why the span itself does not collapse.

A fold happens when the main goal of an episode is met, which is when trailing steps ("remove the debug flag") and parked work are most easily forgotten. So on the next prompt you send after a fold, the harness adds one short message listing the open items of the notes folded since the last such reminder, newest first and capped at about 300 tokens. It is appended once, never edited, so it does not disturb the prompt cache. Turn it off or change the cap in pi's `settings.json`:

```json
{ "episodes": { "openReminder": false, "openReminderMaxTokens": 300 } }
```

Left alone, the model folds late or not at all: the prompt is its only reason to. So once 50k tokens of conversation (at most a quarter of the context window) that a fold could remove have built up since the last fold or compaction (the recent turns a compaction keeps still count), the next prompt you send gets one short message asking the agent to fold first, if that work is finished and the request does not need its raw output, and then to continue. A run that grows by twice as much on its own gets the same message between two requests. When less than that threshold of room is left before Pi compacts (`compaction.reserveTokens`), counting the prompt being sent, a quarter of it is enough. The message names the earliest stamp after the latest note: a fold that starts later leaves the finished work before it in context. Work can also end up behind the latest note without being folded: a fold from a later stamp leaves it there, and a fold with `until` keeps it. It is sent again with every request, and the count above never reaches it. So once the threshold of such work sits behind the latest note, or a quarter of it close to compaction, the message names the stamp where that work starts and asks for one fold from there, which also merges the notes of the folds made since. Declined, it comes again only after a further quarter of the threshold has been added; a fold that removes work done since and still leaves that work behind is asked about at once, and one that only merges notes is not. Turns that a fold has just kept with `until` are the agent's choice: inside the same run they are asked about only after that further quarter or after a fold of work done since, or at your next prompt if the run ended before then and no earlier nudge about that work is still waiting for its quarter. Such a fold rewrites the cache from its start, so what stays after that point, the user messages and what the fold does not reach, is sent uncached once; the work it removes is no longer sent. That repays within a few requests when little stays there, as in the session measured below, and takes longer the more stays: the threshold is a rule for room, not a promise that each such fold pays for itself, and the break-even figure under Numbers says how each one did. Notes take up room too: each stays in context together with the call that wrote it, whose argument is the note again, and a session that folds whenever it is asked can stay far from compaction for hundreds of turns. A fold that covers notes keeps them as written and removes the rest: the calls, what the agent wrote after each fold and what folds left behind them. So once that rest comes to a quarter of the threshold, the message for new work names the stamp of the earliest note, saying which note it ends, or of work left before it, and asks for a fold from there. Notes and their calls never count as new work, so such a fold brings no further message by itself; what the agent wrote after its folds does, since only a fold that covers the note before it removes it. That fold sends the user messages after its start and the notes it keeps uncached once. An earlier version asked for one note in place of all of them instead. Over a 302-turn session the model wrote 400 to 800 tokens each of the three times it was asked, whatever the notes held, and at the end answered 2 of 12 questions about the session from its context, where it answered 12 in a session in which every note stood; that is why a covering fold now keeps the notes. In that other session, which folded on every message and never came close to compaction, 32 notes were 29k and their calls 20k of the 124k tokens in context at the end. Asked at two points of it to fold from its earliest note, the model did in six of six tries once the message said which note the stamp ends, and the next request held about 45k and 56k tokens where the session itself had 76k and 110k ([MEASUREMENTS.md](MEASUREMENTS.md), section 14). In one 302-turn session the model had folded only its last request each time, was asked four times near compaction to fold from the first stamp, did not, and Pi compacted; asked at the same point to fold from where its unfolded work started, it did in five of five tries, and in six of six with an earlier version that named the note before that work ([MEASUREMENTS.md](MEASUREMENTS.md), section 13). Growth is counted from the transcript, not kept in memory, so a resumed session or another branch is measured the same way; an ignored message counts as a new starting point. A fold whose span holds no response is refused; when the stamp it gave is that of the message being answered and unfolded work lies before it, the refusal names the stamp that work starts at. Forks are not nudged. Over three runs of a 97-turn benchmark session, an earlier version of the nudge cut the mean cost from $0.39 to $0.25 and the peak context from 255k to 101k tokens, with no compaction. Turn it off or change the threshold:

```json
{ "episodes": { "foldNudge": false, "foldNudgeTokens": 50000 } }
```

A tool result of 16,000 characters or more is cut before it enters the context: the model gets about 6,000 of them, the start and the end two to one, around a line naming a file next to the session that holds all of it. The result is stored that way, so nothing already cached changes. `recall` searches the whole text once the result is folded, and does the same for a result that another extension cut on arrival and recorded as `details.offload` with the file's `path` and `sha256`, when the file still matches, also when both cut the same result. Results of `read` are left whole, since the file is already there to read a slice of, and so are errors, the results of the extension's own tools, and the calls a codemode script makes, which the script needs in full. The result of a script that folded stands as the note and is left whole too, unless what the script printed next to the note reaches the limit by itself: then it is cut like any output, and the note is added whole after the cut if the cut took part of it. The calls a fork makes are not cut either, and a result with an image between its text blocks is left whole. Sessions without a file (`--no-session`) are not capped. Measured on one long task, the cap lowered cost by about 20% where codemode is the only tool, at the same outcome, and did next to nothing with the built-in tools, where large output comes from `read`: see section 12 of [MEASUREMENTS.md](MEASUREMENTS.md).

```json
{ "episodes": { "outputCap": false, "outputCapChars": 16000, "outputCapKeepChars": 6000 } }
```

<br />

## Tools

| Tool | What it does | Cost |
| --- | --- | --- |
| `fold(since, note, until?)` | Replaces the span starting at `since` with `note` and archives it. The span runs up to the call, or with `until` to the end of that turn. | One tool call. With `until`, the turns kept after the span are sent uncached once. With Anthropic, the cache is rewritten from the newest fold note or user message before the span (see [Prompt caching](#prompt-caching)). |
| `recall(query)` | Searches the archived spans, forks and compacted turns on the current branch: tool outputs, the agent's own messages and reasoning, tool arguments. Results are ranked with BM25+ ([MiniSearch](https://github.com/lucaong/minisearch)): messages holding every word first, in any order, rare words such as error codes weighing more than common ones, with prefixes and small typos tolerated on plain words but never on codes or paths. A `"quoted"` query or `regex: true` matches exact text per line instead. Returns the best line of each match with its handle and turn. With a handle, it searches that thread on any branch. Matches past the limit (30 by default, 200 at most) are written to a file next to the fork transcripts, so a broad query costs a path, not the context. `recall(handle, turn)` with no query returns one turn verbatim instead, the message carrying that stamp and the responses up to the next one: up to 8,000 characters, the rest in a file. A range such as `t14-t16` returns those turns, each message under its own turn: up to 24,000 characters. | No model call. |
| `ask(handle, question)` | Resumes an archived fold or fork with everything it held, asks one question, returns only the answer. Compacted turns cannot be asked: they filled the window when they were compacted. | One model call over the thread's context, cached if the provider still has it. |
| `fork(task)` | Runs a copy of the agent, starting from its current context, until the task is done, and returns its final note plus what changed on disk. Tool calls go through the parent's tools, hooks and permissions. | The child's prefix is the parent's, so it reads from the parent's cache while that is alive; each child turn re-reads the whole prefix at the cached price, and parallel forks each write their own tail. |

Two cues help the agent go back when it does not know it should:

- **File touch:** when a file tool touches a file that a folded span or fork changed or used, the result names the thread, at most the two newest ones, once for each thread and file. It fires on edits after the change is made, and not at all for files touched through the shell; spans that only read a file trigger it too. The file tools are `read`, `edit` and `write`, and the anchor tools of [pi-hashline-edit-pro](https://github.com/YuGiMob/pi-hashline-edit-pro): `replace`, `replace_match`, `insert`, `copy`, `move` and `undo_last_change`. An anchor edit names its file only in its result, so the file is recorded in the result's details (`undo_last_change` takes a path, like pi's own tools); a `move` between two files counts for the file it moved to. A call that failed, or an edit that changed nothing, counts for no file. `grep` is not a file tool, since its path is where it searched.
- **Staleness:** `recall` and `ask` start their result with the files a thread relied on that have changed since it was archived.

`fork` is meant for long work that continues the agent's own, such as making the build pass. Inside, the fork folds its own finished episodes the same way, so a long build loop does not stay in its context either; its result lists those folds, and the archive becomes a tree: the fork's thread holds what it saw at the end, and each fold inside it is a thread of its own. It is deliberately not a reviewer: a review is useful because it does not share the author's assumptions. The other half of the post, a separate agent consulted on purpose and handed the user's requests without the agent's reasoning, is not built here; the prompt only tells the model how to brief one if it has such a tool.

While a fork runs, a widget above the editor shows its turn, elapsed time and current tool. `ctrl+shift+e` or `/forks` opens an overlay with the forks started in this pi process, running or finished, and the child's transcript as it grows: tab switches fork, arrows scroll, esc closes. After a restart, `/episodes`, `recall` and `ask` still reach earlier forks; the overlay does not.

<br />

### Compacted turns

Pi's compaction replaces the oldest turns with a summary. With `pi-episodes`, those turns become one more archived thread, `cp-1a2b3c4d`, holding what the summary replaced as the model saw it (spans folded before stay in their own threads). `recall` searches them like folds and forks, so an exact error string the summary left out can still be found; `ask` cannot resume them, since they filled the context window. The file-touch cue covers the files pi's compaction recorded as modified.

The summary is pi's text and cannot name that handle, so right after a compaction the harness adds one short message saying where the compacted turns went. Compaction has just rewritten the whole prompt, so it costs no cache. Turn it off with `{ "episodes": { "compactionNotice": false } }`.

<br />

## Prompt caching

Each request reads and refreshes only its longest matching cache entry. During a long episode, the entry where the episode started is no longer the longest match, so it expires, and the fold that ends the episode would rewrite everything before it.

With Anthropic, `pi-episodes` keeps a cache breakpoint on the newest fold note or user message, whichever is later. A fold usually starts after the latest note (one that merges notes starts before it), so that is the closest point before the next fold that is already known; the fold rewrites the cache from there on. When one user request holds several episodes, the anchor sits on that request and a fold also rewrites the episodes before it in the same request; how much that costs in practice is measurement 1 in [MEASUREMENTS.md](MEASUREMENTS.md). With an API key, pi leaves one of the four breakpoints free and it is used. With OAuth all four are taken, so the breakpoint of the required "You are Claude Code" block moves instead. Anthropic caches tools, then system, then messages, and the system prompt's own breakpoint already covers that block, so nothing that was cached before stops being cached.

`ask` and `fork` send the same session id, prompt, tools and thinking settings as the main thread, so their message prefixes match it; see [Limitations](#limitations) for what of pi's request pipeline they skip.

<br />

## Numbers

The footer shows context per request as a small graph of the last 16 requests, next to what is archived on the branch:

```text
episodes ▃▄▅▆▇▂▃▄▅▆▂▃ 2 folded (~41k), 1 fork(s)
```

Each bar is the context one main-thread request was sent with, scaled to the largest bar shown. The bar after a fold is green and the one after a compaction is yellow, so the drops can be told apart.

`/episodes stats` prints the rest for the current branch, a line per thing that happened:

```text
context per request  ▃▄▅▆▇▂▃▄▅▆▂▃▄▅▆▇▂▃
folds 2, median ~18k, notes ~1k (1 span picked by you)
fold errors 1, next fold succeeded after 1
context ~41k folded, ~380k fewer tokens sent since, compactions 0
fold cost break-even after median 12 request(s), 1/2 folds past it; ~$0.021 spent on notes and re-sent input, up to ~$0.038 saved since (cached input at 10% of the input price, from recorded costs)
outputs capped 3, ~21k kept out of context
cache missed after fold avg ~24k, anchor placed 2/2, anchor to span start avg ~3k
uncached input ~310k: ~230k new content, ~48k after folds, ~32k sent before and missed in 3 request(s), 1 of them (~21k) after a pause of 5 min or more
recall 3 (1 empty, 1 read back), ask 1, stale 1 (re-read 1)
cues 2, followed 1 (1 read-only)
forks 1, 6 turns, $0.12
added by the extension ~2k (stamps ~1k, cues ~0, reminders and notices ~1k), prompt ~1k per request
```

Token counts are estimates (about four characters a token), except the request sizes and the cache miss after each fold (input plus cache write, so it compares across providers that do and do not report cache writes), which come from the provider's usage. The anchor lines only appear with providers that take cache breakpoints, such as Anthropic. "Fewer tokens sent" counts each fold's span minus its note on every later request, until a compaction. "Fold cost" asks whether each fold has paid for itself: its cost is the note the model wrote, at the output price, plus the input its next request sent uncached, at the full rate instead of the cached one; its span minus its note, and minus the earlier notes it kept, is what every later request saves at the cached rate. The break-even is the number of requests where the two meet, and a fold is past it once it has been in effect that long. The rates are the session's own: what pi recorded as the cost of output, of cached input, and of cache writes where the provider bills them apart, next to uncached input. They are worked out for each model from its own requests; with several models the dollar figures use the mean price of a token over the tokens each model was sent, and a model that read nothing from cache is left out of the price of cached input. Requests without a recorded cost are left out. Without recorded costs it assumes cached input costs a tenth and leaves the note out, and says so; when no request read anything from cache, a fold costs only its note and the line says that. Where cached input is free a fold saves nothing and never breaks even, and the line says so when that holds for half the folds or more. With costs, the line also gives both sides in dollars. The re-sent input includes a miss that would have happened without the fold, such as a cache that expired while you were away. "Saved" is an upper bound: without the folds pi would have compacted before the context grew that far. "Uncached input" sorts every token that was not read from cache by what its request followed, which points at a cause and does not prove one. What the context grew by since the request before is new content, which no cache could hold. All the uncached input of the first request after a fold or a compaction is put under that fold or compaction, the new content of that turn included. The rest is a prefix the provider had been sent and did not return: when the request began five minutes or more after the one before it began, the cache had likely expired, and otherwise the cause is on the provider's side or another extension changed the context. A miss under 1,024 tokens counts as new content, since providers cache in blocks. A cue is followed when a `recall` matching that thread, or an `ask` of it, comes within the next three responses.

The numbers are read from the session entries alone, so older and resumed sessions show them too. Most come from the records already on the tool results; what has no record of its own is written as `episodes-event` entries (fold errors with their cause, and each file-touch cue), and an `episodes-meta` entry at session start records the version, a hash of the prompt and tool descriptions, pi's version and the settings. Pi never sends these entries to the model. Scripts outside pi can compute the same numbers from a session file:

```ts
import { episodeStats, statsLines } from 'pi-episodes/stats';

const stats = episodeStats(branchEntries, cwd);
```

To measure what the cache anchor saves, turn it off with `{ "episodes": { "cacheAnchor": false } }`. [MEASUREMENTS.md](MEASUREMENTS.md) says which of its questions these numbers answer and which still need a person.

<br />

## What is stored where

| Data | Location |
| --- | --- |
| Stamps, notes, fold and fork records, open-item reminders, compaction notices | The session file: on the messages and tool results themselves, plus a marker entry per applied fold and a custom message per reminder or notice. Compacted-turn threads are derived from pi's compaction entries; nothing extra is stored |
| Measurement events and run metadata | The session file, as `episodes-event` and `episodes-meta` custom entries, which are never sent to the model |
| Fork transcripts | `<session>.episodes/<handle>.json` next to the session file, or memory for `--no-session` |
| Recall overflow | `<session>.episodes/recall-<time>-<query>.txt`, or the OS temp directory for `--no-session` |
| Turns read back that are over 8,000 characters, or ranges over 24,000 | `<session>.episodes/<handle>-<turn>.txt`, or the OS temp directory for `--no-session` |
| Tool results cut on arrival | `<session>.episodes/output-<hash>.txt`, named by the hash of the text; the result's `details.episodesOutput` holds the path |
| Recall search index | Memory only: built on the first ranked `recall` of a session from the archive, then added to as threads are archived |
| Working tree snapshots | A private git index and object store in the OS temp directory, using the repository's objects read-only; removed when the session ends. The repository's index, stash and `.git/objects` are never written |

<br />

## Limitations

- Prompting is the only way the model learns the habit; there is no training behind it. Expect folds to be late, early or missing caveats until the prompt is tuned against real sessions. [MEASUREMENTS.md](MEASUREMENTS.md) lists what to measure, in order.
- A snapshot (`git add -A` and `git write-tree` against the private index) runs in the background for every stamped message, and the next tool call waits for it. Unchanged files are not hashed again, but large untracked files are, every time they change. Where that is too slow or unwanted, `{ "episodes": { "snapshots": false } }` turns snapshots off from the next session start: a note then lists the files that file tools touched instead of a diff, a fork's result lists no disk changes, and threads get no staleness notes, as outside a git repository.
- `git add -A` runs the repository's clean filters, so with Git LFS it writes into `.git/lfs/objects` and can be slow on large files. Changes inside submodules show only as the submodule having changed.
- Snapshots only live as long as the session. After a restart, a fold that reaches back past it lists the files that file tools touched instead of a diff, and older threads get no staleness notes. Outside a git repository that is always the case.
- The disk diff is the working tree's, not the agent's. Files you edit in another window while an episode runs are listed in its note as changed. Parallel forks share one working tree the same way.
- Forks run without pi's compaction or run-level retry, so `fork` refuses to start when the context is over 70% full. A fork is stopped after 30 minutes (`episodes.forkTimeoutMs` in pi's settings). A stopped or aborted fork still returns its record, so `recall` and `ask` reach what it did. The transcript is written after every turn, so a crash mid-fork keeps the file, though nothing in the session points at it yet.
- Pi writes a fold's edits one entry at a time. A crash partway leaves part of the span removed, starting from its end, so every request stays valid; the next turn end writes the rest and the fold's marker. Until then, `recall` does not reach that fold, `fold` refuses to run, and calls whose results are already gone show pi's placeholder result.
- A fork runs its tools through pi's `executeTool`, which reaches only callable tools. Tools the parent declares but only the model can call, such as `codemode`, stay declared (so the cache matches) but return an error in the fork, and the child is told which ones up front. Under codemode's `"only"` mode pi hides the direct tools from the parent's requests but not from the fork's, so there the fork's prefix does not match the parent's cache.
- With codemode, `fold`, `recall` and `ask` also work from scripts; under `codemode.mode: "only"` that is the only way to reach them. Their declarations then reach the model only if they fit the `codemode` description's budget, so the prompt section also gives their calls. A fold from a script takes effect only if the script succeeds, its result stands as the note, and one fold per response still holds across scripts. `fork` stays declared to the model under every mode and cannot be called from a script, since it resumes from the response that called it. A `/fold` done from a script does not end the run, because codemode does not pass a nested call's `terminate` on. Files that a script's file tool calls named in their arguments come from the call list pi keeps on the script's result, which is bounded and can drop calls or their arguments, so some can be missing from a fold's files and cues.
- Requests made by `ask` and `fork` go through pi's model registry, not the main agent loop. They skip other extensions' `context`, `before_provider_request` and `before_provider_headers` hooks, pi's attribution headers and the `blockImages` setting. Extensions that rewrite requests or must add headers do not apply to them, and the prefix only matches the main thread's when none are in use.
- Inside a fork, `fork` is unavailable. A fork can `fold` only its own turns, not the parent's it started from. Its folds are kept in the fork's transcript file, as threads of their own that `recall` and `ask` reach under their `ep-` handle; they get no working-tree diff of their own, since the fork's result carries one for all of its work.
- `!` command output inside a span cannot be removed by pi's context edits and stays in context.
- Stamps are part of the saved messages. The transcript hides them on user messages and on tool results whose tool has a renderer of its own; tools that use pi's default rendering still show them on resumed sessions, and so do session pickers.
- **Folded messages do not collapse in the transcript**, the way a subagent's work shows as one block. Pi's TUI rebuilds the transcript from the session entries and ignores `context_edit`, which only shapes the model's view. Extensions can render their own tool calls, custom messages and custom entries, but cannot hide, dim or re-render earlier user, assistant or tool messages, and the chat container is not exposed. Collapsing the span needs a change in pi: render a run of omitted entries as one expandable row, rebuild the chat when a `context_edit` entry is appended, and ideally let the edit carry a label such as the fold's handle. Pi's own overflow recovery omits failed attempts with the same mechanism and leaves them on screen too, so the change is useful on its own. Not sent as a PR yet.

<br />

## Acknowledgements

The design follows [the post](https://arthur.place/subagents-forks-collapsing-context), and through it the research it leans on: Event Segmentation Theory (Zacks & Swallow 2007; Kurby & Zacks 2008) and the goal-activation model (Altmann & Trafton 2002). [MEASUREMENTS.md](MEASUREMENTS.md) turns each into something to measure. The cover image is the post's.

Several details of `fork` were taken from [pi-subagents](https://github.com/nicobailon/pi-subagents) by Nico Bailon, after comparing the two: the live view of a running child (its fleet inspector overlay and status widget), the wall-clock timeout, writing the child's transcript as it grows rather than at the end, and showing tokens and cost on the result card. `pi-episodes` does not depend on it or register its tools. The two make different bets: a pi-subagents child is a full pi session with its own system prompt and agent definition, while a fork is a copy of the running agent that keeps the parent's prompt, tools and folded view byte for byte, so its requests hit the parent's prompt cache. If you want agents with their own prompts, background runs, chains or worktree isolation, use pi-subagents.

<br />

## Contributing

1. Use Node.js 24.16+ and pnpm 12, then install dependencies with `pnpm install`.
2. Run `pnpm test-types`, `pnpm test` and `pnpm lint-ci`. The session tests run the extension inside a real pi session against pi's faux provider, so no API key is needed.
3. Run `pnpm build` and load the result with `pi -e .` to try it by hand.

<br />

## License

[MIT](LICENSE).
