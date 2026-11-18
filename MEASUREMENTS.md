# Measurements

What to measure before believing that folding beats compaction, and how. Only sections 12 to 16 have a result so far.

The order follows the post: try the note before asking the model to notice anything. `/fold` separates the two questions, so note quality comes before boundary detection. Cheap scripted checks come first, so a broken mechanism is not mistaken for a bad model.

Most of these need nothing but session files. Every session pi writes is a JSONL file with the full transcript; `pi-episodes` adds turn stamps, fold and fork records (on the tool results, with handle, stamp range, token counts, open items and files), the `context_edit` entries that hide spans, the open-items reminders, `episodes-event` entries for fold errors and file-touch cues, and an `episodes-meta` entry with the version, prompt hash and settings. The ones that need a human are marked.

`/episodes stats` computes many numbers below for the current branch, and `episodeStats()` from `pi-episodes/stats` computes the same ones from a parsed session file. Each section says what it covers; the rest needs a script or a person. Compare sessions only when their `episodes-meta` matches.

Runs vary a lot between attempts. Unless a section says otherwise, run each condition at least 3 times per task and report the spread, not only the mean.

<br />

## 0. Fold mechanics

**Question.** Does the fold tool do what the model asked, and does the model use it as instructed?

**Method.** Script over session files with folds, real model.

**Report.**
- Rejected folds, by error (`since` made up, compacted or already folded, after the call, behind a note or summary, nothing to fold), and whether the retry that followed was correct. The errors name the earliest foldable stamp, so the retry should be right the first time.
- Folds not alone in their response.
- Folds called in the response that should have been the final reply (the prompt says not to).
- Note size in tokens and the note-to-span ratio. A note over a third of its span saved little.
- Folds of spans under 2k tokens (the prompt's threshold).
- Notes with no first line naming a question and an outcome, and notes missing an Open section where work was left.

**From the stats.** Rejections (causes are in the `fold-rejected` events) and whether the next fold succeeded, folds not alone, folds right before a final reply, spans under 2k, note and span tokens. Checking notes against the template needs a person.

**Pass.** Rejections rare and recovered on the first retry; no fold right before a final reply.

<br />

## 1. Cache

**Question.** Do the cache claims in the README hold?

**Why.** They are reasoning about provider behavior, not measured. Cheap to check.

**Method.** Script over session files. Same long episode, 15+ turns, then a fold. Run on Anthropic with an API key and with OAuth, each with the anchor breakpoint on (default) and off (`{ "episodes": { "cacheAnchor": false } }`).

**Report.**
- `cacheWrite` on the request after the fold, both conditions, next to `cacheRead` on ordinary requests before it. With OAuth, `cacheRead` on ordinary requests must not drop when the identity block loses its breakpoint.
- For each fold, the tokens between the anchor and `since`: the extra write a fold pays when one user request holds several episodes.
- Fork child turns: `cacheRead` close to the parent prefix size on every turn after the first.
- `ask` within the cache lifetime of its thread: `cacheRead` close to the thread size.

**From the stats.** Uncached input (input plus cache write) on the request after each fold, whether the anchor was placed on the request that called fold, and the tokens between the anchor and the span start. Fork and ask cache reads are in the usage on their results (forks as a sum; per turn in the fork's transcript file).

**Pass.** Smaller write after a fold with the anchor; equal reads elsewhere; forks and warm asks read the prefix from cache.

<br />

## 2. Note sufficiency

**Question.** Can the next step proceed from `B + fold(E)` alone? (Post: "Can the model write a note that keeps the caveats, given the template?")

**Why.** This is the post's smallest experiment, and the one the "query works" failure is about. A note that reads well but drops the condition is worse than no fold.

**Method.** Half scripted, half human. After a real episode, run the same probe three ways on copies of the session: no fold, `/fold`, and pi's `/compact`. The probe is a task that depends on something inside the span: once a fact the note should carry, once a detail only the archive has.

**Report.** For each probe and condition: continues correctly from context, goes back (`recall` or `ask`) and then continues correctly, or proceeds on a wrong assumption. A human judge scores each note on the template: headline specific enough to tell the episode from similar ones, conditions kept, rejected paths named, handled user messages listed, open items with a next step.

**Pass.** On facts the note should carry, no more wrong assumptions after `/fold` than with no fold, and fewer than after `/compact`. On archive-only details, the agent goes back rather than guesses. At least 5 probes per condition.

<br />

## 3. Kept user messages

**Question.** After a fold, does a user message kept from inside the span read as already handled?

**Why.** User messages stay verbatim, but every reply to them is gone. A kept question ("why is it failing?") or request ("also add a test") can look unanswered.

**Method.** Script plus a short human pass. In sessions with folds whose spans contain user messages, check the turns after the fold: did the agent act on a kept message again (answer it a second time, redo the change)?

**Report.** Kept messages listed under the note's Handled section; kept messages re-acted on; re-acted on although listed.

**Pass.** Re-acting near zero when the note lists the message.

<br />

## 4. Postcompletion steps and the open-items reminder

**Question.** Do trailing steps survive a fold, and does the reminder help?

**Why.** Altmann and Trafton describe the step after the main goal (take the card after the cash) as one people forget "more often than chance", because the goal that cued it is satisfied. For an agent: remove the debug flag, restore the pinned version, run the formatter. The prompt asks for them under Open, and the harness repeats Open items on the next prompt after a fold.

**Method.** Script plus a short human pass. Find spans where a temporary change was made (a `console.log`, a skipped test, a pinned dependency, a `.only`). Check whether the note lists its reversal under Open and whether it was reversed before the session ended. Run with `episodes.openReminder` on and off, and on the same tasks with pi's compaction instead of folds.

**Report.** Listed and reversed, listed and not reversed, not listed and reversed anyway, forgotten; per condition.

**From the stats.** Reminders sent and their tokens. Whether an item was reversed needs a person.

**Pass.** "Forgotten" near zero with the reminder on, and lower than with compaction.

<br />

## 5. Boundary agreement

**Question.** Does the model fold where a person would? (Post: "Can the model tell when an episode is over and where it started, without me saying so?")

**Why.** Zacks and Swallow report that "observers generally agree on the locations of event boundaries", and that older adults whose segmentation deviated from the group's remembered less of what they watched. Whether that carries over to an agent's own work is the open question; if the model's folds land in odd places, notes and recall are working on the wrong spans.

**Method.** Offline, on fixed transcripts, so the model and the humans judge the same sessions.

1. Take 5 to 10 real sessions of 50+ turns, run without the extension, with stamps added.
2. Give annotators the episode definition from the system prompt, word for word. Zacks and Swallow note that "observers adjust the temporal grain of their segmentation based on explicit instructions", so without the same definition their disagreement measures the instructions, not the model. Two annotators, independently; their agreement is the ceiling.
3. Annotators mark, on the stamps, where they would fold and with which `since`.
4. For the model: at each stamp k, send the transcript up to k with the system prompt and ask whether to fold now, and with which `since`.

**Report.**
- Share of human boundaries the model marked within ±2 turns.
- Share of model boundaries no human boundary is near.
- `since` error in turns, signed: positive means the model folded too little, negative that it reached back into the previous episode.
- Lateness: turns between the human boundary and the model's first "fold now". Tokens that accumulate in that gap are the cost of noticing late.

**Pass.** Agreement with humans at or near human-to-human agreement; median lateness under three turns.

<br />

## 6. Interruptions and returning

**Question.** When the user changes the subject mid-episode, is the episode parked with its next step, and picked up later?

**Why.** Altmann and Trafton's interruption lag: people resume well when they link the suspended goal to a cue before switching. The prompt asks the model to fold at a subject change with the next step under Open, and to say what it returns to.

**Method.** Scripted user turns on real tasks: interrupt an episode at a random point with an unrelated question, answer it, then say "back to it".

**Report.** Folded at the interruption, with the next concrete step under Open; resumed at the right step without redoing finished work; resumed after first recalling or asking the archive.

<br />

## 7. Going back

**Question.** Does the agent use the archive when it should, and not when it should not? (Post: "Does the agent go back to the archive when it should, and does surfacing notes on file touch help or just add noise?")

**Why.** Without a cue, Altmann and Trafton's model does not guess; it falls back on "planning from scratch". For an agent that is re-reading files and re-running commands an archived span already holds. The prompt names four triggers for going back; each is a measurable event.

**Method.** Script over session files, with a short human pass.

1. Find every trigger after a fold: an edit to a file in a fold's file list, a tool output containing an error string from an archived span, a command an archived span already ran, a rejected approach being tried again.
2. Check whether a `recall` or `ask` with a matching query came within the next three turns.
3. Count re-derivation by reading transcripts: work an archived span already did (the same investigation, a rejected fix tried again) redone with no recall before it. Exact repeats of commands are not a useful proxy: re-running a cheap command is often right, and recall would return stale output.
4. List every `recall` and `ask` that no trigger preceded.

**Report.**
- Hit rate: triggers followed by a recall or ask.
- After a compaction: recalls that hit compacted-turn threads (`cp-`), and whether those hits held something the summary had dropped.
- Re-derivation count, per fold.
- Spurious rate: recalls and asks with no trigger, and how many returned nothing.
- File-touch cue: followed by a recall or ask on that handle, or ignored. The cue is capped at two threads per file; the ignored rate at caps of one, two and five threads shows whether the cap is right. Separately for spans that only read the file.

**From the stats.** Cues shown, followed within three responses (a `recall` that matched the thread or an `ask` of it), read-only and capped; empty recalls; recalls that hit compacted turns. Hit rate on the other triggers and spurious recalls need a script; re-derivation needs a person.

**Pass.** Hit rate well above a control run without the triggers in the prompt; re-derivation lower than with compaction. Spurious asks rare, since each is a model call.

<br />

## 8. Staleness

**Question.** When a file a span relied on changes later, does the agent notice before acting on the span's conclusion?

**Method.** Script. Every `recall` and `ask` result records whether a staleness line was added. For each one, check whether the agent re-read the file before acting on the thread's answer.

**Report.** Staleness lines shown and followed by a re-read; shown and ignored; conclusions acted on from a stale thread that turned out wrong (human judged).

**From the stats.** Staleness lines shown, and those followed by a `read` of a changed file within three responses.

<br />

## 9. Cost and outcome against compaction

**Question.** Is `tokens carried per turn + cost of going back` lower with folds than with compaction, at equal outcome? And does a smaller, state-shaped context make the main line reason better, or is that an intuition left over from when windows were small? (Post's first open question.)

**Why.** Everything above measures parts. This measures the whole, and whether the work is as good: a cheaper run that fails more is a loss.

**Method.** Three conditions on the same tasks: no extension, extension with compaction enabled, extension with compaction disabled. Tasks must be long enough that both folds and compaction happen: at least twice the compaction threshold in total tokens. Human judged, or test suites as judge where the task has them. At least 5 tasks, 3 runs each.

**Report.** Per task and condition:
- Context tokens per request, summed over the run (what the model attends over); pi records `input + cacheRead` on every assistant message.
- Cache writes in tokens, and the share of requests that were a full write of the prefix.
- Tokens spent by `ask`, `fork` and compaction summaries.
- Number of folds, median span size, share of folds under 2k tokens.
- Task passed or not; turns to finish; corrections the user had to make.
- What the extension itself adds: stamps, cues, reminders and notices, and its prompt section on every request. Without it the comparison with no extension is not fair.

**From the stats.** Context per request, tokens folded and the estimate of tokens not sent since, compactions, fork turns and cost, and the extension's own additions. Outcome and corrections need a person or a test suite.

**Pass.** Lower summed context per task at an equal or better pass rate.

<br />

## 10. Fork against a clean subagent

**Question.** Does the fork anchor on the parent's wrong guesses more than a clean subagent would, and does it finish in fewer turns? (Post's last open question.)

**Method.** Same delegated task ("make it build", "find where X is defined") from the same parent state, once as `fork`, once as a clean subagent with a prompt the parent writes. Human judged, 3+ runs each.

**Report.**
- Turns and tokens to finish, split into input, cache read, cache write and output.
- Re-reads: files the child reads that the parent had already read.
- Anchoring: when the parent held a wrong guess at fork time, did the child repeat it? Needs tasks seeded with a wrong guess on purpose.
- Clarifications: times the clean subagent had to ask back or re-derive something the parent knew.

**Pass.** Fewer re-reads and clarifications than the clean subagent, with no rise in anchoring on seeded wrong guesses.

<br />

## 11. Folds inside forks

**Question.** Does a fork that folds its own finished episodes stay smaller and finish as well as one that does not?

**Why.** A fork has no compaction of its own and starts as large as its parent; folding is its only way to shed a long build loop.

**Method.** Same delegated task, long enough for the child to have at least one finished episode, run several times. The child decides whether to fold, so compare the runs where it folded with those where it did not. Script over the fork transcripts, which hold each child response's usage, the folds made and the folds refused.

**Report.** Child context per turn and its peak; folds made inside; folds refused for reaching behind the fork's start; turns to finish; whether the final note still mentions what was folded inside.

**From the stats.** Folds made inside forks, and fork turns and cost.

**Pass.** Lower peak context with no more failed or stopped forks.

<br />

## 12. Output cap

**Question.** Does cutting large tool results on arrival lower cost without lowering the outcome, or does the model spend more getting the cut text back than the cut saved?

**Why.** What is cut is never sent, so the saving looks certain. It is not: a model that needed the middle of an output runs the command again or reads the file, and an earlier comparison of a separate offload extension found cost going up with it.

**Method.** Same task and code, `outputCap` on and off, with the built-in tools and with codemode as the only tool, since a script's result is where large output arrives when `read` is left whole. Script over the session file: results cut and their sizes, later tool calls that name a cut result's file, requests each cut result stayed in context for.

**Report.** Cost, peak context, outcome; results cut and tokens kept out; calls that went back to a cut result's file; the saving worked out from the cut sizes alone, to set against the difference between runs.

**From the stats.** Outputs capped and tokens kept out.

**Pass.** Equal outcome, no more tool calls, and a cost difference in the direction the cut sizes predict.

**Result.** 2026-11-10, gpt-6-luna at low thinking, a 97-turn task, pi 1.4.0, commit 125ea30. Three sessions per condition under codemode, one with the built-in tools. Cost includes forks.

| | Sessions | Cost | Peak context | Model calls | Results cut | Tokens kept out | Final quiz | Turn checks of 228 |
|---|---|---|---|---|---|---|---|---|
| codemode, cap on | 3 | $0.209 ($0.194, $0.234, $0.200) | 84k to 94k | 290, 326, 305 | 36, 50, 78 | ~220k, ~333k, ~507k | 12/12 each | 218, 164, 205 |
| codemode, cap off | 3 | $0.261 ($0.278, $0.246, $0.261) | 82k to 87k | 421, 265, 349 | | | 12/12 each | 143, 205, 209 |
| built-in tools, cap on | 1 | $0.305 | 99k | 315 | 3 | ~22k | 12/12 | 223 |
| built-in tools, cap off | 1 | $0.378 | 212k | 299 | | | 12/12 | 226 |

Under codemode the cap lowered cost by 20% on average ($0.052; the two sets of three do not overlap, the closest pair being $0.011 apart and the farthest $0.084), at the same quiz score and with turn checks no worse. Tool calls, counting each script and the calls inside it, were 526, 524 and 554 with the cap and 736, 407 and 568 without, so the cap did not add calls; the table gives model calls, which is a different count. No call in any capped run named a cut result's file. That does not show the model never went back for cut text another way, by running a command again or reading a source again: calls repeated with the same arguments were 34, 61 and 79 with the cap and 39, 44 and 53 without, which overlap but lean higher with it. What such repeats cost is inside the measured difference. The saving worked out from the cut sizes alone was $0.09 to $0.17 per session, two to three times what was measured: without the cap the fold nudge fires sooner, so a large result does not stay as long as that sum assumes. Peak context is the same either way for the same reason.

With the built-in tools the cap is nearly idle: `read` returned 51 results of 16,000 characters or more in the capped session and is not cut. Those two sessions still differ by 24%, which is the spread between runs and says nothing about the cap. One codemode session without the cap stopped at turn 67 on a provider error (`{"detail":"Bad Request"}`) and was rerun; it is not in the table.

**What the cut cost in answers.** 2026-11-15/16, pi 1.5.0, commit c835d91, the same 97-turn task and a 302-turn one, in a comparison with another extension. Under codemode the sessions with this extension failed more turn checks than those with none: 211, 175 and 216 of 228 against 224, 226 and 226. The failed turns were read. A script had printed the 60 log lines asked for, a count and a source file, 18,000 to 22,000 characters, and the cut kept the first 4,000 and the last 2,000: the count, and most of the lines, were in what was left out. One session then answered from the lines in sight as if they were all of them (7 turns). Another answered that the output was truncated in 17 of the 22 turns from 62 to 83, and its next note said "tool output often truncated by harness". Neither opened the file the cut names. Over the 13 codemode sessions run since the first result, 67 of 582 turns with a cut result failed a check, against 42 of about 1,400 turns without one. The first result could not show this: its checks varied as much without the cap (143 to 209) as with it.

**The cut by parts.** For a script's output the cut was changed (see the README): each thing the script printed is cut by itself, a short one stays whole, and the preview opens with a line that says it is a part and where all of it is. Three more sessions of the same task under codemode, 2026-11-16, commit 649c98d, set against the three of c835d91:

| | Sessions | Cost | Peak context | Model calls | Results cut | Tokens kept out | Turns with a cut result that failed a check | Final quiz | Turn checks of 228 |
|---|---|---|---|---|---|---|---|---|---|
| one cut (c835d91) | 3 | $0.198 ($0.235, $0.163, $0.197) | 78k, 90k, 153k | 268, 247, 294 | 25, 43, 34 | ~150k, ~249k, ~236k | 7 of 20, 17 of 39, 1 of 25 | 12, 7, 12 | 211, 175, 216 |
| by parts (649c98d) | 3 | $0.202 ($0.180, $0.181, $0.246) | 79k, 83k, 84k | 276, 309, 357 | 40, 27, 38 | ~210k, ~150k, ~269k | 4 of 31, 1 of 23, 0 of 31 | 9, 12, 12 | 206, 214, 208 |

Turns with a cut result failed in 5 of 85 cases against 25 of 84, at the same cost. Where the answer came straight after a cut result, 1 of 17 turns failed against 18 of 27; 17 of those 18 are the one session that kept answering that the output was truncated. Where another call came after the cut and the answer followed a result that was whole, 4 of 68 failed against 7 of 57; the 7 are the session that answered from the lines in sight. Another call after a cut result is common in both conditions, so the counts do not show that the first line makes the agent go back; in the turns read, the call after a cut by parts printed less of the same thing. One call in the three sessions named the file. Of the five failures, two say that something was not in what was retrieved, in two the agent had the 60 lines uncut and named the wrong three, and one is the count described next.

The checks as a whole are not better by as much, for a reason that has nothing to do with the cut: in two of the three new sessions the agent's script came to count the events with pi's `read` tool, which stops at 50 KB, found 281 of the 400, and kept to that for 16 and 20 turns. The count came from a result that was whole each time; in the first of those turns of one session an earlier result was cut, in the others none. Sessions with no extension and with the other extension did the same in other runs. The sessions by parts differ from the others also by the line of section 15, which they do not write.

What this does not show: a 302-turn session with the cut by parts; whether the first line or the cut by parts did the work, since they were changed together; other models. The sessions ran 649c98d, before the cut came to tell codemode's part lines from lines of the output that read like them.

<br />

**The line at the top of a preview.** 2026-11-17, pi 1.5.2, commit 3b1b928, with the built-in tools, in a comparison with another extension over sessions of 7 to 302 turns. Here a file that is read is cut like any result (since 7e4434e), and the preview of a result that is no script's output had only the line in place of the middle. In the sessions of 13, 49 and 97 turns, 12 of the 13 turns with a failed check were turns where the report had been read and cut: the root cause and the mitigation stand on one line in the middle of a 73 KB report, and the agent, with the start and the end in view, answered that the report "does **not** contain `ROOT_CAUSE` or `MITIGATION` fields". Without this extension pi's `read` returns the first 50 KB, which holds the line.

Two changes were tried.

*Cut after the answer* (branch `read-after-answer`, not merged): the file arrives whole and a context edit puts the preview in its place once the agent has answered. Asked again about a value it had read and given rightly, the agent took the answer back in 26 of 28 sessions (claude-haiku-5-5 16 of 16, gpt-6-luna 7 of 8, claude-sonnet-5-5 3 of 4; claude-haiku-5-5: "my earlier answer was wrong. The read output I got was truncated"), whatever the preview or a line of the agent's own in the calling response said about the file having been in view. A result the agent has seen is not replaced by less: it takes what stands there now for all it ever received.

*A line first* (this commit's form): the cut on arrival stays, and the preview opens with a line that says it is a part and that what is not shown may still be there. Replays of two of the failed turns, gpt-6-luna at low thinking, each replay the turn's prompt sent once to the session as it was before that turn; counted is a replay that passed all checks of its turn.

| Turn | No line first | Line first | Line first, with a sentence on how to search |
| --- | --- | --- | --- |
| 5 of the 13-turn session (owner, ticket, root cause, mitigation) | 16 of 20 | 20 of 20: a patch of the words alone 10 of 10, this code 10 of 10 | 10 of 10 |
| 12 of the 49-turn session (status counts, mitigation, a source file) | 19 of 30 | 29 of 40: the patch 9 of 10, this code 20 of 30 (6 of 10, 14 of 20) | 14 of 20 |

The first ten replays of turn 12 with the line, made with a patch that changed the words of the preview and nothing else, passed in 9; thirty more with this code, which shows the same text, passed in 20, which is what the form without the line does. In the replays that failed with the line the agent did go back to the file, as the line asks, with a search whose pattern matched every event line (`grep -nE 'ROOT_CAUSE|MITIGATION|retry|status=' ... | tail -30`): the one line it wanted was cut out of that result in turn, or dropped by its own `tail`. A sentence added to the line ("for exactly what you need ... A search that matches most lines comes back cut like this") changed nothing and was not kept. Cutting after the answer passed 10 of 10 and 7 of 10 on the same two turns; its three failures counted events from the 50 KB page that was then in view.

Full sessions, one per length, the same three tasks, the cap's default settings, cost includes forks. Before and with the line, `foldAfterRun` was `"always"`, because the harness drives pi over RPC, where the default does not ask for the fold at the end of a run:

| | 13 turns, before | with the line | 49 turns, before | with the line | 97 turns, before | with the line |
| --- | --- | --- | --- | --- | --- | --- |
| Checks passed | 10 of 12 | 12 of 12 | 78 of 84 | 84 of 84 | 221 of 228 | 226 of 228 |
| Turns with a failed check, of them with a cut result | 1, 1 | 0 | 6, 6 | 0 | 6, 5 | 1, 0 |
| Answers from memory | 10 of 12 | 12 of 12 | 12 of 12 | 12 of 12 | 10 of 12 | 12 of 12 |
| Cost | $0.019 | $0.023 | $0.080 | $0.080 | $0.212 | $0.195 |
| Model calls | 35 | 42 | 113 | 119 | 225 | 251 |
| Peak context | 55k | 55k | 73k | 73k | 95k | 94k |
| Results cut | 9 | 7 | 52 | 45 | 25 | 42 |

What this shows: with the line no failed check followed a cut result in three sessions, where 12 turns had one before, at about the same cost. What it does not show: one session per cell, one model; the replays say that at a turn of the second kind the line alone changes little. Why the sessions did better at such turns is not established. In the 97-turn session the agent's first searches at turns 12 and 13 were broad and missed the fields, narrower ones in the same turns found them, and from turn 14 on it made the narrower search first; a replay starts instead from the turns of the earlier session. That is a reading of the traces, not a test. A file whose one needed line stands in the middle is still better served by a search than by a read, and the agent still has to make that search well. Under codemode nothing changes: a script's output has had its own opening line since section 12's cut by parts.

**Output that codemode had cut first.** Codemode cuts what a script prints to a budget of its own, 10,000 tokens unless the script asks for another: it keeps the first and the last 20,000 characters around a line with the count, and writes all of it to a temporary file. With the default budget such a result is still over this extension's limit, so it was cut a second time, from what codemode had passed on. The file the preview named then held about 40,200 characters of an output of, in one case, about 147,000, while the preview's first line said it "holds all of it"; codemode's own line with the count stood in the part left out; and `recall` could not find the middle once the turn was folded. In 15 sessions of 13 to 302 turns with codemode as the only tool (2026-11-17, gpt-6-luna at low thinking; five each with no extension, with another extension, and with this one at 3b1b928 and 4b7e009), 115 turns had a result that codemode cut, 39 of them in this extension's sessions, where all 42 such results were cut twice. A check failed in 4 of the 115 turns and in 46 of all 1,530, so the checks do not show a cost. The change is made for what the line says and for what `recall` finds: a result that arrives over the limit is now cut from codemode's file, and one that arrives under it, as when a script asked for a small budget, is left as it arrived (see the README). In a session with the real codemode on pi 1.5.2, a script printed 64,645 characters in three things, 40,184 arrived, and the preview of 6,504 showed the short thing in the middle whole, which codemode's cut had dropped, and named one file with all of it; after a fold `recall` found a line from the part codemode had left out. Not measured: a full session with the change.

## 13. Which stamp the nudge names

**Question.** When work sits unfolded behind the latest note, does the model fold it when asked, and does it matter which stamp the request names?

**Why.** A fold from a later stamp leaves the work before it behind its note, where the count for new work never reaches it. Until this was measured the nudge asked about it only close to compaction, and named the earliest stamp of the session, so that one fold would merge every note.

**Method.** Take a session in which this went wrong, cut it before a user turn, and let the model answer that turn again with the nudge under test. Read from the session: the stamp the nudge named, the fold calls that followed and what they removed.

**Report.** Per nudge wording: tries, folds from the stamp named, tokens removed, context before and after.

**Pass.** The model folds from the stamp named in most tries, and the context falls well below the compaction threshold.

**Result.** 2026-11-11, gpt-6-luna at low thinking, codemode as the only tool, pi 1.4.0. The session is a 302-turn benchmark run at commit 5d23dc7 in which the model, from about turn 100, folded only its last request each time. By turn 195 about 64k tokens of unfolded work and 85k with the notes between sat behind the latest note, the context was at 206k of 272k, and in the original run four nudges naming `t1` were not followed: the model folded its last request or a span with nothing in it, wrote that `t1` was "outside the available conversation span", and Pi compacted at turn 208. The final memory quiz of that run scored 5 of 12, against 12 of 12 in the six other complete runs of the task.

| Turn replayed | Stamp named | Tries | Folded from it | Removed | Context after |
|---|---|---|---|---|---|
| 195, close to compaction | `t1`, earliest of the session (5d23dc7) | 1 | 0 | nothing | 206k |
| 195, close to compaction | `t1`, with a sentence saying it is a valid start | 2 | 0 | nothing; 49k in one try, from another stamp | 207k; 141k |
| 195, close to compaction | `t239`, the note before the unfolded work | 6 | 6 | 134k each | 35k to 39k |
| 195, close to compaction | `t404`, where the unfolded work starts | 3 | 3 | 123k each | 53k to 58k |
| 150 and 170, far from compaction | `t239` | 2 | 2 | 80k, 98k | 33k |
| 170, far from compaction | `t404` | 1 | 1 | 88k | 50k |

The nudge first named the stamp of the note before the unfolded work, which made the fold cover ten more notes than it had to; it now names the first stamp after that note, `t404` here. Three of the six tries with `t239` had the sentence about a valid start and three did not, so that sentence was dropped. Why the model would not fold from `t1` was not found; that message holds values the task says to remember and not to write down, which may be why. The numbers are from one session and one model: they show that the stamp matters there, not how often.

The rule for when this nudge fires was corrected in review (6bb4293), and the replays were run again with it, one request each. At turn 195 the nudge named `t404` and the model folded from it in 2 of 2 tries (123k each); at turn 170 in 1 of 1 (87k). At turn 150, where 52k of unfolded work sat behind the latest note, the earlier rule sent no nudge, because one had been sent before that note and little was added after it; the corrected rule sent one, and the model folded from `t404` (66k) after one script of its own that failed.

The original run also called fold 104 times with the stamp of the message it was answering, a span with nothing in it, and never tried again. The refusal now names the stamp the unfolded work starts at. Three of the replays got that refusal: the model folded from the named stamp in one and went on without folding in two.

One full run of the 302-turn task with the rule (c0279d4, output cap off, 2026-11-11) did not put this nudge to work. The model answered each of the 29 nudges for new work with a fold from the stamp named, or from the one after it; the work behind the latest note stayed under the threshold, at about 23k tokens by the end, 22k of them tool results; and the nudge for it was never sent. So the rule is tested by the replays only. The run called fold on a span with nothing in it twice, against 104 times in the run above and 2, 4 and 70 times in the three other complete runs of the task under codemode, so the count shows nothing by itself; both calls got the refusal that names a stamp, and neither was followed by a fold from that stamp in the same request. Pi did not compact, the quiz scored 12 of 12, and the cost of $1.115, estimated from list prices, is inside the range of those four runs, $0.882 to $1.407.

<br />

## 14. Room the notes take

**Question.** How much of the context do the notes hold in a long session, does the model fold from the earliest note when asked far from compaction, and what does it still know afterwards?

**Why.** A note stays until a later fold covers it, and so does the response that called its fold, whose argument is the note again. Notes were merged only close to compaction, and a session that folds at every nudge can stay far from it: the one below did for all its 302 turns.

**Method.** For the share: at the end of a session, add up the text still in context by kind (script `/tmp/ctx-composition.py` for the run below). For the fold: the replay of section 13, cut before a turn at which the nudge for new work is due and what stands around the notes weighs over a quarter of the threshold. For what the model still knows: a full session, and at its end the 12 questions of the benchmark about facts from the whole session, answered from context alone.

**Report.** Tokens in context by kind at the end. Per replay: the stamp named, the fold made, the notes covered, the size of the new note, the context before and after. Per full session: questions answered, peak context, cost.

**Pass.** The model folds from the stamp named in most tries, the context after such a fold is clearly smaller, and the questions are answered as well as in a session that never covered a note.

**Result.** 2026-11-11, gpt-6-luna at low thinking, codemode as the only tool, pi 1.4.0, the 302-turn run of section 13 (c0279d4, output cap off), in which the model folded at each of 29 nudges: 39 folds in all, some unasked, of which 32 notes still stood at the end. At the end about 124k tokens of text were in context: tool results 39k, the 302 user messages 31k, 32 notes 29k, the calls of their folds 20k, other responses 2k. Each note was there three times: as the argument of the call, as the script printed it, in JSON, and as the extension added it again because it looked for the plain text (fixed in e4bca6a). The first request after a fold held 6k tokens at turn 12, 36k at turn 100, 76k at turn 205, 110k at turn 256 and 135k at turn 299.

*First rule: one note in place of all (ac7533f to bef8a0a).* A fold that covered notes removed them, and the message asked for a note covering everything from the earliest note on. Replayed at three turns of that run, the nudge named `t19`, the stamp of the earliest note, every time, and the model folded from it in 9 of 9 tries:

| Turn | Tries | Context before | Notes covered | Removed | New note | Next request | Same point in the run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 6 | 111k | 13 | 75k | 463 to 1,014 tokens | 17k to 19k | 57k |
| 205 | 1 | 135k | 21 | 85k | 548 tokens | 25k | 76k |
| 256 | 2 | 174k | 26 | 108k | 386 and 524 tokens | 30k | 110k |

"Same point in the run" is the first request after the fold the original run made at that turn. The tries span three versions of the message's wording and of the count that decides when it is sent (the working tree before ac7533f, 61cc80e, f17a2a5); the stamp named and the fold made were the same in all. The first request after such a fold sent 16k tokens uncached at turn 143, 23k at turn 205 and 28k at turn 256, in the tries that no other try had warmed the cache for: the user messages after `t19`, which stay. The result of the merging fold carries text the model did not write, the stamps of the user messages kept and the handles of the notes covered: about 700 tokens at turn 143 (136 messages, 13 notes).

A full session with that rule (bef8a0a, output cap off, same workload and model, 2026-11-11, n = 1) against the session of section 13 without it:

| | Notes merged only close to compaction (c0279d4) | One note in place of all (bef8a0a) |
| --- | --- | --- |
| Questions answered from context | 12 of 12 | 2 of 12 |
| Questions answered with recall allowed | 12 of 12 | 12 of 12 |
| Peak context | 197k | 112k |
| First request after the last fold | 135k | 44k |
| Notes and their calls in context at the end | 32 notes, 49k | 6 notes, 7k |
| Folds, of which covering earlier notes | 39, 7 | 64, 30 |
| Cost | $1.12 | $1.01 |

The rule did what it was for, and it failed the pass condition. The facts asked about were in the first note, which stood to the end in the session without the rule. With it, three folds answered the message (turns 126, 167 and 252) and put one note of 615, 397 and 771 tokens in place of 15, 7 and 8 notes: the model wrote a note of about the same size whatever it covered. The early facts came through the first two and were left out of the third; the two questions answered at the end were about a user message that stays. The 27 other covering folds were not asked for: they started at the stamp of the note before them, which is the only stamp from which the reply written after that note can be folded, and put a note about the last request in its place. The same happened 7 times in the session without the rule. The 9% difference in cost is inside what two sessions of one setup differ by. An earlier try with this rule ended at turn 287 on a harness time limit and is not counted.

*Second rule: a covering fold keeps the notes.* The harness adds each covered note under the new one, as written, so covering a note loses nothing it said; the message is sent once a quarter of the threshold of what such a fold frees has built up, and no longer counts the notes themselves. Replayed at turns of the same session, with the text of each note put into its record as the package now records it:

| Turn | Tries | Folded from `t19` | Notes kept | Removed | New note | Kept notes | Next request | Same point in the run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 2 | 2 | 13 | 75k | 280 and 860 tokens | 8.6k | 31k and 34k | 57k |
| 205 | 6 | 4 | 21 | 85k | 420 to 660 tokens | 12.4k | 44k to 46k | 76k |
| 256 | 6 | 5 | 26 | 108k | 200 to 620 tokens | 16.1k | 56k to 57k | 110k |

"Kept notes" is their text at four characters a token. They are full of trace ids, and going by the next request the provider counts more for them: 26k more than under the first rule at turn 256.

The tries span three wordings. With the first, which asked for "a note covering only the work that no note describes yet", the model folded from `t19` in 1 of 3 tries and in the others from the stamp where that work started, as the words say. With "a note covering everything from there that your notes do not say already" it did in 4 of 5; in the fifth it wrote that `t19` lay behind a compaction and folded its last request, the answer section 13 reports for `t1`. With the message also saying which note the stamp ends ("the stamp at the end of your note ep-3b4b7333") it did in 6 of 6. In one of those the model then folded once more from the stamp of the new note: a fold of 17k tokens whose note, with the 27 it kept, came to 16k.

One full session with the second rule, same task, model and setup as the two above (package at 8b552cc, which wrote the line over a kept note without its length; n = 1, 2026-11-11):

| | Every note stands | One note in place of all | Covering fold keeps the notes |
| --- | --- | --- | --- |
| Quiz from context | 12 of 12 | 2 of 12 | 8 of 12 |
| Found with `recall` | 12 of 12 | 12 of 12 | 12 of 12 |
| Checks | 802 of 843 | 797 of 843 | 801 of 843 |
| Peak context | 197k | 112k | 141k |
| First request after the last fold | 135k | 44k | 78k |
| Folds, of them covering earlier notes | 39, 7 | 64, 30 | 33, 9 |
| Messages asking for a merge | 0 | 3 | 0 |
| Calls to `fork` | 0 | 6 | 92 |
| Cost, forks included | $1.12 | $1.01 | $1.11 |

What it shows: the 9 covering folds, none of them asked for, each started at the stamp of the note before it and kept that note with the ones already kept under it, up to 3; at the end the text of each of the 33 notes the agent wrote stood once among the results in context (the calls that wrote the notes hold their text too, as far as those calls are still in context). Under the first rule 27 such folds each left a short note about the last request in place of the note before it, and 7 did in the session where every note was meant to stand.

What it does not show:

- That answers hold. Of the 4 questions missed, 3 asked for values that no note of this session ever held: its first note said "counts/docs basenames" where the first note of the other session gave them. The fourth, the latest commit, stood in that first note to the end, and the model answered null. So the quiz here measured what the model wrote down, which differs from one session to the next, more than what folds kept.
- The message that asks for a merge. It was never sent: what a fold of all the notes would free, the calls that wrote them and the replies after them, came to about 13k tokens at the end of the session by a count of characters, and the message needs 12.5k, a quarter of the 50k threshold, when a prompt arrives. For most of the session it was well under that. The replays above remain the only evidence for the message.
- A difference in cost or context. From turn 119 to 209 the model handed each request to a `fork`, 92 in all, which no other session did more than 6 times; the forks cost $0.28 of the $1.11, kept their output out of the main context, and no fold was made in those 90 turns. Peak and cost are not comparable with the other two columns.

A second full session, same task, model and setup, package at 17fb8c7 (n = 1, 2026-11-12), in which the message was sent once:

| | Covering fold keeps the notes, second session |
| --- | --- |
| Quiz from context | 12 of 12 |
| Found with `recall` | 12 of 12 |
| Checks | 769 of 843 |
| Peak context | 144k |
| First request after the last fold | 85k |
| Folds, of them covering earlier notes | 48, 13 |
| Messages asking for a merge | 1 |
| Calls to `fork` | 2 |
| Cost, forks included | $1.11 |

The message came at turn 146, with 116k tokens in context, when what a fold of the notes would free had reached 13k. It named `t18` as the stamp at the end of the first note, and the model folded from there in its next call: 74k tokens removed, and the 17 notes written until then kept under a new note of 385 tokens, 10.4k tokens of them at four characters a token. The next request held 33k tokens; the first request after the fold before it, 11 turns earlier, had held 49k. The 12 other covering folds were not asked for and kept 1 to 7 notes each. When the quiz was asked, the text of each of the 48 notes the agent wrote stood once among the results in context, the first note under the note of turn 146, and all 12 questions were answered from context.

What this adds: one merge that was asked for, made at once from the stamp named, after which nothing the notes said was missing at the end. What it does not show: how often the model follows the message, for which the replays remain the evidence, this being one try; a second merge in one session (the calls and replies that a further one would free had come to about 11k tokens by the end, by a count of characters, under the 12.5k the message needs); or a difference in cost: $1.110 here against $1.115 in the session in which every note stood. The checks passed, 769, compare with 716 to 834 in the seven other complete sessions of this task with codemode.

Not measured: a session with more than one such merge, and several sessions per rule.

<br />

## 15. Whether the agent stands by its note

**Question.** After a fold, does the model treat its note as what it saw, or does it take the missing calls and outputs as a sign that it never did the work?

**Why.** A fold removes the evidence for what the note says. A model that checks its own claims against the transcript then finds a note stating a result and no call that produced it. Sections 1 to 14 were run on one model, which never did this, so none of them shows it.

**Method.** Scripted sessions of a few prompts, one fold each, in a directory with a 200-line log of random tokens. The model runs `cat` on the log, folds that work with a note stating one value from it, and is later asked for that value without tools. Two shapes: *next request*, where the fold comes at a later prompt and stops at `until`, so a user message follows the folded span; and *mid-run*, where the fold call is the response right after the tool result, so the fold call follows the span directly. Read from each session, for everything after the fold: text to the user that retracts the note (made up, wrong, not based on output, the command never ran); text that only says the value cannot be checked now; calls that go back (`recall`, `ask`, or the same command again); and whether the last answer gives the value.

**Report.** Per form of the fold and per shape: sessions, retractions, hedges, sessions that went back, final answers that give the value, and requests a provider rejected.

**Pass.** No retraction of a correct note, and the model does not go back just to check it: a `recall` of the span brings back what the fold removed.

**Result.** 2026-11-15, pi 1.5.2, thinking medium, reasoning kept, six sessions per row with claude-haiku-5-5 unless the row says otherwise. The value in the note was right in every session, and every final answer gave it. No provider rejected a request in any session.

| What the fold leaves of its span | Shape | Sessions | Retracted the note | Only said it cannot check now | Went back |
|---|---|---|---|---|---|
| Nothing (as before this section) | next request | 12 | 12 | 0 | 4 |
| Nothing | mid-run | 6 | 5 | 0 | 6 |
| Nothing; the system prompt says the note is first-hand | next request | 12 | 9 | 0 | 7 |
| Nothing; system prompt as above | mid-run | 6 | 6 | 0 | 6 |
| A line in place of the first response: "the tool calls I made here and their outputs were folded away after I read them" | next request | 6 | 0 | 1 | 4 |
| That line, and the system prompt | next request | 6 | 0 | 1 | 2 |
| A line: "I ran the tool calls that stood here and read their outputs before I folded them away ... my own record ... written with them in view" | next request | 6 | 1 | 4 | 1 |
| That line, and the system prompt | next request | 6 | 0 | 3 | 0 |
| That line, and the system prompt | mid-run | 3 | 2 | 0 | 2 |
| The first call of the span kept, with a line in place of its output; system prompt | next request | 6 | 1 | 0 | 5 |
| The same | mid-run | 6 | 2 | 1 | 5 |
| The same, the line saying "this line is not what the call returned ... shown here in full when it ran" | next request | 6 | 2 | 0 | 5 |
| The same | mid-run | 6 | 2 | 0 | 6 |
| **A line in place of the first response that also says why nothing is shown; system prompt** (the line until 2026-11-16, opening with the handle: "[ep-1a2b3c4d: earlier responses of mine stood here ...") | next request | 12 | 0 | 4 | 1 |
| **The same** | mid-run | 12 | 0 | 2 | 1 |
| The same sentences without the brackets, the handle inside the sentence ("... folded all of it out of this conversation as ep-1a2b3c4d: ...") | next request | 6 | 0 | 0 | 4 |
| The same | mid-run | 6 | 6 | 0 | 5 |
| **The same sentences in brackets, the handle inside the sentence; system prompt** (what a fold does now, see "One line for every model") | next request | 12 | 0 | 3 | 1 |
| **The same** | mid-run | 12 | 0 | 0 | 1 |

What the retractions said, with nothing left of the span: "I never ran `cat data/build.log` for your first request", "the fold note I just wrote states an artifact value and log details that I never actually read", "Please don't rely on it." A sentence added to the fold's result, saying that the note was written with the messages in view, changed nothing (6 of 6 retracted, in an earlier set).

Why the forms that failed did so, from the model's reasoning:

- A line that only claims the work holds when a user message follows it. When the fold call follows it, the provider joins the two responses, and the model read the line as the opening of the response that called fold: "my earlier message went straight to fold", with no call behind the claim. It then ran the command again.
- A kept call with a line in place of its output was read literally: "the tool result for `cat data/build.log` was only a fold placeholder, so I never actually saw that value". Saying in the line that the output had been shown in full did not change that. The model seldom retracted, and went back with `recall` in 21 of 24 sessions, which reads the folded output into context again.
- The line that holds says why nothing is shown: "that is why no call and no output is shown here". With it the reasoning was "that note was written while I was looking at the output, so it's a reliable record", in both shapes.

The same sessions with the final form, one or two each: claude-haiku-5-5 at high thinking (2, one went back), claude-sonnet-5-5 (2, one went back), gpt-6-luna (2) and gemini-3.8-flash (2, one went back and unfolded, as it does unasked): no retraction, the value given in all eight.

A span in which the agent called no tool gets a shorter line that speaks of responses only ("earlier responses of mine stood here (t1-t2). I folded them out of this conversation: that is why they are not shown here."). In four sessions that folded a two-sentence answer and asked for it again, the model restated it as fact each time. After the fold came to record where its line stands, with the wording unchanged, six more sessions of the two shapes gave no retraction and one hedge.

Sessions written by the version before, which stripped provider signatures from the responses a fold kept, still hold those edits. Seven of them (three gpt-6-luna, two claude-haiku-5-5, two gemini-3.8-flash, with 8 to 17 stripped responses in effect each) were continued with this version for three more requests, and no provider rejected one.

**Long sessions.** 2026-11-15, pi 1.5.0, claude-haiku-5-5 at thinking medium, the 97-turn task of section 12 (95 requests of work, a quiz without tools, a retrieval quiz), one session per cell. The agent folds when it chooses to. "Before" is the version without the line (and with the signature stripping the paragraph above describes); cost is pi's estimate from list prices.

| | Plain, before | Plain, with the line | Codemode only, before | Codemode only, with the line |
| --- | --- | --- | --- | --- |
| Folds | 5 | 9 | 3 | 3 |
| Turns in which the agent told the user its own note was wrong, unverified or not to be relied on | 3 | 0 | 3 | 0 |
| Turns with a caveat only | 0 | 0 | 0 | 2 |
| `recall` calls | 1 | 0 | 3 | 1 |
| Cost | $0.634 | $0.539 | $0.426 | $0.344 |
| Output tokens | 166k | 119k | 178k | 142k |
| Peak context | 129k | 133k | 121k | 124k |
| Quiz, retrieval, checks | 12, 12, 228 of 228 | 12, 12, 228 of 228 | 12, 12, 228 of 228 | 12, 12, 228 of 228 |

Without the line the retraction happens in long work too, each time at or right after a fold: "Before I answer incident-73, I need to correct my last fold note. [...] The per-incident figures I recorded for 33 to 72 also came from my own notes rather than from output I can show in this session, so I can't vouch for them." In the codemode session one such turn went on to read nine reports again and "corrected" a right figure in the note to a wrong one. With the line there was none; the two caveats read "my digest of the folded span t1-t60 was written from tool outputs that are no longer visible in this context" and "my earlier folded summaries of incidents 45-57 were not re-checked here". The scores do not show any of this: the checks grade the answer to the request at hand, which the agent worked out from the files each time. The cost difference of 15% and 19% is in part the work the retractions caused and in part the spread between sessions, which one session per cell does not separate.

**gpt-6-luna in long sessions.** 2026-11-15/16, pi 1.5.0, commit c835d91, which wrote the line for every model: six 97-turn and two 302-turn sessions on gpt-6-luna at low thinking, plain and under codemode (the comparison of section 12). In three of the eight the agent answered a request with nothing but a line of this form, with a handle and a range of turns of its own making, and stopped. That is 8 turns: 1 in a 97-turn session, 2 and 5 in the 302-turn sessions, four of the five within five turns of each other. Six came right after the result of a fold, two right after the user's message, and one more turn was answered with a bare stamp. Each failed its checks. In the 20 gpt-6-luna sessions of versions without the line, about 9,000 responses, there is no such answer (three turns were answered with a few words after a fold), and none in the claude-haiku-5-5 sessions above. The line was then written for Claude models only (commits 5b93597 to 0240684); the three codemode sessions of section 12 that ran in that state have no line and no such answer.

**One line for every model.** 2026-11-16. A setting that guesses the model is a poor answer, so the form of the line was looked at again, with two tests.

*Why gpt-6-luna wrote the line.* In its long sessions each fold starts at the stamp of the note before it, so the line of the next fold stands right after the result of the fold before: fold call, note, line, user messages, fold call, note, line. In the 302-turn codemode session 9 to 12 notes in context were followed by a line when the answers went wrong. When the agent folds at the start of a request and has to go on working, all it has seen follow a fold result is a line and then the user. The line also opened with the handle in brackets, as the tags the harness writes do ("[fold ep-...", "[t315]").

*Replays.* A finished session is copied and cut before one user turn, the lines of its earlier folds are rewritten to the form under test, and the model answers that turn once (gpt-6-luna, low thinking, pi 1.5.0, codemode as in the session). The turns are ones the sessions above had answered with a line: 228, 246 and 249 of the codemode 302-turn session, 112 and 238 of the plain one. Counted: replays whose answer is nothing but a line. These turns were picked because they went wrong, so the shares are not a rate per turn. Set against them, the scripted claude-haiku-5-5 sessions of the table above, both shapes together.

| Form of the line | gpt-6-luna: answer is only a line | claude-haiku-5-5: retracted the note | went back |
|---|---|---|---|
| Opening with the handle (until 2026-11-16) | 15 of 34 replays | 0 of 24 | 2 of 24 |
| No line | 0 of 10 | 17 of 18 | 10 of 18 |
| No line where the span stood; the words as a text block of the response that called fold, before or after the call | not run | 8 of 12 | 9 of 12 |
| The line as the text of a response that keeps the span's first call, a mark in place of that call's output | 0 of 10; 2 of 10 opened a response with a line of their own and then called `recall` several times | 1 of 12 | 7 of 12 |
| No response: a note by the harness added to the message before the span | 0 of 10 | not run | |
| No response: the words in front of the next response that stays | 0 of 10 | not run | |
| Prose without brackets, other words ("Earlier in this conversation I made tool calls here ...") | 0 of 34 | 5 of 12, all mid-run | 7 of 12 |
| The sentences of the line without brackets, the handle inside | 1 of 18 | 6 of 12, all mid-run | 9 of 12 |
| **The sentences of the line in brackets, the handle inside** | 2 of 42: one in this form, one in the form that opens with the handle, which the session's own earlier answer showed | 0 of 24 | 2 of 24 |

The two models go by different parts of the form. claude-haiku-5-5 needs the brackets: without them the sentences read as the opening of the response that calls fold, as the line that only claimed the work did. gpt-6-luna copies the line that opens with the handle. So the line keeps its sentences and its brackets and names its fold inside the sentence: "[Earlier responses of mine stood here, with the tool calls I made (t13-t27) and the outputs they returned. I read those outputs, and then I folded all of it out of this conversation as ep-1a2b3c4d: that is why no call and no output is shown here. ...]". It is written for every model again; `episodes.foldLine: false` turns it off. Of its two answers that were only a line, one copied the earlier form: at turn 246 the history holds the answer of turn 228, which the session itself had written in that form, and rewriting the lines of the folds does not change it. On other models, one session of each shape: claude-haiku-5-5 at high thinking, claude-sonnet-5-5, gpt-6-luna at medium and gemini-3.8-flash did not retract, and three of the eight went back once. The shorter line of a span without calls has the same form ("[Earlier responses of mine stood here (t1-t2). I folded them out of this conversation as ep-1a2b3c4d: ...]"); in four claude-haiku-5-5 sessions that folded a two-sentence answer and asked for it again, the model restated it as fact each time. That wording was not part of the replays.

*Sessions written with this line from the start*, one each, pi 1.5.0:

| | gpt-6-luna, low, codemode only, 302 turns | claude-haiku-5-5, medium, plain, 97 turns |
|---|---|---|
| Folds, each leaving a line | 10 | 5 |
| Turns answered with nothing but a line | 1 (turn 123, right after a fold; it failed its checks) | 0 |
| Turns in which the agent took its own note back | 0 | 0 |
| `recall` calls | about 180 | 0 |
| Cost | $0.62 | $0.48 |
| Peak context | 109k | 132k |
| Quiz, retrieval | 12, 12 | 12, 12 |
| Checks | 621 of 843 | 228 of 228 |

The gpt-6-luna session has one such answer in 302 turns, against 5 and 2 in the two 302-turn sessions with the line that opened with the handle; replayed 12 times, that turn was not answered with a line again. So the form makes the answer rare and does not rule it out. The claude-haiku-5-5 session costs less than both sessions of the table further up ($0.54 with the earlier line, $0.63 without one), which one session each does not separate from the spread.

The low check score of the gpt-6-luna session has another cause, which this section did not look for. At turns 62 and 136 the agent folded at the start of a request, as the nudge asked, and then looked for that request's data with `recall` instead of reading the report: "I can't verify `reports/incident-54.log`'s requested lines [...] from the available evidence". It kept to that for 33 and 39 turns, 72 of the 76 turns with a failed check. Replayed ten times each with and without the line, turns 62 and 136 were worked from the archive alone in 5 of 20 tries with it and 10 of 20 without it, so the line is not what causes it. It is older than the line. Of the 33 earlier complete codemode sessions on gpt-6-luna of 97 turns or more, leaving out the two quiz turns at the end, 30 have none to three turns of work done from the archive alone, two have five and eight with no more than three in a row, and one has 30: the first session of section 12 without the cap (143 of 228 checks), from before there was a line, where the same thing began at a fold at the start of turn 58 and lasted 27 turns. In both sessions the first `recall` after the fold found the request being answered in the archive, since the span of such a fold runs up to the fold call and so takes in that request's message. It is a fault of folding at the start of a request. Section 16 takes the archived copy of the request away, which did not help, and measures a fold made between two requests instead.

What this does not show: more than one long session per cell; a session longer than 97 turns on Claude; a long session of gemini-3.8-flash or of any model beyond the four; folds that cover notes with this line (the gpt-6-luna session had none); whether the hedges ("I am going from my note, I cannot re-check it without a tool") would become checks when tools are allowed. gpt-6-luna and gemini-3.8-flash did not retract with nothing left either: for them the line costs about 100 tokens a fold and, on gpt-6-luna, the rare answer above.

<br />

## 16. Where the fold is made: inside a request or between two

**Question.** Does it matter where a fold is made: at the start of a request, where the nudge asked for it, or between two requests?

**Why.** In the 302-turn gpt-6-luna session of section 15 the agent folded at the start of a request twice and then answered from `recall` for 33 and 39 turns instead of reading the files it was asked about. Section 15 left the cause open.

**Method.** Three parts.

1. *Counting*, without model calls. All 62 complete sessions of this extension on gpt-6-luna of 97 turns or more, 34 under `codemode.mode: "only"` and 28 without codemode, the two quiz turns left out. Each turn is put in one of three classes: it begins with a fold; a fold comes after other work in it; no fold. A turn fails when one of the benchmark's checks on its answer fails.
2. *Replays* at turns 62 and 136 of that session. The session file is cut before the turn and the turn's prompt is sent once, in the benchmark's sandbox. For a fold inside the request, the nudge comes with the prompt as it did; variants change what the fold leaves. For a fold between requests, the session is cut after the answer of the turn before, the fold is asked for there, the run ends, and then the prompt is sent. About half a cent a replay.
3. *Full sessions* with the fold asked for at the end of a run as built: two of 97 turns and one of 302, codemode only, gpt-6-luna at low thinking, Pi 1.5.0. The benchmark takes the last response of a turn as its answer, so a copy of it takes the last response before the request for the fold. It drives Pi over RPC, where the fold was then asked for by default; it now takes `"foldAfterRun": "always"` there.

**Report.** The share of turns that fail a check, by class. For replays: tries that read the report asked for, tries that worked from the archive alone, tries with a right answer. For sessions: checks, quiz, cost, peak, turns worked from the archive alone, what followed each request for a fold, and what the provider served from cache after each fold.

**Pass.** Set before the replays, with a second model's review of the plan: a fold between requests is worth building if one form of it gives a right answer in 9 of 10 tries at turn 62 and 4 of 5 at turn 136, with at most one try worked from the archive alone. A full session passes with no run of turns worked from the archive and 95% of its checks.

**Result.**

*Counting.*

| | Turns | Failed a check |
| --- | ---: | ---: |
| Codemode only: turn begins with a fold | 739 | 141 (19%) |
| Codemode only: fold after other work in the turn | 24 | 16 (67%) |
| Codemode only: no fold | 5,337 | 470 (9%) |
| No codemode: turn begins with a fold | 374 | 26 (7%) |
| No codemode: fold after other work in the turn | 41 | 25 (61%) |
| No codemode: no fold | 4,090 | 179 (4%) |

The requests of this benchmark are short and alike, which makes a fold in the middle of one worse than it would be in long work; the middle rows are few turns. The difference between the first and third row of each mode is the one that the rest of this section is about.

*A fold inside the request.* Tries that read the report asked for, and tries that worked from the archive alone:

| What the fold left | Turn 62 | Turn 136 |
| --- | --- | --- |
| As in the session, with the line of section 15 | 5 of 10, 3 | 8 of 10, 2 |
| As in the session, without the line | 2 of 10, 8 | 6 of 10, 2 |
| The span ends before the request's own message, so the archive holds no copy of it | 3 of 11, 6 | 9 of 10, 1 |
| That, and a paragraph in the result saying the request is new work | 10 of 15, 4 | 5 of 5, 0 |
| A paragraph naming the tools that the folded scripts called | 10 of 15, 5 | not run |
| That, with the latest working call of each tool | 7 of 15, 8 | not run |

Taking the archived copy of the request away did not help: without it the agent read the report no more often (3 of 11 and 9 of 10, against 5 of 10 and 8 of 10). No wording fixed it either. The two paragraphs raised the count at turn 62 to 10 of 15, which at these numbers may be chance, and the third lowered it again; none is ruled out as part of the cause by so few tries. In tries that failed, the agent knew which request it was answering and still only called `recall`, or answered an earlier request that the fold had kept, or folded again and called the open request done. Under `codemode.mode: "only"` with no nested tools listed in the prompt, as in the benchmark, the calls of `bash` and `read` leave the context with the span, and what stands last is a row of kept requests that look alike, the fold call and the note.

*A fold between two requests.* Tries with a right answer to the turn after it:

| What the fold left | Right | At turn 62, turn 136 | First call of the next turn |
| --- | --- | --- | --- |
| Nothing: everything up to the answer folded | 12 of 16 | 8 of 11, 4 of 5 | `recall` in 6, a search for tools in 7, `read` in 2, `bash` in 1. 12 searched for tools at some point; 3 counted events with Pi's `read`, which stops at 50 KB |
| The last request, with its calls and outputs | 16 of 16 | 11 of 11, 5 of 5 | `bash` on the report in all 16 |
| Nothing, with a paragraph giving the latest working call of each tool | 15 of 16 | 10 of 11, 5 of 5 | `recall` in 6; one try worked from the archive alone for 39 requests |

By the numbers set above the second form passes and so does the third, with its one try from the archive; the first does not. The third was not built because it still began 6 of 16 turns in the archive and stayed there in one, which is the behaviour this section is about, where the form that keeps the last request began all 16 with `bash` on the report. A right answer here is one that passes the benchmark's checks and gives the right counts. The kept request costs room, 4k to 9k tokens more in the next request, and nothing else measured: the turns after it took half the requests. Where the fold is called from a script the run cannot end on it, so the agent was asked to reply with one word: in 12 of 12 tries the reply was "Folded." and nothing else, and 11 of 12 answers to the next turn were right.

*Full sessions.* With the fold asked for at the end of a run, the last request kept, and each such fold starting at what the one before kept:

| 302 turns, codemode only | Folds inside requests (section 15) | Fold at the end of a run |
| --- | ---: | ---: |
| Checks | 621 of 843 | 839 of 843 |
| Turns worked from the archive alone | 72 | 0 |
| Quiz from context, retrieval | 12 of 12, 12 of 12 | 12 of 12, 12 of 12 |
| Folds | 10 | 21 (20 asked for, all made as asked) |
| Cost | $0.62 | $0.92 |
| Peak context | 109k | 146k |
| Uncached input | 0.96M | 2.43M |
| Output | 119k | 214k |
| Model calls | 827 | 809 |

| 97 turns, codemode only | Folds inside requests (three sessions, section 12) | Fold at the end of a run | The same, second session |
| --- | ---: | ---: | ---: |
| Checks | 206, 214 and 208 of 228 | 149 of 228 | 228 of 228 |
| Turns worked from the archive alone | not counted | 0 | 0 |
| Quiz from context, retrieval | 9, 12 and 12 of 12; 12 of 12 | 12 of 12, 12 of 12 | 12 of 12, 12 of 12 |
| Folds | not counted | 5 (5 asked for, all made as asked) | 5 (5 asked for, all made as asked) |
| Cost | $0.18, $0.18, $0.25 | $0.24 | $0.21 |
| Peak context | 79k, 83k, 84k | 94k | 94k |
| Model calls | 276, 309, 357 | 312 | 268 |

By the rule set above the 302-turn session passes, with 99.5% of its checks and no turn worked from the archive. Of the two 97-turn sessions the first fails, with 65% of its checks, and the second passes, with all of them; why they differ is below. One long session and two shorter ones say little about any of these numbers.

In the turn after 18 of the 20 folds of the 302-turn session the first call was `bash` on the report; in one it was `read` and then `bash`, and one was the quiz, where no tool can be called. None failed a check, none called `recall` first. The six earlier 302-turn sessions of this extension under codemode cost $0.59 to $0.95 and passed 621 to 839 checks, so this one is at the top of both: one of the six, with folds inside requests, passed 839 as well. Part of the higher output is the answers themselves: the session it is compared with gave short refusals for 72 turns.

*What a fold at the end of a run costs.* After 18 of the 20 folds the provider served 0 or 2,048 tokens of the next request from cache; that request held 13k to 73k tokens, 11k to 70k of them uncached. The 20 requests after folds took 745k tokens of uncached input, about $0.07 of the $0.92. In the session of section 15 the request after a fold was served from cache up to the fold's first edit (38k of 46k tokens at turn 289). A probe that hashed every item of each request showed that the two requests around the fold at turn 264 began with the same 237 items, about 30k tokens, and 2,048 were served from cache; with the line of the covered fold left in place they began with the same 252 items, and it was 2,048 again. A reading that fits most of it: beyond a short head, this provider serves a prefix from cache only up to a place where an earlier request ended, not any common prefix. That is an inference from these counts and not something the provider documents, and it leaves things unexplained. The 2,048 tokens it served are fewer than any request of the session held, so the head is not a place where a request ended. In the first 97-turn session the request after two of the five folds was served 2,816 tokens, and the first request of turn 49, with no fold before it, 2,048 of 14,211. On that reading a fold that starts after the note before it edits the context where an earlier request ended, and from the third on a fold that removes what the one before it kept edits a place that stands before every request since that fold and after that fold's own edits, where no request ended. Keeping the last request in view went with right answers in the replays and in the 302-turn session, and removing it one fold later is what costs the cache; leaving several kept requests to be removed together would re-send less often and carry more at the cached price in between, which comes to about the same on these sessions and was not built.

*One fold was followed by more.* At turn 278 the reply after the fold was "Folded." together with another call: the agent folded the request it had been left, read it back with `recall` twice, and then did the work for the next report, which had not been asked for, in ten requests. The request for a fold at turn 288 then started after that stray note, so its fold covered no earlier note and edited nothing before it; the fold at turn 300 covered the note of turn 288 only. After those two folds the provider served 68k tokens from cache. Since then a call made once such a fold has taken effect, and before anything new arrives, is not run, and the run ends when every call of the response was such a call. Six replays of that turn with an earlier form of the change, which refused the call and let the run go on, gave the one-word reply each time, so neither form has been seen at work with a model.

*The first 97-turn session.* $0.24, peak 94k, five requests for a fold, five folds, five replies of the one word, no turn worked from the archive, quiz 12 of 12 — and 149 of 228 checks. From turn 14 to the end the agent counted the events of each report in a script that read it with Pi's `read`, which stops at 50 KB: 281 events of 400, in 76 of the 78 turns that failed. It began three turns after the first fold with the tool's notice "Showing lines 1-285 of 406 (50.0KB limit)" in view. Wrong counts that last are not new with this change: answers with wrong counts of events or statuses run through 11 stretches of the 63 earlier sessions, one of 112 turns, and not all from this cause (that one reports 274 events, others 1,000 events, or 343 and 57). But what this change does was part of how it lasted here. The first fold removed every `bash` call of turns 1 to 10 and kept turn 11, a comparison of source files done with `read` alone, so that was the one workspace tool in view when the reports began at turn 12, which took 12 requests. Each later fold kept the last turn in view, by then one that counted with `read`, and from the second fold on the notes recorded "281 records, not 400" as a fact about the reports, which every later fold kept as written. So the session fails the rule set above, and the kept request and the kept notes can carry a wrong method as well as a right one.

*The second 97-turn session.* Run after the review of this change, with the code as it stands: $0.21, peak 94k, five requests for a fold, five folds, five replies of the one word, no turn worked from the archive, quiz and retrieval 12 of 12, and 228 of 228 checks. What was changed between the two sessions does not bear on the miscount: it concerns a request that got no answer, calls made after the fold, and `until` at a stamp behind the request. What differed is the turn that the first fold kept. Here the agent had done turn 11 with `read` and then with `bash` (`rg`, `sed`), so calls of `bash` were in view when the reports began. Turn 12 took 5 requests where the first session took 12, read the report with `bash`, and the counts were right from there to the end; no note records 281. So in both sessions the tool the agent went on with was the tool of the turn that the first fold kept in view. That fits a kept request passing its method on, and does not show it: the sessions differ in more than that turn, their notes among it, and the first took up its counting script three turns after the fold, not at once. Two sessions cannot say how often either course is taken.

What this does not show: more than one session of 302 turns or two of 97; a session without codemode, where the direct fold ends the run and the reply is not needed (one 13-turn session made one such fold); any model but gpt-6-luna; long work in which a run is one long task, where a fold in the middle is the agent's own to make; how often keeping the last request in view, and notes that record a wrong count as fact, make a wrong method last longer than it would have; a person typing while the fold is being made, a run stopped on the request for the fold, and a call after the fold, each of which is tested with the test provider only.

<br />

## Keeping notes on what was found

Record results here, under the measurement, with the date, the model, the number of sessions and the number: not "it works", but "7 of 9 folds within ±2 turns of a human boundary, median lateness 2, 1 spurious; n = 4 sessions, claude-sonnet-4-5, 2026-10". A later prompt change should be checked against the same tasks.

<br />

## References

- Zacks & Swallow, *Event Segmentation*, Current Directions in Psychological Science 16(2), 2007. Observers generally agree on boundary locations; the agreed coarse boundaries mark the end of subgoals; older adults whose segmentation deviated from the group's remembered less; the grain of segmentation follows instructions.
- Kurby & Zacks, *Segmentation in the perception and memory of events*, Trends in Cognitive Sciences 12(2), 2008. Event models are updated at boundaries, which frees information from working memory; recall holds up when boundary content is kept; fine-grained segmentation goes with better recall of detail than coarse.
- Altmann & Trafton, *Memory for goals: an activation-based model*, Cognitive Science 26, 2002. A suspended goal is resumed only through a cue encoded with it before suspension; cues work when obvious and not shared by many goals; without one, people plan from scratch; postcompletion steps are forgotten more often than chance; the interruption lag is when the cue can be formed.
