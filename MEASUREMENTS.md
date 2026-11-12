# Measurements

What to measure before believing that folding beats compaction, and how. Only sections 12, 13 and 14 have a result so far.

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

<br />

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

Not measured: a full session in which a merge is asked for and the notes are kept.

<br />

## Keeping notes on what was found

Record results here, under the measurement, with the date, the model, the number of sessions and the number: not "it works", but "7 of 9 folds within ±2 turns of a human boundary, median lateness 2, 1 spurious; n = 4 sessions, claude-sonnet-4-5, 2026-10". A later prompt change should be checked against the same tasks.

<br />

## References

- Zacks & Swallow, *Event Segmentation*, Current Directions in Psychological Science 16(2), 2007. Observers generally agree on boundary locations; the agreed coarse boundaries mark the end of subgoals; older adults whose segmentation deviated from the group's remembered less; the grain of segmentation follows instructions.
- Kurby & Zacks, *Segmentation in the perception and memory of events*, Trends in Cognitive Sciences 12(2), 2008. Event models are updated at boundaries, which frees information from working memory; recall holds up when boundary content is kept; fine-grained segmentation goes with better recall of detail than coarse.
- Altmann & Trafton, *Memory for goals: an activation-based model*, Cognitive Science 26, 2002. A suspended goal is resumed only through a cue encoded with it before suspension; cues work when obvious and not shared by many goals; without one, people plan from scratch; postcompletion steps are forgotten more often than chance; the interruption lag is when the cue can be formed.
