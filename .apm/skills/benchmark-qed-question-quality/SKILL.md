---
name: benchmark-qed-question-quality
description: >
  Evaluate the quality of autoq-generated benchmark questions
  (activity_global, activity_local, data_global, data_local) against the
  criteria encoded in the autoq generator prompts, and compare two or more
  question sets head-to-head. Use when: the user asks to "evaluate", "score",
  "audit", "grade", "review", or "QA" autoq question outputs; when they want
  to compare different question sets, generator runs, prompt variants, model
  outputs, or A/B experiments; when they ask which set is "better" or want a
  per-question rubric; even if they don't say "autoq" or "quality" explicitly.
---

# Benchmark-QED Question Quality Evaluation & Comparison

Use this skill to (a) score the quality of an autoq-generated question set
against the criteria baked into the autoq generator prompts, and (b) compare
two or more question sets head-to-head (e.g., different prompt variants,
model versions, or datasets).

## Inputs

Accept any of:
- JSON files produced by autoq (typically under
  `<workspace>/output/.../questions/*.json`) containing arrays of questions.
- CSV/JSONL files with one question per row.
- A raw list of questions pasted in chat.

For each input, the user should specify (or you should infer) the **question
type**: one of `activity_global`, `activity_local`, `data_global`,
`data_local`. The criteria differ per type.

If comparing sets, ask the user to label each set (e.g., `A`, `B`, "prompt
v1", "gpt-4o") so the report is unambiguous.

## Workflow

1. **Identify the type** of each question set (ask if ambiguous).
2. **Score each question** against the per-type checklist below. Each
   criterion is **pass / fail / N/A**, plus a one-line justification when
   failing.
3. **Aggregate per set**: pass rate per criterion, overall pass rate, and
   counts of the most common failure modes.
4. **Compare sets** (if more than one): a side-by-side table with the
   per-criterion pass rate for each set, plus a "winner" column when one
   set is clearly stronger on that criterion. Call out ties.
5. **Surface examples**: for each criterion where sets diverge, show 1–2
   concrete examples from each set.
6. **Summarize**: a short verdict ("Set B is stronger on self-containment
   and varied starters; Set A is stronger on length compliance"),
   followed by actionable recommendations (e.g., "tighten the
   no-counting rule in Set A's prompt").

## Output Format

Always produce, in this order:

1. **Per-set summary table** — one row per criterion, columns:
   `Criterion | Pass rate | Sample failures (n)`.
2. **Comparison table** (when comparing) — one row per criterion,
   one column per set with pass rate, plus a "Winner" column.
3. **Failure examples** — grouped by criterion, with the offending
   question verbatim and a one-line reason.
   - For **repetition / distinct-and-diverse / no-duplicate** failures
     (e.g. `C13: No repetition (set)`, `C8: Distinct and diverse (set)`,
     `C11: Distinct and diverse (set)`), the reason MUST name the
     specific sibling question(s) it overlaps with by their number in
     the set, e.g. `Overlaps Q1 and Q8 on healthcare policy debates` or
     `Verbatim duplicate of local_test Q4`. Never write a bare "overlaps
     other questions" without citing which.
4. **Verdict & recommendations** — 3–6 bullets.

Use markdown tables. Do not invent scores — every percentage must be
derivable from the per-question scoring.

## Where to save the report

If the host (e.g. the benchmark-UI Copilot popup) specifies an exact
target path in the initial prompt, **use that path verbatim** — write
`QUALITY_REPORT.md` only there. Do not write copies to the compared
folders unless the prompt explicitly lists multiple destinations.

If no destination path was provided, ask the user where to save the
report before writing. As a last resort fallback (only when the user
declines to choose), save it inside the first compared folder.

Never write the report — or any helper scripts, intermediate JSON, or
scratch files — to the current working directory or any unrelated
location.

Any helper scripts you generate while scoring must be placed inside
the destination folder (a `.tmp/` subfolder is fine) and cleaned up
when you are done; never leave Python scripts or JSON dumps in the
working directory.

## Per-Type Quality Criteria

The criteria are derived directly from the autoq generator prompts. The
source files are listed at the bottom of this skill — re-read them if a user
disputes a criterion.

### `activity_global_questions`

- **Global**: requires holistic dataset understanding; not answerable by a
  single passage or keyword lookup.
- **Asks what the dataset contains**, not what the persona should do
  ("What advice do guests give…" ✅ vs. "How should I handle…" ❌).
- **Persona/task aligned**: directly serves the persona's stated task.
- **Length ≤ 10 words** (count strictly).
- **Self-contained**: no pronouns (`they/their/them`) without an explicit
  referent in the same question. Uses explicit subjects (`guests`,
  `speakers`, `experts`, `interviewees`).
- **Single-intent**: no compound questions ("…and why", "…and how").
- **Natural phrasing**: short, simple, direct.
- **Varied starters**: mix of `What`, `Which`, `How` across the set.
- **No "Why" questions** (they presuppose claims).
- **No meta-questions** about other questions or interview techniques.
  The bar is the *subject* of the question, not stray prepositional
  phrases. A question is meta only if it asks about the dataset/corpus
  itself as an artifact ("What topics are covered in the dataset?",
  "Which questions are repeated?"). A content question that merely
  contains the phrase "in the dataset" or "in the articles" is NOT
  meta and must not be flagged — e.g. "What new drug regulations do
  journalists report in the dataset?" is a valid content question.
- **No counting / ranking / frequency** language. Banned phrases:
  `most often`, `most common`, `most frequently`, `least often`,
  `how many`, `frequency`.
- **No NLP/ML operations** (sentiment, keyword extraction).
- **No repetition** of the same information category across the set.
- **Specific** to the dataset, persona, and task (not generic).

### `activity_local_questions`

- **References ≥1 named entity** from the provided list, using real names
  (not generic `guests`/`speakers`).
- **Locally answerable**: answerable from that entity's description alone,
  no cross-source synthesis.
- **Persona/task aligned**.
- **Length 10–20 words**.
- **Varied starters**: `What does X say…`, `How does X describe…`,
  `Which … does X mention…`, `According to X…`, `In X, how/what…`. Penalize
  repetition of one pattern.
- **Single-intent**.
- **Does not telegraph the answer** (no embedded resolution).
- **Stands alone** without the entity list or sibling questions.
- **Absolute references** (`summer 2024`, not `this summer`); fully-qualified
  people/places/things (`Governor Newsom's school funding proposal`, not
  `the proposal`).
- **No "Why" starters** (use `How` / `What`).
- **Not a disguised global** question (no patterns/themes/trends across the
  dataset).
- **No forced compound concepts**, no awkward analogies, no cramming
  entities just to cover them.
- **No marketing/academic tone**, no buzzword chains.
- **Grounded in the source data** — does not invent facts.
- **No meta-questions**.
- **Diverse and distinct** across the set.

### `data_global_questions`

- **Requires cross-dataset synthesis** (not a specific fact/entity/event).
- **Relevant to the shared abstract category** of the source local questions,
  and to **all** of them (not a subset).
- **Grounded in input topics** — clearly tied to the subject matter, not
  vacuously abstract.
- **Self-contained**: no demonstratives (`these cases`, `this issue`)
  without an explicit referent.
- **Length ~10–15 words**, simple, direct, natural.
- **Single-intent**.
- **Varied question types**: mix of `who, what, where, when, how, which, why`.
- **Includes 1–3 listing/enumeration questions** per batch (people,
  organizations, places, events) — including WHO-about-people when
  applicable.
- **No counting / ranking / frequency** (same banned phrase list as
  `activity_global`).
- **No NLP/ML operations**.
- **Distinct and diverse**.
- **Appropriate scope**: assumes only general dataset awareness.

### `data_local_questions`

- **Background paragraph** present and coherent, synthesizing
  who/what/where/when/why/how across the input texts (may link multiple
  texts).
- **Single-fact target**: each question targets one extractable fact
  (number, name, date, reason, stated position) — not a long descriptive
  answer.
- **Specific to entities/events/concepts** from the input texts.
- **Fully answerable from the input texts** — no external knowledge.
- **Stands alone** without the background paragraph or sibling questions.
- **Absolute temporal references** (`in 2019`, `at the time of the
  interview`) — never `currently / today / right now`.
- **Single-intent**.
- **Concise & natural** — under 30 words.
- **Distinct and diverse** across the set.

## Scoring Rubric

For each question and each applicable criterion, assign:

- ✅ **pass** — question fully meets the criterion.
- ❌ **fail** — question violates the criterion (include a ≤15-word reason).
- ➖ **n/a** — criterion does not apply (e.g., set-level criteria like
  "varied starters" don't apply to a single question; score them once per
  set).

Per-set criterion pass rate = `passes / (passes + fails)` (exclude n/a).

When comparing N sets:
- Declare a **winner** for a criterion when the leading set's pass rate
  exceeds the next set's by ≥ 10 percentage points (configurable; state
  the threshold you used).
- Otherwise mark as **tie**.

## Tips & Common Pitfalls

- Banned-phrase checks are deterministic — grep first, then read.
- Length checks must count words, not characters. Hyphenated tokens count
  as one word.
- "Self-contained" failures are the most common across all four types;
  always include 1–2 concrete examples in the report.
- Don't conflate "does not telegraph" with "vague". A question can be
  specific and concise without giving away the answer.
- When a user asks "which set is better?" without specifying criteria,
  default to overall pass rate **plus** explicit per-criterion winners —
  don't reduce to a single number.

## Source Prompts (re-read if criteria are disputed)

- benchmark_qed/autoq/prompts/activity_questions/global_questions/activity_global_gen_system_prompt.txt
- benchmark_qed/autoq/prompts/activity_questions/local_questions/activity_local_gen_system_prompt.txt
- benchmark_qed/autoq/prompts/data_questions/global_questions/data_global_gen_system_prompt.txt
- benchmark_qed/autoq/prompts/data_questions/local_questions/data_local_gen_system_prompt.txt
