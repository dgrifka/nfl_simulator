# Pre-publication checklist

This repo is written to be public-safe from day one, but **making it public —
or adding any git remote at all — is a gated decision, not a push.** The gate
is a full manual review, run as its own checklisted task, with the explicit
option to rescind or soften whole portions before anything leaves the machine.

Nothing below is automated away. A grep is the floor, not the review.

## 1. Read every tracked file

- [ ] Comb through every tracked file by hand — docs, code, configs — reading
      for anything private, machine-specific, or better left unsaid. The
      author reserves the right to rescind entire documents or sections.
- [ ] `git ls-files -z | xargs -0 grep -nE '/(Users|home)/'` returns nothing.
- [ ] No usernames, emails, credentials, tokens, or absolute paths anywhere.
- [ ] `CLAUDE.local.md` is untracked (gitignored) and stays that way.

## 2. The private sibling

- [ ] The private repository is referenced **only** as "the private
      orchestration repo" — verify with a search for any other phrasing.

## 3. Git history

- [ ] Decide the history question before the first push: earlier commits
      contain local absolute paths (in `CLAUDE.md` and one research script,
      both since fixed). Either scrub history (`git filter-repo`) or publish
      from a fresh history. **Do not push the unscrubbed history without a
      deliberate decision.**

## 4. Data and outputs

- [ ] `data/`, `research/outputs/`, `*.parquet`, `*.nc` are gitignored and
      absent from tracking — cached pulls and fitted traces never ship.
- [ ] No figure, notebook output, or committed artifact embeds a local path.

## 5. Provenance

- [ ] Ideas that also circulate publicly (e.g. dropped interceptions, dropped
      passes, turnover luck) are documented through internal lineage — the
      numbered research documents and their timestamped pre-registrations are
      the record of independent derivation. No external attribution is needed
      and none should be implied.

## 6. Final pass

- [ ] Fresh clone into a clean directory; run the test suite and one research
      script; confirm nothing depends on anything outside the repo.
- [ ] Read the README and `CLAUDE.md` as a stranger would.
- [ ] Re-trace the article's final pasted text (added 2026-09-02): any hand
      edit to the Medium draft after a claim trace reopens every row, so the
      last text that will actually be published gets one more read-only trace
      per `handoff-2026-09-02-claim-trace.md` — extract the final text, run
      the trace, zero MISMATCH rows before the Publish button.

## 5. Spelling and voice (added 2026-08-27)

- [ ] American English in **prose and rendered strings**: "defense" not
      "defence", "neutralize" not "neutralise", "color" not "colour",
      "modeling" not "modelling". Code *identifiers* are exempt (ruled
      2026-09-03): v1.4.0 shipped with British-spelled identifiers in 83
      files, and renaming them churns history for zero function — new
      identifiers use American spelling, existing ones stay.
- [ ] No personal names beyond the author's own handle; no coworkers.
