---
name: research-issue
description: Research a paper or article linked from a dwim GitHub issue and draft a comment on it, with a summary, strengths, weaknesses, relevance to dwim, and what to try. Use when asked to research, read up on, or write up an issue that points at a paper. Drafts only; posts on a go-ahead.
---

# Research an issue

The argument is an issue number or URL in `dwim-sh/dwim`. The issue is
usually a link and a title, and the comment you draft is the write-up.

## Read

1. `gh issue view <n> --repo dwim-sh/dwim --json title,body,comments` for
   the link and anything already said.
2. Fetch the paper as text. For arXiv, `https://arxiv.org/html/<id>` is
   the version to read: there is no `pdftotext` or PDF renderer on this
   machine. Save it under the scratchpad, strip the tags, and read the
   abstract, the body, and the appendices that give prompts, harnesses, or
   results on open-weight models. Don't write from the abstract alone.
3. Check every claim you make about dwim against the code: the harness
   under `harness/`, the tools under `tools/`, the model under `inference/`,
   and the manual page. A number about dwim says whether it was measured
   or worked out from the code.

## Write

The comment has five sections, in this order, under `##` headings:

- **Summary**: one paragraph, under 150 words. What the paper claims, how
  it tests it, and the numbers that carry the claim.
- **Strengths**: bullets.
- **Weaknesses**: bullets. Where the evidence is thin, what was tuned, what
  was not evaluated.
- **Relevance to dwim**: bullets. What dwim already does of this, what
  differs, and what the paper's evidence does and does not say about our
  model and setup.
- **What to try**: a numbered list in order of cost, starting with how to
  measure. One or two sentences each. End with what the paper leaves open
  for us.

Keep the whole comment under 700 words. Plain sentences, no em-dashes.
Name a file, function, or flag only where the reader has to go there.

## Deliver

Write the draft to the scratchpad and show it whole in the reply. Don't
post it. On a go-ahead, post it as a comment with
`gh issue comment <n> --repo dwim-sh/dwim --body-file <draft>`, and say
that you did.
