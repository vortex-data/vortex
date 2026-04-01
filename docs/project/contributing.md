# Contributing

Vortex welcomes contributions of all kinds — code, documentation, bug reports, and feature requests.
The full contributing guide lives in the repository:

**[CONTRIBUTING.md on GitHub](https://github.com/vortex-data/vortex/blob/develop/CONTRIBUTING.md)**

Below is a brief summary.

## Reporting Issues

Bugs should be filed as [GitHub Issues](https://github.com/vortex-data/vortex/issues). Open-ended
questions and feature requests should be filed as
[GitHub Discussions](https://github.com/vortex-data/vortex/discussions).

## Code Contributions

1. Start a discussion on GitHub (unless the change is trivial).
2. Implement the change, including tests for new functionality or bug fixes.
3. Open a pull request — ensure CI passes and that you sign off your commits (see below).
   CI requires approval from a committer for first-time contributors.

For larger changes, consider opening a draft PR prefixed with `[WIP]` to get early feedback.

## Developer Certificate of Origin

All contributions require a
[Developer Certificate of Origin](https://developercertificate.org/) (DCO) sign-off.
If you have `user.name` and `user.email` configured in git, you can sign your commits with:

```bash
git commit -s
```

## AI Assistance

The Vortex project permits and embraces AI-assisted contributions. Today, most PRs involve some
degree of AI assistance — whether through IDE autocomplete, conversational AI, or autonomous agents.
This section describes our expectations around disclosure, review, and accountability.

### Repository Setup

The repository has [Claude Code](https://docs.anthropic.com/en/docs/claude-code) configured as a
GitHub Action. Users with write access to the repository can mention `@claude` in PR comments or
issue comments to trigger AI-powered code reviews, request changes, or generate PRs. Configuration
for Claude's behavior lives in `CLAUDE.md` at the repository root.

### Disclosure

Contributors should disclose AI usage in the PR description when conversational or agentic AI tools
(e.g., Claude, ChatGPT, Claude Code) were used to produce code, documentation, or tests. Standard
IDE autocomplete (e.g., Copilot tab-completion) does not require disclosure.

AI-translated content should note that it was translated with AI assistance.

### Human vs. Agent PRs

We distinguish between two kinds of AI-assisted PRs:

- **Human PRs** — A human writes the PR, possibly with significant AI assistance. The human author
  is accountable for the code quality and correctness, just as they would be for any handwritten
  code. Standard review rules apply (one approving reviewer).

- **Agent PRs** — An autonomous AI agent (e.g., triggered via `@claude` or a scheduled GitHub
  Action) opens the PR with minimal human steering. Agent PRs require **two human reviewers**
  before merge.

The distinction is straightforward: if a human opened the PR, it's a human PR. If an automated
agent opened it, it's an agent PR.

### Review Standards

AI-assisted code should receive extra scrutiny during review. AI tools can produce code that is
superficially correct but subtly wrong — off-by-one errors, incorrect edge case handling, or tests
that pass without actually exercising the intended behavior. Reviewers should pay particular
attention to:

- Correctness of logic, not just whether it compiles and passes CI.
- Tests that genuinely validate behavior rather than merely achieving coverage.
- Unnecessary complexity or over-abstraction.

The project may use AI-powered review tools on PRs. Reviewers are free to use AI to assist their
reviews without disclosure.

### AI Agents

AI agents (bots, scheduled actions, etc.) are permitted to:

- Open pull requests, with a human assigned as the responsible party.
- Post review comments on pull requests.

AI agents must **not**:

- Merge or approve pull requests.
- Push code to protected branches.

A human is accountable for all actions taken by an agent operating under their authority.

### AI-Generated Tests

AI-generated tests are welcome, but contributors must verify that tests actually exercise the
intended behavior — not just pass. A green test suite produced by AI can give a false sense of
coverage if the assertions are trivial or the setup doesn't reflect real conditions.

## Coding Style

CI enforces `clippy` lints and `cargo fmt` formatting. Additional style guidelines are documented
in [STYLE.md](https://github.com/vortex-data/vortex/blob/develop/STYLE.md).
