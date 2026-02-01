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

AI-assisted contributions are permitted but must be disclosed in the pull request, along with the
extent of use. Contributors must be able to understand and reason about AI-generated output.

## Coding Style

CI enforces `clippy` lints and `cargo fmt` formatting. Additional style guidelines are documented
in [STYLE.md](https://github.com/vortex-data/vortex/blob/develop/STYLE.md).
