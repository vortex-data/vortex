# Contributing to Vortex

Welcome, and thank you for your interest in contributing to Vortex! We are delighted to receive all forms of community contributions (issues, pull requests, questions).

We ask that you read the guidelines below in order to make the process as streamlined as possible.

## AI Assistance

> [!IMPORTANT]
> The Vortex project permits and embraces AI-assisted contributions. Contributors
> should disclose usage of conversational or agentic AI tools in the PR description.
>
> For the full AI policy — including disclosure requirements, review standards for
> AI-generated PRs, and rules for autonomous agents — see the
> [contributing guide](https://docs.vortex.dev/project/contributing#ai-assistance).

## Code Contributions

The contribution process is outlined below:

1. Start a discussion by creating or commenting on a GitHub Issue (unless it's a very minor change).

2. Implement the change.
   - If the change is large, consider posting a draft pull request (PR)
     with the title prefixed with [WIP], and share with the team to get early feedback.
   - Give the PR a clear, brief description; this will be the commit
     message when the PR is merged.
   - For significant new functionality, ensure that you write tests to cover that new functionality. Similarly,
     for bugfixes, include a test that reproduces the original bug (and that should now pass after the fix).
   - Make sure the PR passes all CI tests.

3. Open a PR to indicate that the change is ready for review.
   - Ensure that you sign your work via DCO (see below).
   - Disclose LLM usage as described in [AI Assistance](#ai-assistance).
   - CI requires approval from external committers.

## Development Workflows

The repository uses [`uv`](https://docs.astral.sh/uv/) to manage its Python workspace. From the
repository root, create or update the development environment with:

```bash
uv sync --all-packages
```

### Rust toolchain and MSRV

Workspace declares a Minimum Supported Rust Version through `rust-version`
in the root `Cargo.toml`, which the `Rust (MSRV)` CI job verifies by building
the publishable crates with exactly that toolchain:

```bash
cargo hack --rust-version --no-dev-deps --ignore-private check --all-features
```

Read the [Rust version compatibility policy](README.md#rust-version-compatibility-policy) before
changing `rust-version`; when the MSRV job fails, the usual fix is in the code or the dependency
update, not the MSRV.

### Python bindings

`vortex-data` is a mixed Python and Rust package. `uv` manages its Python environment, and
[Maturin](https://www.maturin.rs/) compiles the PyO3 code in `vortex-python/src/` into the
`vortex._lib` extension.

Python-only changes do not require an explicit extension rebuild. Run the affected tests directly:

```bash
uv run --all-packages pytest <changed-python-tests>
```

After changing the PyO3 Rust code, `vortex-python/Cargo.toml`, or relevant Cargo features, rebuild
the extension before testing. For a single build-and-test cycle, ask `uv` to reinstall the local
package, which invokes its Maturin build backend:

```bash
uv run --all-packages --reinstall-package vortex-data pytest <changed-python-tests>
```

For repeated Rust edits, install the extension into the development environment explicitly:

```bash
(cd vortex-python && uv run maturin develop)
uv run --all-packages pytest <changed-python-tests>
```

Rerun `maturin develop` after each Rust change. Pass Cargo features through Maturin when needed:

```bash
(cd vortex-python && uv run maturin develop --features <feature>)
```

Use one rebuild path per test cycle. Before handing off broad Python binding changes, run the full
Python check:

```bash
./vortex-python/check.sh
```

This synchronizes the workspace, rebuilds the extension, runs Python formatting, linting, type
checks and tests, and validates the documentation.

### Documentation

Use `docs/Makefile` as the canonical interface for Sphinx. From the repository root, run the full
clean build and doctest flow with:

```bash
uv run --all-packages make -C docs check
```

For live-reloading development, run:

```bash
make -C docs serve
```

Use `make -C docs help` to list focused targets, and finish documentation changes with `check`.

## Governance

Vortex is an independent open-source project and not controlled by any single company. The Vortex Project is a sub-project of the Linux Foundation Projects. As such, the governance is subject to the terms of the [Technical Charter](https://vortex.dev/charter.pdf).

## Project Roles

- Contributor: anyone who contributes intellectual property to the common endeavor of the project under the project license.
- Committer: a subset of Contributors, who collectively determine the project's technical direction. Committers have permissions to review & merge code contributions. Unless they are also Maintainers, Committers are non-voting members of the Technical Steering Committee (TSC).
- Maintainer: a subset of Committers, who are also _voting_ members of the Technical Steering Committee (TSC). In practice, Maintainers' primary responsibility is to manage membership of the Committers/Maintainers group over time and ensure the long-term health of the project.

### Committers

At the time of writing, the following individuals serve as Committers (non-voting TSC members) on the project:

1. Adam Gutglick
2. Alexander Droste
3. Andy Pavlo
4. Connor Tsui
5. Daniel King
6. Dmitrii Blaginin
7. Joseph Isaacs
8. Marko Bakovic
9. Onur Satici
10. Xinyu Zeng
11. Alfonso Subiotto
12. Lorenz Hübschle

### Maintainers

At the time of writing, the following individuals serve as Committers & Maintainers (voting TSC members) on the project:

1. Andrew Duffy
2. Carlo Curino
3. Nicholas Gates
4. Robert Kruszewski
5. Wes McKinney
6. Will Manning (chair)
7. Frederic Branczyk
8. Benjamin Wagner

## Coding style

Our CI process enforces an extensive set of linter (e.g., `clippy`) rules, as well as language-specific formatters (e.g., `cargo fmt`). Beyond that, we document additional style guidelines in [STYLE.md](STYLE.md).

## Issues and Questions

Bugs, feature requests, and questions should all be filed as
[GitHub Issues](https://github.com/vortex-data/vortex/issues/new/choose). We strongly prefer that
you use one of the provided issue templates rather than opening a blank issue; templates make sure
we get the information needed to act on your report. For quick questions, the
[Vortex Slack channel](https://vortex.dev/slack) is also a good option.

## Developer Certificate of Origin (DCO)

The Vortex project, like all Linux Foundation projects, uses Developer Certificates of Origin to ensure
compliance with the project license for submitted patches. Signing off a patch certifies that you have
the right to submit it as an open-source patch.

From <https://developercertificate.org>, only sign & submit patches where you can
certify that:

```git
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Signing off is simple, simply add this line to every commit message:

```git
Signed-off-by: Your Real Name <your.real.email@email.com>
```

Please note that pseudonyms and fake email addresses are not allowed.

If you have configured `user.name` and `user.email` in git, then you can sign your commit with `git commit -s`.
Similarly `git rebase -s` can be used to sign commits in bulk.
