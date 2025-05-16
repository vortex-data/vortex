# Contributing to Vortex

Welcome, and thank you for your interest in contributing to Vortex! We are  delighted to receive all forms of community contributions (issues, pull requests, questions).

We ask that you read the guidelines below in order to
make the process as streamlined as possible.

## Governance

Vortex is an independent open-source project and not controlled by any single company. The Vortex Project is a sub-project of the Linux Foundation Projects. As such, the governance is subject to the terms of the [Technical Charter](https://vortex.dev/charter.pdf).

## Coding style

Our CI process enforces an extensive set of linter (e.g., `clippy`) rules, as well as language-specific formatters (e.g., `cargo fmt`). Beyond that,
we document additional style guidelines in [STYLE.md](STYLE.md).

## Reporting Issues

Found a bug? Have an improvement to suggest? Please file a
[GitHub issue](https://github.com/vortex-data/vortex/issues).
Before you create a new issue, please ensure that a relevant issue doesn't
already exist by running a quick search of existing issues.
If you're unable to find an open issue, then please open a new one.

## Code Contributions

The contribution process is outlined below:

1. Start a discussion by creating or commenting on a GitHub Issue (unless it's a very minor change).

2. Implement the change.
    * If the change is large, consider posting a draft pull request (PR)
      with the title prefixed with [WIP], and share with the team to get early feedback.
    * Give the PR a clear, brief description; this will be the commit
      message when the PR is merged.
    * Make sure the PR passes all CI tests.

3. Open a PR to indicate that the change is ready for review.
    * Ensure that you sign your work via DCO (see below).

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
