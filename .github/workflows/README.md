# Release runbook

A release takes two manual steps: one workflow run, then the publication of its draft release.

| Step | What you do                                            | Result                                        |
|------|--------------------------------------------------------|-----------------------------------------------|
| 1    | Run [Release Tag][tag-runs] on `develop` with `commit` | A tag at the commit and a draft with binaries |
| 2    | Publish the draft by hand                              | [Publish][publish-runs] uploads every package |

## Before you start

- Choose the commit on `develop` to release. CI should be green on it.
- Each pull request's `changelog/*` label puts it in a section of the notes and sets the version
  bump, as [`.github/release-drafter.yml`][drafter-config] defines. A `changelog/feature`,
  `changelog/break` or `changelog/edition` pull request raises the minor version, a `major` label
  raises the major version, and anything else raises the patch version. Fix a wrong label before
  step 1, or run step 1 again after you fix it.
- To see the version and notes without changing anything, run [Release Tag][tag-runs] with
  `dry-run` checked. The run log shows the notes; the summary shows the version.

## 1. Prepare the release

From the [Release Tag][tag-runs] Actions page, choose **Run workflow** on `develop` and set
`commit` to the commit SHA. For a pre-release, also set `prerelease-identifier`, such as `rc`.
The [workflow][tag-src] runs these jobs in order:

1. `prepare` checks that the commit is on `develop`.
2. `checks` runs the [Release Checks][checks-src] at the commit. These are the checks that pull
   requests do not run: packaging dry runs for Python and crates.io, compatibility with the last
   release's files, musl, Windows, the CPU feature tiers, the minimum supported Rust version,
   minimal dependency versions, per-crate builds with all features and with the tokio dispatcher,
   Clippy without default features, sanitizers and Miri. If any check fails, the workflow stops before
   it drafts or tags anything. Fix the failure on `develop` and run the workflow for the new commit,
   or re-run the failed jobs if the failure was not in the code.
3. `tag` runs Release Drafter, which drafts the notes for the pull requests merged between the
   last published release and the commit, and resolves the version. It then tags the commit with
   that version. The run summary names the tag and links the draft.
4. `binaries` runs [Release Binaries][binaries-src], which builds `vx` at the tag's commit for
   macOS and Linux on ARM64 and x86-64, and Windows on x86-64. It attaches all five archives to the
   draft only after every build succeeds.

To run a check before a release only, add it to `release-checks.yml`.

## 2. Publish

Open the draft on the [releases page][releases], check the notes and the five attached binaries,
and publish it. Publication starts [Publish][publish-src] ([runs][publish-runs]), which sets the
workspace version from the tag and publishes:

- the Rust crates to [crates.io][crates];
- the Python wheels to [PyPI][pypi];
- `dev.vortex` artifacts to [Maven Central][maven];
- the compatibility fixtures for the version.

The PyPI and compatibility jobs run in the `push-to-pypi` and `compat-upload` GitHub environments.
If an environment requires approval, approve its job on the run page.

## If a step fails

- **Release Tag** can run again for the same commit, or you can re-run only its failed jobs from
  the run page. It updates the same draft, and accepts a tag that already exists at the commit. It
  fails if the tag exists at another commit; delete that tag only if nothing was published from it.
- **Binaries**: while the release is a draft, re-run the failed `binaries` jobs, or run
  [Release Binaries][binaries-runs] by hand with `tag` set to the tag. Either replaces the
  attached archives.
- **Publish** runs after publication, so its jobs can only be re-run from the run page. The PyPI job
  skips files that are already uploaded. When release immutability is enabled, a published release
  cannot take new binaries, so fix build failures in step 1, before you publish.

[drafter-config]: ../release-drafter.yml
[tag-src]: release-tag.yml
[checks-src]: release-checks.yml
[tag-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-tag.yml
[binaries-src]: release-binaries.yml
[binaries-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-binaries.yml
[publish-src]: publish.yml
[publish-runs]: https://github.com/vortex-data/vortex/actions/workflows/publish.yml
[releases]: https://github.com/vortex-data/vortex/releases
[crates]: https://crates.io/crates/vortex
[pypi]: https://pypi.org/p/vortex-data
[maven]: https://central.sonatype.com/namespace/dev.vortex
