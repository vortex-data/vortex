# Release runbook

A release takes two workflow runs and a manual publication. Each run is started from its
Actions page with **Run workflow**, on the `develop` branch.

| Step | Workflow                          | Input    | Result                                        |
|------|-----------------------------------|----------|-----------------------------------------------|
| 1    | [Release Tag][tag-runs]           | `commit` | A draft release and a tag at the commit       |
| 2    | [Release Binaries][binaries-runs] | `tag`    | The `vx` binaries attached to the draft       |
| 3    | Publish the draft by hand         |          | [Publish][publish-runs] uploads every package |

## Before you start

- Choose the commit on `develop` to release. CI should be green on it.
- Each pull request's `changelog/*` label puts it in a section of the notes and sets the version
  bump, as [`.github/release-drafter.yml`][drafter-config] defines. A `changelog/feature`,
  `changelog/break` or `changelog/edition` pull request raises the minor version, a `major` label
  raises the major version, and anything else raises the patch version. Fix a wrong label before
  step 1, or run step 1 again after you fix it.
- To see the version and notes without changing anything, run [Release Tag][tag-runs] with
  `dry-run` checked. The run log shows the notes; the summary shows the version.

## 1. Tag the commit

Run [Release Tag][tag-src] ([runs][tag-runs]) with `commit` set to the commit SHA. For a
pre-release, also set `prerelease-identifier`, such as `rc`.

The workflow checks that the commit is on `develop`. It then runs Release Drafter, which drafts
the notes for the pull requests merged between the last published release and the commit, and
resolves the version. It tags the commit with that version. The run summary names the tag and links
the draft.

Release checks that should run only before a release go in this workflow, as jobs between
`prepare` and `tag`.

## 2. Attach the binaries

Run [Release Binaries][binaries-src] ([runs][binaries-runs]) with `tag` set to the tag from
step 1. The workflow builds `vx` at the tag's commit for macOS and Linux on ARM64 and x86-64, and
Windows on x86-64. It attaches all five archives to the draft only after every build succeeds.

## 3. Publish

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

- **Release Tag** can run again for the same commit. It updates the same draft, and accepts a tag
  that already exists at the commit. It fails if the tag exists at another commit; delete that tag
  only if nothing was published from it.
- **Release Binaries** can run again while the release is a draft. It replaces the attached
  archives.
- **Publish** runs after publication, so its jobs can only be re-run from the run page. The PyPI job
  skips files that are already uploaded. When release immutability is enabled, a published release
  cannot take new binaries, so fix build failures in step 2, before you publish.

[drafter-config]: ../release-drafter.yml
[tag-src]: release-tag.yml
[tag-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-tag.yml
[binaries-src]: release-binaries.yml
[binaries-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-binaries.yml
[publish-src]: publish.yml
[publish-runs]: https://github.com/vortex-data/vortex/actions/workflows/publish.yml
[releases]: https://github.com/vortex-data/vortex/releases
[crates]: https://crates.io/crates/vortex
[pypi]: https://pypi.org/p/vortex-data
[maven]: https://central.sonatype.com/namespace/dev.vortex
