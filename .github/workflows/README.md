# Releases

A release takes two workflow runs, then a manual publication:

1. Run the [Release Tag][tag-src] workflow from its [Actions page][tag-runs] and supply the
   `commit` input, a commit on `develop`. The workflow runs [Release Drafter][drafter-config],
   which drafts the release notes for the pull requests merged between the last published release
   and the commit, and resolves the version from their `changelog/*` labels. It then tags the
   commit with this version. For a pre-release, also supply `prerelease-identifier`, such as `rc`.
   The workflow fails if the commit is not on `develop` or if the tag already exists at another
   commit. If the run fails after it drafts the release, run it again: it updates the same draft.
   Set `dry-run` to see the version and notes without drafting or tagging. Pull requests that
   change the workflow or `.github/release-drafter.yml` run it as a dry run at their base commit.
2. Run the [Release Binaries][binaries-src] workflow from its [Actions page][binaries-runs] and
   supply the `tag` input. The workflow gets the commit from the tag. It fails if the tag does not
   exist or if no draft release has this tag. All five platform builds use the same commit: macOS
   and Linux on ARM64 and x86-64, and Windows on x86-64.
3. Wait until the full workflow is successful. The workflow attaches the full set of binaries to
   the draft only after all builds are successful. You can run the workflow again while the
   release is a draft.
4. Examine the draft and publish it manually. Publication starts the [Publish][publish-src]
   workflow ([runs][publish-runs]), which publishes to crates.io, PyPI, Maven Central, and the
   compatibility fixtures.

[drafter-config]: ../release-drafter.yml
[tag-src]: release-tag.yml
[tag-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-tag.yml
[binaries-src]: release-binaries.yml
[binaries-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-binaries.yml
[publish-src]: publish.yml
[publish-runs]: https://github.com/vortex-data/vortex/actions/workflows/publish.yml
