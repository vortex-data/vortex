# Releases

[Release Drafter][drafter-src] ([runs][drafter-runs]) keeps a draft release with release notes
as changes land on `develop`. A release takes two workflow runs, then a manual publication:

1. Run the [Release Tag][tag-src] workflow from its [Actions page][tag-runs] and supply the
   `commit` input, a commit on `develop`. The workflow takes the version from the draft release
   and creates that tag at the commit. It fails if the commit is not on `develop`, if the tag
   already exists, or if there is not exactly one draft release. If there are several drafts,
   supply the `tag` input to choose one.
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

[drafter-src]: release-drafter.yml
[drafter-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-drafter.yml
[tag-src]: release-tag.yml
[tag-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-tag.yml
[binaries-src]: release-binaries.yml
[binaries-runs]: https://github.com/vortex-data/vortex/actions/workflows/release-binaries.yml
[publish-src]: publish.yml
[publish-runs]: https://github.com/vortex-data/vortex/actions/workflows/publish.yml
