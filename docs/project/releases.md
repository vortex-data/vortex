# Releases

Vortex releases are made every two weeks.

Release Drafter maintains draft release notes as changes land on `develop`. Prepare the binaries
before publishing the draft:

1. Choose the draft release's version tag and the full commit SHA to release. The commit must
   include the draft preparation workflow described here.
2. Run the **Release Binaries** workflow from the Actions tab, supplying `tag` and `commit`.
   The workflow requires an existing draft, creates its tag at that commit if needed, and rejects
   an existing tag that points elsewhere. All five platform builds use the same commit: macOS and
   Linux on ARM64 and x86-64, plus Windows on x86-64.
3. Wait for the entire workflow to succeed. It attaches the complete set of binaries to the draft
   only after all builds succeed. Failed runs can be retried while the release remains a draft.
4. Review the draft and publish it manually. This triggers the **Publish** workflow for crates.io,
   PyPI, Maven Central, and compatibility fixtures.

Enable **Settings → General → Releases → Enable release immutability** after this workflow is in
place and any older release binary uploads have finished. GitHub then locks the tag and assets of
future releases at publication. Do not publish a draft before its binary workflow succeeds;
missing or incorrect binaries after publication require a new release version.
