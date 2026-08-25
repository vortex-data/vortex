# DuckDB extension for Vortex

User documentation is on the [docs website](https://docs.vortex.dev/user-guide/duckdb).

## Build

Install `ninja`, `cmake`, and a C++20 compiler.

- For debug builds, launch `VX_DUCKDB_DEBUG=1 cargo build -p vortex-duckdb`.
- For ASAN/TSAN builds, launch `VX_DUCKDB_DEBUG=1 VX_DUCKDB_SAN=1 cargo build -p
  vortex-duckdb`.
- For builds against a custom DuckDB tag or commit, change `build.rs`'s
  `DEFAULT_DUCKDB_VERSION`.

## Test

Our sqllogic tests use a build which links against `libduckdb.so`. This means
you don't get an extension file to load in DuckDB. If you want to test just
sqllogic, run `cargo test -p vortex-sqllogictest --test sqllogictests`. If you
want to test a full setup,

1. Clone [duckdb-vortex](https://github.com/vortex-data/duckdb-vortex).
2. Update duckdb-vortex's submodules. Update vortex reference in duckdb-vortex's
   `vortex-extension/Cargo.toml` with this Vortex checkout.
4. Inside duckdb-vortex, run `make reldebug -j`.

`./build/reldebug/duckdb` will be a DuckDB shell with Vortex linked statically.

## Update extension with same DuckDB version

1. Merge the PR with your changes, if any.
2. Make a new Vortex release.
3. Get the commit of the release, i.e.
   `afb005379dd9a1b7dc7ae2cb49f4f465d91cc2b3` is the commit for
   [0.85](https://github.com/vortex-data/vortex/releases/tag/0.85.0).
4. In duckdb-vortex, update vortex reference in `vortex-extension/Cargo.toml` to
   the release commit hash.
5. Open a PR in duckdb-vortex, check the tests and merge it.
6. Ensure the PR has been backported to the current release branch, i.e. for
   duckdb 1.5.* the release branch is
   [v1.5-variegata](https://github.com/vortex-data/duckdb-vortex/tree/v1.5-variegata).
   This should happen automatically.
7. Notify DuckDB maintainers in the Discord `#spiraldb` channel of the new
   commit and ask to update the extension version. This will likely take some
   days since they need to bump the extension version, trigger a deploy job and
   update the artifacts on their website. Provide the commit in duckdb-vortex's
   release branch, i.e.
   [`5390e62`](https://github.com/vortex-data/duckdb-vortex/commit/5390e6270da964f04afbd6df4a5c9bc83a2e7880).
8. Check `UPDATE EXTENSIONS; INSTALL vortex; LOAD vortex` in your local duckdb
   loads the latest version.

## Update duckdb version

- Update the version in `build.rs`'s `DEFAULT_DUCKDB_VERSION`.
- Create a PR with your changes and ensure it builds. Merge the PR. If your PR
  is for a pre-release build, CI will build DuckDB from source and upload the
  artifacts to R2 so you can reuse them locally.
- Create a new Vortex release.
- In duckdb-vortex, change all occurrences of old version to the new version. See
  [1.5.5](https://github.com/vortex-data/duckdb-vortex/pull/105/changes) upgrade
  as an example.
- In duckdb-vortex, upgrade submodules to the new version.
- Follow from step (3) in extension update plan.

## Build artifacts storage

CI uses duckdb prebuilt (`.so/.a/.dylib`) artifacts from Github releases.
However, release downloads from Github are very unstable, and we're observed a
~25% build rate failure when doing so. Thus we mirror release archives to R2.

A job is triggered on every PR which checks artifacts are present in R2, and if
they aren't, tries to mirror them from DuckDB Github releases via a deployment
request. If `DEFAULT_DUCKDB_VERSION` is a hash and files weren't present, the
job builds DuckDB from source and uploads the artifacts to R2 as well.
