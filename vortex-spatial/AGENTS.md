# Spatial Guidance

Applies to files under `vortex-spatial/`, and to Arrow and DataFusion upgrades that touch its
dependencies.

## Arrow upgrades and geoarrow

`vortex-spatial` depends on `geoarrow` and `geoarrow-cast`, which are built for a single Arrow
major version. When an Arrow or DataFusion upgrade moves Vortex to an Arrow version that no
released geoarrow supports, do not publish `vortex-spatial` until one does.

Cargo applies `[patch]` only to builds of this workspace and drops it from published crates. A
published crate whose dependencies rely on a patch therefore fails to build for everyone who
depends on it from crates.io. That is how `vortex-spatial` 0.86.1 to 0.93.0 shipped depending on
geoarrow 0.8.0, which uses arrow 58, while Vortex used arrow 59.

Until geoarrow releases a compatible version:

1. Keep the published requirement in the root `Cargo.toml`, such as `geoarrow = "0.8.0"`, and add
   a `[patch.crates-io]` entry that points `geoarrow`, `geoarrow-array`, `geoarrow-cast` and
   `geoarrow-schema` at a git revision that supports the new Arrow version. Pin a `rev`, and give
   each entry the version that revision declares, such as `version = "=0.8.0"`, which must satisfy
   the workspace requirement.
2. Set `publish = false` in `vortex-spatial/Cargo.toml`, with a comment that names the patch and
   says when to remove it. Releases then skip `vortex-spatial`, which leaves the previous version
   on crates.io in place of a broken new one.
3. Once geoarrow publishes a compatible release, set the requirement to it, remove the
   `[patch.crates-io]` entries and `publish = false`, and update `Cargo.lock`.

The patch keeps every build that runs from the workspace working, both local builds and CI's
build, lint and test jobs, so neither needs further changes. The `Rust publish dry-run` job in
`.github/workflows/publish-dry-runs.yml` builds each published crate from its package, outside the
workspace and without the patch. It fails if `vortex-spatial` is still published while it relies
on the patch.

`publish = false` works only while no published crate depends on `vortex-spatial`. Before setting
it, check that every crate depending on `vortex-spatial` is also `publish = false`. Today those are
`vortex-bench`, `vortex-duckdb`, `vortex-jni`, `xtask` and `fuzz`.
