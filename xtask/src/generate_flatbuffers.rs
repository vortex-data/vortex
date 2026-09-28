// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regenerates the Rust bindings for the FlatBuffers schemas owned by the Vortex crates.
//!
//! Every workspace member with a `flatbuffers` directory owns the `.fbs` schemas under it, and
//! gets their bindings checked in under `src/flatbuffers/generated/`, so building the crates needs
//! no `flatc`. CI regenerates them with the pinned `flatc` and fails when the checked-in copies
//! differ.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;

/// The `flatc` release the checked-in bindings are generated with. Generated code is not
/// source-compatible across `flatc` releases, so regeneration insists on this exact version.
///
/// Keep in sync with the default `flatc_version` of `.github/actions/setup-flatc`.
pub const FLATC_VERSION: &str = "25.12.19";

/// Module path, relative to the crate root, where generated code looks for the schemas it
/// includes from other crates. Crates with cross-crate includes define it by hand; see
/// `vortex-ipc/src/flatbuffers.rs`.
const INCLUDE_PREFIX: &str = "flatbuffers::deps";

/// Directory under each crate root holding its schemas.
const SCHEMA_DIR: &str = "flatbuffers";

const HEADER: &str = "\
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Regenerate with `cargo run -p xtask -- generate-flatbuffers`.

";

/// The FlatBuffers schemas owned by one workspace crate.
struct Schemas {
    crate_dir: PathBuf,
    schemas: Vec<PathBuf>,
}

pub fn generate_flatbuffers() -> anyhow::Result<()> {
    let flatc = env::var_os("FLATC").map_or_else(|| PathBuf::from("flatc"), PathBuf::from);
    check_version(&flatc)?;

    let root = workspace_root()?;
    let owners = schema_owners(&root)?;
    if owners.is_empty() {
        bail!("no `<crate>/flatbuffers/**/*.fbs` schemas found among the workspace members");
    }

    for owner in &owners {
        let out_dir = owner.crate_dir.join("src/flatbuffers/generated");
        fs::create_dir_all(&out_dir)
            .with_context(|| format!("failed to create {}", out_dir.display()))?;

        let mut command = Command::new(&flatc);
        command
            .arg("--rust")
            // Vortex modules are named for the schema, so drop flatc's `_generated` suffix.
            .args(["--filename-suffix", ""])
            .args(["--include-prefix", INCLUDE_PREFIX])
            .arg("-o")
            .arg(&out_dir);
        // Schemas include other crates' schemas by their path under that crate's `flatbuffers`
        // directory, so every crate's directory is an include root.
        for other in &owners {
            command.arg("-I").arg(other.crate_dir.join(SCHEMA_DIR));
        }
        command.args(&owner.schemas);

        let status = command
            .status()
            .with_context(|| format!("failed to run {}", flatc.display()))?;
        if !status.success() {
            bail!("{} failed with {status}", flatc.display());
        }

        for schema in &owner.schemas {
            let name = schema
                .file_stem()
                .with_context(|| format!("schema has no file name: {}", schema.display()))?;
            let generated = out_dir.join(name).with_extension("rs");
            let code = fs::read_to_string(&generated)
                .with_context(|| format!("flatc did not write {}", generated.display()))?;
            fs::write(&generated, format!("{HEADER}{code}"))
                .with_context(|| format!("failed to write {}", generated.display()))?;
            println!("wrote {}", generated.display());
        }
    }
    Ok(())
}

/// Every workspace member with a `flatbuffers` directory, with the `.fbs` files under it.
fn schema_owners(root: &Path) -> anyhow::Result<Vec<Schemas>> {
    let manifest = root.join("Cargo.toml");
    let manifest: toml::Table = toml::from_str(
        &fs::read_to_string(&manifest)
            .with_context(|| format!("failed to read {}", manifest.display()))?,
    )
    .with_context(|| format!("failed to parse {}", manifest.display()))?;
    let members = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .context("Cargo.toml has no `workspace.members`")?;

    let mut owners = Vec::new();
    for member in members {
        let member = member
            .as_str()
            .with_context(|| format!("workspace member is not a path: {member}"))?;
        let crate_dir = root.join(member);
        let schema_dir = crate_dir.join(SCHEMA_DIR);
        if !schema_dir.is_dir() {
            continue;
        }
        let mut schemas = Vec::new();
        collect_schemas(&schema_dir, &mut schemas)?;
        if !schemas.is_empty() {
            schemas.sort();
            owners.push(Schemas { crate_dir, schemas });
        }
    }
    Ok(owners)
}

fn collect_schemas(dir: &Path, schemas: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_schemas(&path, schemas)?;
        } else if path.extension().is_some_and(|ext| ext == "fbs") {
            schemas.push(path);
        }
    }
    Ok(())
}

fn check_version(flatc: &Path) -> anyhow::Result<()> {
    let output = Command::new(flatc)
        .arg("--version")
        .output()
        .with_context(|| {
            format!(
                "failed to run {}. Install flatc {FLATC_VERSION} from \
             https://github.com/google/flatbuffers/releases, or set FLATC to its location.",
                flatc.display()
            )
        })?;
    let version = String::from_utf8_lossy(&output.stdout);
    let expected = format!("flatc version {FLATC_VERSION}");
    if version.trim() != expected {
        bail!(
            "{} reports `{}`, but the checked-in bindings are generated with `{expected}`. \
             Install that release from https://github.com/google/flatbuffers/releases, or set \
             FLATC to its location.",
            flatc.display(),
            version.trim()
        );
    }
    Ok(())
}

fn workspace_root() -> anyhow::Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask lives directly under the workspace root")
}
