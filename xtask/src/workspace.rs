// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Locating the schemas owned by workspace crates.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;

/// The schemas owned by one workspace crate.
pub struct Schemas {
    pub crate_dir: PathBuf,
    pub schemas: Vec<PathBuf>,
}

/// Every workspace member with a `schema_dir` directory, with the `.<extension>` files under it.
pub fn schema_owners(schema_dir: &str, extension: &str) -> anyhow::Result<Vec<Schemas>> {
    let root = workspace_root()?;
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
        let dir = crate_dir.join(schema_dir);
        if !dir.is_dir() {
            continue;
        }
        let mut schemas = Vec::new();
        collect_with_extension(&dir, extension, &mut schemas)?;
        if !schemas.is_empty() {
            schemas.sort();
            owners.push(Schemas { crate_dir, schemas });
        }
    }
    Ok(owners)
}

/// Appends every file under `dir` with the given extension to `files`.
pub fn collect_with_extension(
    dir: &Path,
    extension: &str,
    files: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_with_extension(&path, extension, files)?;
        } else if path.extension().is_some_and(|ext| ext == extension) {
            files.push(path);
        }
    }
    Ok(())
}

pub fn workspace_root() -> anyhow::Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask lives directly under the workspace root")
}
