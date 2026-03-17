// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Local dataset operations: init, manifest generation, and validation.

use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use tracing::info;

use super::dataset::DatasetDescriptor;
use super::manifest::Manifest;
use super::remote::build_manifest_from_dir;

/// Initialize a new dataset directory with a template `dataset.yaml` and empty `data/`.
pub fn init(dir: &Path, name: &str) -> Result<()> {
    if dir.exists() {
        bail!("directory already exists: {}", dir.display());
    }

    std::fs::create_dir_all(dir)?;
    std::fs::create_dir_all(dir.join("data"))?;

    // Try to get author from git config.
    let author = git_author().unwrap_or_default();
    let descriptor = DatasetDescriptor::template(name, &author);
    descriptor.write_to_file(dir.join("dataset.yaml"))?;

    info!(name, path = %dir.display(), "initialized dataset");
    Ok(())
}

/// Generate `manifest.json` by scanning the `data/` directory.
pub fn manifest(dir: &Path) -> Result<Manifest> {
    let descriptor = DatasetDescriptor::from_file(dir.join("dataset.yaml"))?;
    let data_dir = dir.join("data");
    if !data_dir.exists() {
        bail!("data/ directory not found in {}", dir.display());
    }

    let manifest = build_manifest_from_dir(&descriptor.name, &data_dir)?;
    let bytes = manifest.to_json()?;
    std::fs::write(dir.join("manifest.json"), &bytes)?;

    info!(
        name = descriptor.name,
        files = manifest.total_files(),
        total_bytes = manifest.total_size_bytes(),
        "generated manifest"
    );

    Ok(manifest)
}

/// Validate a dataset directory: check `dataset.yaml` and that files match manifest.
pub fn validate(dir: &Path) -> Result<Vec<String>> {
    let mut problems = Vec::new();

    // Check dataset.yaml exists and is valid.
    let descriptor_path = dir.join("dataset.yaml");
    if !descriptor_path.exists() {
        problems.push("dataset.yaml not found".to_string());
        return Ok(problems);
    }

    let descriptor = match DatasetDescriptor::from_file(&descriptor_path) {
        Ok(d) => d,
        Err(e) => {
            problems.push(format!("failed to parse dataset.yaml: {e}"));
            return Ok(problems);
        }
    };

    problems.extend(descriptor.validate());

    // Check data/ directory exists.
    let data_dir = dir.join("data");
    if !data_dir.exists() {
        problems.push("data/ directory not found".to_string());
        return Ok(problems);
    }

    // Check data/ has at least one file.
    let has_files = walkdir(data_dir.as_path());
    if !has_files {
        problems.push("data/ directory contains no files".to_string());
    }

    // If manifest.json exists, check it matches current files.
    let manifest_path = dir.join("manifest.json");
    if manifest_path.exists() {
        match (
            Manifest::from_json(&std::fs::read(&manifest_path)?),
            build_manifest_from_dir(&descriptor.name, &data_dir),
        ) {
            (Ok(existing), Ok(current)) => {
                let existing_json = existing.to_json()?;
                let current_json = current.to_json()?;
                if existing_json != current_json {
                    problems
                        .push("manifest.json is stale — run `manifest` to regenerate".to_string());
                }
            }
            (Err(e), _) => {
                problems.push(format!("failed to parse existing manifest.json: {e}"));
            }
            (_, Err(e)) => {
                problems.push(format!("failed to scan data/ directory: {e}"));
            }
        }
    }

    Ok(problems)
}

/// Recursively check if a directory contains at least one file.
fn walkdir(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_file() {
            return true;
        }
        if ft.is_dir() && walkdir(&entry.path()) {
            return true;
        }
    }
    false
}

/// Try to get "Name <email>" from git config.
fn git_author() -> Option<String> {
    let name = std::process::Command::new("git")
        .args(["config", "user.name"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;
    let email = std::process::Command::new("git")
        .args(["config", "user.email"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;
    Some(format!("{name} <{email}>"))
}
