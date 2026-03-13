// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use cargo_metadata::MetadataCommand;
use cargo_metadata::Package;
use xshell::Shell;
use xshell::cmd;

/// Returns true if the package is published to a registry (i.e. `publish` is not disabled).
fn is_published(pkg: &Package) -> bool {
    pkg.publish.as_ref().map(|v| !v.is_empty()).unwrap_or(true)
}

/// Discover all published library crates in the workspace.
fn published_crates() -> anyhow::Result<Vec<Package>> {
    let metadata = MetadataCommand::new().no_deps().exec()?;
    Ok(metadata
        .workspace_packages()
        .into_iter()
        .filter(|v| is_published(v))
        .filter(|p| p.targets.iter().any(|target| target.is_lib()))
        .cloned()
        .collect())
}

/// Get the repo root directory.
fn repo_root() -> anyhow::Result<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    root.canonicalize().map_err(Into::into)
}

/// Parse a `public-api.lock` file's contents into a set of API items.
///
/// Each non-empty line (after stripping blank separator lines) is one API item.
fn parse_lock_items(contents: &str) -> BTreeSet<String> {
    contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

/// Regenerate `public-api.lock` files for all published crates.
///
/// 1. Runs `cargo +nightly doc` for all published crates to produce rustdoc JSON
/// 2. Feeds each crate's rustdoc JSON into `public_api::Builder`
/// 3. Writes the rendered API to `<crate_path>/public-api.lock`
pub fn public_api() -> anyhow::Result<()> {
    let repo_root = repo_root()?;
    let published = published_crates()?;

    println!("Found {} published crates.", published.len());

    // Generate rustdoc JSON for all published crates in a single parallel invocation.
    let sh = Shell::new()?;
    sh.change_dir(&repo_root);
    let pkg_flags: Vec<String> = published
        .iter()
        .flat_map(|c| ["-p".to_string(), c.name.to_string()])
        .collect();
    println!("Generating rustdoc JSON...");
    cmd!(sh, "cargo +nightly doc {pkg_flags...} --no-deps")
        .env("RUSTDOCFLAGS", "-Z unstable-options --output-format json")
        .run()?;

    // For each published crate, build the public API from JSON and write the lock file.
    let doc_dir = repo_root.join("target/doc");
    let mut updated = 0;
    for krate in &published {
        let json_name = krate.name.replace('-', "_");
        let json_path = doc_dir.join(format!("{json_name}.json"));

        if !json_path.exists() {
            anyhow::bail!(
                "rustdoc JSON not found for crate '{}' at {}",
                krate.name,
                json_path.display()
            );
        }

        let api = public_api::Builder::from_rustdoc_json(&json_path)
            .omit_blanket_impls(true)
            .omit_auto_trait_impls(true)
            .build()?;

        // Insert blank lines between items to reduce git conflicts.
        let formatted = api
            .items()
            .map(|item| item.to_string())
            .collect::<Vec<_>>()
            .join("\n\n");

        let crate_dir = krate
            .manifest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no parent dir for {}", krate.manifest_path))?;
        let lock_path = crate_dir.join("public-api.lock");
        fs::write(&lock_path, format!("{formatted}\n"))?;

        updated += 1;
    }

    println!("Done. Updated {updated} public-api.lock files.");
    Ok(())
}

/// Check backward compatibility of the current public API against the latest release.
///
/// Finds the latest semver git tag, reads the `public-api.lock` files from that tag,
/// and compares them against the current on-disk lock files. Any API item that existed
/// in the released version but is missing from the current version is reported as a
/// breaking change.
pub fn check_backcompat() -> anyhow::Result<()> {
    let repo_root = repo_root()?;
    let published = published_crates()?;

    let sh = Shell::new()?;
    sh.change_dir(&repo_root);

    // Find the latest semver git tag.
    let tags_output = cmd!(sh, "git tag --sort=-v:refname").read()?;
    let latest_tag = tags_output
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no git tags found; cannot determine baseline version"))?
        .trim();

    println!("Checking backward compatibility against {latest_tag}...");

    let mut total_removed = 0usize;
    let mut breaking_crates: Vec<(String, Vec<String>)> = Vec::new();

    for krate in &published {
        let crate_dir = krate
            .manifest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no parent dir for {}", krate.manifest_path))?;
        let lock_path = crate_dir.join("public-api.lock");

        // Compute the repo-relative path for `git show`.
        let repo_root_str = repo_root.to_string_lossy();
        let lock_str = lock_path.as_str();
        let relative_str = lock_str
            .strip_prefix(repo_root_str.as_ref())
            .unwrap_or(lock_str)
            .trim_start_matches('/');

        // Read the baseline lock file from the latest tag.
        let git_show_ref = format!("{latest_tag}:{relative_str}");
        let baseline = match cmd!(sh, "git show {git_show_ref}").read() {
            Ok(contents) => contents,
            Err(_) => {
                // Crate is new since the last release — no backcompat concern.
                println!(
                    "  {}: new crate (not in {latest_tag}), skipping",
                    krate.name
                );
                continue;
            }
        };

        // Read the current lock file.
        let current = match fs::read_to_string(&lock_path) {
            Ok(contents) => contents,
            Err(_) => {
                anyhow::bail!(
                    "current public-api.lock not found for '{}' at {}. \
                     Run `cargo xtask public-api` first.",
                    krate.name,
                    lock_path
                );
            }
        };

        let baseline_items = parse_lock_items(&baseline);
        let current_items = parse_lock_items(&current);

        // Items in the baseline but not in the current version are removals.
        let removed: Vec<_> = baseline_items.difference(&current_items).cloned().collect();

        if removed.is_empty() {
            println!("  {}: ok", krate.name);
        } else {
            println!(
                "  {}: {} removed API item(s) (BREAKING)",
                krate.name,
                removed.len()
            );
            total_removed += removed.len();
            breaking_crates.push((krate.name.to_string(), removed));
        }
    }

    if breaking_crates.is_empty() {
        println!("\nNo breaking API changes detected relative to {latest_tag}.");
        Ok(())
    } else {
        eprintln!("\n=== Breaking API changes relative to {latest_tag} ===\n");
        for (crate_name, removed) in &breaking_crates {
            eprintln!("{crate_name}:");
            for item in removed {
                eprintln!("  - {item}");
            }
            eprintln!();
        }
        anyhow::bail!(
            "{total_removed} API item(s) removed across {} crate(s). \
             This is a backward-incompatible change.",
            breaking_crates.len()
        )
    }
}
