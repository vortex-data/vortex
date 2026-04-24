// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

/// Regenerate `public-api.lock` files for all published crates.
///
/// 1. Runs `cargo +nightly doc` for all published crates to produce rustdoc JSON
/// 2. Feeds each crate's rustdoc JSON into `public_api::Builder`
/// 3. Writes the rendered API to `<crate_path>/public-api.lock`
pub fn public_api() -> anyhow::Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let repo_root = repo_root.canonicalize()?;

    // 1. Use cargo metadata to discover published library crates.
    let metadata = MetadataCommand::new().no_deps().exec()?;

    let published: Vec<_> = metadata
        .workspace_packages()
        .into_iter()
        .filter(|v| is_published(v))
        // Only keep crates that publish Rust libs
        .filter(|p| p.targets.iter().any(|target| target.is_lib()))
        .collect();

    println!("Found {} published crates.", published.len());

    // 2. Generate rustdoc JSON for all published crates in a single parallel invocation.
    //    Uses -p flags (not --workspace) to exclude non-published crates that may
    //    have special build requirements (CUDA, Python, etc.).
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

    // 3. For each published crate, build the public API from JSON and write the lock file.
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
