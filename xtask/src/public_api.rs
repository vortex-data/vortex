// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::PathBuf;

use xshell::Shell;
use xshell::cmd;

/// A published crate extracted from the workspace Cargo.toml.
struct PublishedCrate {
    name: String,
    path: String,
}

/// Regenerate `public-api.lock` files for all published crates.
///
/// 1. Runs `cargo +nightly doc` for all published crates to produce rustdoc JSON
/// 2. Feeds each crate's rustdoc JSON into `public_api::Builder`
/// 3. Writes the rendered API to `<crate_path>/public-api.lock`
pub fn public_api() -> anyhow::Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let repo_root = repo_root.canonicalize()?;

    // 1. Parse workspace Cargo.toml for published crates.
    let cargo_toml = fs::read_to_string(repo_root.join("Cargo.toml"))?;
    let crates = parse_published_crates(&cargo_toml)?;
    println!("Found {} published crates.", crates.len());

    // 2. Generate rustdoc JSON for all published crates in a single parallel invocation.
    //    Uses -p flags (not --workspace) to exclude non-published crates that may
    //    have special build requirements (CUDA, Python, etc.).
    let sh = Shell::new()?;
    sh.change_dir(&repo_root);
    let pkg_flags: Vec<String> = crates
        .iter()
        .flat_map(|c| ["-p".to_string(), c.name.clone()])
        .collect();
    println!("Generating rustdoc JSON...");
    cmd!(sh, "cargo +nightly doc {pkg_flags...} --no-deps")
        .env("RUSTDOCFLAGS", "-Z unstable-options --output-format json")
        .run()?;

    // 3. For each published crate, build the public API from JSON and write the lock file.
    let doc_dir = repo_root.join("target/doc");
    let mut updated = 0;
    for krate in &crates {
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

        let lock_path = repo_root.join(&krate.path).join("public-api.lock");
        fs::write(&lock_path, format!("{formatted}\n"))?;

        updated += 1;
    }

    println!("Done. Updated {updated} public-api.lock files.");
    Ok(())
}

/// Extract published crate names and paths from workspace Cargo.toml.
///
/// Looks for lines between `# BEGIN crates published` and `# END crates published`
/// markers. Each line has the form:
///   `vortex-alp = { version = "0.1.0", path = "./encodings/alp", ... }`
fn parse_published_crates(cargo_toml: &str) -> anyhow::Result<Vec<PublishedCrate>> {
    let mut in_section = false;
    let mut crates = Vec::new();

    for line in cargo_toml.lines() {
        if line.starts_with("# BEGIN crates published") {
            in_section = true;
            continue;
        }
        if line.starts_with("# END crates published") {
            break;
        }
        if !in_section {
            continue;
        }

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse: name = { ... path = "..." ... }
        let name = line
            .split('=')
            .next()
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("failed to parse crate name from: {line}"))?
            .to_string();

        let path = line
            .find("path")
            .and_then(|i| {
                let rest = &line[i..];
                let start = rest.find('"')? + 1;
                let end = start + rest[start..].find('"')?;
                Some(rest[start..end].to_string())
            })
            .ok_or_else(|| anyhow::anyhow!("failed to parse path from: {line}"))?;

        // Strip leading "./" if present.
        let path = path.strip_prefix("./").unwrap_or(&path).to_string();

        crates.push(PublishedCrate { name, path });
    }

    if crates.is_empty() {
        anyhow::bail!("no published crates found between BEGIN/END markers in Cargo.toml");
    }

    Ok(crates)
}
