// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generate optimal workspace member ordering for cargo-hack.
//!
//! Orders packages by topological sort, prioritizing packages with more dependents
//! at each level to maximize Cargo cache reuse when using tools like `cargo-hack`.
//! which executes commands per crate by their order in the members array.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

#[derive(Debug, Clone)]
struct Package {
    #[expect(dead_code)]
    name: String,
    path: String,
    normal_deps: HashSet<String>,
}

pub fn sort_workspace(check_only: bool) -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Failed to find workspace root")?
        .to_path_buf();

    // Run cargo metadata to get dependency information
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&workspace_root)
        .output()
        .context("Failed to run cargo metadata")?;

    if !output.status.success() {
        bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("Failed to parse cargo metadata")?;

    let packages = metadata["packages"]
        .as_array()
        .context("Expected packages array")?;

    let workspace_members: HashSet<String> = packages
        .iter()
        .filter_map(|p| p["name"].as_str().map(String::from))
        .collect();

    // Build package info
    let mut pkg_map: HashMap<String, Package> = HashMap::new();
    let mut reverse_deps: HashMap<String, HashSet<String>> = HashMap::new();

    for name in &workspace_members {
        reverse_deps.insert(name.clone(), HashSet::new());
    }

    for pkg in packages {
        let name = pkg["name"]
            .as_str()
            .context("Package missing name")?
            .to_string();
        let manifest_path = pkg["manifest_path"].as_str().unwrap_or("");

        // Extract relative path from workspace root
        let path = if let Some(idx) = manifest_path.find("/vortex/") {
            manifest_path[idx + 8..]
                .strip_suffix("/Cargo.toml")
                .unwrap_or_else(|| &manifest_path[idx + 8..])
                .to_string()
        } else {
            name.clone()
        };

        let mut normal_deps = HashSet::new();

        if let Some(deps) = pkg["dependencies"].as_array() {
            for dep in deps {
                let dep_name = dep["name"].as_str().unwrap_or("");
                if !workspace_members.contains(dep_name) || dep_name == name {
                    continue;
                }

                // Only consider normal dependencies (kind is null in JSON)
                if dep["kind"].is_null() {
                    normal_deps.insert(dep_name.to_string());
                    reverse_deps.get_mut(dep_name).unwrap().insert(name.clone());
                }
            }
        }

        pkg_map.insert(
            name.clone(),
            Package {
                name,
                path,
                normal_deps,
            },
        );
    }

    // Topological sort using Kahn's algorithm
    // At each step, pick the package with the most dependents (reverse deps)
    let mut in_degree: HashMap<String, usize> = pkg_map
        .iter()
        .map(|(name, pkg)| (name.clone(), pkg.normal_deps.len()))
        .collect();

    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(name, _)| name.clone())
        .collect();

    let mut sorted_pkgs: Vec<String> = Vec::new();

    while !queue.is_empty() {
        // Sort queue by number of reverse deps (descending), then by name (ascending) for stability
        let mut queue_vec: Vec<_> = queue.drain(..).collect();
        queue_vec.sort_by(|a, b| {
            let ra = reverse_deps.get(a).map(|s| s.len()).unwrap_or(0);
            let rb = reverse_deps.get(b).map(|s| s.len()).unwrap_or(0);
            match rb.cmp(&ra) {
                std::cmp::Ordering::Equal => a.cmp(b), // alphabetical for stability
                other => other,
            }
        });

        let pkg = queue_vec.remove(0);
        queue.extend(queue_vec);

        sorted_pkgs.push(pkg.clone());

        if let Some(dependents) = reverse_deps.get(&pkg) {
            for dependent in dependents {
                let deg = in_degree.get_mut(dependent).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(dependent.clone());
                }
            }
        }
    }

    if sorted_pkgs.len() != pkg_map.len() {
        bail!(
            "Cycle detected! Only sorted {} of {} packages",
            sorted_pkgs.len(),
            pkg_map.len()
        );
    }

    // Categorize packages
    let mut core_pkgs = Vec::new();
    let mut encoding_pkgs = Vec::new();
    let mut benchmark_pkgs = Vec::new();

    for name in &sorted_pkgs {
        let pkg = &pkg_map[name];
        if pkg.path.starts_with("encodings/") {
            encoding_pkgs.push(pkg.clone());
        } else if pkg.path.starts_with("benchmarks/") {
            benchmark_pkgs.push(pkg.clone());
        } else {
            core_pkgs.push(pkg.clone());
        }
    }

    // Generate the members list
    let mut members = Vec::new();
    members.push("    # Core crates in dependency order".to_string());
    for pkg in &core_pkgs {
        members.push(format!("    \"{}\",", pkg.path));
    }
    members.push("    # Encodings".to_string());
    for pkg in &encoding_pkgs {
        members.push(format!("    \"{}\",", pkg.path));
    }
    members.push("    # Benchmarks".to_string());
    for pkg in &benchmark_pkgs {
        members.push(format!("    \"{}\",", pkg.path));
    }

    let new_members_block = members.join("\n");

    // Read current Cargo.toml
    let cargo_toml_path = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml_path).context("Failed to read Cargo.toml")?;

    // Parse and update the members section
    let new_content = update_members_section(&content, &new_members_block)?;

    if check_only {
        if content == new_content {
            println!("Workspace members are in optimal order.");
            return Ok(());
        } else {
            // Show diff
            println!("Workspace members are NOT in optimal order.");
            println!("Run `cargo xtask sort-workspace` to fix.");
            println!("\nExpected order:");
            println!("[workspace]");
            println!("members = [");
            println!("{}", new_members_block);
            println!("]");
            bail!("Workspace members need reordering");
        }
    }

    if content == new_content {
        println!("Workspace members are already in optimal order.");
        return Ok(());
    }

    fs::write(&cargo_toml_path, &new_content).context("Failed to write Cargo.toml")?;
    println!("Updated Cargo.toml with optimal member ordering.");
    println!("\nNew order:");
    println!("[workspace]");
    println!("members = [");
    println!("{}", new_members_block);
    println!("]");

    Ok(())
}

fn update_members_section(content: &str, new_members: &str) -> Result<String> {
    // Find the members = [ ... ] section and replace it
    let members_start = content
        .find("members = [")
        .context("Could not find 'members = [' in Cargo.toml")?;

    // Find the matching closing bracket
    let after_start = &content[members_start..];
    let bracket_start = after_start.find('[').unwrap();
    let mut depth = 0;
    let mut bracket_end = None;

    for (i, c) in after_start[bracket_start..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    bracket_end = Some(bracket_start + i);
                    break;
                }
            }
            _ => {}
        }
    }

    let bracket_end = bracket_end.context("Could not find closing ']' for members")?;

    let before = &content[..members_start];
    let after = &content[members_start + bracket_end + 1..];

    Ok(format!(
        "{}members = [\n{}\n]{}",
        before, new_members, after
    ))
}
