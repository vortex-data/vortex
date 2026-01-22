// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check for cyclic dependencies in the workspace.
//!
//! Dev-dependencies are allowed to form cycles (this is common for test utilities),
//! but normal dependencies must form a DAG.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

#[derive(Debug)]
struct Package {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    path: String,
    normal_deps: HashSet<String>,
    dev_deps: HashSet<String>,
}

pub fn check_deps() -> Result<()> {
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

    // Build set of workspace members
    let packages = metadata["packages"]
        .as_array()
        .context("Expected packages array")?;

    let workspace_members: HashSet<String> = packages
        .iter()
        .filter_map(|p| p["name"].as_str().map(String::from))
        .collect();

    // Build package info with separated normal and dev deps
    let mut pkg_map: HashMap<String, Package> = HashMap::new();

    for pkg in packages {
        let name = pkg["name"]
            .as_str()
            .context("Package missing name")?
            .to_string();
        let manifest_path = pkg["manifest_path"].as_str().unwrap_or("");
        let path = manifest_path
            .strip_suffix("/Cargo.toml")
            .unwrap_or(manifest_path)
            .to_string();

        let mut normal_deps = HashSet::new();
        let mut dev_deps = HashSet::new();

        if let Some(deps) = pkg["dependencies"].as_array() {
            for dep in deps {
                let dep_name = dep["name"].as_str().unwrap_or("");
                if !workspace_members.contains(dep_name) || dep_name == name {
                    continue;
                }

                let kind = dep["kind"].as_str();
                match kind {
                    None => {
                        // Normal dependency (kind is null in JSON)
                        normal_deps.insert(dep_name.to_string());
                    }
                    Some("dev") => {
                        dev_deps.insert(dep_name.to_string());
                    }
                    Some("build") => {
                        // Build deps are like normal deps for cycle detection
                        normal_deps.insert(dep_name.to_string());
                    }
                    _ => {}
                }
            }
        }

        pkg_map.insert(
            name.clone(),
            Package {
                name,
                path,
                normal_deps,
                dev_deps,
            },
        );
    }

    // Find cycles in normal dependencies using Tarjan's algorithm
    let normal_cycles = find_cycles(&pkg_map, false);
    // Find cycles that involve dev dependencies
    let all_cycles = find_cycles(&pkg_map, true);

    let mut has_errors = false;
    let mut has_warnings = false;

    // Report normal dependency cycles as errors
    for cycle in &normal_cycles {
        if cycle.len() > 1 {
            has_errors = true;
            eprintln!("ERROR: Cyclic normal dependency detected:");
            for pkg in cycle {
                let p = &pkg_map[pkg];
                let cycle_deps: Vec<_> = p
                    .normal_deps
                    .iter()
                    .filter(|d| cycle.contains(*d))
                    .collect();
                eprintln!("  {} -> {:?}", pkg, cycle_deps);
            }
            eprintln!();
        }
    }

    // Find dev-dependency cycles (cycles that exist with dev deps but not without)
    let normal_cycle_set: HashSet<_> = normal_cycles
        .iter()
        .filter(|c| c.len() > 1)
        .flatten()
        .collect();

    for cycle in &all_cycles {
        if cycle.len() > 1 {
            // Check if this cycle is purely dev-dependency based
            let is_new_cycle = !cycle.iter().any(|p| normal_cycle_set.contains(p));
            if is_new_cycle {
                has_warnings = true;
                eprintln!("WARNING: Dev-dependency cycle detected (allowed but noted):");
                for pkg in cycle {
                    let p = &pkg_map[pkg];
                    let cycle_deps: Vec<_> = p
                        .dev_deps
                        .iter()
                        .filter(|d| cycle.contains(*d))
                        .collect();
                    if !cycle_deps.is_empty() {
                        eprintln!("  {} --(dev)--> {:?}", pkg, cycle_deps);
                    }
                }
                eprintln!();
            }
        }
    }

    if has_errors {
        bail!("Cyclic dependencies detected in normal dependencies!");
    }

    if has_warnings {
        eprintln!("Dev-dependency cycles found (these are allowed but may slow builds)");
    } else {
        eprintln!("No dependency cycles found.");
    }

    Ok(())
}

/// Find strongly connected components using Tarjan's algorithm.
/// If `include_dev` is true, dev dependencies are included in the graph.
fn find_cycles(pkg_map: &HashMap<String, Package>, include_dev: bool) -> Vec<Vec<String>> {
    let mut index_counter = 0;
    let mut stack = Vec::new();
    let mut lowlinks: HashMap<String, usize> = HashMap::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut sccs = Vec::new();

    fn strongconnect(
        v: &str,
        pkg_map: &HashMap<String, Package>,
        include_dev: bool,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        lowlinks: &mut HashMap<String, usize>,
        index: &mut HashMap<String, usize>,
        on_stack: &mut HashSet<String>,
        sccs: &mut Vec<Vec<String>>,
    ) {
        index.insert(v.to_string(), *index_counter);
        lowlinks.insert(v.to_string(), *index_counter);
        *index_counter += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        if let Some(pkg) = pkg_map.get(v) {
            let deps: HashSet<_> = if include_dev {
                pkg.normal_deps.union(&pkg.dev_deps).cloned().collect()
            } else {
                pkg.normal_deps.clone()
            };

            for w in deps {
                if !index.contains_key(&w) {
                    strongconnect(
                        &w,
                        pkg_map,
                        include_dev,
                        index_counter,
                        stack,
                        lowlinks,
                        index,
                        on_stack,
                        sccs,
                    );
                    let lw = *lowlinks.get(&w).unwrap();
                    let lv = *lowlinks.get(v).unwrap();
                    lowlinks.insert(v.to_string(), lv.min(lw));
                } else if on_stack.contains(&w) {
                    let iw = *index.get(&w).unwrap();
                    let lv = *lowlinks.get(v).unwrap();
                    lowlinks.insert(v.to_string(), lv.min(iw));
                }
            }
        }

        if lowlinks.get(v) == index.get(v) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for v in pkg_map.keys() {
        if !index.contains_key(v) {
            strongconnect(
                v,
                pkg_map,
                include_dev,
                &mut index_counter,
                &mut stack,
                &mut lowlinks,
                &mut index,
                &mut on_stack,
                &mut sccs,
            );
        }
    }

    sccs
}
