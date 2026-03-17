// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::PathBuf;

use chrono::Utc;
use clap::Parser;
use vortex_compat::fixtures::all_fixtures;
use vortex_compat::manifest::Manifest;
use vortex_error::VortexResult;

#[derive(Parser)]
#[command(
    name = "compat-gen",
    about = "Generate Vortex backward-compat fixture files"
)]
struct Cli {
    /// Version tag for this fixture set (e.g. "0.62.0").
    #[arg(long)]
    version: String,

    /// Output directory for generated fixture files.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    if cli.output.exists() {
        let is_empty = cli
            .output
            .read_dir()
            .map_err(|e| vortex_error::vortex_err!("failed to read output dir: {e}"))?
            .next()
            .is_none();
        if !is_empty {
            vortex_error::vortex_bail!(
                "output directory '{}' is not empty; use a fresh directory",
                cli.output.display()
            );
        }
    } else {
        fs::create_dir_all(&cli.output)
            .map_err(|e| vortex_error::vortex_err!("failed to create output dir: {e}"))?;
    }

    let fixtures = all_fixtures();
    let mut entries = Vec::new();

    for fixture in &fixtures {
        let new_entries = fixture.write(&cli.output)?;
        for entry in &new_entries {
            eprintln!("  generated file: {}", entry.name);
        }
        entries.extend(new_entries);
    }

    let manifest = Manifest {
        version: cli.version.clone(),
        generated_at: Utc::now(),
        fixtures: entries,
    };
    let manifest_path = cli.output.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| vortex_error::vortex_err!("failed to serialize manifest: {e}"))?;
    fs::write(&manifest_path, manifest_json)
        .map_err(|e| vortex_error::vortex_err!("failed to write manifest: {e}"))?;
    eprintln!("  wrote manifest.json");

    eprintln!(
        "done: {} fixtures for v{}",
        manifest.fixtures.len(),
        cli.version
    );
    Ok(())
}
