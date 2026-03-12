// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod adapter;
mod fixtures;
mod manifest;

use std::path::PathBuf;

use chrono::Utc;
use clap::Parser;

use crate::fixtures::all_fixtures;
use crate::manifest::Manifest;

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

fn main() -> vortex_error::VortexResult<()> {
    let cli = Cli::parse();

    std::fs::create_dir_all(&cli.output)
        .map_err(|e| vortex_error::vortex_err!("failed to create output dir: {e}"))?;

    let fixtures = all_fixtures();
    let mut fixture_names = Vec::with_capacity(fixtures.len());

    for fixture in &fixtures {
        let chunks = fixture.build()?;
        let path = cli.output.join(fixture.name());
        adapter::write_file(&path, chunks)?;
        fixture_names.push(fixture.name().to_string());
        eprintln!("  wrote {}", fixture.name());
    }

    let manifest = Manifest {
        version: cli.version.clone(),
        generated_at: Utc::now(),
        fixtures: fixture_names,
    };
    let manifest_path = cli.output.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| vortex_error::vortex_err!("failed to serialize manifest: {e}"))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| vortex_error::vortex_err!("failed to write manifest: {e}"))?;
    eprintln!("  wrote manifest.json");

    eprintln!("done: {} fixtures for v{}", fixtures.len(), cli.version);
    Ok(())
}
