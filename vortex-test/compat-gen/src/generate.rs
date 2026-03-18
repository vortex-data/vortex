// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::all_fixtures;
use crate::manifest::FixtureEntry;

#[derive(Serialize)]
struct FixturesJson {
    fixtures: Vec<FixtureEntry>,
}

/// Write all fixture files into `output_dir`, returning name, description, and sha256 for each.
pub fn write_fixtures(output_dir: &Path, exclude: &[String]) -> VortexResult<Vec<FixtureEntry>> {
    let fixtures = all_fixtures();
    let fixtures: Vec<_> = fixtures
        .into_iter()
        .filter(|f| {
            let name = f.name();
            !exclude.iter().any(|pat| name.contains(pat.as_str()))
        })
        .collect();

    if !exclude.is_empty() {
        eprintln!("excluding: {}", exclude.join(", "));
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create output dir: {e}"))?;

    eprintln!("generating {} fixtures...", fixtures.len());

    let mut infos = Vec::new();
    for fixture in &fixtures {
        let entries = fixture.write(output_dir)?;
        for entry in entries {
            eprintln!("  wrote {}", entry.name);
            infos.push(entry);
        }
    }

    Ok(infos)
}

/// Write the `fixtures.json` manifest from previously collected fixture info.
pub fn write_manifest(output_dir: &Path, infos: Vec<FixtureEntry>) -> VortexResult<()> {
    let fixtures_json = FixturesJson { fixtures: infos };
    let json = serde_json::to_string_pretty(&fixtures_json)
        .map_err(|e| vortex_err!("failed to serialize fixtures.json: {e}"))?;
    std::fs::write(output_dir.join("fixtures.json"), format!("{json}\n"))
        .map_err(|e| vortex_err!("failed to write fixtures.json: {e}"))?;
    eprintln!("  wrote fixtures.json");

    eprintln!(
        "\ndone: {} fixtures in {}",
        fixtures_json.fixtures.len(),
        output_dir.display()
    );
    Ok(())
}

/// Generate all fixtures into `output_dir` and write the manifest.
pub fn generate(output_dir: &Path, exclude: &[String]) -> VortexResult<()> {
    let infos = write_fixtures(output_dir, exclude)?;
    write_manifest(output_dir, infos)
}
