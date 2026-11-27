// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use base16ct::HexDisplay;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex::VortexSessionDefault;
use vortex::editions::EditionId;
use vortex::editions::EditionSessionExt;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::fixtures::all_fixtures;

#[derive(Serialize)]
struct FixturesJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    edition: Option<String>,
    fixtures: Vec<FixtureInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    skipped: Vec<SkippedFixture>,
}

#[derive(Serialize)]
pub struct FixtureInfo {
    pub name: String,
    pub description: String,
    pub sha256: String,
}

#[derive(Serialize)]
pub struct SkippedFixture {
    pub name: String,
    pub reason: String,
}

#[derive(Serialize)]
pub struct EditionRecord {
    pub id: String,
    pub min_library_version: String,
}

/// List all frozen `core` editions in chronological order, along with their minimum reader version.
pub fn list_core_editions() -> VortexResult<()> {
    let session = VortexSession::default();
    let mut editions: Vec<_> = session
        .editions()
        .editions()
        .into_iter()
        .filter(|e| e.id.family == "core" && e.min_library_version.is_some())
        .collect();
    editions.sort_by_key(|e| e.id);

    let records: Vec<EditionRecord> = editions
        .into_iter()
        .map(|e| EditionRecord {
            id: e.id.to_string(),
            min_library_version: e.min_library_version.unwrap_or_default().to_string(),
        })
        .collect();

    let json = serde_json::to_string_pretty(&records)
        .map_err(|e| vortex_err!("failed to serialize editions: {e}"))?;
    println!("{json}");
    Ok(())
}

/// Resolve an edition name (e.g. `core2025.05.0`) against the session's registered editions.
pub fn resolve_edition(session: &VortexSession, name: &str) -> VortexResult<EditionId> {
    if let Some(edition) = session.editions().find_by_name(name) {
        return Ok(edition.id);
    }

    let available: Vec<String> = session
        .editions()
        .editions()
        .into_iter()
        .map(|e| e.id.to_string())
        .collect();
    vortex_bail!(
        "unknown edition {name:?} (available: {})",
        available.join(", ")
    );
}

fn is_edition_unsupported_error(err: &vortex_error::VortexError) -> bool {
    err.to_string().contains("not permitted by ctx")
}

/// Write all fixture files into `output_dir`, returning written fixtures and any encoding-specific
/// fixtures skipped because their serialized wire ID is not permitted by `edition`.
pub fn write_fixtures(
    output_dir: &Path,
    exclude: &[String],
    edition: Option<&str>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<FixtureInfo>, Vec<SkippedFixture>)> {
    let fixtures = all_fixtures();
    let fixtures: Vec<_> = fixtures
        .into_iter()
        .filter(|f| {
            let excluded = exclude.iter().any(|pat| f.name().contains(pat.as_str()));
            if excluded {
                eprintln!("  skip {} (excluded)", f.name());
            }
            !excluded
        })
        .collect();

    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create {}: {e}", output_dir.display()))?;

    eprintln!("generating {} fixtures...", fixtures.len());

    let mut infos = Vec::new();
    let mut skipped = Vec::new();
    for fixture in &fixtures {
        let entries = match fixture.write(output_dir, ctx) {
            Ok(entries) => entries,
            Err(err)
                if edition.is_some()
                    && !fixture.required_encodings().is_empty()
                    && is_edition_unsupported_error(&err) =>
            {
                let name = fixture.name().to_string();
                drop(std::fs::remove_file(output_dir.join(&name)));
                let reason = err
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_string();
                eprintln!("  skip {name} ({reason})");
                skipped.push(SkippedFixture { name, reason });
                continue;
            }
            Err(err) => return Err(err),
        };
        for entry in entries {
            let path = output_dir.join(&entry.name);
            let file_bytes = std::fs::read(&path)
                .map_err(|e| vortex_err!("failed to read {}: {e}", path.display()))?;
            let digest = Sha256::digest(&file_bytes);
            let sha256 = format!("{:x}", HexDisplay(&digest));
            eprintln!("  wrote {} ({} bytes)", entry.name, file_bytes.len());
            infos.push(FixtureInfo {
                name: entry.name,
                description: entry.description,
                sha256,
            });
        }
    }

    Ok((infos, skipped))
}

/// Write the `fixtures.json` manifest from previously collected fixture info.
pub fn write_manifest(
    output_dir: &Path,
    edition: Option<&str>,
    infos: Vec<FixtureInfo>,
    skipped: Vec<SkippedFixture>,
) -> VortexResult<()> {
    let fixtures_json = FixturesJson {
        edition: edition.map(ToOwned::to_owned),
        fixtures: infos,
        skipped,
    };
    let json = serde_json::to_string_pretty(&fixtures_json)
        .map_err(|e| vortex_err!("failed to serialize fixtures.json: {e}"))?;
    std::fs::write(output_dir.join("fixtures.json"), format!("{json}\n"))
        .map_err(|e| vortex_err!("failed to write fixtures.json: {e}"))?;
    eprintln!("  wrote fixtures.json");

    eprintln!(
        "\ndone: {} fixtures written, {} skipped in {}",
        fixtures_json.fixtures.len(),
        fixtures_json.skipped.len(),
        output_dir.display()
    );
    Ok(())
}

/// Generate all fixtures into `output_dir` and write the manifest.
pub fn generate(output_dir: &Path, exclude: &[String], edition: Option<&str>) -> VortexResult<()> {
    let session = VortexSession::default();
    if let Some(edition_name) = edition {
        let edition_id = resolve_edition(&session, edition_name)?;
        session.enable_edition(edition_id)?;
        eprintln!("enabled edition: {edition_id}");
    }
    let mut ctx = session.create_execution_ctx();
    let (infos, skipped) = write_fixtures(output_dir, exclude, edition, &mut ctx)?;
    write_manifest(output_dir, edition, infos, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_edition_finds_known_and_rejects_unknown() -> VortexResult<()> {
        let session = VortexSession::default();
        let resolved = resolve_edition(&session, "core2025.05.0")?;
        assert_eq!(resolved.to_string(), "core2025.05.0");
        assert!(resolve_edition(&session, "core1999.01.0").is_err());
        Ok(())
    }

    #[test]
    fn generate_with_old_edition_writes_v1_and_skips_unsupported() -> VortexResult<()> {
        let dir = tempfile::tempdir()?;
        let exclude = vec!["tpch".to_string(), "clickbench".to_string()];
        generate(dir.path(), &exclude, Some("core2025.05.0"))?;

        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("fixtures.json"))?)
                .map_err(|e| vortex_err!("failed to parse fixtures.json: {e}"))?;
        assert_eq!(manifest["edition"], "core2025.05.0");

        let written: Vec<&str> = manifest["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["name"].as_str())
            .collect();
        let skipped: Vec<&str> = manifest["skipped"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["name"].as_str())
            .collect();

        assert!(written.contains(&"primitives.vortex"));
        assert!(written.contains(&"bitpacked.vortex"));
        assert!(written.contains(&"for.vortex"));
        assert!(skipped.contains(&"map.vortex"));
        assert!(skipped.contains(&"pco.vortex"));
        assert!(skipped.contains(&"rle.vortex"));
        assert!(skipped.contains(&"zstd.vortex"));
        assert!(!dir.path().join("map.vortex").exists());
        Ok(())
    }
}
