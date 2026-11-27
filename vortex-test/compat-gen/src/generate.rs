// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::StructArray as ArrowStructArray;
use arrow_ipc::writer::FileWriter as ArrowIpcFileWriter;
use arrow_schema::Field as ArrowField;
use arrow_schema::Schema as ArrowSchema;
use base16ct::HexDisplay;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex::VortexSessionDefault;
use vortex::editions::ComponentKind;
use vortex::editions::EditionId;
use vortex::editions::EditionSessionExt;
use vortex_array::ArrayId;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::session::ArraySessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::adapter;
use crate::fixtures::all_fixtures;

#[derive(Serialize, Deserialize)]
struct FixturesJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    edition: Option<String>,
    fixtures: Vec<FixtureInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    skipped: Vec<SkippedFixture>,
}

#[derive(Serialize, Deserialize)]
pub struct FixtureInfo {
    pub name: String,
    pub description: String,
    pub sha256: String,
}

#[derive(Serialize, Deserialize)]
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

/// Write expected Arrow IPC files from HEAD for every non-excluded fixture.
///
/// Each fixture is written via `write_expected()` and decoded with HEAD's reader so oldest-reader
/// checks compare historical wheels against HEAD's own oracle rather than another historical wheel.
pub fn dump_expected(output_dir: &Path, exclude: &[String]) -> VortexResult<()> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create {}: {e}", output_dir.display()))?;

    let fixtures: Vec<_> = all_fixtures()
        .into_iter()
        .filter(|f| !exclude.iter().any(|pat| f.name().contains(pat.as_str())))
        .collect();

    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    let tmp_dir = tempfile::tempdir()?;

    eprintln!(
        "dumping expected Arrow IPC for {} fixtures...",
        fixtures.len()
    );
    for fixture in &fixtures {
        let entries = fixture.write_expected(tmp_dir.path(), &mut ctx)?;
        for entry in entries {
            let bytes = ByteBuffer::from(std::fs::read(tmp_dir.path().join(&entry.name))?);
            let array = adapter::read_file(bytes)?;
            let arrow_array = array.execute_arrow(None, &mut ctx)?;
            let batch =
                if let Some(struct_arr) = arrow_array.as_any().downcast_ref::<ArrowStructArray>() {
                    RecordBatch::from(struct_arr)
                } else {
                    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "value",
                        arrow_array.data_type().clone(),
                        true,
                    )]));
                    RecordBatch::try_new(schema, vec![arrow_array]).map_err(|e| {
                        vortex_err!("failed to build RecordBatch for {}: {e}", entry.name)
                    })?
                };
            let out_path = output_dir.join(format!("{}.arrow", entry.name));
            let file = File::create(&out_path)?;
            let mut writer = ArrowIpcFileWriter::try_new(file, &batch.schema())
                .map_err(|e| vortex_err!("failed to create IPC writer for {}: {e}", entry.name))?;
            writer
                .write(&batch)
                .map_err(|e| vortex_err!("failed to write IPC batch for {}: {e}", entry.name))?;
            writer
                .finish()
                .map_err(|e| vortex_err!("failed to finish IPC file for {}: {e}", entry.name))?;
            eprintln!("  dumped {}.arrow", entry.name);
        }
    }
    Ok(())
}

fn encoding_supported_in_edition(
    session: &VortexSession,
    allowed: &BTreeSet<String>,
    id: &ArrayId,
) -> bool {
    if let Some(plugin) = session.arrays().registry().get(id) {
        plugin
            .write_ids()
            .iter()
            .any(|wire_id| allowed.contains(wire_id.as_ref()))
    } else {
        allowed.contains(id.as_ref())
    }
}

/// Write all fixture files into `output_dir`, returning written fixtures and any encoding-specific
/// fixtures skipped up front because none of their serializer's `write_ids()` are members of
/// `edition`.
///
/// Any write error on a fixture whose required encodings are all permitted by `edition` (including
/// a disallowed encoding emitted by the writer) fails generation.
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

    let prev_manifest_files: Vec<String> =
        std::fs::read_to_string(output_dir.join("fixtures.json"))
            .ok()
            .and_then(|raw| serde_json::from_str::<FixturesJson>(&raw).ok())
            .map(|prev| prev.fixtures.into_iter().map(|f| f.name).collect())
            .unwrap_or_default();

    let allowed_arrays: Option<BTreeSet<String>> = edition.map(|_| {
        ctx.session()
            .enabled_component_ids(ComponentKind::Array)
            .into_iter()
            .map(|id| id.as_str().to_string())
            .collect()
    });

    eprintln!("generating {} fixtures...", fixtures.len());

    let mut infos = Vec::new();
    let mut skipped = Vec::new();
    for fixture in &fixtures {
        if let (Some(edition_name), Some(allowed)) = (edition, &allowed_arrays) {
            let missing: Vec<String> = fixture
                .required_encodings()
                .into_iter()
                .filter(|id| !encoding_supported_in_edition(ctx.session(), allowed, id))
                .map(|id| id.to_string())
                .collect();
            if !missing.is_empty() {
                let name = fixture.name().to_string();
                let reason = format!(
                    "required encoding(s) not in {edition_name}: {}",
                    missing.join(", ")
                );
                eprintln!("  skip {name} ({reason})");
                skipped.push(SkippedFixture { name, reason });
                continue;
            }
        }

        let entries = fixture.write(output_dir, ctx)?;
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

    let written_names: BTreeSet<&str> = infos.iter().map(|info| info.name.as_str()).collect();
    for stale_name in prev_manifest_files {
        if !written_names.contains(stale_name.as_str()) {
            let stale_path = output_dir.join(&stale_name);
            if stale_path.is_file() {
                std::fs::remove_file(&stale_path).map_err(|e| {
                    vortex_err!("failed to remove stale {}: {e}", stale_path.display())
                })?;
            }
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

    #[test]
    fn generate_reusing_dir_removes_stale_manifest_files_only() -> VortexResult<()> {
        let dir = tempfile::tempdir()?;
        let exclude = vec!["tpch".to_string(), "clickbench".to_string()];
        let unrelated = dir.path().join("notes.txt");
        std::fs::write(&unrelated, "keep me")?;

        generate(dir.path(), &exclude, None)?;
        assert!(dir.path().join("pco.vortex").exists());

        generate(dir.path(), &exclude, Some("core2025.05.0"))?;
        assert!(!dir.path().join("pco.vortex").exists());
        assert!(dir.path().join("primitives.vortex").exists());
        assert!(unrelated.exists());
        Ok(())
    }
}
