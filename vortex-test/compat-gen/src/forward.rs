// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Forward-compatibility fixtures: files written by the current writer under each frozen
//! edition, for older releases to read.
//!
//! A frozen edition promises that a file written with only its members enabled is readable by
//! every release from the edition's `min_library_version` on. [`generate_forward`] writes every
//! fixture once per frozen `core` edition, with that edition as the only enabled `core`
//! edition and the default write strategy, so the files carry everything a real writer emits:
//! compressed encodings, layouts, zone maps, and file statistics. The orchestrator then reads
//! each file with every released reader the edition covers.
//!
//! Older readers are driven through the published Python wheels, so the expected values are
//! written once per fixture as an Arrow IPC file and compared in Arrow space.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_ipc::writer::FileWriter;
use futures::stream;
use serde::Serialize;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::editions::EditionId;
use vortex::editions::EditionSessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::fixtures::dataset_fixtures;
use crate::fixtures::synthetic_fixtures;

/// The edition family whose frozen editions are checked.
const FAMILY: &str = "core";

/// Directory, relative to the output directory, holding the expected values.
const EXPECTED_DIR: &str = "expected";

/// The manifest written next to the generated files.
const MANIFEST: &str = "forward.json";

#[derive(Serialize)]
struct ForwardManifest {
    fixtures: Vec<ForwardFixture>,
    editions: Vec<ForwardEdition>,
}

#[derive(Serialize)]
struct ForwardFixture {
    /// File name of the fixture inside each edition directory, e.g. `primitives.vortex`.
    name: String,
    description: String,
    /// Path of the Arrow IPC file holding the expected values, relative to the output directory.
    expected: String,
}

#[derive(Serialize)]
struct ForwardEdition {
    /// The edition id, which is also the directory holding its files, e.g. `core2025.05.0`.
    edition: String,
    /// The first release whose reader must decode every file of this edition.
    min_library_version: String,
    /// Fixtures written under this edition.
    written: Vec<String>,
    /// Fixtures this edition cannot represent, with the writer's error.
    unwritable: Vec<UnwritableFixture>,
}

#[derive(Serialize)]
struct UnwritableFixture {
    name: String,
    error: String,
}

/// A fixture's logical contents, independent of how any edition encodes them.
struct ForwardInput {
    name: String,
    description: String,
    array: ArrayRef,
}

/// Write the forward-compatibility fixtures into `output_dir`.
///
/// Layout:
///
/// ```text
/// <output_dir>/
/// ├── forward.json
/// ├── expected/<fixture>.arrow
/// └── <edition>/<fixture>.vortex
/// ```
///
/// A fixture may be unrepresentable in an old edition, e.g. an extension dtype that joined a later
/// edition; it is recorded as unwritable for that edition. Once an edition can write a fixture,
/// every later edition must also be able to, and the newest frozen edition must write all of them.
pub fn generate_forward(output_dir: &Path, exclude: &[String]) -> VortexResult<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    let editions: Vec<(EditionId, String)> = session
        .editions()
        .editions()
        .into_iter()
        .filter(|edition| edition.id.family == FAMILY)
        .filter_map(|edition| {
            edition
                .min_library_version
                .map(|version| (edition.id, version.to_string()))
        })
        .collect();
    if editions.is_empty() {
        vortex_bail!("no frozen {FAMILY} editions are registered");
    }

    let inputs = forward_inputs(&session, exclude, &mut ctx)?;

    std::fs::create_dir_all(output_dir.join(EXPECTED_DIR))
        .map_err(|e| vortex_err!("failed to create {}: {e}", output_dir.display()))?;

    let mut fixtures = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let expected = format!("{EXPECTED_DIR}/{}.arrow", stem(&input.name));
        write_arrow_ipc(&session, &output_dir.join(&expected), &input.array, &mut ctx)?;
        fixtures.push(ForwardFixture {
            name: input.name.clone(),
            description: input.description.clone(),
            expected,
        });
    }

    let mut manifest_editions: Vec<ForwardEdition> = Vec::with_capacity(editions.len());
    for (edition, min_library_version) in editions {
        let edition_dir = output_dir.join(edition.to_string());
        std::fs::create_dir_all(&edition_dir)
            .map_err(|e| vortex_err!("failed to create {}: {e}", edition_dir.display()))?;

        let edition_session = VortexSession::default();
        edition_session.enable_edition(edition)?;

        eprintln!("writing {edition} (min reader {min_library_version})...");
        let mut written = Vec::new();
        let mut unwritable = Vec::new();
        for input in &inputs {
            match write_vortex(&edition_session, input.array.clone()) {
                Ok(bytes) => {
                    let roundtrip = read_vortex(&edition_session, bytes.clone())?;
                    assert_arrays_eq!(roundtrip, input.array.clone(), &mut ctx);
                    std::fs::write(edition_dir.join(&input.name), bytes.as_slice())
                        .map_err(|e| vortex_err!("failed to write {}: {e}", input.name))?;
                    eprintln!("  wrote {edition}/{}", input.name);
                    written.push(input.name.clone());
                }
                Err(error) => {
                    let error = error_message(&error);
                    eprintln!("  unwritable {edition}/{}: {error}", input.name);
                    unwritable.push(UnwritableFixture {
                        name: input.name.clone(),
                        error,
                    });
                }
            }
        }

        if let Some(previous) = manifest_editions.last()
            && let Some(lost) = unwritable
                .iter()
                .find(|fixture| previous.written.contains(&fixture.name))
        {
            vortex_bail!(
                "{} is writable under {} but not under the later edition {edition}: {}",
                lost.name,
                previous.edition,
                lost.error,
            );
        }

        manifest_editions.push(ForwardEdition {
            edition: edition.to_string(),
            min_library_version,
            written,
            unwritable,
        });
    }

    if let Some(newest) = manifest_editions.last()
        && let Some(fixture) = newest.unwritable.first()
    {
        vortex_bail!(
            "the newest frozen edition {} cannot write {}: {}",
            newest.edition,
            fixture.name,
            fixture.error,
        );
    }

    let manifest = ForwardManifest {
        fixtures,
        editions: manifest_editions,
    };
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| vortex_err!("failed to serialize {MANIFEST}: {e}"))?;
    std::fs::write(output_dir.join(MANIFEST), format!("{json}\n"))
        .map_err(|e| vortex_err!("failed to write {MANIFEST}: {e}"))?;

    eprintln!(
        "\ndone: {} fixtures x {} editions in {}",
        manifest.fixtures.len(),
        manifest.editions.len(),
        output_dir.display()
    );
    Ok(())
}

/// Collect the logical contents of every synthetic and dataset fixture.
///
/// Each array is canonicalized so the writer chooses every encoding itself, under the edition it
/// writes with, rather than serializing the encodings a backward-compatibility fixture pins.
fn forward_inputs(
    session: &VortexSession,
    exclude: &[String],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<ForwardInput>> {
    let excluded = |name: &str| exclude.iter().any(|pat| name.contains(pat.as_str()));

    let mut inputs = Vec::new();
    for fixture in synthetic_fixtures() {
        if excluded(fixture.name()) {
            continue;
        }
        inputs.push(ForwardInput {
            name: fixture.name().to_string(),
            description: fixture.description().to_string(),
            array: fixture.build(ctx)?,
        });
    }
    for fixture in dataset_fixtures() {
        let name = format!("{}.vortex", fixture.name());
        if excluded(&name) {
            continue;
        }
        inputs.push(ForwardInput {
            name,
            description: fixture.description().to_string(),
            array: fixture.build(&session.arrow())?,
        });
    }

    for input in &mut inputs {
        input.array = input.array.clone().execute::<Canonical>(ctx)?.into_array();
    }
    Ok(inputs)
}

/// An error's message without the backtrace its display form may carry.
fn error_message(error: &VortexError) -> String {
    let message = error.to_string();
    match message.split_once("\nBacktrace:") {
        Some((message, _)) => message.to_string(),
        None => message,
    }
}

/// Strip the `.vortex` extension from a fixture file name.
fn stem(name: &str) -> &str {
    name.strip_suffix(".vortex").unwrap_or(name)
}

fn runtime() -> VortexResult<Runtime> {
    Runtime::new().map_err(|e| vortex_err!("failed to create tokio runtime: {e}"))
}

/// Write `array` with the session's default write strategy, restricted to its enabled editions.
fn write_vortex(session: &VortexSession, array: ArrayRef) -> VortexResult<ByteBuffer> {
    let stream = ArrayStreamAdapter::new(array.dtype().clone(), stream::iter([Ok(array)]));
    let session = session.clone().with_tokio();
    runtime()?.block_on(async move {
        let mut bytes = Vec::new();
        session.write_options().write(&mut bytes, stream).await?;
        Ok(ByteBuffer::from(bytes))
    })
}

fn read_vortex(session: &VortexSession, bytes: ByteBuffer) -> VortexResult<ArrayRef> {
    let session = session.clone().with_tokio();
    runtime()?.block_on(async move {
        session
            .open_options()
            .open_buffer(bytes)?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await
    })
}

/// Write `array`, a struct array, as a single-batch Arrow IPC file.
fn write_arrow_ipc(
    session: &VortexSession,
    path: &Path,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let arrow = session.arrow().execute_arrow(array.clone(), None, ctx)?;
    let Some(struct_array) = arrow.as_struct_opt() else {
        vortex_bail!(
            "{} must be a struct array to export, got {}",
            path.display(),
            arrow.data_type()
        );
    };
    let batch = RecordBatch::from(struct_array);

    let file =
        File::create(path).map_err(|e| vortex_err!("failed to create {}: {e}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &Arc::clone(batch.schema_ref()))
        .map_err(|e| vortex_err!("failed to start {}: {e}", path.display()))?;
    writer
        .write(&batch)
        .map_err(|e| vortex_err!("failed to write {}: {e}", path.display()))?;
    writer
        .finish()
        .map_err(|e| vortex_err!("failed to finish {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use vortex_error::VortexResult;

    use super::*;

    /// The newest frozen edition must write every synthetic fixture, with its expected values.
    #[test]
    fn newest_edition_writes_every_synthetic_fixture() -> VortexResult<()> {
        let dir = tempfile::tempdir().map_err(|e| vortex_err!("tempdir: {e}"))?;
        // Dataset fixtures generate or download large inputs; the synthetic ones suffice here.
        generate_forward(dir.path(), &["tpch".to_string(), "clickbench".to_string()])?;

        let manifest = std::fs::read(dir.path().join(MANIFEST))
            .map_err(|e| vortex_err!("failed to read {MANIFEST}: {e}"))?;
        let manifest: Value = serde_json::from_slice(&manifest)
            .map_err(|e| vortex_err!("failed to parse {MANIFEST}: {e}"))?;
        let newest = manifest["editions"]
            .as_array()
            .and_then(|editions| editions.last())
            .ok_or_else(|| vortex_err!("{MANIFEST} lists no editions"))?;
        let edition = newest["edition"]
            .as_str()
            .ok_or_else(|| vortex_err!("edition id is not a string"))?;

        let names: Vec<String> = synthetic_fixtures()
            .iter()
            .map(|fixture| fixture.name().to_string())
            .collect();
        let written: Vec<&str> = newest["written"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(written, names);
        for name in &names {
            assert!(dir.path().join(edition).join(name).exists());
            let expected = format!("{}.arrow", stem(name));
            assert!(dir.path().join(EXPECTED_DIR).join(expected).exists());
        }
        Ok(())
    }
}
