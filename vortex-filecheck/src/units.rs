// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flattening a Vortex file into text units, one per flat layout.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context as _;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use vortex::array::ArrayRef;
use vortex::array::display::BufferExtractor;
use vortex::array::display::EncodingSummaryExtractor;
use vortex::array::display::MetadataExtractor;
use vortex::array::display::NbytesExtractor;
use vortex::array::display::StatsExtractor;
use vortex::array::serde::SerializedArray;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
#[cfg(not(target_arch = "wasm32"))]
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutRef;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::segments::SegmentSource;
use vortex::session::VortexSession;

#[cfg(not(target_arch = "wasm32"))]
use crate::check::CheckFile;
use crate::check::Glob;
#[cfg(not(target_arch = "wasm32"))]
use crate::check::SectionOutcome;
#[cfg(not(target_arch = "wasm32"))]
use crate::check::run_checks;

/// The path of the only unit of a file whose root layout is flat.
const ROOT_PATH: &str = "<root>";

/// How many segments to fetch and decode concurrently.
const FETCH_CONCURRENCY: usize = 16;

/// Which details to render for each array node.
#[derive(Debug, Clone, Copy, Default)]
pub struct DumpOptions {
    /// Include array metadata.
    pub metadata: bool,
    /// Include array sizes.
    pub nbytes: bool,
    /// Include array buffers.
    pub buffers: bool,
    /// Include array statistics.
    pub stats: bool,
}

/// One flat layout of a file, rendered as text.
#[derive(Debug, Clone)]
pub struct Unit {
    /// Path of the unit in the layout tree, such as `l_comment/data[3]`.
    pub path: String,
    /// The rendered unit: a `unit:` header line followed by the array encoding tree.
    pub text: String,
}

/// Check the Vortex file at `path` against the sections of a parsed check file.
///
/// Units whose path matches one of `excludes` are skipped without being read.
///
/// Call [`SectionOutcome::passed`] on each outcome, or [`crate::format_report`] to describe
/// them.
///
/// # Errors
///
/// Returns an error if the file cannot be read or a directive is malformed. Failing checks
/// are reported in the outcomes, not as an error.
#[cfg(not(target_arch = "wasm32"))]
pub async fn check_vortex_file(
    session: &VortexSession,
    path: &Path,
    check: &CheckFile,
    options: DumpOptions,
    excludes: &[Glob],
) -> anyhow::Result<Vec<SectionOutcome>> {
    let units = load_units(session, path, options, excludes).await?;
    run_checks(&units, &check.sections)
}

/// Open the Vortex file at `path` and collect its units, minus those matching `excludes`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or a segment cannot be decoded.
#[cfg(not(target_arch = "wasm32"))]
pub async fn load_units(
    session: &VortexSession,
    path: &Path,
    options: DumpOptions,
    excludes: &[Glob],
) -> anyhow::Result<Vec<Unit>> {
    let file = session
        .open_options()
        .open_path(path)
        .await
        .with_context(|| format!("opening {}", path.display()))?;
    collect_units(session, &file, options, excludes)
        .await
        .with_context(|| format!("reading {}", path.display()))
}

/// Decode every flat layout of `file`, except those whose path matches one of `excludes`, and
/// render it as a [`Unit`], in layout order.
///
/// The segment of every included unit is read, but no array is decompressed. At most a few
/// segments are held in memory at once.
///
/// # Errors
///
/// Returns an error if a segment cannot be read or decoded.
pub async fn collect_units(
    session: &VortexSession,
    file: &VortexFile,
    options: DumpOptions,
    excludes: &[Glob],
) -> VortexResult<Vec<Unit>> {
    let mut flats = Vec::new();
    collect_flat_layouts(file.footer().layout(), "", "", &mut flats)?;
    flats.retain(|flat| !excludes.iter().any(|glob| glob.matches(&flat.path)));

    let segment_source = file.segment_source();
    futures::stream::iter(flats)
        .map(|flat| {
            let segment_source = Arc::clone(&segment_source);
            async move {
                let array = decode_flat(session, segment_source, &flat.layout).await?;
                VortexResult::Ok(Unit {
                    text: render_unit(&flat.path, &flat.layouts, &array, options),
                    path: flat.path,
                })
            }
        })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

struct FlatLeaf {
    path: String,
    layouts: String,
    layout: LayoutRef,
}

fn collect_flat_layouts(
    layout: &LayoutRef,
    path: &str,
    parent_layouts: &str,
    out: &mut Vec<FlatLeaf>,
) -> VortexResult<()> {
    let layouts = if parent_layouts.is_empty() {
        layout.encoding_id().to_string()
    } else {
        format!("{parent_layouts}/{}", layout.encoding_id())
    };
    if layout.is::<Flat>() {
        out.push(FlatLeaf {
            // A file whose root layout is flat has a single unit.
            path: if path.is_empty() { ROOT_PATH } else { path }.to_string(),
            layouts,
            layout: Arc::clone(layout),
        });
        return Ok(());
    }
    for (child, child_type) in layout.children()?.iter().zip(layout.child_types()) {
        collect_flat_layouts(child, &child_path(path, &child_type), &layouts, out)?;
    }
    Ok(())
}

fn child_path(parent: &str, child: &LayoutChildType) -> String {
    match child {
        LayoutChildType::Field(name) if parent.is_empty() => name.to_string(),
        LayoutChildType::Field(name) => format!("{parent}.{name}"),
        LayoutChildType::Chunk((idx, _)) => format!("{parent}[{idx}]"),
        LayoutChildType::Auxiliary(name) | LayoutChildType::Transparent(name)
            if parent.is_empty() =>
        {
            name.to_string()
        }
        LayoutChildType::Auxiliary(name) | LayoutChildType::Transparent(name) => {
            format!("{parent}/{name}")
        }
    }
}

async fn decode_flat(
    session: &VortexSession,
    segment_source: Arc<dyn SegmentSource>,
    layout: &LayoutRef,
) -> VortexResult<ArrayRef> {
    let flat = layout.as_::<Flat>();
    let row_count = flat.row_count();
    let len = usize::try_from(row_count)
        .map_err(|_| vortex_err!("row count {row_count} does not fit in usize"))?;
    let segment = segment_source.request(flat.segment_id()).await?;
    let parts = match flat.array_tree() {
        Some(array_tree) => {
            SerializedArray::from_flatbuffer_and_segment(array_tree.clone(), segment)?
        }
        None => SerializedArray::try_from(segment)?,
    };
    parts.decode(flat.dtype(), len, flat.array_ctx(), session)
}

fn render_unit(path: &str, layouts: &str, array: &ArrayRef, options: DumpOptions) -> String {
    let mut display = array.tree_display_builder().with(EncodingSummaryExtractor);
    if options.nbytes {
        display = display.with(NbytesExtractor);
    }
    if options.stats {
        display = display.with(StatsExtractor);
    }
    if options.metadata {
        display = display.with(MetadataExtractor);
    }
    if options.buffers {
        display = display.with(BufferExtractor {
            show_percent: false,
        });
    }
    format!(
        "unit: {path} rows={} layouts={layouts}\n{display}",
        array.len()
    )
}
