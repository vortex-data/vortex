// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Display segment information for Vortex files.

use std::path::PathBuf;

use serde::Serialize;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;

use crate::segment_tree::collect_segment_tree;

/// Command-line arguments for the segments command.
#[derive(Debug, clap::Parser)]
pub struct SegmentsArgs {
    /// Path to the Vortex file
    pub file: PathBuf,
}

#[derive(Serialize)]
struct SegmentsOutput {
    /// Columns in display order
    columns: Vec<ColumnInfo>,
}

#[derive(Serialize)]
struct ColumnInfo {
    /// Field name (column header)
    name: String,
    /// Segments within this column
    segments: Vec<SegmentInfo>,
}

#[derive(Serialize)]
struct SegmentInfo {
    /// Segment name (e.g., "[0]", "data", etc.)
    name: String,
    /// Row range start
    row_offset: u64,
    /// Number of rows
    row_count: u64,
    /// Byte offset in file
    byte_offset: u64,
    /// Length in bytes
    byte_length: u32,
    /// Alignment requirement
    alignment: usize,
    /// Gap from previous segment end
    byte_gap: u64,
}

/// Display segment information for a Vortex file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub async fn exec_segments(session: &VortexSession, args: SegmentsArgs) -> VortexResult<()> {
    let vxf = session.open_options().open(args.file).await?;

    let footer = vxf.footer();
    let mut segment_tree = collect_segment_tree(footer.layout().as_ref(), footer.segment_map());

    // Convert to output format
    let columns: Vec<ColumnInfo> = segment_tree
        .segment_ordering
        .iter()
        .filter_map(|name| {
            let mut segments = segment_tree.segments.remove(name)?;

            // Sort by byte offset
            segments.sort_by(|a, b| a.spec.offset.cmp(&b.spec.offset));

            // Convert to output format, computing byte gaps
            let mut current_offset = 0u64;
            let segment_infos: Vec<SegmentInfo> = segments
                .into_iter()
                .map(|seg| {
                    let byte_gap = if current_offset == 0 {
                        0
                    } else {
                        seg.spec.offset.saturating_sub(current_offset)
                    };
                    current_offset = seg.spec.offset + seg.spec.length as u64;

                    SegmentInfo {
                        name: seg.name.to_string(),
                        row_offset: seg.row_offset,
                        row_count: seg.row_count,
                        byte_offset: seg.spec.offset,
                        byte_length: seg.spec.length,
                        alignment: *seg.spec.alignment,
                        byte_gap,
                    }
                })
                .collect();

            Some(ColumnInfo {
                name: name.to_string(),
                segments: segment_infos,
            })
        })
        .collect();

    let output = SegmentsOutput { columns };

    let json_output = serde_json::to_string_pretty(&output)
        .map_err(|e| vortex::error::vortex_err!("Failed to serialize JSON: {e}"))?;
    println!("{json_output}");

    Ok(())
}
