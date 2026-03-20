// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Analyze vortex file footer sizes, requires files written with FLAT_LAYOUT_INLINE_ARRAY_NODE.
//!
//! Usage:
//!   cargo run -p vortex-file --bin vortex-footer-analyze -- <path> [<path> ...]
//!
//! Each <path> can be a .vortex file or a directory (scanned recursively for .vortex files).

#![allow(clippy::cast_possible_truncation)]

use std::env;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::process;

use vortex_flatbuffers::array as fba;
use vortex_flatbuffers::footer as fb;
use vortex_flatbuffers::layout as fbl;

const EOF_SIZE: usize = 8;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: vortex-footer-analyze <path> [<path> ...]");
        eprintln!("  Each <path> can be a .vortex file or a directory.");
        eprintln!();
        eprintln!("Files must be written with FLAT_LAYOUT_INLINE_ARRAY_NODE=1");
        process::exit(1);
    }

    let mut files = Vec::new();
    for arg in &args {
        let path = PathBuf::from(arg);
        if path.is_dir() {
            collect_vortex_files(&path, &mut files);
        } else if path.is_file() {
            files.push(path);
        } else {
            eprintln!("warning: {arg} is not a file or directory, skipping");
        }
    }

    files.sort();

    if files.is_empty() {
        eprintln!("No .vortex files found.");
        process::exit(1);
    }

    println!("Found {} vortex file(s)\n", files.len());

    let mut results: Vec<(String, FileAnalysis)> = Vec::new();

    for path in &files {
        let file_bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error reading {}: {e}", path.display());
                continue;
            }
        };
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        match analyze(&file_bytes) {
            Ok(info) => results.push((name, info)),
            Err(e) => eprintln!("error analyzing {name}: {e}"),
        }
    }

    if results.is_empty() {
        eprintln!("No files could be analyzed.");
        process::exit(1);
    }

    // Per-file table
    println!(
        "{:<30} {:>10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>6} {:>10}",
        "file", "size", "layout", "footer", "dtype", "stats", "segs", "seg data", "flats", "AN total",
    );
    println!("{}", "-".repeat(120));

    let mut totals = FileAnalysis::default();

    for (name, a) in &results {
        totals.accumulate(a);
        println!(
            "{:<30} {:>10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>6} {:>10}",
            truncate_name(name, 30),
            HumanBytes(a.file_size),
            HumanBytes(a.layout_len),
            HumanBytes(a.footer_len),
            HumanBytes(a.dtype_len),
            HumanBytes(a.stats_len),
            a.n_segments,
            HumanBytes(a.total_segment_data),
            a.n_flat_layouts,
            HumanBytes(a.an.total()),
        );
    }

    println!("{}", "-".repeat(120));
    println!(
        "{:<30} {:>10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>6} {:>10}",
        "TOTAL",
        HumanBytes(totals.file_size),
        HumanBytes(totals.layout_len),
        HumanBytes(totals.footer_len),
        HumanBytes(totals.dtype_len),
        HumanBytes(totals.stats_len),
        totals.n_segments,
        HumanBytes(totals.total_segment_data),
        totals.n_flat_layouts,
        HumanBytes(totals.an.total()),
    );

    // ArrayNode field breakdown
    let an = &totals.an;
    let an_total = an.total();
    println!();
    println!("ArrayNode breakdown (across all {} FlatLayouts):", totals.n_flat_layouts);
    println!("  {:>6} nodes ({} root, {} children)", an.n_nodes, totals.n_flat_layouts, an.n_nodes - totals.n_flat_layouts);
    println!();
    let fs = totals.file_size;
    println!("  {:<20} {:>10}  {:>6}  {:>8}", "field", "bytes", "% of AN", "% of file");
    println!("  {}", "-".repeat(52));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "encoding", HumanBytes(an.encoding_bytes), pct(an.encoding_bytes, an_total), pct(an.encoding_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "metadata", HumanBytes(an.metadata_bytes), pct(an.metadata_bytes, an_total), pct(an.metadata_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "buffers", HumanBytes(an.buffers_bytes), pct(an.buffers_bytes, an_total), pct(an.buffers_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "stats.min", HumanBytes(an.stats_min_bytes), pct(an.stats_min_bytes, an_total), pct(an.stats_min_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "stats.max", HumanBytes(an.stats_max_bytes), pct(an.stats_max_bytes, an_total), pct(an.stats_max_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "stats.sum", HumanBytes(an.stats_sum_bytes), pct(an.stats_sum_bytes, an_total), pct(an.stats_sum_bytes, fs));
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "stats.other", HumanBytes(an.stats_other_bytes), pct(an.stats_other_bytes, an_total), pct(an.stats_other_bytes, fs));
    let stats_total = an.stats_min_bytes + an.stats_max_bytes + an.stats_sum_bytes + an.stats_other_bytes;
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "stats (total)", HumanBytes(stats_total), pct(stats_total, an_total), pct(stats_total, fs));
    let fb_overhead = an_total.saturating_sub(an.encoding_bytes + an.metadata_bytes + an.buffers_bytes + stats_total);
    println!("  {:<20} {:>10}  {:>5.1}%  {:>7.3}%", "fb overhead", HumanBytes(fb_overhead), pct(fb_overhead, an_total), pct(fb_overhead, fs));
    println!("  {}", "-".repeat(52));
    println!("  {:<20} {:>10}  100.0%  {:>7.3}%", "TOTAL", HumanBytes(an_total), pct(an_total, fs));

    // File-level summary
    println!();
    let total_footer = totals.layout_len + totals.footer_len + totals.dtype_len + totals.stats_len + EOF_SIZE;
    println!("File summary:");
    println!("  Total footer: {:>10} ({:.2}% of file)", HumanBytes(total_footer), pct(total_footer, totals.file_size));
    println!("    ArrayNode:  {:>10} ({:.2}% of file)", HumanBytes(an_total), pct(an_total, totals.file_size));
    println!("  Segment data: {:>10} ({:.2}% of file)", HumanBytes(totals.total_segment_data), pct(totals.total_segment_data, totals.file_size));

    if totals.n_flat_layouts_without_inline > 0 {
        println!();
        eprintln!(
            "WARNING: {} FlatLayout(s) did NOT have inline ArrayNode metadata.",
            totals.n_flat_layouts_without_inline
        );
        eprintln!("  These files may not have been written with FLAT_LAYOUT_INLINE_ARRAY_NODE=1");
    }
    println!();
}

fn collect_vortex_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: cannot read directory {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_vortex_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "vortex") {
            out.push(path);
        }
    }
}

/// Per-ArrayNode field byte counts, accumulated across all nodes in all FlatLayouts.
#[derive(Default)]
struct ArrayNodeBreakdown {
    n_nodes: usize,
    encoding_bytes: usize,
    metadata_bytes: usize,
    buffers_bytes: usize,
    stats_min_bytes: usize,
    stats_max_bytes: usize,
    stats_sum_bytes: usize,
    /// is_sorted, is_strict_sorted, is_constant, null_count, uncompressed_size, nan_count
    /// plus the stats vtable overhead
    stats_other_bytes: usize,
    /// Raw prost + flatbuffer bytes for the whole ArrayNode tree (the ground truth)
    raw_bytes: usize,
}

impl ArrayNodeBreakdown {
    fn total(&self) -> usize {
        self.raw_bytes
    }

    fn accumulate(&mut self, other: &ArrayNodeBreakdown) {
        self.n_nodes += other.n_nodes;
        self.encoding_bytes += other.encoding_bytes;
        self.metadata_bytes += other.metadata_bytes;
        self.buffers_bytes += other.buffers_bytes;
        self.stats_min_bytes += other.stats_min_bytes;
        self.stats_max_bytes += other.stats_max_bytes;
        self.stats_sum_bytes += other.stats_sum_bytes;
        self.stats_other_bytes += other.stats_other_bytes;
        self.raw_bytes += other.raw_bytes;
    }
}

#[derive(Default)]
struct FileAnalysis {
    file_size: usize,
    layout_len: usize,
    footer_len: usize,
    dtype_len: usize,
    stats_len: usize,
    n_segments: usize,
    total_segment_data: usize,
    n_flat_layouts: usize,
    n_flat_layouts_without_inline: usize,
    an: ArrayNodeBreakdown,
}

impl FileAnalysis {
    fn accumulate(&mut self, other: &FileAnalysis) {
        self.file_size += other.file_size;
        self.layout_len += other.layout_len;
        self.footer_len += other.footer_len;
        self.dtype_len += other.dtype_len;
        self.stats_len += other.stats_len;
        self.n_segments += other.n_segments;
        self.total_segment_data += other.total_segment_data;
        self.n_flat_layouts += other.n_flat_layouts;
        self.n_flat_layouts_without_inline += other.n_flat_layouts_without_inline;
        self.an.accumulate(&other.an);
    }
}

fn analyze(file_bytes: &[u8]) -> Result<FileAnalysis, String> {
    let file_size = file_bytes.len();
    if file_size < EOF_SIZE {
        return Err("file too small".into());
    }

    let eof_start = file_size - EOF_SIZE;
    let ps_size = u16::from_le_bytes(
        file_bytes[eof_start + 2..eof_start + 4].try_into().unwrap(),
    ) as usize;
    let ps_bytes = &file_bytes[eof_start - ps_size..eof_start];
    let ps = flatbuffers::root::<fb::Postscript>(ps_bytes).map_err(|e| e.to_string())?;

    let layout_seg = ps.layout().ok_or("missing layout segment")?;
    let footer_seg = ps.footer().ok_or("missing footer segment")?;
    let dtype_len = ps.dtype().map(|s| s.length() as usize).unwrap_or(0);
    let stats_len = ps.statistics().map(|s| s.length() as usize).unwrap_or(0);
    let layout_len = layout_seg.length() as usize;
    let footer_len = footer_seg.length() as usize;

    let footer_offset = footer_seg.offset() as usize;
    let footer_bytes = &file_bytes[footer_offset..footer_offset + footer_len];
    let fb_footer = flatbuffers::root::<fb::Footer>(footer_bytes).map_err(|e| e.to_string())?;

    let n_segments = fb_footer.segment_specs().map(|s| s.len()).unwrap_or(0);

    let total_segment_data: usize = fb_footer
        .segment_specs()
        .iter()
        .flat_map(|specs| specs.iter())
        .map(|seg| seg.length() as usize)
        .sum();

    let flat_encoding_idx: Option<u16> = fb_footer.layout_specs().and_then(|specs| {
        specs
            .iter()
            .position(|s| s.id() == "vortex.flat")
            .map(|i| i as u16)
    });

    let layout_offset = layout_seg.offset() as usize;
    let layout_bytes = &file_bytes[layout_offset..layout_offset + layout_len];
    let layout_root =
        flatbuffers::root::<fbl::Layout>(layout_bytes).map_err(|e| e.to_string())?;

    let mut n_flat_layouts = 0usize;
    let mut n_flat_layouts_without_inline = 0usize;
    let mut an = ArrayNodeBreakdown::default();

    walk_layout(
        &layout_root,
        flat_encoding_idx,
        &mut n_flat_layouts,
        &mut n_flat_layouts_without_inline,
        &mut an,
    );

    Ok(FileAnalysis {
        file_size,
        layout_len,
        footer_len,
        dtype_len,
        stats_len,
        n_segments,
        total_segment_data,
        n_flat_layouts,
        n_flat_layouts_without_inline,
        an,
    })
}

fn walk_layout(
    layout: &fbl::Layout<'_>,
    flat_encoding_idx: Option<u16>,
    n_flat: &mut usize,
    n_flat_no_inline: &mut usize,
    an: &mut ArrayNodeBreakdown,
) {
    if Some(layout.encoding()) == flat_encoding_idx {
        *n_flat += 1;
        if let Some(metadata) = layout.metadata() {
            let metadata_bytes = metadata.bytes();
            if metadata_bytes.is_empty() {
                *n_flat_no_inline += 1;
            } else {
                an.raw_bytes += metadata_bytes.len();
                // The metadata is prost-encoded FlatLayoutMetadata.
                // Field 1 = array_encoding_tree: bytes.
                // Prost encoding: tag (1 byte) + varint length + payload.
                // Extract the payload (the raw ArrayNode flatbuffer).
                if let Some(array_node_fb) = extract_prost_bytes_field(metadata_bytes) {
                    analyze_array_node_fb(array_node_fb, an);
                }
            }
        } else {
            *n_flat_no_inline += 1;
        }
    }

    if let Some(children) = layout.children() {
        for child in children.iter() {
            walk_layout(&child, flat_encoding_idx, n_flat, n_flat_no_inline, an);
        }
    }
}

/// Extract the bytes payload from a prost-encoded message with a single bytes field (tag=1).
/// Prost encodes `bytes` as: tag (0x0a = field 1, wire type 2) + varint length + raw bytes.
fn extract_prost_bytes_field(prost_bytes: &[u8]) -> Option<&[u8]> {
    if prost_bytes.is_empty() {
        return None;
    }
    // tag should be 0x0a (field 1, wire type 2 = length-delimited)
    if prost_bytes[0] != 0x0a {
        return None;
    }
    let (len, consumed) = decode_varint(&prost_bytes[1..])?;
    let start = 1 + consumed;
    let end = start + len;
    if end > prost_bytes.len() {
        return None;
    }
    Some(&prost_bytes[start..end])
}

fn decode_varint(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut result = 0usize;
    let mut shift = 0;
    for (i, &b) in bytes.iter().enumerate() {
        result |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Parse an ArrayNode flatbuffer (wrapped in Array root table) and accumulate field sizes.
fn analyze_array_node_fb(fb_bytes: &[u8], an: &mut ArrayNodeBreakdown) {
    let Ok(array) = flatbuffers::root::<fba::Array>(fb_bytes) else {
        return;
    };
    let Some(root) = array.root() else {
        return;
    };
    walk_array_node(&root, an);
}

fn walk_array_node(node: &fba::ArrayNode<'_>, an: &mut ArrayNodeBreakdown) {
    an.n_nodes += 1;

    // encoding: u16 = 2 bytes
    an.encoding_bytes += 2;

    // metadata: [ubyte] — vector header (4 bytes for length) + data
    if let Some(m) = node.metadata() {
        an.metadata_bytes += 4 + m.bytes().len();
    }

    // buffers: [uint16] — vector header (4 bytes) + 2 bytes per element
    if let Some(b) = node.buffers() {
        an.buffers_bytes += 4 + b.len() * 2;
    }

    // stats
    if let Some(stats) = node.stats() {
        if let Some(min) = stats.min() {
            an.stats_min_bytes += 4 + min.bytes().len();
        }
        if let Some(max) = stats.max() {
            an.stats_max_bytes += 4 + max.bytes().len();
        }
        if let Some(sum) = stats.sum() {
            an.stats_sum_bytes += 4 + sum.bytes().len();
        }
        // bool fields (1 byte each) + u64 fields (8 bytes each) + precision enums (1 byte each)
        // is_sorted, is_strict_sorted, is_constant = 3 bools = 3 bytes
        // null_count, uncompressed_size, nan_count = 3 u64s = 24 bytes
        // min_precision, max_precision = 2 bytes
        // Plus the stats vtable (~20 bytes) and table offset (4 bytes)
        an.stats_other_bytes += 3 + 24 + 2 + 24;
    }

    // Recurse into children
    if let Some(children) = node.children() {
        for child in children.iter() {
            walk_array_node(&child, an);
        }
    }
}

fn pct(num: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        100.0 * num as f64 / denom as f64
    }
}

fn truncate_name(name: &str, max: usize) -> String {
    if name.len() <= max {
        name.to_string()
    } else {
        format!("..{}", &name[name.len() - max + 2..])
    }
}

struct HumanBytes(usize);

impl fmt::Display for HumanBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0;
        if b < 1024 {
            write!(f, "{}B", b)
        } else if b < 1024 * 1024 {
            write!(f, "{:.1}KB", b as f64 / 1024.0)
        } else if b < 1024 * 1024 * 1024 {
            write!(f, "{:.1}MB", b as f64 / (1024.0 * 1024.0))
        } else {
            write!(f, "{:.1}GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}
