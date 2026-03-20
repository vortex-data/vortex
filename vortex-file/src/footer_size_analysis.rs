// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Analysis of footer/segment/ArrayNode sizes for real Vortex files.
//! Run with: cargo test -p vortex-file -- footer_size_analysis --nocapture

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::WriteOptionsSessionExt;
use crate::EOF_SIZE;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<vortex_array::scalar_fn::session::ScalarFnSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&mut session);
    session
});

/// Analyze a vortex file's bytes and report section sizes plus dictionary compression estimates.
fn analyze_file(name: &str, file_bytes: &[u8]) {
    let file_size = file_bytes.len();

    // Parse EOF
    let eof_start = file_size - EOF_SIZE;
    let ps_size = u16::from_le_bytes(
        file_bytes[eof_start + 2..eof_start + 4]
            .try_into()
            .unwrap(),
    ) as usize;

    // Parse postscript
    let ps_bytes = &file_bytes[eof_start - ps_size..eof_start];
    let ps = flatbuffers::root::<vortex_flatbuffers::footer::Postscript>(ps_bytes).unwrap();

    let layout_seg = ps.layout().unwrap();
    let footer_seg = ps.footer().unwrap();
    let dtype_seg = ps.dtype();
    let stats_seg = ps.statistics();

    // Parse footer
    let footer_offset = footer_seg.offset() as usize;
    let footer_len = footer_seg.length() as usize;
    let footer_bytes = &file_bytes[footer_offset..footer_offset + footer_len];
    let fb_footer = flatbuffers::root::<vortex_flatbuffers::footer::Footer>(footer_bytes).unwrap();
    let n_segments = fb_footer
        .segment_specs()
        .map(|s| s.len())
        .unwrap_or(0);
    let n_array_specs = fb_footer
        .array_specs()
        .map(|s| s.len())
        .unwrap_or(0);

    let layout_len = layout_seg.length() as usize;

    // Extract and deduplicate ArrayNodes
    let mut total_array_node_bytes = 0usize;
    let mut segment_count = 0usize;
    let mut min_an = usize::MAX;
    let mut max_an = 0usize;
    // For raw byte dedup
    let mut unique_trees: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut total_unique_bytes = 0usize;
    // For shape-only dedup (ignoring stats and buffer indices)
    let mut unique_shapes: HashMap<u64, (usize, usize)> = HashMap::new(); // hash -> (count, shape_size)
    let mut total_stats_bytes = 0usize;

    if let Some(segments) = fb_footer.segment_specs() {
        for seg in segments.iter() {
            let seg_offset = seg.offset() as usize;
            let seg_len = seg.length() as usize;
            if seg_len < 4 {
                continue;
            }
            let seg_end = seg_offset + seg_len;
            if seg_end > file_size {
                continue;
            }
            let fb_len_bytes = &file_bytes[seg_end - 4..seg_end];
            let fb_len = u32::from_le_bytes(fb_len_bytes.try_into().unwrap()) as usize;
            if fb_len > 0 && fb_len < seg_len && fb_len < 100_000 {
                let node_bytes = fb_len + 4;
                total_array_node_bytes += node_bytes;
                min_an = min_an.min(node_bytes);
                max_an = max_an.max(node_bytes);
                segment_count += 1;

                let fb_start = seg_end - 4 - fb_len;
                let fb_content = &file_bytes[fb_start..fb_start + fb_len];

                // Raw byte dedup
                *unique_trees.entry(fb_content.to_vec()).or_insert(0) += 1;

                // Shape-only dedup
                if let Some(shape_hash) = array_node_shape_hash(fb_content) {
                    let shape_size = array_node_shape_size(fb_content).unwrap_or(0);
                    let entry = unique_shapes.entry(shape_hash).or_insert((0, shape_size));
                    entry.0 += 1;
                    // Stats = full ArrayNode bytes - shape-only bytes (rough estimate)
                    total_stats_bytes += fb_len.saturating_sub(shape_size);
                }
            }
        }
    }

    // Raw byte dictionary compression estimate
    for (tree_bytes, _count) in &unique_trees {
        total_unique_bytes += tree_bytes.len();
    }
    let dict_raw_overhead = total_unique_bytes
        + unique_trees.len() * 8
        + segment_count * 2;

    // Shape-only dictionary compression estimate
    let total_unique_shape_bytes: usize = unique_shapes.values().map(|(_, sz)| *sz).sum();
    let dict_shape_overhead = total_unique_shape_bytes
        + unique_shapes.len() * 8  // per-shape overhead
        + segment_count * 2;       // u16 index per flat layout

    let total_metadata = layout_len
        + footer_len
        + ps_size
        + EOF_SIZE
        + dtype_seg.map(|s| s.length() as usize).unwrap_or(0)
        + stats_seg.map(|s| s.length() as usize).unwrap_or(0);

    println!("=== {name} ===");
    println!("  File size:       {:>12} bytes ({:.1} MB)", file_size, file_size as f64 / 1_048_576.0);
    println!("  Metadata total:  {:>12} bytes ({:.2}% of file)", total_metadata, pct(total_metadata, file_size));
    println!("    Layout fb:     {:>12} bytes", layout_len);
    println!("    Footer fb:     {:>12} bytes", footer_len);
    if let Some(ds) = dtype_seg {
        println!("    DType fb:      {:>12} bytes", ds.length());
    }
    if let Some(ss) = stats_seg {
        println!("    Stats fb:      {:>12} bytes", ss.length());
    }
    println!();
    println!("  Segments:        {:>12}", n_segments);
    println!("    SegmentSpecs:  {:>12} bytes", n_segments * 16);
    println!("    array_specs:   {:>12}", n_array_specs);
    println!();
    println!("  ArrayNode in segments:");
    println!("    count:         {:>12}", segment_count);
    println!("    total:         {:>12} bytes ({:.2}% of file)", total_array_node_bytes, pct(total_array_node_bytes, file_size));
    if segment_count > 0 {
        println!("    avg:           {:>12} bytes", total_array_node_bytes / segment_count);
        println!("    min:           {:>12} bytes", min_an);
        println!("    max:           {:>12} bytes", max_an);
    }
    println!();
    println!("  Raw byte dedup (full ArrayNode including stats):");
    println!("    unique:     {:>8} / {segment_count}", unique_trees.len());
    let raw_savings = total_array_node_bytes.saturating_sub(dict_raw_overhead);
    println!("    savings:    {:>8} bytes ({:.1}% of ArrayNode)", raw_savings, pct(raw_savings, total_array_node_bytes));
    println!();
    println!("  Shape-only dedup (encoding+metadata, NO stats/buffers):");
    println!("    unique:     {:>8} / {segment_count}", unique_shapes.len());
    println!("    shape-only: {:>8} bytes (sum of unique shapes)", total_unique_shape_bytes);
    println!("    est stats:  {:>8} bytes (stats+buffers portion)", total_stats_bytes);
    println!("    dict cost:  {:>8} bytes (unique shapes + u16 indices)", dict_shape_overhead);
    println!();

    // The real savings question: if we store shapes in footer, stats stay in segment
    // We save the per-segment shape bytes but still need stats in the segment.
    // Actually the stats are already small. Let's show what it means.
    let shape_bytes_in_segments: usize = unique_shapes.values()
        .map(|(count, sz)| count * sz)
        .sum();
    let shape_saving = shape_bytes_in_segments.saturating_sub(
        total_unique_shape_bytes + unique_shapes.len() * 8 + segment_count * 2
    );
    println!("  ** Proposed: move tree shapes to footer, keep stats in segments **");
    println!("    shape bytes currently in segments: {:>8} bytes", shape_bytes_in_segments);
    println!("    shape bytes after dict in footer:  {:>8} bytes", total_unique_shape_bytes + unique_shapes.len() * 8 + segment_count * 2);
    println!("    NET SAVINGS:                       {:>8} bytes ({:.3}% of file)", shape_saving, pct(shape_saving, file_size));

    // Show unique shapes by frequency
    let mut by_count: Vec<_> = unique_shapes.iter().collect();
    by_count.sort_by(|a, b| (b.1).0.cmp(&(a.1).0));
    if unique_shapes.len() > 1 && unique_shapes.len() < segment_count {
        println!();
        println!("    Shape frequency distribution:");
        for (i, (_hash, (count, sz))) in by_count.iter().take(10).enumerate() {
            println!("      #{}: ~{} bytes x {} occurrences", i + 1, sz, count);
        }
    }
    println!();
}

fn pct(num: usize, denom: usize) -> f64 {
    if denom == 0 { 0.0 } else { 100.0 * num as f64 / denom as f64 }
}

/// Deep analysis of stats within ArrayNode trees.
/// Walks every node recursively and measures per-field stats byte usage.
fn analyze_stats_in_file(name: &str, file_bytes: &[u8]) {
    use vortex_flatbuffers::array as fba;

    let file_size = file_bytes.len();
    let eof_start = file_size - EOF_SIZE;
    let ps_size = u16::from_le_bytes(
        file_bytes[eof_start + 2..eof_start + 4].try_into().unwrap(),
    ) as usize;
    let ps_bytes = &file_bytes[eof_start - ps_size..eof_start];
    let ps = flatbuffers::root::<vortex_flatbuffers::footer::Postscript>(ps_bytes).unwrap();
    let footer_seg = ps.footer().unwrap();
    let footer_offset = footer_seg.offset() as usize;
    let footer_len = footer_seg.length() as usize;
    let footer_bytes = &file_bytes[footer_offset..footer_offset + footer_len];
    let fb_footer = flatbuffers::root::<vortex_flatbuffers::footer::Footer>(footer_bytes).unwrap();

    // Accumulators
    let mut total_nodes = 0usize;
    let mut root_nodes = 0usize;
    let mut nodes_with_stats = 0usize;
    let mut nodes_without_stats = 0usize;
    let mut child_nodes_with_stats = 0usize;
    let mut child_nodes_without_stats = 0usize;

    // Per-field counters
    let mut has_min = 0usize;
    let mut has_max = 0usize;
    let mut has_sum = 0usize;
    let mut has_is_sorted = 0usize;
    let mut has_is_strict_sorted = 0usize;
    let mut has_is_constant = 0usize;
    let mut has_null_count = 0usize;
    let mut has_uncompressed_size = 0usize;
    let mut has_nan_count = 0usize;

    // Byte sizes
    let mut total_min_bytes = 0usize;
    let mut total_max_bytes = 0usize;
    let mut total_sum_bytes = 0usize;
    let mut total_stats_fb_overhead = 0usize; // estimated vtable + field offsets

    fn walk_node(
        node: &fba::ArrayNode<'_>,
        is_root: bool,
        total_nodes: &mut usize,
        root_nodes: &mut usize,
        nodes_with_stats: &mut usize,
        nodes_without_stats: &mut usize,
        child_nodes_with_stats: &mut usize,
        child_nodes_without_stats: &mut usize,
        has_min: &mut usize, has_max: &mut usize, has_sum: &mut usize,
        has_is_sorted: &mut usize, has_is_strict_sorted: &mut usize,
        has_is_constant: &mut usize, has_null_count: &mut usize,
        has_uncompressed_size: &mut usize, has_nan_count: &mut usize,
        total_min_bytes: &mut usize, total_max_bytes: &mut usize,
        total_sum_bytes: &mut usize, total_stats_fb_overhead: &mut usize,
    ) {
        *total_nodes += 1;
        if is_root { *root_nodes += 1; }

        if let Some(stats) = node.stats() {
            if is_root { *nodes_with_stats += 1; } else { *child_nodes_with_stats += 1; }

            // Measure each field
            if let Some(min) = stats.min() {
                *has_min += 1;
                *total_min_bytes += min.bytes().len();
            }
            if let Some(max) = stats.max() {
                *has_max += 1;
                *total_max_bytes += max.bytes().len();
            }
            if let Some(sum) = stats.sum() {
                *has_sum += 1;
                *total_sum_bytes += sum.bytes().len();
            }
            // For scalar bool/u64 fields, we can't easily distinguish "not present"
            // from default in flatbuffers. Count them as present if stats exists.
            // Each costs 1-8 bytes inline + 2 byte vtable entry.
            *has_is_sorted += 1;
            *has_is_strict_sorted += 1;
            *has_is_constant += 1;
            *has_null_count += 1;
            *has_uncompressed_size += 1;
            *has_nan_count += 1;

            // Estimate flatbuffer overhead for this stats table:
            // vtable (variable, ~20-30 bytes) + inline fields
            // Conservative: 24 bytes base + 2 per present field
            *total_stats_fb_overhead += 24;
        } else {
            if is_root { *nodes_without_stats += 1; } else { *child_nodes_without_stats += 1; }
        }

        if let Some(children) = node.children() {
            for child in children.iter() {
                walk_node(
                    &child, false,
                    total_nodes, root_nodes,
                    nodes_with_stats, nodes_without_stats,
                    child_nodes_with_stats, child_nodes_without_stats,
                    has_min, has_max, has_sum,
                    has_is_sorted, has_is_strict_sorted,
                    has_is_constant, has_null_count,
                    has_uncompressed_size, has_nan_count,
                    total_min_bytes, total_max_bytes,
                    total_sum_bytes, total_stats_fb_overhead,
                );
            }
        }
    }

    if let Some(segments) = fb_footer.segment_specs() {
        for seg in segments.iter() {
            let seg_offset = seg.offset() as usize;
            let seg_len = seg.length() as usize;
            if seg_len < 4 { continue; }
            let seg_end = seg_offset + seg_len;
            if seg_end > file_size { continue; }
            let fb_len = u32::from_le_bytes(
                file_bytes[seg_end - 4..seg_end].try_into().unwrap(),
            ) as usize;
            if fb_len == 0 || fb_len >= seg_len || fb_len >= 100_000 { continue; }

            let fb_start = seg_end - 4 - fb_len;
            let fb_content = &file_bytes[fb_start..fb_start + fb_len];

            if let Ok(array) = flatbuffers::root::<fba::Array>(fb_content) {
                if let Some(root) = array.root() {
                    walk_node(
                        &root, true,
                        &mut total_nodes, &mut root_nodes,
                        &mut nodes_with_stats, &mut nodes_without_stats,
                        &mut child_nodes_with_stats, &mut child_nodes_without_stats,
                        &mut has_min, &mut has_max, &mut has_sum,
                        &mut has_is_sorted, &mut has_is_strict_sorted,
                        &mut has_is_constant, &mut has_null_count,
                        &mut has_uncompressed_size, &mut has_nan_count,
                        &mut total_min_bytes, &mut total_max_bytes,
                        &mut total_sum_bytes, &mut total_stats_fb_overhead,
                    );
                }
            }
        }
    }

    let total_value_bytes = total_min_bytes + total_max_bytes + total_sum_bytes;
    let total_all_stats = total_value_bytes + total_stats_fb_overhead;

    println!("=== STATS ANALYSIS: {name} ===");
    println!("  Total ArrayNode nodes:     {:>8} ({root_nodes} roots, {} children)",
        total_nodes, total_nodes - root_nodes);
    println!("  Root nodes with stats:     {:>8} / {root_nodes}", nodes_with_stats);
    println!("  Root nodes without stats:  {:>8} / {root_nodes}", nodes_without_stats);
    println!("  Child nodes with stats:    {:>8} / {}", child_nodes_with_stats, total_nodes - root_nodes);
    println!("  Child nodes without stats: {:>8} / {}", child_nodes_without_stats, total_nodes - root_nodes);
    println!();
    println!("  Stats field presence (across all nodes with stats):");
    println!("    min:        {:>6} nodes, {:>8} value bytes, avg {:.0} bytes/val",
        has_min, total_min_bytes, if has_min > 0 { total_min_bytes as f64 / has_min as f64 } else { 0.0 });
    println!("    max:        {:>6} nodes, {:>8} value bytes, avg {:.0} bytes/val",
        has_max, total_max_bytes, if has_max > 0 { total_max_bytes as f64 / has_max as f64 } else { 0.0 });
    println!("    sum:        {:>6} nodes, {:>8} value bytes",
        has_sum, total_sum_bytes);
    println!("    (bool/u64 fields: is_sorted, is_strict_sorted, is_constant, null_count, etc.)");
    println!();
    println!("  Estimated stats byte breakdown:");
    println!("    min/max/sum values:      {:>8} bytes", total_value_bytes);
    println!("    fb overhead (~24B/node): {:>8} bytes", total_stats_fb_overhead);
    println!("    TOTAL stats estimate:    {:>8} bytes", total_all_stats);
    println!();

    // What if we strip stats from child nodes entirely?
    let child_stats_nodes = child_nodes_with_stats;
    // Rough estimate: child stats are similar size to root stats on average
    let avg_stats_per_node = if (nodes_with_stats + child_nodes_with_stats) > 0 {
        total_all_stats / (nodes_with_stats + child_nodes_with_stats)
    } else {
        0
    };
    let child_stats_cost = child_stats_nodes * avg_stats_per_node;
    println!("  ** Opportunity: strip stats from child nodes **");
    println!("    Child nodes carrying stats:  {child_stats_nodes}");
    println!("    Est child stats cost:        {:>8} bytes", child_stats_cost);
    println!("    (These stats are for sub-arrays like validity, offsets, values)");
    println!("    (Only root-level stats are used for pruning)");
    println!();

    // What if we strip stats entirely from segments (they're already in the file-level stats)?
    println!("  ** Opportunity: remove ALL per-segment stats (rely on file-level stats) **");
    println!("    Total stats in segments:     {:>8} bytes", total_all_stats);
    println!("    This is {:.2}% of file", pct(total_all_stats, file_size));
    println!();
}

/// Extract the "structural shape" of an ArrayNode, ignoring stats and buffer indices.
/// Returns a canonical representation: (encoding_id, metadata_bytes, children_shapes).
/// This lets us see how many truly structurally-distinct encoding trees exist.
fn array_node_shape_hash(fb_bytes: &[u8]) -> Option<u64> {
    use vortex_flatbuffers::array as fba;
    // Parse as Array (the root table), then get its root ArrayNode.
    let array = flatbuffers::root::<fba::Array>(fb_bytes).ok()?;
    let root = array.root()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_array_node_shape(&root, &mut hasher);
    Some(hasher.finish())
}

fn hash_array_node_shape(node: &vortex_flatbuffers::array::ArrayNode<'_>, hasher: &mut impl Hasher) {
    // Hash the encoding ID
    node.encoding().hash(hasher);
    // Hash metadata bytes
    if let Some(metadata) = node.metadata() {
        metadata.bytes().hash(hasher);
    } else {
        0u8.hash(hasher);
    }
    // Recursively hash children shapes
    if let Some(children) = node.children() {
        children.len().hash(hasher);
        for child in children.iter() {
            hash_array_node_shape(&child, hasher);
        }
    } else {
        0usize.hash(hasher);
    }
    // Deliberately NOT hashing: buffers (always sequential), stats (per-chunk)
}

/// Estimate the size of a "shape-only" ArrayNode (encoding + metadata + children, no stats/buffers).
fn array_node_shape_size(fb_bytes: &[u8]) -> Option<usize> {
    use vortex_flatbuffers::array as fba;
    let array = flatbuffers::root::<fba::Array>(fb_bytes).ok()?;
    let root = array.root()?;
    Some(measure_node_shape(&root))
}

fn measure_node_shape(node: &vortex_flatbuffers::array::ArrayNode<'_>) -> usize {
    // Approximate flatbuffer cost of this node without stats and buffers:
    // vtable (~8 bytes) + encoding u16 (2) + metadata vector (4 + len) + children offset (4)
    let base = 8 + 2;
    let metadata_cost = node.metadata().map(|m| 4 + m.bytes().len()).unwrap_or(0);
    let children_cost: usize = node.children()
        .map(|children| {
            4 + children.iter().map(|c| measure_node_shape(&c)).sum::<usize>()
        })
        .unwrap_or(0);
    base + metadata_cost + children_cost
}

/// Write a file using a simple non-repartitioning strategy.
async fn write_file_simple(array: vortex_array::ArrayRef) -> VortexResult<Vec<u8>> {
    let flat: Arc<dyn vortex_layout::LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let chunked = ChunkedLayoutStrategy::new(flat.clone());
    let table = TableStrategy::new(flat, Arc::new(chunked));

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(Arc::new(table))
        .with_file_statistics(vec![])
        .write(&mut buf, array.to_array_stream())
        .await?;
    Ok(buf.freeze().to_vec())
}

/// Write using the default strategy
async fn write_file_default(array: vortex_array::ArrayRef) -> VortexResult<Vec<u8>> {
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;
    Ok(buf.freeze().to_vec())
}

#[tokio::test]
async fn footer_size_analysis() -> VortexResult<()> {
    // ========================================================
    // Part 1: Analyze real TPCH Vortex files on disk
    // ========================================================
    println!("\n======== REAL TPCH VORTEX FILES ========\n");

    let tpch_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-bench/data/tpch/1.0/vortex-file-compressed");

    if tpch_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&tpch_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "vortex"))
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();
            analyze_file(name, &file_bytes);
        }

        // Stats deep-dive on the biggest files
        println!("\n======== STATS DEEP DIVE ========\n");
        for entry in &entries {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();
            analyze_stats_in_file(name, &file_bytes);
        }
    } else {
        println!("  (TPCH data not found at {:?}, skipping)", tpch_dir);
    }

    // ========================================================
    // Part 2: Synthetic files for comparison
    // ========================================================
    println!("\n======== SYNTHETIC FILES (simple strategy, no compression) ========\n");

    // Struct(20 cols), 1000 chunks x 1000 rows
    {
        let make_struct = || {
            let names: Vec<FieldName> = (0..20)
                .map(|i| FieldName::from(format!("col_{i}")))
                .collect();
            let fields: Vec<_> = (0..20)
                .map(|_| {
                    PrimitiveArray::new(
                        (0..1000).collect::<Buffer<i32>>(),
                        Validity::AllValid,
                    )
                    .into_array()
                })
                .collect();
            StructArray::try_new(
                FieldNames::from(names),
                fields,
                1000,
                Validity::NonNullable,
            )
            .unwrap()
            .into_array()
        };
        let chunks: Vec<_> = (0..1000).map(|_| make_struct()).collect();
        let array = ChunkedArray::from_iter(chunks).into_array();
        let bytes = write_file_simple(array).await?;
        analyze_file("Synthetic: Struct(20 cols), 1000 chunks x 1000 rows", &bytes);
    }

    println!("\n======== SYNTHETIC FILES (default strategy, compressed) ========\n");

    // Struct(10 cols), 1M rows via default strategy
    {
        let make_struct = |offset: i32| {
            let names: Vec<FieldName> = (0..10)
                .map(|i| FieldName::from(format!("col_{i}")))
                .collect();
            let fields: Vec<_> = (0..10)
                .map(|_| {
                    PrimitiveArray::new(
                        (offset..offset + 100000).collect::<Buffer<i32>>(),
                        Validity::AllValid,
                    )
                    .into_array()
                })
                .collect();
            StructArray::try_new(
                FieldNames::from(names),
                fields,
                100000,
                Validity::NonNullable,
            )
            .unwrap()
            .into_array()
        };
        let chunks: Vec<_> = (0..10).map(|i| make_struct(i * 100000)).collect();
        let array = ChunkedArray::from_iter(chunks).into_array();
        let bytes = write_file_default(array).await?;
        analyze_file("Synthetic DEFAULT: Struct(10 cols), 1M rows", &bytes);
    }

    // ========================================================
    // Part 3: Analyze real ClickBench Vortex files
    // ========================================================
    println!("\n======== REAL CLICKBENCH VORTEX FILES ========\n");

    let clickbench_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-bench/data/clickbench_partitioned/vortex-file-compressed");

    if clickbench_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&clickbench_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "vortex"))
            .collect();
        entries.sort_by_key(|e| e.file_name());

        // Aggregate stats across all files
        let mut total_file_size = 0usize;
        let mut total_array_node = 0usize;
        let mut total_segments = 0usize;
        let mut total_metadata = 0usize;

        let n_files = entries.len();
        println!("  Found {} clickbench vortex files\n", n_files);

        // Show detailed analysis for first 3 files, summary for rest
        for (i, entry) in entries.iter().enumerate() {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();

            let file_size = file_bytes.len();
            total_file_size += file_size;

            if i < 3 {
                analyze_file(name, &file_bytes);
            }

            // Accumulate for summary
            let eof_start = file_size - EOF_SIZE;
            let ps_size = u16::from_le_bytes(
                file_bytes[eof_start + 2..eof_start + 4].try_into().unwrap(),
            ) as usize;
            let ps_bytes = &file_bytes[eof_start - ps_size..eof_start];
            let ps =
                flatbuffers::root::<vortex_flatbuffers::footer::Postscript>(ps_bytes).unwrap();
            let footer_seg = ps.footer().unwrap();
            let layout_seg = ps.layout().unwrap();
            let footer_offset = footer_seg.offset() as usize;
            let footer_len = footer_seg.length() as usize;
            let footer_bytes = &file_bytes[footer_offset..footer_offset + footer_len];
            let fb_footer =
                flatbuffers::root::<vortex_flatbuffers::footer::Footer>(footer_bytes).unwrap();

            let n_segs = fb_footer
                .segment_specs()
                .map(|s| s.len())
                .unwrap_or(0);
            total_segments += n_segs;
            total_metadata += footer_len
                + layout_seg.length() as usize
                + ps_size
                + EOF_SIZE
                + ps.dtype().map(|s| s.length() as usize).unwrap_or(0)
                + ps.statistics().map(|s| s.length() as usize).unwrap_or(0);

            // Count ArrayNode bytes in segments
            if let Some(segments) = fb_footer.segment_specs() {
                for seg in segments.iter() {
                    let seg_offset = seg.offset() as usize;
                    let seg_len = seg.length() as usize;
                    if seg_len < 4 {
                        continue;
                    }
                    let seg_end = seg_offset + seg_len;
                    if seg_end > file_size {
                        continue;
                    }
                    let fb_len = u32::from_le_bytes(
                        file_bytes[seg_end - 4..seg_end].try_into().unwrap(),
                    ) as usize;
                    if fb_len > 0 && fb_len < seg_len && fb_len < 100_000 {
                        total_array_node += fb_len + 4;
                    }
                }
            }
        }

        println!("\n======== CLICKBENCH AGGREGATE SUMMARY ({n_files} files) ========\n");
        println!(
            "  Total file size:       {:>12} bytes ({:.1} MB)",
            total_file_size,
            total_file_size as f64 / 1_048_576.0
        );
        println!(
            "  Total metadata:        {:>12} bytes ({:.2}% of total)",
            total_metadata,
            pct(total_metadata, total_file_size)
        );
        println!(
            "  Total segments:        {:>12}",
            total_segments
        );
        println!(
            "  Total ArrayNode bytes: {:>12} bytes ({:.2}% of total)",
            total_array_node,
            pct(total_array_node, total_file_size)
        );
        println!(
            "  Avg per file:          {:>12} bytes ({:.1} MB)",
            total_file_size / n_files.max(1),
            total_file_size as f64 / n_files.max(1) as f64 / 1_048_576.0
        );
        println!();

        // Stats deep-dive on first file
        println!("======== CLICKBENCH STATS DEEP DIVE (first file) ========\n");
        if let Some(entry) = entries.first() {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();
            analyze_stats_in_file(name, &file_bytes);
        }
    } else {
        println!(
            "  (ClickBench data not found at {:?}, skipping)",
            clickbench_dir
        );
        println!("  Generate with: cargo run -p vortex-bench --bin data-gen --release -- clickbench --formats vortex");
    }

    Ok(())
}

/// Given a vortex file's raw bytes, compute the size impact of inlining ArrayNode into footer.
///
/// When FLAT_LAYOUT_INLINE_ARRAY_NODE is enabled:
/// - The ArrayNode flatbuffer is removed from each segment (segment shrinks)
/// - The ArrayNode flatbuffer is stored in the FlatLayout metadata (layout fb grows)
///   - Each FlatLayout gains a prost-encoded field: tag(1) + varint_len + array_node_bytes
///
/// Returns (current_file_size, estimated_inline_size, segment_savings, footer_growth)
fn estimate_inline_size(file_bytes: &[u8]) -> (usize, usize, usize, usize) {
    let file_size = file_bytes.len();
    let eof_start = file_size - EOF_SIZE;
    let ps_size = u16::from_le_bytes(
        file_bytes[eof_start + 2..eof_start + 4].try_into().unwrap(),
    ) as usize;
    let ps_bytes = &file_bytes[eof_start - ps_size..eof_start];
    let ps = flatbuffers::root::<vortex_flatbuffers::footer::Postscript>(ps_bytes).unwrap();
    let footer_seg = ps.footer().unwrap();
    let footer_offset = footer_seg.offset() as usize;
    let footer_len = footer_seg.length() as usize;
    let footer_bytes = &file_bytes[footer_offset..footer_offset + footer_len];
    let fb_footer = flatbuffers::root::<vortex_flatbuffers::footer::Footer>(footer_bytes).unwrap();

    let mut total_array_node_removed = 0usize; // bytes removed from segments
    let mut total_metadata_added = 0usize; // bytes added to layout metadata

    if let Some(segments) = fb_footer.segment_specs() {
        for seg in segments.iter() {
            let seg_offset = seg.offset() as usize;
            let seg_len = seg.length() as usize;
            if seg_len < 4 {
                continue;
            }
            let seg_end = seg_offset + seg_len;
            if seg_end > file_size {
                continue;
            }
            let fb_len = u32::from_le_bytes(
                file_bytes[seg_end - 4..seg_end].try_into().unwrap(),
            ) as usize;
            if fb_len > 0 && fb_len < seg_len && fb_len < 100_000 {
                // This ArrayNode would be removed from the segment
                // The segment stores: [data buffers] [padding] [ArrayNode fb] [u32 fb_len]
                // With inline, only the ArrayNode fb moves out; the u32 length stays (becomes 0)
                total_array_node_removed += fb_len;

                // In the layout metadata (prost), the ArrayNode is stored as:
                // field tag (1 byte) + varint length (1-3 bytes) + raw bytes
                let prost_overhead = 1 + varint_len(fb_len) + fb_len;
                total_metadata_added += prost_overhead;
            }
        }
    }

    let estimated_inline_size = file_size - total_array_node_removed + total_metadata_added;
    (
        file_size,
        estimated_inline_size,
        total_array_node_removed,
        total_metadata_added,
    )
}

fn varint_len(value: usize) -> usize {
    if value < 128 {
        1
    } else if value < 16384 {
        2
    } else if value < 2_097_152 {
        3
    } else {
        4
    }
}

fn print_inline_comparison(name: &str, file_bytes: &[u8]) {
    let (current, inline, seg_savings, footer_growth) = estimate_inline_size(file_bytes);
    let diff = if inline > current {
        inline - current
    } else {
        current - inline
    };
    let sign = if inline > current { "+" } else { "-" };
    println!(
        "  {:<40} {:>10} -> {:>10} ({}{} bytes, {}{:.3}%)",
        name,
        current,
        inline,
        sign,
        diff,
        sign,
        pct(diff, current),
    );
    println!(
        "    segment savings: -{} bytes, footer growth: +{} bytes",
        seg_savings, footer_growth,
    );
}

#[tokio::test]
async fn inline_size_comparison() -> VortexResult<()> {
    println!("\n======== INLINE ArrayNode SIZE COMPARISON ========\n");
    println!("  Shows file size with vs without FLAT_LAYOUT_INLINE_ARRAY_NODE\n");

    // TPCH files
    let tpch_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-bench/data/tpch/1.0/vortex-file-compressed");

    if tpch_dir.exists() {
        println!("  --- TPCH ---");
        let mut entries: Vec<_> = std::fs::read_dir(&tpch_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "vortex"))
            .collect();
        entries.sort_by_key(|e| e.file_name());

        let mut total_current = 0usize;
        let mut total_inline = 0usize;
        for entry in &entries {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();
            let (current, inline, _, _) = estimate_inline_size(&file_bytes);
            total_current += current;
            total_inline += inline;
            print_inline_comparison(name, &file_bytes);
        }
        let diff = total_inline.abs_diff(total_current);
        let sign = if total_inline > total_current {
            "+"
        } else {
            "-"
        };
        println!(
            "\n  {:<40} {:>10} -> {:>10} ({}{} bytes, {}{:.3}%)\n",
            "TPCH TOTAL",
            total_current,
            total_inline,
            sign,
            diff,
            sign,
            pct(diff, total_current),
        );
    }

    // ClickBench files
    let clickbench_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-bench/data/clickbench_partitioned/vortex-file-compressed");

    if clickbench_dir.exists() {
        println!("  --- ClickBench (100 files) ---");
        let mut entries: Vec<_> = std::fs::read_dir(&clickbench_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "vortex"))
            .collect();
        entries.sort_by_key(|e| e.file_name());

        let mut total_current = 0usize;
        let mut total_inline = 0usize;
        // Show first 5, then summary
        for (i, entry) in entries.iter().enumerate() {
            let path = entry.path();
            let file_bytes = std::fs::read(&path)?;
            let name = path.file_name().unwrap().to_str().unwrap();
            let (current, inline, _, _) = estimate_inline_size(&file_bytes);
            total_current += current;
            total_inline += inline;
            if i < 5 {
                print_inline_comparison(name, &file_bytes);
            }
        }
        if entries.len() > 5 {
            println!("  ... ({} more files)", entries.len() - 5);
        }
        let diff = total_inline.abs_diff(total_current);
        let sign = if total_inline > total_current {
            "+"
        } else {
            "-"
        };
        println!(
            "\n  {:<40} {:>10} -> {:>10} ({}{} bytes, {}{:.3}%)",
            "CLICKBENCH TOTAL",
            total_current,
            total_inline,
            sign,
            diff,
            sign,
            pct(diff, total_current),
        );
        println!(
            "  {:<40} {:>10}    {:>10}",
            "",
            format!("({:.1} MB)", total_current as f64 / 1_048_576.0),
            format!("({:.1} MB)", total_inline as f64 / 1_048_576.0),
        );
    }

    println!();
    Ok(())
}
