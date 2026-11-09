// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Size audit: per-column Vortex vs Parquet size breakdown for a compress-bench source file.
//!
//! Usage: `cargo run -p compress-bench --profile bench --example size_audit -- <file.parquet> [noenc]`

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use serde_json::json;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::DynLayout;
use vortex::layout::LayoutChildType;
use vortex::editions::EditionSessionExt;
use vortex::editions::UNSTABLE;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;

fn walk_layout(
    l: &dyn DynLayout,
    cat: &str,
    segs: &[vortex::file::SegmentSpec],
    out: &mut BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let enc = l.encoding_id().to_string();
    let mine: u64 = l
        .segment_ids()
        .iter()
        .map(|s| segs[**s as usize].length as u64)
        .sum();
    if mine > 0 {
        *out.entry(format!("{cat}|{enc}")).or_default() += mine;
    }
    for (child, ty) in l.children()?.into_iter().zip(l.child_types()) {
        let c = match ty {
            LayoutChildType::Auxiliary(n) => format!("{cat}/{n}"),
            LayoutChildType::Transparent(n) => format!("{cat}/{n}"),
            LayoutChildType::Chunk(_) => cat.to_string(),
            LayoutChildType::Field(n) => format!("{cat}.{n}"),
        };
        walk_layout(child.as_ref(), &c, segs, out)?;
    }
    Ok(())
}

fn walk_array(
    a: &ArrayRef,
    path: &str,
    seen: &mut std::collections::HashSet<(usize, usize)>,
    out: &mut BTreeMap<String, u64>,
) {
    let enc = a.encoding_id().to_string();
    let enc = enc.trim_start_matches("vortex.").to_string();
    let p = if enc == "chunked" {
        path.to_string()
    } else if path.is_empty() {
        enc
    } else {
        format!("{path}>{enc}")
    };
    let own: u64 = a
        .buffers()
        .iter()
        .filter(|b| seen.insert((b.as_ptr() as usize, b.len())))
        .map(|b| b.len() as u64)
        .sum();
    if own > 0 {
        *out.entry(p.clone()).or_default() += own;
    }
    for (name, child) in a.named_children() {
        let cp = if a.encoding_id().to_string() == "vortex.chunked" {
            p.clone()
        } else {
            format!("{p}[{name}]")
        };
        walk_array(&child, &cp, seen, out);
    }
}

async fn write_vortex(array: ArrayRef, compact: bool) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut opts = SESSION.write_options();
    if compact {
        let strategy = WriteStrategyBuilder::from_session(&SESSION)
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::from_session(&SESSION).with_compact())
            .build();
        opts = opts.with_strategy(strategy);
    }
    opts.write(&mut Cursor::new(&mut buf), array.to_array_stream())
        .await?;
    Ok(buf)
}

async fn vortex_columns(
    buf: Vec<u8>,
    encodings: bool,
) -> anyhow::Result<(u64, BTreeMap<String, serde_json::Value>)> {
    let total = buf.len() as u64;
    let file = SESSION.open_options().open_buffer(Bytes::from(buf))?;
    let footer = file.footer();
    let segs = footer.segment_map().clone();
    let layout = footer.layout().clone();
    let mut cols = BTreeMap::new();
    let dtype = file.dtype().clone();
    for (child, ty) in layout.children()?.into_iter().zip(layout.child_types()) {
        let LayoutChildType::Field(name) = ty else {
            let mut m = BTreeMap::new();
            walk_layout(child.as_ref(), "", &segs, &mut m)?;
            cols.insert("<root-aux>".to_string(), json!({"layout": m}));
            continue;
        };
        let mut m = BTreeMap::new();
        walk_layout(child.as_ref(), "", &segs, &mut m)?;
        let mut arr = BTreeMap::new();
        if encodings {
            let proj = get_item(name.clone(), root()).bind(&dtype)?;
            let mut stream = file
                .scan()?
                .with_projection(proj)
                .into_array_stream()?
                .boxed();
            let mut seen = std::collections::HashSet::new();
            while let Some(a) = stream.next().await {
                walk_array(&a?, "", &mut seen, &mut arr);
            }
        }
        cols.insert(name.to_string(), json!({"layout": m, "arrays": arr}));
    }
    Ok((total, cols))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::var_os("VX_UNSTABLE").is_some() {
        SESSION.enable_edition(UNSTABLE)?;
    }
    let path = PathBuf::from(std::env::args().nth(1).expect("parquet path"));
    let encodings = std::env::args().nth(2).as_deref() != Some("noenc");

    // Parquet: same writer properties as the bench's ParquetCompressor (sequential ArrowWriter).
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
    let schema = Arc::clone(builder.schema());
    let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut pq = Vec::new();
    let mut w = ArrowWriter::try_new(&mut pq, Arc::clone(&schema), Some(props))?;
    for b in &batches {
        w.write(b)?;
    }
    let meta = w.close()?;
    drop(batches);
    let mut pcols: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for rg in meta.row_groups() {
        for c in rg.columns() {
            let root_name = c.column_path().parts()[0].clone();
            let e = pcols
                .entry(root_name)
                .or_insert_with(|| json!({"bytes": 0u64, "uncompressed": 0u64, "encodings": [], "dict_page": 0u64}));
            e["bytes"] = json!(e["bytes"].as_u64().unwrap() + c.compressed_size() as u64);
            e["uncompressed"] =
                json!(e["uncompressed"].as_u64().unwrap() + c.uncompressed_size() as u64);
            if c.dictionary_page_offset().is_some() {
                e["dict_page"] = json!(1);
            }
            let encs: Vec<String> = c.encodings().map(|x| format!("{x:?}")).collect();
            let mut cur: Vec<String> = e["encodings"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            for x in encs {
                if !cur.contains(&x) {
                    cur.push(x);
                }
            }
            e["encodings"] = json!(cur);
        }
    }
    let pq_total = pq.len() as u64;
    drop(pq);

    let chunks = parquet_to_vortex_chunks(path.clone()).await?.into_array();
    let default = write_vortex(chunks.clone(), false).await?;
    let compact = write_vortex(chunks.clone(), true).await?;
    drop(chunks);
    let (d_total, d_cols) = vortex_columns(default, encodings).await?;
    let (c_total, c_cols) = vortex_columns(compact, encodings).await?;

    let out = json!({
        "file": path.display().to_string(),
        "rows": meta.file_metadata().num_rows(),
        "parquet_total": pq_total,
        "vortex_total": d_total,
        "compact_total": c_total,
        "parquet": pcols,
        "vortex": d_cols,
        "compact": c_cols,
    });
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
