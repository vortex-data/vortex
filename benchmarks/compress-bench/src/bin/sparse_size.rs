// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Throwaway: measure sparse-wide file sizes when null/false-dominated chunks use Sparse.
use std::path::PathBuf;
use std::sync::Arc;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::sparse::SparseExt;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::scalar::Scalar;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;

fn sparse_or_default(
    btr: &BtrBlocksCompressor,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
    do_bool: bool,
) -> VortexResult<ArrayRef> {
    let len = chunk.len();
    if len > 0 {
        let fill = match chunk.dtype() {
            DType::Primitive(..) if chunk.dtype().is_nullable() => {
                let nulls = chunk.validity()?.execute_mask(len, ctx)?.false_count();
                (nulls as f64 > 0.9 * len as f64).then(|| None)
            }
            DType::Bool(_) if do_bool && !chunk.dtype().is_nullable() => {
                let b = chunk.clone().execute::<BoolArray>(ctx)?;
                let t = b.to_bit_buffer().true_count();
                ((t as f64) < 0.1 * len as f64 && t > 0).then(|| Some(Scalar::from(false)))
            }
            _ => None,
        };
        if let Some(fill) = fill {
            let enc = Sparse::encode(chunk, fill, ctx)?;
            if let Some(s) = enc.as_opt::<Sparse>() {
                let p = s.patches();
                let idx = btr.compress(p.indices(), ctx)?;
                let vals = btr.compress(p.values(), ctx)?;
                return Ok(Sparse::try_new(idx, vals, len, s.fill_scalar().clone(), ctx)?.into_array());
            }
            return Ok(enc);
        }
    }
    btr.compress(chunk, ctx)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap();
    if mode == "trace" {
        tracing_subscriber::fmt()
            .with_env_filter("vortex_compressor::encode=debug")
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_writer(std::io::stderr)
            .init();
        use vortex::array::VortexSessionExecute;
        use vortex::array::arrays::StructArray;
        use vortex::array::arrays::struct_::StructArrayExt;
        let mut ctx = SESSION.create_execution_ctx();
        let array = parquet_to_vortex_chunks(PathBuf::from(args.next().unwrap())).await?.into_array();
        let st = array.execute::<StructArray>(&mut ctx)?;
        let col = st.unmasked_field_by_name("g050_f0")?.clone();
        let col = col.execute::<vortex::array::Canonical>(&mut ctx)?.into_array();
        let out = BtrBlocksCompressorBuilder::from_session(&SESSION).build().compress(&col, &mut ctx)?;
        println!("{}", out.encoding_id());
        return Ok(());
    }
    if mode == "probe" {
        return probe(&args.next().unwrap()).await;
    }
    for path in args {
        let array = parquet_to_vortex_chunks(PathBuf::from(&path)).await?.into_array();
        let btr = Arc::new(BtrBlocksCompressorBuilder::from_session(&SESSION).build());
        let do_bool = mode == "sparse+validity";
        let mut opts = SESSION.write_options();
        if mode != "default" {
            let b = Arc::clone(&btr);
            let strategy = WriteStrategyBuilder::from_session(&SESSION)
                .with_compressor(move |c: &ArrayRef, ctx: &mut ExecutionCtx| {
                    sparse_or_default(&b, c, ctx, do_bool)
                })
                .build();
            opts = opts.with_strategy(strategy);
        }
        let mut buf = Vec::new();
        opts.write(&mut std::io::Cursor::new(&mut buf), array.to_array_stream()).await?;
        let out = format!("{path}.{mode}.vortex");
        std::fs::write(&out, &buf)?;
        println!("{mode}\t{path}\t{}", buf.len());
    }
    Ok(())
}

async fn probe(path: &str) -> anyhow::Result<()> {
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::integer::BitPackingScheme;
    let mut ctx = SESSION.create_execution_ctx();
    let array = parquet_to_vortex_chunks(PathBuf::from(path)).await?.into_array();
    let st = array.execute::<StructArray>(&mut ctx)?;
    for name in ["g000_f0", "g050_f0"] {
        let col = st.unmasked_field_by_name(name)?.clone();
        let col = col.execute::<vortex::array::Canonical>(&mut ctx)?.into_array();
        let default = BtrBlocksCompressorBuilder::from_session(&SESSION).build();
        let no_bp = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .exclude_schemes([BitPackingScheme::v1().id(), BitPackingScheme::v2().id()])
            .build();
        for (label, c) in [("default", default), ("no-bitpacking", no_bp)] {
            let out = c.compress(&col, &mut ctx)?;
            println!("{name} {label}: {} nbytes={}", out.encoding_id(), out.nbytes());
        }
    }
        static BP: BitPackingScheme = BitPackingScheme::v2();
    let bp_only = BtrBlocksCompressorBuilder::empty().with_new_scheme(&BP).build();
    let col = st.unmasked_field_by_name("g050_f0")?.clone();
    for (label, start) in [("all-null 1024 rows", 0usize), ("1024 rows incl 500 valid", 50_000 - 524)] {
        let s = col.slice(start..start + 1024)?.execute::<vortex::array::Canonical>(&mut ctx)?.into_array();
        let out = bp_only.compress(&s, &mut ctx)?;
        println!("{label}: in nbytes={} -> {} nbytes={}", s.nbytes(), out.encoding_id(), out.nbytes());
        println!("{}", out.display_tree());
    }
        // Emulate the sampler: 16 slices of 64 rows, one per 6250-row partition, none hitting the valid block.
    let slices = (0..16)
        .map(|i| col.slice(i * 6250 + 100..i * 6250 + 164))
        .collect::<VortexResult<Vec<_>>>()?;
    let chunked = vortex::array::arrays::ChunkedArray::try_new(slices, col.dtype().clone())?;
    let sample = chunked
        .into_array()
        .execute::<vortex::array::arrays::PrimitiveArray>(&mut ctx)?;
    println!("sample nbytes={} validity={:?}", sample.clone().into_array().nbytes(), sample.clone().into_array().validity()?);
    let bp = vortex::encodings::fastlanes::bitpack_compress::bitpack_encode(&sample, 0, None, &mut ctx)?;
    println!("bitpacked(w=0) sample nbytes={}", bp.into_array().nbytes());
    Ok(())
}
