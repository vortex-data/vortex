// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Throwaway: measure sparse-wide file sizes when null/false-dominated chunks use Sparse.
use std::path::PathBuf;
use std::sync::Arc;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
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
                let b = chunk.clone().execute::<vortex::array::arrays::BoolArray>(ctx)?;
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
        let _ = SESSION.create_execution_ctx();
    }
    Ok(())
}
