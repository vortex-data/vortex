// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::pin_mut;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::Canonical;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::FieldPath;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::sparse::SparseExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::collect::CollectStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::dtype::FieldNames;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::v2::ScanDriver;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::compress::read_projection;
use vortex_bench::conversions::parquet_to_vortex_chunks;

/// Compressor implementation for Vortex format.
pub struct VortexCompressor;

#[async_trait]
impl Compressor for VortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn load(&self, parquet_path: &Path) -> Result<Uncompressed> {
        let chunks = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
        Ok(Uncompressed::Vortex(chunks.into_array()))
    }

    async fn compress(&self, input: &Uncompressed) -> Result<Compressed> {
        let array = input.vortex()?.clone();

        // Run the timed work as a task so it uses only the runtime's worker threads, the same
        // pool the Parquet compressor runs on.
        let start = Instant::now();
        let buf = tokio::spawn(async move {
            let mut buf = Vec::new();
            let mut options = SESSION.write_options();
            if let Some(strategy) = sparse_struct_strategy(&array)? {
                options = options.with_strategy(strategy);
            }
            options
                .write(&mut Cursor::new(&mut buf), array.to_array_stream())
                .await?;
            anyhow::Ok(buf)
        })
        .await??;
        let elapsed = start.elapsed();

        Ok(Compressed {
            size: buf.len() as u64,
            data: CompressedData::Bytes(Bytes::from(buf)),
            elapsed,
        })
    }

    async fn decompress(&self, compressed: &Compressed) -> Result<Duration> {
        let CompressedData::Bytes(data) = &compressed.data else {
            bail!("Vortex decompression expects in-memory bytes");
        };
        let data = data.clone();

        let start = Instant::now();
        tokio::spawn(vortex_decompress_read(data)).await??;
        Ok(start.elapsed())
    }
}

/// EXPERIMENT: top-level fields that are more than 90% null are written as a single flat segment
/// holding a null-fill Sparse array, so everything nested beneath a null row is dropped and the
/// children hold only the present rows. Returns `None`, keeping the default writer, when no field
/// qualifies.
fn sparse_struct_strategy(array: &ArrayRef) -> Result<Option<Arc<dyn LayoutStrategy>>> {
    let mut ctx = SESSION.create_execution_ctx();
    let table = array.clone().execute::<StructArray>(&mut ctx)?;
    let len = table.len();
    let mut builder = WriteStrategyBuilder::from_session(&SESSION);
    let mut any = false;
    for (name, field) in table.names().iter().zip(table.iter_unmasked_fields()) {
        let nulls = field.validity()?.execute_mask(len, &mut ctx)?.false_count();
        if len == 0 || nulls == len || (nulls as f64) <= 0.9 * len as f64 {
            continue;
        }
        let btr = BtrBlocksCompressorBuilder::from_session(&SESSION).build();
        let writer = CollectStrategy::new(CompressingStrategy::new(
            FlatLayoutStrategy::default(),
            move |chunk: &ArrayRef, ctx: &mut ExecutionCtx| sparse_encode(&btr, chunk, ctx),
        ));
        builder = builder.with_field_writer(FieldPath::from_name(name.clone()), Arc::new(writer));
        any = true;
    }
    Ok(any.then(|| builder.build()))
}

fn sparse_encode(
    btr: &BtrBlocksCompressor,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> vortex::error::VortexResult<ArrayRef> {
    let canonical = chunk.clone().execute::<Canonical>(ctx)?.into_array();
    let encoded = Sparse::encode(&canonical, None, ctx)?;
    let Some(sparse) = encoded.as_opt::<Sparse>() else {
        return btr.compress(&encoded, ctx);
    };
    let patches = sparse.patches();
    let indices = btr.compress(patches.indices(), ctx)?;
    let values = btr.compress(patches.values(), ctx)?;
    Ok(Sparse::try_new(
        indices,
        values,
        canonical.len(),
        sparse.fill_scalar().clone(),
        ctx,
    )?
    .into_array())
}

/// Scan every record batch out of an in-memory Vortex file.
async fn vortex_decompress_read(data: Bytes) -> Result<()> {
    let file = SESSION.open_options().open_buffer(data)?;
    let mut scan = ScanBuilder::new(SESSION.clone(), file.scan_file());
    let source_dtype = scan.dtype().clone();
    let root_columns = source_dtype
        .as_struct_fields_opt()
        .map_or(0, |fields| fields.nfields());
    if let Some(cols) = read_projection(root_columns) {
        // Columns are named "0".."num_columns-1"; project the given subset.
        let names: FieldNames = cols.iter().map(|i| i.to_string()).collect();
        let projection = select(names, root())
            .bind(&source_dtype)?
            .optimize_recursive()?;
        scan = scan.with_projection(projection);
    }
    let stream = ScanDriver::new(SESSION.handle(), 4, true).run_record_batches(scan, None);
    pin_mut!(stream);

    while let Some(batch) = stream.next().await {
        let _batch = batch?;
    }
    Ok(())
}
