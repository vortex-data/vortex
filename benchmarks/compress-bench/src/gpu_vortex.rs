// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
#[cfg(target_os = "linux")]
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;

/// Vortex compressor whose decompression measurement executes CUDA-compatible files on the GPU.
pub struct GpuVortexCompressor;

#[async_trait]
impl Compressor for GpuVortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn compress(&self, _parquet_path: &Path) -> Result<(u64, Duration)> {
        anyhow::bail!("GPU compress-bench only supports decompression measurements")
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        register_cuda_layout(&SESSION);

        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
        let gpu_file = NamedTempFile::new()?;
        let mut output = tokio::fs::File::create(gpu_file.path()).await?;
        let strategy = WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().only_cuda_compatible())
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
            .build();
        SESSION
            .write_options()
            .with_strategy(strategy)
            .write(&mut output, uncompressed.into_array().to_array_stream())
            .await?;
        output.sync_all().await?;
        drop(output);

        let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
        let start = Instant::now();
        let open_options = SESSION.open_options().with_cuda();
        // Direct IO keeps repeated iterations measuring storage bandwidth rather than
        // page-cache hits. It is only available on Linux.
        #[cfg(target_os = "linux")]
        let open_options =
            open_options.with_read_at_options(PooledFileReadAtOptions::default().with_direct_io());
        let file = open_options.open_path(gpu_file.path()).await?;
        let mut batches = file.scan()?.into_array_stream()?;

        while let Some(batch) = batches.next().await {
            let record = batch?.execute::<StructArray>(cuda_ctx.execution_ctx())?;
            for field in record.iter_unmasked_fields() {
                black_box(field.clone().execute_cuda(&mut cuda_ctx).await?);
            }
        }
        cuda_ctx.synchronize_stream()?;

        Ok(start.elapsed())
    }
}
