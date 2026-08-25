// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use tempfile::NamedTempFile;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::scan::split_by::SplitBy;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::conversions::parquet_to_vortex_chunks_with_batch_size;
use vortex_cuda::CanonicalCudaExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
#[cfg(target_os = "linux")]
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;

use crate::gpu::writer::GPU_ROW_GROUP_SIZE;

/// Vortex compressor whose decompression measurement executes CUDA-compatible files on the GPU.
pub struct GpuVortexCompressor {
    verify: bool,
    direct_io: bool,
}

impl GpuVortexCompressor {
    /// Create the backend.
    ///
    /// When `verify` is set, each GPU-decoded field is copied back to the host and compared
    /// against the same field decoded on the CPU before the timed scan runs. The verification is
    /// not itself timed, so a verifying run reports the same measurement a plain one would — it
    /// just takes longer to get there.
    pub fn new(verify: bool, direct_io: bool) -> Self {
        Self { verify, direct_io }
    }
}

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

        // Rebatch to the same partition size the GPU Parquet file is written with. Left alone,
        // the Arrow reader hands back ~8K-row batches, each of which becomes its own Vortex
        // chunk and its own set of kernel launches.
        let uncompressed = parquet_to_vortex_chunks_with_batch_size(
            parquet_path.to_path_buf(),
            Some(GPU_ROW_GROUP_SIZE),
        )
        .await?;
        let gpu_file = NamedTempFile::new()?;
        let mut output = tokio::fs::File::create(gpu_file.path()).await?;
        // Write those batches straight through as root chunks, so a chunk on disk is one
        // partition rather than whatever the default strategy would regroup them into.
        let strategy = Arc::new(ChunkedLayoutStrategy::new(CompressingStrategy::new(
            CudaFlatLayoutStrategy::default(),
            BtrBlocksCompressorBuilder::default()
                .only_cuda_compatible()
                .build(),
        )));
        SESSION
            .write_options()
            .with_strategy(strategy)
            .write(&mut output, uncompressed.into_array().to_array_stream())
            .await?;
        output.sync_all().await?;
        drop(output);

        // Verification is a precondition on the measurement below, not a substitute for it. It
        // used to return its own elapsed time, which bundled a file copy, a second host scan and
        // every Arrow conversion into a number the table then published as a decode time.
        if self.verify {
            verify_against_host_scan(gpu_file.path(), self.direct_io).await?;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
        let start = Instant::now();
        let file = open_gpu(gpu_file.path(), self.direct_io).await?;
        // Split reads on the same boundary the file was written with, so a scan batch is one
        // partition instead of a sub-slice of one.
        let mut batches = file
            .scan()?
            .with_split_by(SplitBy::RowCount(GPU_ROW_GROUP_SIZE))
            .into_array_stream()?;

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

/// Opens a Vortex file for CUDA execution.
///
/// `direct_io` is off by default so this backend is comparable with the cuDF one: cuDF reads
/// through the page cache after an untimed warm-up read, so leaving direct IO on would compare
/// a Vortex read of the disk against a cuDF read of RAM. Turning it on measures storage
/// bandwidth instead, which is a different question and not comparable across the two.
///
/// Only the direct-IO path is Linux-only — `O_DIRECT` has no portable equivalent, so
/// [`PooledFileReadAtOptions`] only offers it there. The rest of this module is not: CUDA runs on
/// Windows too, and the whole crate still has to compile on a developer's macOS machine. Asking
/// for `--gpu-direct-io` where it cannot be honoured is an error rather than a silent no-op,
/// because the flag changes what the resulting number means.
async fn open_gpu(path: &Path, direct_io: bool) -> Result<vortex::file::VortexFile> {
    let open_options = SESSION.open_options().with_cuda();

    #[cfg(target_os = "linux")]
    let open_options = if direct_io {
        open_options.with_read_at_options(PooledFileReadAtOptions::default().with_direct_io())
    } else {
        open_options
    };

    #[cfg(not(target_os = "linux"))]
    anyhow::ensure!(
        !direct_io,
        "--gpu-direct-io needs O_DIRECT, which is only available on Linux"
    );

    Ok(open_options.open_path(path).await?)
}

/// Decodes the same file on the GPU and on the CPU and fails on the first difference.
///
/// The CPU reference comes from a second, host-only scan rather than from re-decoding the
/// GPU scan's arrays: a CUDA scan hands back arrays whose buffers live in device memory,
/// which the host decoders cannot read.
///
/// This times nothing and reports no measurement: the caller runs its own timed scan afterwards,
/// so a verifying run publishes the same kind of number as a plain one.
async fn verify_against_host_scan(path: &Path, direct_io: bool) -> Result<()> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
    // Everything on the reference side — the host scan and both Arrow conversions — has to run
    // through a plain host context. A CUDA context allocates its outputs in device memory, and
    // the Arrow conversion then reads those buffers on the host.
    let mut host_ctx = SESSION.create_execution_ctx();

    // The host scan reads a copy rather than the same path. The session's segment cache is
    // keyed by URI, and the CUDA reader deliberately bypasses it because its buffers are
    // device-resident; running both scans against one URI risks the two sharing entries.
    let host_path = NamedTempFile::new()?;
    std::fs::copy(path, host_path.path())?;

    let gpu_file = open_gpu(path, direct_io).await?;
    let mut gpu_batches = gpu_file
        .scan()?
        .with_split_by(SplitBy::RowCount(GPU_ROW_GROUP_SIZE))
        .into_array_stream()?;
    let host_file = SESSION.open_options().open_path(host_path.path()).await?;
    let mut host_batches = host_file
        .scan()?
        .with_split_by(SplitBy::RowCount(GPU_ROW_GROUP_SIZE))
        .into_array_stream()?;

    let mut fields_checked = 0usize;
    let mut batch_index = 0usize;
    while let Some((gpu_batch, host_batch)) =
        next_batch_pair(&mut gpu_batches, &mut host_batches).await?
    {
        fields_checked += verify_batch(
            gpu_batch,
            host_batch,
            &mut cuda_ctx,
            &mut host_ctx,
            batch_index,
        )
        .await?;
        batch_index += 1;
    }
    cuda_ctx.synchronize_stream()?;

    tracing::info!("verified {fields_checked} GPU-decoded Vortex fields against the CPU decode");
    Ok(())
}

/// Verifies every field of one batch, returning the number of fields checked.
///
/// A GPU decode is enqueued rather than immediate, so each field synchronises the stream before
/// its buffers are read back to the host.
async fn verify_batch(
    gpu_batch: ArrayRef,
    host_batch: ArrayRef,
    cuda_ctx: &mut CudaExecutionCtx,
    host_ctx: &mut ExecutionCtx,
    batch_index: usize,
) -> Result<usize> {
    let gpu_record = gpu_batch.execute::<StructArray>(cuda_ctx.execution_ctx())?;
    let host_record = host_batch.execute::<StructArray>(host_ctx)?;
    ensure!(
        gpu_record.len() == host_record.len(),
        "batch {batch_index} length differs between the GPU and CPU scans: {} vs {}",
        gpu_record.len(),
        host_record.len()
    );

    let gpu_fields = gpu_record.iter_unmasked_fields();
    let host_fields = host_record.iter_unmasked_fields();
    let nfields = gpu_fields.len();
    ensure!(
        nfields == host_fields.len(),
        "batch {batch_index} field count differs between the GPU and CPU scans"
    );

    for (field_index, (gpu_field, host_field)) in gpu_fields.zip(host_fields).enumerate() {
        let decoded = gpu_field.clone().execute_cuda(cuda_ctx).await?;
        cuda_ctx.synchronize_stream()?;
        let decoded = decoded.into_host().await?.into_array();
        verify_field(host_field, decoded, host_ctx, batch_index, field_index)?;
    }

    Ok(nfields)
}

/// Pulls one batch from each scan, or `None` once both are exhausted.
///
/// Both streams cover the same file, so one ending before the other is itself a failure rather
/// than a stopping condition — that is why this returns `Result<Option<_>>` instead of letting
/// the caller zip the two streams together.
async fn next_batch_pair(
    gpu: &mut (impl Stream<Item = VortexResult<ArrayRef>> + Unpin),
    host: &mut (impl Stream<Item = VortexResult<ArrayRef>> + Unpin),
) -> Result<Option<(ArrayRef, ArrayRef)>> {
    match (gpu.next().await, host.next().await) {
        (Some(gpu_batch), Some(host_batch)) => Ok(Some((gpu_batch?, host_batch?))),
        (None, None) => Ok(None),
        _ => bail!("the GPU and CPU scans of the same file produced different batch counts"),
    }
}

/// Fails unless a GPU-decoded field matches the same field decoded on the CPU.
fn verify_field(
    host: &ArrayRef,
    gpu: ArrayRef,
    ctx: &mut ExecutionCtx,
    batch_index: usize,
    field_index: usize,
) -> Result<()> {
    let expected = SESSION.arrow().execute_arrow(host.clone(), None, ctx)?;
    // Pin the Arrow target type so the two sides cannot land on different but equivalent
    // encodings of the same logical values.
    let target = Field::new("", expected.data_type().clone(), gpu.dtype().is_nullable());
    let actual = SESSION.arrow().execute_arrow(gpu, Some(&target), ctx)?;

    if expected.to_data() == actual.to_data() {
        return Ok(());
    }

    bail!(
        "GPU decode of a {} field does not match the CPU decode \
         (batch {batch_index}, field {field_index}){}",
        host.encoding_id(),
        describe_mismatch(&expected, &actual)
    )
}

/// Builds a human-readable description of how two Arrow arrays differ.
fn describe_mismatch(expected: &ArrowArrayRef, actual: &ArrowArrayRef) -> String {
    let mut description = format!(
        "\n  cpu: type={:?} len={} nulls={}\n  gpu: type={:?} len={} nulls={}",
        expected.data_type(),
        expected.len(),
        expected.null_count(),
        actual.data_type(),
        actual.len(),
        actual.null_count(),
    );

    if expected.data_type() != actual.data_type() || expected.len() != actual.len() {
        return description;
    }

    // Binary search for the shortest prefix that already differs; its last element is the
    // first mismatching row.
    let (mut low, mut high) = (0usize, expected.len());
    while low < high {
        let mid = low + (high - low) / 2 + 1;
        if expected.slice(0, mid).to_data() == actual.slice(0, mid).to_data() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    if low < expected.len() {
        description.push_str(&format!(
            "\n  first difference at row {low}:\n    cpu: {:?}\n    gpu: {:?}",
            expected.slice(low, 1),
            actual.slice(low, 1),
        ));
    }

    description
}
