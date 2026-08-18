// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use async_trait::async_trait;
use clap::ValueEnum;
use cudarc::driver::CudaEvent;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::nvtx::safe::scoped_range;
use futures::StreamExt;
use serde::Serialize;
use tempfile::NamedTempFile;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
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
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
#[cfg(target_os = "linux")]
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::executor::CudaExecutionCtx;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;

use crate::gpu_writer::GPU_ROW_GROUP_SIZE;

/// Optional diagnostics for the Vortex GPU decompression path.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum GpuVortexProfile {
    /// Record host wall time for each stage and encoding dispatch group.
    Wall,
    /// Also bracket each field dispatch with CUDA events and report device-stream elapsed time.
    Gpu,
    /// Add NVTX ranges for correlating field dispatches with an Nsight Systems capture.
    Nsys,
}

impl GpuVortexProfile {
    fn records_gpu_spans(self) -> bool {
        matches!(self, Self::Gpu)
    }

    fn records_nsys_ranges(self) -> bool {
        matches!(self, Self::Nsys)
    }
}

/// Vortex compressor whose decompression measurement executes CUDA-compatible files on the GPU.
pub struct GpuVortexCompressor {
    verify: bool,
    direct_io: bool,
    profile: Option<GpuVortexProfile>,
    dataset: Arc<str>,
    iteration: AtomicUsize,
}

impl GpuVortexCompressor {
    /// Create the backend.
    ///
    /// When `verify` is set, each GPU-decoded field is copied back to the host and compared
    /// against the same field decoded on the CPU. Verification runs inline, so timings from a
    /// verifying run are not comparable to a plain one.
    pub fn new(
        verify: bool,
        direct_io: bool,
        profile: Option<GpuVortexProfile>,
        dataset: &str,
    ) -> Self {
        Self {
            verify,
            direct_io,
            profile,
            dataset: dataset.into(),
            iteration: AtomicUsize::new(0),
        }
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

        // Match the Parquet writer's row-group size so both formats expose the same number of
        // independently readable row partitions to their GPU decoder. The default Arrow reader
        // batch size is only 8K rows, which otherwise creates hundreds of tiny CUDA-flat layouts
        // and thousands of tiny kernel launches for Vortex.
        let uncompressed = parquet_to_vortex_chunks_with_batch_size(
            parquet_path.to_path_buf(),
            Some(GPU_ROW_GROUP_SIZE),
        )
        .await?;
        let gpu_file = NamedTempFile::new()?;
        let mut output = tokio::fs::File::create(gpu_file.path()).await?;
        // Preserve the exact input partitions at the root. The general-purpose file strategy
        // splits struct fields and may introduce field-specific chunk boundaries, which makes a
        // fixed-row GPU scan yield ChunkedArray fields that have no CUDA execution kernel.
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

        if self.verify {
            return verify_against_host_scan(gpu_file.path(), self.direct_io).await;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
        // Match cuDF's untimed read: pay CUDA module loading, allocator initialization and page
        // faults before measuring. The helper opens and scans the file afresh each time, and the
        // CUDA opener disables its data-segment cache, so no decoded arrays are reused.
        decode_gpu_file(
            gpu_file.path(),
            self.direct_io,
            &mut cuda_ctx,
            "warmup",
            None,
        )
        .await?;

        let profile = self.profile.map(|mode| ProfileRun {
            mode,
            dataset: self.dataset.as_ref(),
            iteration: self.iteration.fetch_add(1, Ordering::Relaxed),
        });
        let start = Instant::now();
        decode_gpu_file(
            gpu_file.path(),
            self.direct_io,
            &mut cuda_ctx,
            "timed",
            profile,
        )
        .await?;
        Ok(start.elapsed())
    }
}

/// Runtime information attached to one profiled timed decode.
#[derive(Clone, Copy)]
struct ProfileRun<'a> {
    mode: GpuVortexProfile,
    dataset: &'a str,
    iteration: usize,
}

struct FieldTiming {
    field_name: String,
    rows: usize,
    encoding: String,
    tree: String,
    wall_time: Duration,
    gpu_events: Option<(CudaEvent, CudaEvent)>,
}

#[derive(Default)]
struct EncodingAggregate {
    calls: usize,
    rows: usize,
    fields: BTreeSet<String>,
    wall_time: Duration,
    gpu_time_us: Option<u64>,
}

#[derive(Serialize)]
struct EncodingProfileRecord {
    encoding: String,
    tree: String,
    fields: Vec<String>,
    calls: usize,
    rows: usize,
    wall_us: u64,
    gpu_us: Option<u64>,
}

#[derive(Serialize)]
struct StageProfileRecord {
    total_us: u64,
    open_us: u64,
    scan_plan_us: u64,
    read_us: u64,
    struct_dispatch_us: u64,
    field_dispatch_us: u64,
    final_sync_us: u64,
    profile_overhead_us: u64,
}

#[derive(Serialize)]
struct GpuProfileRecord<'a> {
    record: &'static str,
    version: u8,
    dataset: &'a str,
    iteration: usize,
    mode: &'static str,
    direct_io: bool,
    file_bytes: u64,
    data_segment_bytes: u64,
    data_segments: usize,
    root_layout: String,
    root_layout_children: usize,
    file_rows: u64,
    decoded_rows: usize,
    batches: usize,
    fields_per_batch: usize,
    field_dispatches: usize,
    stages: StageProfileRecord,
    encodings: Vec<EncodingProfileRecord>,
}

/// Decode every row and column from a fresh file open into device-resident canonical arrays.
async fn decode_gpu_file(
    path: &Path,
    direct_io: bool,
    cuda_ctx: &mut CudaExecutionCtx,
    phase: &'static str,
    profile: Option<ProfileRun<'_>>,
) -> Result<()> {
    let total_start = profile.map(|_| Instant::now());
    let open_start = Instant::now();
    let file = open_gpu(path, direct_io).await?;
    let open_time = open_start.elapsed();

    let file_bytes = profile
        .is_some()
        .then(|| std::fs::metadata(path).map(|metadata| metadata.len()))
        .transpose()?;
    let (data_segments, data_segment_bytes, root_layout, root_layout_children, file_rows) =
        if profile.is_some() {
            let footer = file.footer();
            (
                footer.segment_map().len(),
                footer
                    .segment_map()
                    .iter()
                    .map(|segment| u64::from(segment.length))
                    .sum(),
                footer.layout().encoding_id().to_string(),
                footer.layout().nchildren(),
                file.row_count(),
            )
        } else {
            (0, 0, String::new(), 0, 0)
        };

    let scan_start = Instant::now();
    let mut batches = file
        .scan()?
        .with_split_by(SplitBy::RowCount(GPU_ROW_GROUP_SIZE))
        .into_array_stream()?;
    let scan_time = scan_start.elapsed();

    let mut read_time = Duration::ZERO;
    let mut struct_time = Duration::ZERO;
    let mut execute_time = Duration::ZERO;
    let mut batch_count = 0usize;
    let mut decoded_rows = 0usize;
    let mut fields_per_batch = 0usize;
    let mut field_count = 0usize;
    let profile_gpu_spans = profile.is_some_and(|run| run.mode.records_gpu_spans());
    let profile_nsys = profile.is_some_and(|run| run.mode.records_nsys_ranges());
    let mut field_timings = Vec::new();
    loop {
        let read_start = Instant::now();
        let Some(batch) = batches.next().await else {
            read_time += read_start.elapsed();
            break;
        };
        read_time += read_start.elapsed();

        let struct_start = Instant::now();
        let record = batch?.execute::<StructArray>(cuda_ctx.execution_ctx())?;
        struct_time += struct_start.elapsed();
        batch_count += 1;
        decoded_rows += record.len();
        if batch_count == 1 {
            fields_per_batch = record.struct_fields().names().len();
        }
        if phase == "warmup" && std::env::var_os("VORTEX_GPU_DUMP_ARRAY_TREES").is_some() {
            eprintln!(
                "VORTEX_GPU_ARRAY_TREE batch={}\n{}",
                batch_count - 1,
                record.clone().into_array().display_tree()
            );
        }

        for (field_index, (field, field_name)) in record
            .iter_unmasked_fields()
            .zip(record.struct_fields().names().iter())
            .enumerate()
        {
            let metadata = profile.map(|_| {
                (
                    field.encoding_id().to_string(),
                    field
                        .display_tree_encodings_only()
                        .to_string()
                        .replace('\n', " | "),
                )
            });
            let before = profile_gpu_spans
                .then(|| cuda_ctx.stream().record_event(Some(CU_EVENT_DEFAULT)))
                .transpose()?;
            let nsys_range = profile_nsys.then(|| {
                scoped_range(format!(
                    "vortex_field batch={} field={field_index} name={field_name} encoding={}",
                    batch_count - 1,
                    field.encoding_id(),
                ))
            });
            let execute_start = Instant::now();
            black_box(field.clone().execute_cuda(cuda_ctx).await?);
            let wall_time = execute_start.elapsed();
            drop(nsys_range);
            execute_time += wall_time;
            let gpu_events = if let Some(before) = before {
                let after = cuda_ctx.stream().record_event(Some(CU_EVENT_DEFAULT))?;
                Some((before, after))
            } else {
                None
            };
            if let Some((encoding, tree)) = metadata {
                field_timings.push(FieldTiming {
                    field_name: field_name.to_string(),
                    rows: field.len(),
                    encoding,
                    tree,
                    wall_time,
                    gpu_events,
                });
            }
            field_count += 1;
        }
    }

    let sync_start = Instant::now();
    cuda_ctx.synchronize_stream()?;
    let sync_time = sync_start.elapsed();
    let total_time = total_start.map(|start| start.elapsed());

    if let Some(profile) = profile {
        let mut encodings: BTreeMap<(String, String), EncodingAggregate> = BTreeMap::new();
        for timing in field_timings {
            let gpu_time_us = timing
                .gpu_events
                .map(|(before, after)| before.elapsed_ms(&after))
                .transpose()?
                .map(|milliseconds| duration_us(Duration::from_secs_f32(milliseconds / 1_000.0)));
            let aggregate = encodings.entry((timing.encoding, timing.tree)).or_default();
            aggregate.calls += 1;
            aggregate.rows += timing.rows;
            aggregate.fields.insert(timing.field_name);
            aggregate.wall_time += timing.wall_time;
            aggregate.gpu_time_us = match (aggregate.gpu_time_us, gpu_time_us) {
                (Some(total), Some(elapsed)) => Some(total.saturating_add(elapsed)),
                (None, Some(elapsed)) => Some(elapsed),
                (total, None) => total,
            };
        }
        let encodings = encodings
            .into_iter()
            .map(|((encoding, tree), aggregate)| EncodingProfileRecord {
                encoding,
                tree,
                fields: aggregate.fields.into_iter().collect(),
                calls: aggregate.calls,
                rows: aggregate.rows,
                wall_us: duration_us(aggregate.wall_time),
                gpu_us: aggregate.gpu_time_us,
            })
            .collect();
        let accounted_time =
            open_time + scan_time + read_time + struct_time + execute_time + sync_time;
        let total_time = total_time.unwrap_or_default();
        let record = GpuProfileRecord {
            record: "vortex_gpu_decompress_profile",
            version: 1,
            dataset: profile.dataset,
            iteration: profile.iteration,
            mode: match profile.mode {
                GpuVortexProfile::Wall => "wall",
                GpuVortexProfile::Gpu => "gpu",
                GpuVortexProfile::Nsys => "nsys",
            },
            direct_io,
            file_bytes: file_bytes.unwrap_or_default(),
            data_segment_bytes,
            data_segments,
            root_layout,
            root_layout_children,
            file_rows,
            decoded_rows,
            batches: batch_count,
            fields_per_batch,
            field_dispatches: field_count,
            stages: StageProfileRecord {
                total_us: duration_us(total_time),
                open_us: duration_us(open_time),
                scan_plan_us: duration_us(scan_time),
                read_us: duration_us(read_time),
                struct_dispatch_us: duration_us(struct_time),
                field_dispatch_us: duration_us(execute_time),
                final_sync_us: duration_us(sync_time),
                profile_overhead_us: duration_us(total_time.saturating_sub(accounted_time)),
            },
            encodings,
        };
        eprintln!("{}", serde_json::to_string(&record)?);
    }

    tracing::debug!(
        phase,
        batch_count,
        field_count,
        ?open_time,
        ?scan_time,
        ?read_time,
        ?struct_time,
        ?execute_time,
        ?sync_time,
        "GPU Vortex decode stages"
    );
    Ok(())
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// Opens a Vortex file for CUDA execution.
///
/// `direct_io` is off by default so this backend is comparable with the cuDF one: cuDF reads
/// through the page cache after an untimed warm-up read, so leaving direct IO on would compare
/// a Vortex read of the disk against a cuDF read of RAM. Turning it on measures storage
/// bandwidth instead, which is a different question and not comparable across the two.
async fn open_gpu(path: &Path, direct_io: bool) -> Result<vortex::file::VortexFile> {
    let open_options = SESSION.open_options().with_cuda();
    #[cfg(target_os = "linux")]
    let open_options = if direct_io {
        open_options.with_read_at_options(PooledFileReadAtOptions::default().with_direct_io())
    } else {
        open_options
    };
    #[cfg(not(target_os = "linux"))]
    let _ = direct_io;
    Ok(open_options.open_path(path).await?)
}

/// Decodes the same file on the GPU and on the CPU and fails on the first difference.
///
/// The CPU reference comes from a second, host-only scan rather than from re-decoding the
/// GPU scan's arrays: a CUDA scan hands back arrays whose buffers live in device memory,
/// which the host decoders cannot read.
///
/// Verification runs inline, so the returned duration is not comparable to a plain run.
async fn verify_against_host_scan(path: &Path, direct_io: bool) -> Result<Duration> {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&SESSION)?;
    // Everything on the reference side — the host scan and both Arrow conversions — has to run
    // through a plain host context. A CUDA context allocates its outputs in device memory, and
    // the Arrow conversion then reads those buffers on the host.
    let mut host_ctx = SESSION.create_execution_ctx();
    let start = Instant::now();

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
    loop {
        let (gpu_batch, host_batch) = (gpu_batches.next().await, host_batches.next().await);
        let (gpu_batch, host_batch) = match (gpu_batch, host_batch) {
            (Some(gpu_batch), Some(host_batch)) => (gpu_batch?, host_batch?),
            (None, None) => break,
            _ => bail!("the GPU and CPU scans of the same file produced different batch counts"),
        };

        let gpu_record = gpu_batch.execute::<StructArray>(cuda_ctx.execution_ctx())?;
        let host_record = host_batch.execute::<StructArray>(&mut host_ctx)?;
        ensure!(
            gpu_record.len() == host_record.len(),
            "batch {batch_index} length differs between the GPU and CPU scans: {} vs {}",
            gpu_record.len(),
            host_record.len()
        );

        let gpu_fields = gpu_record
            .iter_unmasked_fields()
            .cloned()
            .collect::<Vec<_>>();
        let host_fields = host_record
            .iter_unmasked_fields()
            .cloned()
            .collect::<Vec<_>>();
        ensure!(
            gpu_fields.len() == host_fields.len(),
            "batch {batch_index} field count differs between the GPU and CPU scans"
        );

        for (field_index, (gpu_field, host_field)) in
            gpu_fields.into_iter().zip(host_fields).enumerate()
        {
            let decoded = gpu_field.execute_cuda(&mut cuda_ctx).await?;
            // The decode is enqueued, not complete: make the writes visible before reading
            // the buffers back to the host.
            cuda_ctx.synchronize_stream()?;
            let decoded = decoded.into_host().await?.into_array();
            verify_field(
                &host_field,
                decoded,
                &mut host_ctx,
                batch_index,
                field_index,
            )?;
            fields_checked += 1;
        }

        batch_index += 1;
    }
    cuda_ctx.synchronize_stream()?;

    tracing::info!("verified {fields_checked} GPU-decoded Vortex fields against the CPU decode");
    Ok(start.elapsed())
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
