// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use cudarc::driver::CudaEvent;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::nvtx::safe::scoped_range;
use futures::StreamExt;
use serde_json::json;
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

struct FieldTiming {
    encoding: String,
    tree: String,
    rows: usize,
    wall: Duration,
    events: Option<(CudaEvent, CudaEvent)>,
}

#[derive(Default)]
struct EncodingTiming {
    calls: usize,
    rows: usize,
    wall: Duration,
    gpu_us: Option<u64>,
}

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
        let profile = std::env::var("VORTEX_GPU_PROFILE").ok();
        anyhow::ensure!(
            profile
                .as_deref()
                .is_none_or(|mode| matches!(mode, "wall" | "gpu" | "nsys" | "nsys-ranges")),
            "VORTEX_GPU_PROFILE must be wall, gpu, nsys, or nsys-ranges"
        );
        if profile.is_none() {
            let start = Instant::now();
            let open_options = SESSION.open_options().with_cuda();
            // Direct IO keeps repeated iterations measuring storage bandwidth rather than
            // page-cache hits. It is only available on Linux.
            #[cfg(target_os = "linux")]
            let open_options = open_options
                .with_read_at_options(PooledFileReadAtOptions::default().with_direct_io());
            let file = open_options.open_path(gpu_file.path()).await?;
            let mut batches = file.scan()?.into_array_stream()?;

            while let Some(batch) = batches.next().await {
                let record = batch?.execute::<StructArray>(cuda_ctx.execution_ctx())?;
                for field in record.iter_unmasked_fields() {
                    black_box(field.clone().execute_cuda(&mut cuda_ctx).await?);
                }
            }
            cuda_ctx.synchronize_stream()?;

            return Ok(start.elapsed());
        }

        let profile_gpu = profile.as_deref() == Some("gpu");
        let profile_nsys = matches!(profile.as_deref(), Some("nsys" | "nsys-ranges"));
        let profile_trees = profile.as_deref() != Some("nsys-ranges");
        let start = Instant::now();
        let open_start = Instant::now();
        let open_options = SESSION.open_options().with_cuda();
        #[cfg(target_os = "linux")]
        let open_options =
            open_options.with_read_at_options(PooledFileReadAtOptions::default().with_direct_io());
        let file = open_options.open_path(gpu_file.path()).await?;
        let open_time = open_start.elapsed();

        let (file_bytes, data_segments, data_segment_bytes, root_layout, root_children, file_rows) =
            if profile.is_some() {
                let footer = file.footer();
                (
                    std::fs::metadata(gpu_file.path())?.len(),
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
                (0, 0, 0, String::new(), 0, 0)
            };

        let scan_start = Instant::now();
        let mut batches = file.scan()?.into_array_stream()?;
        let scan_time = scan_start.elapsed();
        let mut read_time = Duration::ZERO;
        let mut struct_time = Duration::ZERO;
        let mut field_time = Duration::ZERO;
        let mut batch_count = 0usize;
        let mut decoded_rows = 0usize;
        let mut field_count = 0usize;
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
            for field in record.iter_unmasked_fields() {
                let metadata = profile_trees.then(|| {
                    (
                        field.encoding_id().to_string(),
                        field
                            .display_tree_encodings_only()
                            .to_string()
                            .replace('\n', " | "),
                    )
                });
                let before = profile_gpu
                    .then(|| cuda_ctx.stream().record_event(Some(CU_EVENT_DEFAULT)))
                    .transpose()?;
                let range = profile_nsys.then(|| {
                    scoped_range(format!("vortex_field encoding={}", field.encoding_id()))
                });
                let field_start = Instant::now();
                black_box(field.clone().execute_cuda(&mut cuda_ctx).await?);
                let wall = field_start.elapsed();
                drop(range);
                field_time += wall;
                let events = if let Some(before) = before {
                    Some((
                        before,
                        cuda_ctx.stream().record_event(Some(CU_EVENT_DEFAULT))?,
                    ))
                } else {
                    None
                };
                if let Some((encoding, tree)) = metadata {
                    field_timings.push(FieldTiming {
                        encoding,
                        tree,
                        rows: field.len(),
                        wall,
                        events,
                    });
                }
                field_count += 1;
            }
        }
        let sync_start = Instant::now();
        cuda_ctx.synchronize_stream()?;
        let sync_time = sync_start.elapsed();
        let total = start.elapsed();

        if let Some(mode) = profile {
            let mut encodings = BTreeMap::<(String, String), EncodingTiming>::new();
            for timing in field_timings {
                let gpu_us = timing
                    .events
                    .map(|(before, after)| before.elapsed_ms(&after))
                    .transpose()?
                    .map(|ms| duration_us(Duration::from_secs_f32(ms / 1_000.0)));
                let aggregate = encodings.entry((timing.encoding, timing.tree)).or_default();
                aggregate.calls += 1;
                aggregate.rows += timing.rows;
                aggregate.wall += timing.wall;
                aggregate.gpu_us = match (aggregate.gpu_us, gpu_us) {
                    (Some(total), Some(value)) => Some(total.saturating_add(value)),
                    (None, value) => value,
                    (value, None) => value,
                };
            }
            let accounted =
                open_time + scan_time + read_time + struct_time + field_time + sync_time;
            let encodings = encodings
                .into_iter()
                .map(|((encoding, tree), timing)| {
                    json!({
                        "encoding": encoding,
                        "tree": tree,
                        "calls": timing.calls,
                        "rows": timing.rows,
                        "wall_us": duration_us(timing.wall),
                        "gpu_us": timing.gpu_us,
                    })
                })
                .collect::<Vec<_>>();
            eprintln!(
                "{}",
                json!({
                    "record": "vortex_gpu_decompress_profile",
                    "version": 1,
                    "dataset_path": parquet_path,
                    "mode": mode,
                    "file_bytes": file_bytes,
                    "data_segments": data_segments,
                    "data_segment_bytes": data_segment_bytes,
                    "root_layout": root_layout,
                    "root_layout_children": root_children,
                    "file_rows": file_rows,
                    "decoded_rows": decoded_rows,
                    "batches": batch_count,
                    "field_dispatches": field_count,
                    "stages": {
                        "total_us": duration_us(total),
                        "open_us": duration_us(open_time),
                        "scan_plan_us": duration_us(scan_time),
                        "read_us": duration_us(read_time),
                        "struct_dispatch_us": duration_us(struct_time),
                        "field_dispatch_us": duration_us(field_time),
                        "final_sync_us": duration_us(sync_time),
                        "profile_overhead_us": duration_us(total.saturating_sub(accounted)),
                    },
                    "encodings": encodings,
                })
            );
        }

        Ok(total)
    }
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
