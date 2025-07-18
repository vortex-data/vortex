// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(dead_code)]
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_io::VortexReadAt; // Assuming this trait is from your vortex_io crate

#[derive(Debug, Clone)]
pub struct PerformanceHint {
    pub coalescing_window: u64,
    pub max_read: Option<u64>,
}

/// A wrapper struct for [`VortexReadAt`] that tracks performance metrics and provides a hint for
/// coalescing.
pub struct PerformanceHintingReadAt<T> {
    inner: T,
    config: PerformanceHintingConfig,
    metrics: Arc<RwLock<MetricsState>>,
}

pub struct PerformanceHintingConfig {
    /// E.g., 0.90 for P90 latency
    pub target_latency_quantile: f64,
    /// E.g., 0.50 for P50 throughput
    pub target_throughput_quantile: f64,
    /// How many times target_latency should be covered by throughput
    pub target_latency_amplification_factor: f64,
    /// Ignore reads smaller than this for statistics
    pub min_read_size_for_metrics: u64,
}

impl Default for PerformanceHintingConfig {
    fn default() -> Self {
        Self {
            // P90 latency
            target_latency_quantile: 0.90,
            // P50 throughput (median)
            target_throughput_quantile: 0.50,
            // Target transfer time is 5x P90 latency
            target_latency_amplification_factor: 5.0,
            // Ignore reads smaller than 4KB (page size) for metrics
            min_read_size_for_metrics: 4096,
        }
    }
}

struct MetricsState {
    /// Stores latency in microseconds
    latency_sketch: DDSketch,
    /// Stores throughput in Bytes/sec
    throughput_sketch: DDSketch,
    // Add a timestamp to allow for periodic resetting/decaying of metrics if desired.
    // This was in a previous version and is useful for long-running systems.
    // last_reset_time: Instant,
}

impl<T> PerformanceHintingReadAt<T>
where
    T: VortexReadAt,
{
    /// Creates a new tracking reader with default configuration.
    pub fn new(inner: T) -> Self {
        Self::with_config(inner, Default::default())
    }

    /// Creates a new tracking reader with custom configuration.
    pub fn with_config(inner: T, config: PerformanceHintingConfig) -> Self {
        Self {
            inner,
            config,
            metrics: Arc::new(RwLock::new(MetricsState {
                latency_sketch: DDSketch::default(),
                throughput_sketch: DDSketch::default(),
            })),
        }
    }

    /// Computes the current performance hint based on collected DDSketch metrics.
    /// This method is intended to be called by an external mechanism (e.g., a background task)
    /// or periodically to refresh the hint.
    pub fn compute_performance_hint(&self) -> PerformanceHint {
        let metrics = self.metrics.read();

        let p_latency_micros = metrics
            .latency_sketch
            .quantile(self.config.target_latency_quantile)
            .ok()
            .flatten()
            .unwrap_or(0.0); // Default to 0 if no data yet

        let p_throughput_bytes_per_sec = metrics
            .throughput_sketch
            .quantile(self.config.target_throughput_quantile)
            .ok()
            .flatten()
            .unwrap_or(0.0);

        log::debug!(
            "Throughput: {} bytes/s  Latency: {} us",
            p_throughput_bytes_per_sec,
            p_latency_micros
        );

        let mut coalescing_window = self.config.min_read_size_for_metrics;
        let max_read: Option<u64>;

        // Heuristic for coalescing window
        if p_throughput_bytes_per_sec > 0.0 && p_latency_micros > 0.0 {
            let p_latency_secs = p_latency_micros / 1_000_000.0;
            let target_transfer_time_secs =
                p_latency_secs * self.config.target_latency_amplification_factor;

            let estimated_coalescing_window_f64 =
                target_transfer_time_secs * p_throughput_bytes_per_sec;

            coalescing_window = estimated_coalescing_window_f64.round() as u64;
            coalescing_window = coalescing_window.max(self.config.min_read_size_for_metrics);
        }

        // --- Refined max_read determination ---
        let mut potential_max_read_mb;

        if p_throughput_bytes_per_sec > 0.0 {
            let target_max_read_transfer_duration_secs = 0.100; // 100 milliseconds

            let calculated_max_read_bytes =
                p_throughput_bytes_per_sec * target_max_read_transfer_duration_secs;

            let calculated_max_read_mb =
                (calculated_max_read_bytes / (1024.0 * 1024.0)).round() as u64;

            // Start with a base, e.g., 4MB, if throughput is decent
            potential_max_read_mb = 4;

            if calculated_max_read_mb > potential_max_read_mb {
                potential_max_read_mb = calculated_max_read_mb;
            }

            potential_max_read_mb = potential_max_read_mb.min(64); // Cap at 64MB
            potential_max_read_mb = potential_max_read_mb.max(1); // At least 1MB if throughput is positive

            max_read = Some(potential_max_read_mb * 1024 * 1024);
        } else {
            max_read = None;
        }

        let hint = PerformanceHint {
            coalescing_window,
            max_read,
        };

        log::debug!(
            "HINT window: {} bytes, max_read: {:?} bytes",
            hint.coalescing_window,
            hint.max_read
        );

        hint
    }
}

// Add async_trait to the VortexReadAt implementation for the wrapper, as it enables
// 'impl Future + Send' in the trait definition.
// This is necessary because your `VortexReadAt` trait is defined to return `impl Future<Output = io::Result<ByteBuffer>>`,
// and `async_trait` helps ensure that this `impl Future` is `Send` when using `async fn` in the wrapper.
// Also, the inner `T` needs to be `Send + Sync` for the wrapper to be.
impl<T> VortexReadAt for PerformanceHintingReadAt<T>
where
    T: VortexReadAt + Send + Sync,
{
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let start_time = Instant::now();
        let result = self.inner.read_byte_range(range.clone(), alignment).await;
        let end_time = Instant::now();

        if let Ok(ref buffer) = result {
            let bytes_read = buffer.len() as u64;

            if bytes_read >= self.config.min_read_size_for_metrics {
                let duration = end_time.duration_since(start_time);
                let latency_micros = duration.as_micros() as f64;
                let throughput_bytes_per_sec = if duration.as_secs_f64() > 0.0 {
                    bytes_read as f64 / duration.as_secs_f64()
                } else {
                    f64::INFINITY // Instant reads, effectively infinite throughput
                };

                let mut metrics = self.metrics.write(); // Acquire write lock here
                metrics.latency_sketch.add(latency_micros);
                metrics.throughput_sketch.add(throughput_bytes_per_sec);
            }
        }

        result
    }

    async fn size(&self) -> io::Result<u64> {
        self.inner.size().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

    // A mock implementation of VortexReadAt for testing
    // Needs async_trait macro if the trait itself uses it or returns `impl Future + Send`
    impl VortexReadAt for MockVortexReader {
        async fn read_byte_range(
            &self,
            range: Range<u64>,
            _alignment: Alignment,
        ) -> io::Result<ByteBuffer> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            let start_idx = range.start as usize;
            let end_idx = range.end as usize;

            if start_idx >= self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
            }

            let actual_end_idx = end_idx.min(self.data.len());
            let bytes_to_read = (actual_end_idx - start_idx) as u64;

            let mut total_delay_micros = self.read_delay_micros;

            // Simulate throughput limit based on bytes_to_read
            if self.throughput_limit_mbps > 0 {
                let throughput_bytes_per_sec = self.throughput_limit_mbps * 1024 * 1024;
                if throughput_bytes_per_sec > 0 {
                    let time_for_transfer_secs =
                        bytes_to_read as f64 / throughput_bytes_per_sec as f64;
                    let time_for_transfer_micros =
                        (time_for_transfer_secs * 1_000_000.0).round() as u64;
                    total_delay_micros += time_for_transfer_micros;
                }
            }

            if total_delay_micros > 0 {
                sleep(Duration::from_micros(total_delay_micros)).await;
            }

            let slice = &self.data[start_idx..actual_end_idx];
            Ok(ByteBuffer::from(slice.to_vec()))
        }

        async fn size(&self) -> io::Result<u64> {
            Ok(self.data.len() as u64)
        }
    }

    // A mock implementation of VortexReadAt for testing
    struct MockVortexReader {
        data: Vec<u8>,
        read_delay_micros: u64,     // Simulate base latency
        throughput_limit_mbps: u64, // Simulate throughput limit in MB/s
        call_count: AtomicUsize,
    }

    #[tokio::test]
    async fn test_tracking_reader_with_ddsketch() {
        let mock_data = vec![0; 1024 * 1024 * 20]; // 20 MB of data
        let mock_reader = MockVortexReader {
            data: mock_data,
            read_delay_micros: 200,     // 200 us base latency
            throughput_limit_mbps: 100, // 100 MB/s throughput
            call_count: AtomicUsize::new(0),
        };

        let tracking_reader = PerformanceHintingReadAt::new(mock_reader);

        // Perform many reads with varying sizes to populate the sketches
        for i in 0..50 {
            let offset = (i as u64 * 1024 * 100) % (1024 * 1024 * 10); // Loop within first 10MB
            let len = match i % 3 {
                0 => 1024 * 4,   // Small read (4KB)
                1 => 1024 * 64,  // Medium read (64KB)
                _ => 1024 * 512, // Large read (512KB)
            };
            let result = tracking_reader
                .read_byte_range(offset..offset + len, Alignment::new(1))
                .await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), len as usize);
            // No sleep here, let it run fast to get more samples in DDSketch quickly
        }

        // Wait a little for any background processing if there were any, though with DDSketch
        // it's mostly about collecting enough samples.
        sleep(Duration::from_millis(100)).await;

        let hint = tracking_reader.compute_performance_hint();
        println!("DDSketch Performance Hint: {:?}", hint);

        // Acquire a read lock to inspect metrics for assertions
        let metrics_guard = tracking_reader.metrics.read();

        // Print quantiles for debugging
        println!(
            "Latency (us) P50: {:?}",
            metrics_guard.latency_sketch.quantile(0.5)
        );
        println!(
            "Latency (us) P90: {:?}",
            metrics_guard.latency_sketch.quantile(0.9)
        );
        println!(
            "Throughput (B/s) P50: {:?}",
            metrics_guard.throughput_sketch.quantile(0.5)
        );
        println!(
            "Throughput (B/s) P90: {:?}",
            metrics_guard.throughput_sketch.quantile(0.9)
        );

        // Assertions based on expected values given mock config
        let p90_latency = metrics_guard
            .latency_sketch
            .quantile(0.9)
            .ok()
            .flatten()
            .unwrap_or(0.0);
        assert!(
            p90_latency > 150.0 && p90_latency < 500.0,
            "P90 Latency should be around 200-500 us, got {}",
            p90_latency
        );

        let p50_throughput = metrics_guard
            .throughput_sketch
            .quantile(0.5)
            .ok()
            .flatten()
            .unwrap_or(0.0);
        assert!(
            p50_throughput > 50_000_000.0 && p50_throughput < 150_000_000.0,
            "P50 Throughput should be around 100 MB/s, got {}",
            p50_throughput
        );

        // Check coalescing window heuristic based on the expected values
        assert!(
            hint.coalescing_window > 50 * 1024,
            "Coalescing window should be significant, got {}",
            hint.coalescing_window
        );
        assert!(
            hint.coalescing_window < 1024 * 1024,
            "Coalescing window should be less than 1MB, got {}",
            hint.coalescing_window
        );
    }

    #[tokio::test]
    async fn test_tracking_reader_no_reads_yet_ddsketch() {
        let mock_data = vec![0; 10];
        let mock_reader = MockVortexReader {
            data: mock_data,
            read_delay_micros: 0,
            throughput_limit_mbps: 0,
            call_count: AtomicUsize::new(0),
        };
        let tracking_reader = PerformanceHintingReadAt::new(mock_reader);

        let hint = tracking_reader.compute_performance_hint();
        println!("DDSketch Performance Hint (no reads): {:?}", hint);

        assert_eq!(
            hint.coalescing_window,
            tracking_reader.config.min_read_size_for_metrics
        );
        assert_eq!(hint.max_read, None);
    }
}
