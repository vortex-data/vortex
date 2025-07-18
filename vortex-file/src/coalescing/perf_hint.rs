// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(dead_code)]
use parking_lot::RwLock;
use sketches_ddsketch::{Config as DDSketchConfig, DDSketch};
use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_io::VortexReadAt;

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
    /// E.g., 0.90 for P90 fixed latency (T0)
    pub target_fixed_latency_quantile: f64,
    /// E.g., 0.50 for P50 bandwidth (B)
    pub target_bandwidth_quantile: f64,
    /// How many times target_fixed_latency should be covered by throughput
    pub target_latency_amplification_factor: f64,
    /// Ignore reads smaller than this for statistics
    pub min_read_size_for_metrics: u64,
    /// Configuration for the underlying DDSketches
    pub ddsketch_config: DDSketchConfig,
}

impl Default for PerformanceHintingConfig {
    fn default() -> Self {
        Self {
            // P90 fixed latency (T0)
            target_fixed_latency_quantile: 0.90,
            // P50 bandwidth (median is more stable)
            target_bandwidth_quantile: 0.50,
            // Target transfer time is 5x P90 fixed latency
            target_latency_amplification_factor: 5.0,
            // Ignore reads smaller than 4KB (page size) for metrics
            min_read_size_for_metrics: 4096,
            ddsketch_config: DDSketchConfig::default(),
        }
    }
}

struct MetricsState {
    /// Stores estimated fixed overhead (T0) in microseconds
    fixed_overhead_sketch: DDSketch,
    /// Stores estimated bandwidth (B) in Bytes/sec
    bandwidth_sketch: DDSketch,
    // Store the last calculated P50 bandwidth to use for T0 estimation on subsequent reads.
    last_p50_bandwidth_bytes_per_sec: f64,
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
        let ddsketch_config_for_init = config.ddsketch_config.clone();
        Self {
            inner,
            config,
            metrics: Arc::new(RwLock::new(MetricsState {
                fixed_overhead_sketch: DDSketch::new(ddsketch_config_for_init.clone()),
                bandwidth_sketch: DDSketch::new(ddsketch_config_for_init),
                last_p50_bandwidth_bytes_per_sec: 0.0, // Initialize to 0
            })),
        }
    }

    /// Computes the current performance hint based on collected DDSketch metrics.
    /// This method is intended to be called by an external mechanism (e.g., a background task)
    /// or periodically to refresh the hint.
    pub fn compute_performance_hint(&self) -> PerformanceHint {
        let mut metrics = self.metrics.write(); // Acquire write lock as we update last_p50_bandwidth

        let p_fixed_latency_micros = metrics
            .fixed_overhead_sketch
            .quantile(self.config.target_fixed_latency_quantile)
            .ok()
            .flatten()
            .unwrap_or(0.0); // Default to 0 if no data yet

        let p_bandwidth_bytes_per_sec = metrics
            .bandwidth_sketch
            .quantile(self.config.target_bandwidth_quantile)
            .ok()
            .flatten()
            .unwrap_or(0.0);

        // Update the cached bandwidth for subsequent T0 calculations
        metrics.last_p50_bandwidth_bytes_per_sec = p_bandwidth_bytes_per_sec;

        log::debug!(
            "Bandwidth (B): {} bytes/s  Fixed Latency (T0): {} us",
            p_bandwidth_bytes_per_sec,
            p_fixed_latency_micros
        );

        let mut coalescing_window = self.config.min_read_size_for_metrics;
        let max_read: Option<u64>;

        // Heuristic for coalescing window using estimated T0 and B
        if p_bandwidth_bytes_per_sec > 0.0 && p_fixed_latency_micros > 0.0 {
            let p_fixed_latency_secs = p_fixed_latency_micros / 1_000_000.0;
            let target_transfer_time_secs =
                p_fixed_latency_secs * self.config.target_latency_amplification_factor;

            let estimated_coalescing_window_f64 =
                target_transfer_time_secs * p_bandwidth_bytes_per_sec;

            coalescing_window = estimated_coalescing_window_f64.round() as u64;
            coalescing_window = coalescing_window.max(self.config.min_read_size_for_metrics);
        }

        // --- Refined max_read determination ---
        let mut potential_max_read_mb;

        if p_bandwidth_bytes_per_sec > 0.0 {
            let target_max_read_transfer_duration_secs = 0.100; // 100 milliseconds

            let calculated_max_read_bytes =
                p_bandwidth_bytes_per_sec * target_max_read_transfer_duration_secs;

            let calculated_max_read_mb =
                (calculated_max_read_bytes / (1024.0 * 1024.0)).round() as u64;

            // Start with a base, e.g., 8MB, if throughput is decent
            potential_max_read_mb = 8;

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
                let total_duration_micros = end_time.duration_since(start_time).as_micros() as f64;

                // --- Calculate Bandwidth (B) ---
                let estimated_bandwidth_for_this_read_bytes_per_sec = if total_duration_micros > 0.0
                {
                    bytes_read as f64 / (total_duration_micros / 1_000_000.0)
                } else {
                    f64::INFINITY // Instant reads
                };

                // --- Calculate Fixed Overhead (T0) ---
                let estimated_t0_micros = {
                    let metrics_read_guard = self.metrics.read(); // Temporarily read lock to get last_p50_bandwidth
                    let current_p50_bandwidth = metrics_read_guard.last_p50_bandwidth_bytes_per_sec;
                    drop(metrics_read_guard); // Release read lock quickly

                    let calculated_t0;
                    if current_p50_bandwidth > 0.0 {
                        let transfer_time_micros =
                            (bytes_read as f64 / current_p50_bandwidth) * 1_000_000.0;
                        calculated_t0 = total_duration_micros - transfer_time_micros;
                    } else {
                        // If no bandwidth estimate yet, or very low, treat total_duration as T0
                        calculated_t0 = total_duration_micros;
                    }
                    // T0 cannot be negative. If calculation makes it negative (e.g., due to noise), cap at 0.
                    calculated_t0.max(0.0)
                };

                let mut metrics_write_guard = self.metrics.write(); // Acquire write lock
                metrics_write_guard
                    .bandwidth_sketch
                    .add(estimated_bandwidth_for_this_read_bytes_per_sec);
                metrics_write_guard
                    .fixed_overhead_sketch
                    .add(estimated_t0_micros);
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

    // MockVortexReader remains the same
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

    struct MockVortexReader {
        data: Vec<u8>,
        read_delay_micros: u64,     // Simulate base latency (T0)
        throughput_limit_mbps: u64, // Simulate bandwidth (B)
        call_count: AtomicUsize,
    }

    #[tokio::test]
    async fn test_tracking_reader_with_t0_and_b_estimation() {
        // We need a longer test with various read sizes to stabilize the estimates
        let mock_data = vec![0; 1024 * 1024 * 50]; // 50 MB of data
        let mock_reader = MockVortexReader {
            data: mock_data,
            read_delay_micros: 200,     // Simulate base latency (T0) = 200 us
            throughput_limit_mbps: 100, // Simulate bandwidth (B) = 100 MB/s
            call_count: AtomicUsize::new(0),
        };

        // Use a config that targets 90th percentile for T0 and 50th for B
        let config = PerformanceHintingConfig {
            target_fixed_latency_quantile: 0.90,
            target_bandwidth_quantile: 0.50,
            ..Default::default()
        };
        let tracking_reader = PerformanceHintingReadAt::with_config(mock_reader, config);

        // Perform many reads with varying sizes to populate the sketches
        // This is crucial for separating T0 and B
        for i in 0..100 {
            // Increased samples
            let offset = (i as u64 * 1024 * 200) % (1024 * 1024 * 40); // Loop within first 40MB
            let len = match i % 5 {
                // More variety in sizes
                0 => 1024 * 4,    // 4KB - heavily T0-dominated
                1 => 1024 * 16,   // 16KB
                2 => 1024 * 64,   // 64KB
                3 => 1024 * 256,  // 256KB
                _ => 1024 * 1024, // 1MB - heavily B-dominated
            };
            let result = tracking_reader
                .read_byte_range(offset..offset + len, Alignment::new(1))
                .await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), len as usize);
            // No sleep here, let it run fast to get more samples in DDSketch quickly
        }

        // Allow some time for metrics updates
        sleep(Duration::from_millis(100)).await;

        let hint = tracking_reader.compute_performance_hint();
        println!("\n--- DDSketch Performance Hint (T0/B Estimation) ---");
        println!("Hint: {:?}", hint);

        let metrics_guard = tracking_reader.metrics.read();

        println!(
            "Fixed Overhead (us) P90: {:?}",
            metrics_guard.fixed_overhead_sketch.quantile(0.9)
        );
        println!(
            "Bandwidth (B/s) P50: {:?}",
            metrics_guard.bandwidth_sketch.quantile(0.5)
        );

        // Assertions based on expected T0=200us and B=100MB/s from Mock
        let p90_fixed_latency = metrics_guard
            .fixed_overhead_sketch
            .quantile(0.9)
            .ok()
            .flatten()
            .unwrap_or(0.0);
        assert!(
            // T0 should be close to 200us. Allow for some noise (e.g., 150us to 300us)
            p90_fixed_latency > 150.0 && p90_fixed_latency < 300.0,
            "P90 Fixed Latency (T0) should be around 200us, got {}",
            p90_fixed_latency
        );

        let p50_bandwidth = metrics_guard
            .bandwidth_sketch
            .quantile(0.5)
            .ok()
            .flatten()
            .unwrap_or(0.0);
        assert!(
            // B should be close to 100MB/s (100,000,000 B/s). Allow for some range.
            p50_bandwidth > 80_000_000.0 && p50_bandwidth < 120_000_000.0,
            "P50 Bandwidth (B) should be around 100 MB/s, got {}",
            p50_bandwidth
        );

        // Check coalescing window heuristic based on the new T0 and B estimates
        // Ideal window = B * T0 * amplification (100MB/s * 200us * 5) = 100e6 * 200e-6 * 5 = 100,000 bytes (100KB)
    }
}
