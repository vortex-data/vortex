// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct array-level codec microbenchmarks.
//!
//! This module deliberately measures the selected encoder's train + compress
//! path over one canonical whole-column array. It does not exercise the Vortex
//! file layout, per-chunk dictionaries, child compression, or file I/O.

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::VarBinViewArray;
use vortex_bench::Format;
use vortex_bench::measurements::CustomUnitMeasurement;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_onpair::Config;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::MaxDictBits;
use vortex_onpair::onpair_compress;

use crate::SESSION;
use crate::StringColumn;
use crate::StringEncoder;
use crate::duration_ms;
use crate::median;
use crate::onpair_label;
use crate::prepare_column;
use crate::throughput;
use crate::verify_canonicalized;

/// A fully configured direct array-compression candidate: an encoder family
/// plus the fixed configuration that path needs.
pub enum DirectCandidate {
    /// OnPair with a deterministic config, whose dictionary budget is the only
    /// thing the benchmark varies.
    OnPair(Config),
    /// FSST (no configuration).
    Fsst,
}

impl DirectCandidate {
    /// OnPair with a dictionary budget of up to `2^max_dict_bits` tokens,
    /// reusing every other default. `max_dict_bits` must be in `9..=16`; OnPair
    /// stores codes as `u16`, so all of those fit.
    pub fn on_pair(max_dict_bits: u8) -> Result<Self> {
        let max_dict_bits = MaxDictBits::new(max_dict_bits).map_err(|e| {
            anyhow::anyhow!(
                "invalid maximum dictionary bit width {max_dict_bits} (want 9..=16): {e}"
            )
        })?;
        Ok(Self::OnPair(Config {
            max_dict_bits,
            ..DEFAULT_DICT12_CONFIG
        }))
    }

    /// The encoder family, for the post-compression type check.
    fn family(&self) -> StringEncoder {
        match self {
            Self::OnPair(_) => StringEncoder::OnPair,
            Self::Fsst => StringEncoder::Fsst,
        }
    }

    /// Stable label used in benchmark output, e.g. `onpair-12` or `fsst`.
    fn label(&self) -> String {
        match self {
            Self::OnPair(config) => onpair_label(config),
            Self::Fsst => StringEncoder::Fsst.label().to_string(),
        }
    }

    /// Run the encoder's direct array-level train + compress path.
    pub(crate) fn compress(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        match self {
            Self::OnPair(config) => Ok(onpair_compress(array, *config, ctx)?),
            Self::Fsst => {
                let compressor = fsst_train_compressor(array, ctx)?;
                Ok(fsst_compress(array, &compressor, ctx)?.into_array())
            }
        }
    }
}

/// Measured in-memory statistics for a single string column under one encoder
/// configuration (an OnPair dictionary size, or FSST).
pub struct ColumnResult {
    /// Column identifier.
    pub name: String,
    /// Encoder configuration label, e.g. `onpair-12` or `fsst`.
    pub encoder: String,
    /// Number of rows.
    pub rows: usize,
    /// Canonical uncompressed array bytes used to normalize size and
    /// throughput: one 16-byte view per row plus the bytes of the strings too
    /// long to inline.
    pub uncompressed_bytes: u64,
    /// Buffer bytes referenced by the encoded array.
    pub encoded_bytes: u64,
    /// Direct array-level train + compress times, one per iteration.
    pub compression_runs: Vec<Duration>,
}

impl ColumnResult {
    /// Median direct array compression time across iterations.
    fn compression_median(&self) -> Duration {
        median(&self.compression_runs)
    }

    /// Encoded array buffer bytes as a percentage of canonical uncompressed
    /// array bytes. Lower is better.
    ///
    /// This direct representation has not had its children compressed by the
    /// file writer.
    pub fn encoded_size_pct(&self) -> f64 {
        self.encoded_bytes as f64 / self.uncompressed_bytes as f64 * 100.0
    }

    /// Direct compression throughput in MB/s of canonical uncompressed array
    /// bytes, from the median run.
    pub fn compression_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, self.compression_median()) / 1e6
    }

    /// Emit lower-is-better timings and size percentages as Vortex custom-unit
    /// metrics, named `codec/<metric>/<input>/<encoder>`.
    ///
    /// The `codec/` prefix keeps these distinct from the tracked file metrics,
    /// which CI reports: this suite is a local diagnostic, so its measurements
    /// only reach `gh-json` when it is explicitly selected.
    ///
    /// `Format` has no in-memory Vortex variant, so these measurements use
    /// `Format::OnDiskVortex` as their reporting target.
    pub fn measurements(&self) -> Vec<CustomUnitMeasurement> {
        let suffix = format!("{}/{}", self.name, self.encoder);
        vec![
            CustomUnitMeasurement {
                name: format!("codec/size/{suffix}"),
                format: Format::OnDiskVortex,
                unit: "%".into(),
                value: self.encoded_size_pct(),
            },
            CustomUnitMeasurement {
                name: format!("codec/compress/{suffix}"),
                format: Format::OnDiskVortex,
                unit: "ms".into(),
                value: duration_ms(self.compression_median()),
            },
        ]
    }
}

/// Time `candidate`'s direct array-level train + compress path on `column` across
/// `iterations` runs, recording every run so the caller can report the median.
///
/// Fails if the column contains nulls or does not compress to `candidate`'s
/// encoding, rather than silently changing the measured workload.
pub fn bench_column(
    column: &StringColumn,
    iterations: usize,
    warmup: usize,
    candidate: &DirectCandidate,
    verify: bool,
) -> Result<ColumnResult> {
    crate::validate_iterations(iterations)?;
    let mut ctx = SESSION.create_execution_ctx();

    let (canonical, uncompressed_bytes) = prepare_column(column, &mut ctx)?;
    let input = canonical.clone().into_array();
    let rows = canonical.len();

    // At least one warm-up produces the reference array for the encoding check,
    // size metric, and canonicalization verification. Extra runs stabilize timings.
    let mut compressed = candidate.compress(&input, &mut ctx)?;
    for _ in 1..warmup.max(1) {
        let mut warm_ctx = SESSION.create_execution_ctx();
        compressed = candidate.compress(&input, &mut warm_ctx)?;
    }

    if !candidate.family().matches(&compressed) {
        bail!(
            "column {} did not compress to {} (got {}); its data may be unsupported \
             by the encoder",
            column.name,
            candidate.label(),
            compressed.encoding_id(),
        );
    }
    let encoded_bytes = compressed.nbytes();

    // One-time correctness check before timing: a byte-wrong canonicalization
    // must not be reported as a fast one.
    if verify {
        let canonicalized = compressed.execute::<VarBinViewArray>(&mut ctx)?;
        verify_canonicalized(
            &format!("{} [{}]", column.name, candidate.label()),
            &canonical,
            &canonicalized,
            &mut ctx,
        )?;
    }

    // Each timed run re-compresses the source array from scratch; the fresh
    // context is cheap per-run isolation, not a cache reset (`ExecutionCtx`
    // holds no cross-run result cache). `black_box` keeps the result live.
    let mut compression_runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut ctx = SESSION.create_execution_ctx();
        let start = Instant::now();
        let compressed = candidate.compress(&input, &mut ctx)?;
        compression_runs.push(start.elapsed());
        black_box(compressed);
    }

    Ok(ColumnResult {
        name: column.name.clone(),
        encoder: candidate.label(),
        rows,
        uncompressed_bytes,
        encoded_bytes,
        compression_runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_benchmark_smoke_tests_each_encoder() -> Result<()> {
        let (column, expected_uncompressed_bytes) = crate::repeated_fixture();

        for (candidate, expected_label) in [
            (DirectCandidate::on_pair(12)?, "onpair-12"),
            (DirectCandidate::Fsst, "fsst"),
        ] {
            let result = bench_column(&column, 1, 0, &candidate, true)?;

            assert_eq!(result.encoder, expected_label);
            assert_eq!(result.rows, 128);
            assert_eq!(result.uncompressed_bytes, expected_uncompressed_bytes);
            assert!(result.encoded_bytes > 0);
            assert_eq!(result.compression_runs.len(), 1);
        }
        Ok(())
    }

    #[test]
    fn machine_readable_metric_schema_is_stable() {
        let result = ColumnResult {
            name: "fixture".to_string(),
            encoder: "fsst".to_string(),
            rows: 1,
            uncompressed_bytes: 2,
            encoded_bytes: 1,
            compression_runs: vec![Duration::from_millis(12)],
        };

        assert_eq!(
            crate::measurement_rows(&result.measurements()),
            [
                "codec/size/fixture/fsst % 50",
                "codec/compress/fixture/fsst ms 12",
            ]
        );
    }
}
