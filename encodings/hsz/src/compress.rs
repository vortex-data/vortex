// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::stage::BlockSummary;
use crate::stage::Hsz;

/// Configuration for [`Hsz::compress`].
#[derive(Clone, Copy, Debug)]
pub struct HszConfig {
    /// Number of elements per predictor block. Larger blocks improve the
    /// compression ratio of the predictor stage but coarsen the zone-map and
    /// hurt range-predicate skipping.
    pub block_size: u32,
    /// Absolute reconstruction error bound. Non-outlier elements decode to
    /// within `eps` of the original. Must be strictly positive and finite.
    pub eps: f64,
}

impl Default for HszConfig {
    /// Defaults: 1024-element blocks and an `eps` of `1e-3`. The block size
    /// matches the FastLanes vector width family and the error bound is a
    /// reasonable starting point for sensor and simulation data.
    fn default() -> Self {
        Self {
            block_size: 1024,
            eps: 1e-3,
        }
    }
}

const OUTLIER_RESIDUAL_PLACEHOLDER: u32 = 0;

impl Hsz {
    /// Compress an `f64` slice into a homomorphic encoding.
    ///
    /// Returns an [`Hsz`] whose [`Hsz::decompress`] is accurate to within
    /// `config.eps` for every non-outlier position.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.eps` is non-positive or non-finite, or if
    /// `config.block_size` is zero.
    pub fn compress(values: &[f64], config: HszConfig) -> VortexResult<Self> {
        if !config.eps.is_finite() || config.eps <= 0.0 {
            vortex_bail!(
                "HszConfig::eps must be positive and finite, got {}",
                config.eps
            );
        }
        if config.block_size == 0 {
            vortex_bail!("HszConfig::block_size must be non-zero");
        }
        if values.len() > u32::MAX as usize {
            vortex_bail!(
                "Hsz currently supports at most {} elements, got {}",
                u32::MAX,
                values.len()
            );
        }

        let block_size = config.block_size as usize;
        let eps = config.eps;
        let n_blocks = values.len().div_ceil(block_size);

        let mut blocks = Vec::with_capacity(n_blocks);
        let mut block_offsets: Vec<u32> = Vec::with_capacity(n_blocks + 1);
        block_offsets.push(0);
        let mut residuals = BufferMut::<u32>::with_capacity(values.len());
        let mut outlier_indices: Vec<u64> = Vec::new();
        let mut outlier_values: Vec<f64> = Vec::new();

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            let mut summary = BlockSummary::empty();
            for &v in block {
                if v.is_finite() {
                    summary.observe(v);
                }
            }
            if summary.count == 0 {
                // All values in this block were non-finite. Pin min/max to a
                // representable value so downstream zone-map arithmetic stays
                // well defined. Every element becomes an outlier.
                summary.min = 0.0;
                summary.max = 0.0;
            }

            let predictor = summary.min;
            for (i, &v) in block.iter().enumerate() {
                let global_idx = (start + i) as u64;
                if !v.is_finite() || has_non_finite_unrepresentable(v) {
                    residuals.push(OUTLIER_RESIDUAL_PLACEHOLDER);
                    outlier_indices.push(global_idx);
                    outlier_values.push(v);
                    continue;
                }
                let quantum = ((v - predictor) / eps).round();
                if quantum < 0.0 || quantum > u32::MAX as f64 {
                    residuals.push(OUTLIER_RESIDUAL_PLACEHOLDER);
                    outlier_indices.push(global_idx);
                    outlier_values.push(v);
                    continue;
                }
                let q = quantum as u32;
                let reconstructed = predictor + (q as f64) * eps;
                if (reconstructed - v).abs() > eps {
                    // The quantiser cannot represent this value within eps
                    // (typically because eps is much larger than the block
                    // range and round-to-nearest overshoots). Fall back to
                    // exact storage.
                    residuals.push(OUTLIER_RESIDUAL_PLACEHOLDER);
                    outlier_indices.push(global_idx);
                    outlier_values.push(v);
                } else {
                    residuals.push(q);
                }
            }

            blocks.push(summary);
            block_offsets.push(u32::try_from(residuals.len())?);
        }

        Ok(Hsz {
            block_size: config.block_size,
            eps,
            len: values.len(),
            blocks,
            block_offsets,
            residuals: residuals.freeze(),
            outlier_indices,
            outlier_values,
        })
    }
}

fn has_non_finite_unrepresentable(v: f64) -> bool {
    // Reserved for future quantiser-side checks (e.g. subnormals when we move
    // to f32 storage). Today every finite value is at least representable as
    // an outlier.
    !v.is_finite()
}
