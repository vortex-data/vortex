// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressor traits for type-specific compression.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::BtrBlocksCompressor;
use crate::CompressorContext;
use crate::CompressorStats;
use crate::Scheme;

pub(crate) mod decimal;
pub(crate) mod float;
pub(crate) mod integer;
mod patches;
mod rle;
pub(crate) mod string;
pub(crate) mod temporal;

/// Maximum cascade depth for compression.
pub(crate) const MAX_CASCADE: usize = 3;

/// A compressor for a particular input type.
///
/// This trait defines the interface for type-specific compressors that can adaptively
/// choose and apply compression schemes based on data characteristics. Compressors
/// analyze input arrays, select optimal compression schemes, and handle cascading
/// compression with multiple encoding layers.
///
/// The compressor works by generating statistics on the input data, evaluating
/// available compression schemes, and selecting the one with the best compression ratio.
pub trait Compressor {
    /// The VTable type for arrays this compressor operates on.
    type ArrayVTable: VTable;
    /// The compression scheme type used by this compressor.
    type SchemeType: Scheme<StatsType = Self::StatsType> + ?Sized;
    /// The statistics type used to analyze arrays for compression.
    type StatsType: CompressorStats<ArrayVTable = Self::ArrayVTable>;

    /// Generates statistics for the given array to guide compression scheme selection.
    fn gen_stats(&self, array: &<Self::ArrayVTable as VTable>::Array) -> Self::StatsType;

    /// Returns all available compression schemes for this compressor.
    fn schemes(&self) -> &[&'static Self::SchemeType];
    /// Returns the default fallback compression scheme.
    fn default_scheme(&self) -> &'static Self::SchemeType;
}

/// Extension trait providing scheme selection and compression for compressors.
pub trait CompressorExt: Compressor
where
    Self::SchemeType: 'static,
{
    /// Selects the best compression scheme based on expected compression ratios.
    ///
    /// Evaluates all available schemes against the provided statistics and returns
    /// the one with the highest compression ratio. Falls back to the default scheme
    /// if no scheme provides compression benefits.
    #[allow(clippy::cognitive_complexity)]
    fn choose_scheme(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
    ) -> VortexResult<&'static Self::SchemeType> {
        let mut best_ratio = 1.0;
        let mut best_scheme: Option<&'static Self::SchemeType> = None;

        // logging helpers
        let depth = MAX_CASCADE - ctx.allowed_cascading;

        for scheme in self.schemes().iter() {
            // Skip excluded schemes
            if excludes.contains(&scheme.code()) {
                continue;
            }

            // We never choose Constant for a sample
            if ctx.is_sample && scheme.is_constant() {
                continue;
            }

            tracing::trace!(
                is_sample = ctx.is_sample,
                depth,
                is_constant = scheme.is_constant(),
                ?scheme,
                "Trying compression scheme"
            );

            let ratio = scheme.expected_compression_ratio(compressor, stats, ctx, excludes)?;
            tracing::trace!(
                is_sample = ctx.is_sample,
                depth,
                ratio,
                ?scheme,
                "Expected compression result"
            );

            if !(ratio.is_subnormal() || ratio.is_infinite() || ratio.is_nan()) {
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_scheme = Some(*scheme);
                }
            } else {
                tracing::trace!(
                    "Calculated invalid compression ratio {ratio} for scheme: {scheme:?}. Must not be sub-normal, infinite or nan."
                );
            }
        }

        tracing::trace!(depth, scheme = ?best_scheme, ratio = best_ratio, "best scheme found");

        if let Some(best) = best_scheme {
            Ok(best)
        } else {
            Ok(self.default_scheme())
        }
    }

    /// Compresses an array using this compressor.
    ///
    /// Generates statistics on the input array, selects the best compression scheme,
    /// and applies it. Returns the original array if compression would increase size.
    fn compress(
        &self,
        btr_blocks_compressor: &BtrBlocksCompressor,
        array: &<<Self as Compressor>::ArrayVTable as VTable>::Array,
        ctx: CompressorContext,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
    ) -> VortexResult<ArrayRef> {
        // Avoid compressing empty arrays.
        if array.is_empty() {
            return Ok(array.to_array());
        }

        // Avoid compressing all-null arrays.
        if array.all_invalid()? {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        // Generate stats on the array directly.
        let stats = self.gen_stats(array);
        let best_scheme = self.choose_scheme(btr_blocks_compressor, &stats, ctx, excludes)?;

        let output = best_scheme.compress(btr_blocks_compressor, &stats, ctx, excludes)?;
        if output.nbytes() < array.nbytes() {
            Ok(output)
        } else {
            tracing::debug!("resulting tree too large: {}", output.encoding_id());
            Ok(array.to_array())
        }
    }
}

// Blanket implementation for all Compressor types with 'static SchemeType
impl<T: Compressor> CompressorExt for T where T::SchemeType: 'static {}
