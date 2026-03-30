// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-specific compressor traits that drive scheme selection and compression.
//!
//! [`Compressor`] defines the interface: generate statistics for an array via
//! [`Compressor::gen_stats`], and provide available [`Scheme`]s via [`Compressor::schemes`].
//!
//! [`CompressorExt`] is blanket-implemented for all `Compressor`s and adds the core logic:
//!
//! - [`CompressorExt::choose_scheme`] iterates all schemes, skips excluded ones, and calls
//!   [`Scheme::expected_compression_ratio`] on each. It returns the scheme with the highest ratio
//!   above 1.0, or falls back to the default. See the [`scheme`](crate::scheme) module for how
//!   ratio estimation works.
//! - [`CompressorExt::compress`] generates stats, calls `choose_scheme()`, and applies the
//!   result. If compression did not shrink the array, the original is returned.

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

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
    fn gen_stats(&self, array: &<Self::ArrayVTable as VTable>::ArrayData) -> Self::StatsType;

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
        array: &<<Self as Compressor>::ArrayVTable as VTable>::ArrayData,
        ctx: CompressorContext,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
    ) -> VortexResult<ArrayRef> {
        let array_ref = array.clone().into_array();

        // Avoid compressing empty arrays.
        if array_ref.is_empty() {
            return Ok(array_ref);
        }

        // Avoid compressing all-null arrays.
        if array_ref.all_invalid()? {
            return Ok(ConstantArray::new(
                Scalar::null(array_ref.dtype().clone()),
                array_ref.len(),
            )
            .into_array());
        }

        // Generate stats on the array directly.
        let stats = self.gen_stats(array);
        let best_scheme = self.choose_scheme(btr_blocks_compressor, &stats, ctx, excludes)?;

        let output = best_scheme.compress(btr_blocks_compressor, &stats, ctx, excludes)?;
        if output.nbytes() < array_ref.nbytes() {
            Ok(output)
        } else {
            tracing::debug!("resulting tree too large: {}", output.encoding_id());
            Ok(array_ref)
        }
    }
}

// Blanket implementation for all Compressor types with 'static SchemeType
impl<T: Compressor> CompressorExt for T where T::SchemeType: 'static {}
