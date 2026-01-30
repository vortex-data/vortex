// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexResult;
use vortex_fastlanes::RLEArray;

use crate::BtrBlocksCompressor;
use crate::CanonicalCompressor;
use crate::CompressorStats;
use crate::Excludes;
use crate::IntCode;
use crate::Scheme;
use crate::estimate_compression_ratio_with_sampling;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub const RUN_LENGTH_THRESHOLD: u32 = 4;

/// Trait for accessing RLE-specific statistics.
pub trait RLEStats {
    fn value_count(&self) -> u32;
    fn average_run_length(&self) -> u32;
    fn source(&self) -> &PrimitiveArray;
}

/// Configuration trait for RLE schemes.
///
/// Implement this trait to define the behavior of an RLE scheme for a specific
/// stats and code type combination.
pub trait RLEConfig: Debug + Send + Sync + 'static {
    /// The statistics type used by this RLE scheme.
    type Stats: RLEStats + CompressorStats;
    /// The code type used to identify schemes.
    type Code: Copy + Clone + Debug + Hash + PartialEq + Eq;

    /// The unique code identifying this RLE scheme.
    const CODE: Self::Code;

    /// Compress the values array after RLE encoding.
    fn compress_values(
        compressor: &BtrBlocksCompressor,
        values: &PrimitiveArray,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[Self::Code],
    ) -> VortexResult<ArrayRef>;
}

/// RLE scheme that is generic over a configuration type.
///
/// This is a ZST (zero-sized type) - all behavior is defined by the `RLEConfig` trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RLEScheme<C: RLEConfig>(PhantomData<C>);

impl<C: RLEConfig> RLEScheme<C> {
    /// Creates a new RLE scheme.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<C: RLEConfig> Default for RLEScheme<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: RLEConfig> Scheme for RLEScheme<C> {
    type StatsType = C::Stats;
    type CodeType = C::Code;

    fn code(&self) -> C::Code {
        C::CODE
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[C::Code],
    ) -> VortexResult<f64> {
        // RLE is only useful when we cascade it with another encoding.
        if allowed_cascading == 0 {
            return Ok(0.0);
        }

        // Don't compress all-null or empty arrays.
        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        // Check whether RLE is a good fit, based on the average run length.
        if stats.average_run_length() < RUN_LENGTH_THRESHOLD {
            return Ok(0.0);
        }

        // Run compression on a sample to see how it performs.
        estimate_compression_ratio_with_sampling(
            self,
            compressor,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[C::Code],
    ) -> VortexResult<ArrayRef> {
        let rle_array = RLEArray::encode(RLEStats::source(stats))?;

        if allowed_cascading == 0 {
            return Ok(rle_array.into_array());
        }

        // Prevent RLE recursion.
        let mut new_excludes = vec![self.code()];
        new_excludes.extend_from_slice(excludes);

        let compressed_values = C::compress_values(
            compressor,
            &rle_array.values().to_primitive(),
            is_sample,
            allowed_cascading - 1,
            &new_excludes,
        )?;

        let compressed_indices = compressor.compress_canonical(
            Canonical::Primitive(rle_array.indices().to_primitive().narrow()?),
            is_sample,
            allowed_cascading - 1,
            Excludes::int_only(&[IntCode::Dict]),
        )?;

        let compressed_offsets = compressor.compress_canonical(
            Canonical::Primitive(rle_array.values_idx_offsets().to_primitive().narrow()?),
            is_sample,
            allowed_cascading - 1,
            Excludes::int_only(&[IntCode::Dict]),
        )?;

        // SAFETY: Recursive compression doesn't affect the invariants.
        unsafe {
            Ok(RLEArray::new_unchecked(
                compressed_values,
                compressed_indices,
                compressed_offsets,
                rle_array.dtype().clone(),
                rle_array.offset(),
                rle_array.len(),
            )
            .into_array())
        }
    }
}
