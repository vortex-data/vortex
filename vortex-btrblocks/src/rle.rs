// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::arrays::PrimitiveArray;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_error::VortexResult;
use vortex_fastlanes::{DeltaArray, RLEArray, delta_compress};

use crate::integer::{IntCode, IntCompressor};
use crate::{Compressor, CompressorStats, Scheme, estimate_compression_ratio_with_sampling};

/// Threshold for the average run length in an array before we consider run-length encoding.
pub const RUN_LENGTH_THRESHOLD: u32 = 4;

pub trait RLEStats {
    fn value_count(&self) -> u32;
    fn average_run_length(&self) -> u32;
    fn source(&self) -> &PrimitiveArray;
}

/// RLE scheme that is generic over stats and code.
#[derive(Debug, Clone, Copy)]
pub struct RLEScheme<Stats, Code> {
    pub code: Code,
    /// Function to compress values
    pub compress_values_fn: fn(&PrimitiveArray, bool, usize, &[Code]) -> VortexResult<ArrayRef>,
    /// Phantom data to tie the scheme to specific stats type
    _phantom: std::marker::PhantomData<Stats>,
}

impl<S, C> RLEScheme<S, C> {
    pub const fn new(
        code: C,
        compress_values_fn: fn(&PrimitiveArray, bool, usize, &[C]) -> VortexResult<ArrayRef>,
    ) -> Self {
        Self {
            code,
            compress_values_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, C> Scheme for RLEScheme<S, C>
where
    S: RLEStats + CompressorStats,
    C: Copy + Clone + Debug + Hash + PartialEq + Eq,
{
    type StatsType = S;
    type CodeType = C;

    fn code(&self) -> C {
        self.code
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[C],
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
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[C],
    ) -> VortexResult<ArrayRef> {
        let rle_array = RLEArray::encode(RLEStats::source(stats))?;

        if allowed_cascading == 0 {
            return Ok(rle_array.into_array());
        }

        // Prevent RLE recursion.
        let mut new_excludes = vec![self.code()];
        new_excludes.extend_from_slice(excludes);

        let compressed_values = (self.compress_values_fn)(
            &rle_array.values().to_primitive(),
            is_sample,
            allowed_cascading - 1,
            &new_excludes,
        )?;

        // Delta in an unstable encoding, once we deem it stable we can switch over to this always.
        #[cfg(feature = "unstable_encodings")]
        // For indices and offsets, we always use integer compression without dictionary encoding.
        let compressed_indices = try_compress_delta(
            &rle_array.indices().to_primitive().narrow()?,
            is_sample,
            allowed_cascading - 1,
            &[],
        )?;

        #[cfg(not(feature = "unstable_encodings"))]
        let compressed_indices = IntCompressor::compress_no_dict(
            &rle_array.indices().to_primitive().narrow()?,
            is_sample,
            allowed_cascading - 1,
            &[],
        )?;

        let compressed_offsets = IntCompressor::compress_no_dict(
            &rle_array.values_idx_offsets().to_primitive().narrow()?,
            is_sample,
            allowed_cascading - 1,
            &[],
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

#[cfg(feature = "unstable_encodings")]
fn try_compress_delta(
    primitive_array: &PrimitiveArray,
    is_sample: bool,
    allowed_cascading: usize,
    excludes: &[IntCode],
) -> VortexResult<ArrayRef> {
    let (bases, deltas) = delta_compress(primitive_array)?;
    let compressed_bases = IntCompressor::compress(&bases, is_sample, allowed_cascading, excludes)?;
    let compressed_deltas =
        IntCompressor::compress_no_dict(&deltas, is_sample, allowed_cascading, excludes)?;

    DeltaArray::try_from_delta_compress_parts(compressed_bases, compressed_deltas)
        .map(DeltaArray::into_array)
}
