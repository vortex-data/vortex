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
use crate::CompressorContext;
use crate::CompressorStats;
use crate::Excludes;
use crate::IntCode;
use crate::Scheme;
use crate::SchemeExt;

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
    type Code: Copy + Clone + Debug + Hash + Eq + Ord;

    /// The unique code identifying this RLE scheme.
    const CODE: Self::Code;

    /// Compress the values array after RLE encoding.
    fn compress_values(
        compressor: &BtrBlocksCompressor,
        values: &PrimitiveArray,
        ctx: CompressorContext,
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
        ctx: CompressorContext,
        excludes: &[C::Code],
    ) -> VortexResult<f64> {
        // RLE is only useful when we cascade it with another encoding.
        if ctx.allowed_cascading == 0 {
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
        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[C::Code],
    ) -> VortexResult<ArrayRef> {
        let rle_array = RLEArray::encode(RLEStats::source(stats))?;

        if ctx.allowed_cascading == 0 {
            return Ok(rle_array.into_array());
        }

        // Prevent RLE recursion.
        let mut new_excludes = vec![self.code()];
        new_excludes.extend_from_slice(excludes);

        let compressed_values = C::compress_values(
            compressor,
            &rle_array.values().to_primitive(),
            ctx.descend(),
            &new_excludes,
        )?;

        // Delta in an unstable encoding, once we deem it stable we can switch over to this always.
        #[cfg(feature = "unstable_encodings")]
        let compressed_indices = try_compress_delta(
            &rle_array.indices().to_primitive().narrow()?,
            compressor,
            ctx.descend(),
            Excludes::from(&[IntCode::Dict]),
        )?;

        #[cfg(not(feature = "unstable_encodings"))]
        let compressed_indices = compressor.compress_canonical(
            Canonical::Primitive(rle_array.indices().to_primitive().narrow()?),
            ctx.descend(),
            Excludes::from(&[IntCode::Dict]),
        )?;

        let compressed_offsets = compressor.compress_canonical(
            Canonical::Primitive(rle_array.values_idx_offsets().to_primitive().narrow()?),
            ctx.descend(),
            Excludes::from(&[IntCode::Dict]),
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
    compressor: &BtrBlocksCompressor,
    ctx: CompressorContext,
    excludes: Excludes,
) -> VortexResult<ArrayRef> {
    use vortex_array::VortexSessionExecute;

    let (bases, deltas) = vortex_fastlanes::delta_compress(
        primitive_array,
        &mut vortex_array::LEGACY_SESSION.create_execution_ctx(),
    )?;

    let compressed_bases =
        compressor.compress_canonical(Canonical::Primitive(bases), ctx, excludes)?;
    let compressed_deltas =
        compressor.compress_canonical(Canonical::Primitive(deltas), ctx, excludes)?;

    vortex_fastlanes::DeltaArray::try_new(
        compressed_bases,
        compressed_deltas,
        0,
        primitive_array.len(),
    )
    .map(vortex_fastlanes::DeltaArray::into_array)
}
