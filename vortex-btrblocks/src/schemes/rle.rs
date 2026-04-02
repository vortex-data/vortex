// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::builtins::is_float_primitive;
use vortex_compressor::builtins::is_integer_primitive;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
#[cfg(feature = "unstable_encodings")]
use vortex_compressor::scheme::SchemeId;
use vortex_compressor::stats::FloatStats;
use vortex_compressor::stats::IntegerStats;
use vortex_error::VortexResult;
use vortex_fastlanes::RLE;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::estimate_compression_ratio_with_sampling;
use crate::schemes::integer::IntDictScheme;
use crate::schemes::integer::SparseScheme;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub const RUN_LENGTH_THRESHOLD: u32 = 4;

/// RLE scheme for integer compression.
pub const RLE_INTEGER_SCHEME: RLEScheme<IntRLEConfig> = RLEScheme::new();

/// RLE scheme for float compression.
pub const RLE_FLOAT_SCHEME: RLEScheme<FloatRLEConfig> = RLEScheme::new();

/// Configuration for integer RLE compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntRLEConfig;

/// Configuration for float RLE compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatRLEConfig;

/// Configuration trait for RLE schemes.
///
/// Implement this trait to define the behavior of an RLE scheme for a specific
/// stats type.
pub trait RLEConfig: Debug + Send + Sync + 'static {
    /// The statistics type used by this RLE scheme.
    type Stats: RLEStats + 'static;

    /// The globally unique name for this RLE scheme.
    const SCHEME_NAME: &'static str;

    /// Whether this scheme can compress the given canonical array.
    fn matches(canonical: &Canonical) -> bool;

    /// Generates statistics for the given array.
    fn generate_stats(array: &ArrayRef) -> Self::Stats;
}

impl RLEConfig for IntRLEConfig {
    type Stats = IntegerStats;

    const SCHEME_NAME: &'static str = "vortex.int.rle";

    fn matches(canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn generate_stats(array: &ArrayRef) -> IntegerStats {
        IntegerStats::generate(&array.to_primitive())
    }
}

impl RLEConfig for FloatRLEConfig {
    type Stats = FloatStats;

    const SCHEME_NAME: &'static str = "vortex.float.rle";

    fn matches(canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    fn generate_stats(array: &ArrayRef) -> FloatStats {
        FloatStats::generate(&array.to_primitive())
    }
}

/// Trait for accessing RLE-specific statistics.
pub trait RLEStats {
    /// Returns the number of non-null values.
    fn value_count(&self) -> u32;
    /// Returns the average run length.
    fn average_run_length(&self) -> u32;
    /// Returns the underlying source array.
    fn source(&self) -> &PrimitiveArray;
}

impl RLEStats for IntegerStats {
    fn value_count(&self) -> u32 {
        self.value_count()
    }

    fn average_run_length(&self) -> u32 {
        self.average_run_length()
    }

    fn source(&self) -> &PrimitiveArray {
        self.source()
    }
}

impl RLEStats for FloatStats {
    fn value_count(&self) -> u32 {
        FloatStats::value_count(self)
    }

    fn average_run_length(&self) -> u32 {
        FloatStats::average_run_length(self)
    }

    fn source(&self) -> &PrimitiveArray {
        FloatStats::source(self)
    }
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
    fn scheme_name(&self) -> &'static str {
        C::SCHEME_NAME
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        C::matches(canonical)
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    /// RLE indices (child 1) and offsets (child 2) are monotonically increasing positions
    /// with all unique values. Dict, RunEnd, and Sparse are all pointless on such data.
    /// Self-exclusion already prevents RLE on RLE children.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::Many(&[1, 2]),
            },
            // TODO(connor): This is wrong for some reason?
            // DescendantExclusion {
            //     excluded: RunEndScheme.id(),
            //     children: ChildSelection::Many(&[1, 2]),
            // },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::Many(&[1, 2]),
            },
        ]
    }

    /// Dict values (child 0) are all unique by definition, so RLE is pointless on them.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(0),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(0),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(0),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        // RLE is only useful when we cascade it with another encoding.
        let array = data.array().clone();
        let stats = data.get_or_insert_with::<C::Stats>(|| C::generate_stats(&array));

        // Don't compress all-null or empty arrays.
        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        // Check whether RLE is a good fit, based on the average run length.
        if stats.average_run_length() < RUN_LENGTH_THRESHOLD {
            return Ok(0.0);
        }

        // Run compression on a sample to see how it performs.
        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let array = data.array().clone();
        let stats = data.get_or_insert_with::<C::Stats>(|| C::generate_stats(&array));
        let rle_array = RLE::encode(RLEStats::source(stats))?;

        let compressed_values = compressor.compress_child(
            &rle_array.values().to_primitive().into_array(),
            &ctx,
            self.id(),
            0,
        )?;

        // Delta in an unstable encoding, once we deem it stable we can switch over to this always.
        #[cfg(feature = "unstable_encodings")]
        let compressed_indices = try_compress_delta(
            compressor,
            &rle_array.indices().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;

        #[cfg(not(feature = "unstable_encodings"))]
        let compressed_indices = compressor.compress_child(
            &rle_array.indices().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;

        let compressed_offsets = compressor.compress_child(
            &rle_array
                .values_idx_offsets()
                .to_primitive()
                .narrow()?
                .into_array(),
            &ctx,
            self.id(),
            2,
        )?;

        // SAFETY: Recursive compression doesn't affect the invariants.
        unsafe {
            Ok(RLE::new_unchecked(
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
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    parent_ctx: &CompressorContext,
    parent_id: SchemeId,
    child_index: usize,
) -> VortexResult<ArrayRef> {
    let (bases, deltas) =
        vortex_fastlanes::delta_compress(&child.to_primitive(), &mut compressor.execution_ctx())?;

    let compressed_bases =
        compressor.compress_child(&bases.into_array(), parent_ctx, parent_id, child_index)?;
    let compressed_deltas =
        compressor.compress_child(&deltas.into_array(), parent_ctx, parent_id, child_index)?;

    vortex_fastlanes::DeltaData::try_new(compressed_bases, compressed_deltas, 0, child.len())
        .map(IntoArray::into_array)
}
