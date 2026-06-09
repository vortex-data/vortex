// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ZigZag integer encoding for signed integers.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArrayExt;
use vortex_zigzag::zigzag_encode;

use super::RunEndScheme;
use super::SparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// ZigZag encoding for negative integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZigZagScheme;

impl Scheme for ZigZagScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.zigzag"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    /// Children: encoded=0.
    fn num_children(&self) -> usize {
        1
    }

    /// ZigZag is a bijective value transform that preserves cardinality, run patterns, and value
    /// dominance. If Dict, RunEnd, or Sparse lost on the original array, they will lose on ZigZag's
    /// output too, so we skip evaluating them.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::All,
            },
        ]
    }

    /// Dict codes are unsigned integers (0..cardinality). ZigZag only helps negatives.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // ZigZag only transforms negative values to positive. Without further compression,
        // the output is the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);

        // ZigZag is only useful when there are negative values.
        if !stats.erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(data.array_as_primitive())?;
        let encoded = zag.encoded().clone().execute::<PrimitiveArray>(exec_ctx)?;

        let compressed = compressor.compress_child(
            &encoded.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        Ok(ZigZag::try_new(compressed)?.into_array())
    }
}
