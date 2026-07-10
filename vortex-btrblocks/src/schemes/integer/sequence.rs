// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sequence integer encoding for sequential patterns.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CandidateEstimate;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DeferredEvaluation;
use vortex_compressor::scheme::ResolvedEvaluation;
use vortex_compressor::scheme::SchemeEvaluation;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_sequence::sequence_encode;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Sequence encoding for sequential patterns.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SequenceScheme;

impl Scheme for SequenceScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.sequence"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    /// Sequence encoding on dictionary codes just adds a layer of indirection without compressing
    /// the data. Dict codes are compact integers that benefit from BitPacking or FoR, not from
    /// sequence detection.
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

    fn evaluate(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        // It is pointless checking if a sample is a sequence since it will not correspond to the
        // entire array.
        if compress_ctx.is_sample() {
            return SchemeEvaluation::Skip;
        }
        let stats = data.integer_stats(exec_ctx);

        // `SequenceArray` does not support nulls.
        if stats.null_count() > 0 {
            return SchemeEvaluation::Skip;
        }

        // If the distinct_values_count was computed, and not all values are unique, then this
        // cannot be encoded as a sequence array.
        if stats
            .distinct_count()
            .is_some_and(|count| count as usize != data.array_len())
        {
            return SchemeEvaluation::Skip;
        }

        // TODO(connor): `sequence_encode` allocates the encoded array just to confirm feasibility.
        // A cheaper `is_sequence` probe would let us skip the allocation entirely.
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, data, _ctx, exec_ctx| {
                // `SequenceArray` stores exactly two scalars (base and multiplier), so the best
                // achievable compression ratio is `array_len / 2`.
                let compressed_size = 2usize;
                let max_ratio = data.array_len() as f64 / compressed_size as f64;

                // TODO(connor): We should pass this array back to the compressor in the case that
                // we do want to sequence encode this so that we do not need to recompress.
                if sequence_encode(data.array_as_primitive(), exec_ctx)?.is_none() {
                    return Ok(ResolvedEvaluation::Skip);
                }
                // TODO(connor): Should we get the actual ratio here?
                Ok(ResolvedEvaluation::Candidate(
                    CandidateEstimate::from_compression_ratio(max_ratio),
                ))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats(exec_ctx);

        if stats.null_count() > 0 {
            vortex_bail!("sequence encoding does not support nulls");
        }
        sequence_encode(data.array_as_primitive(), exec_ctx)?
            .ok_or_else(|| vortex_err!("cannot sequence encode array"))
    }
}
