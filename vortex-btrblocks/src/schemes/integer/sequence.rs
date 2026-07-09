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
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // It is pointless checking if a sample is a sequence since it will not correspond to the
        // entire array.
        if compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);

        // `SequenceArray` does not support nulls.
        if stats.null_count() > 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the distinct_values_count was computed, and not all values are unique, then this
        // cannot be encoded as a sequence array.
        if stats
            .distinct_count()
            .is_some_and(|count| count as usize != data.array_len())
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // TODO(connor): `sequence_encode` allocates the encoded array just to confirm feasibility.
        // A cheaper `is_sequence` probe would let us skip the allocation entirely.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, threshold, _ctx, exec_ctx| {
                // `SequenceArray` stores exactly two scalars (base and multiplier), so the best
                // achievable compression ratio is `array_len / 2`.
                let compressed_size = 2usize;
                let max_ratio = data.array_len() as f64 / compressed_size as f64;

                // If we cannot beat the best so far, then we do not want to even try sequence
                // encoding the data.
                if threshold.best_case_ratio_cannot_win(max_ratio) {
                    return Ok(EstimateVerdict::Skip);
                }

                // TODO(connor): We should pass this array back to the compressor in the case that
                // we do want to sequence encode this so that we do not need to recompress.
                if sequence_encode(data.array_as_primitive(), exec_ctx)?.is_none() {
                    return Ok(EstimateVerdict::Skip);
                }
                // TODO(connor): Should we get the actual ratio here?
                Ok(EstimateVerdict::Ratio(max_ratio))
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::estimate::EstimateVerdict;
    use vortex_sequence::Sequence;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// Number of values in the test arrays; Sequence's best case is `LEN / 2`.
    const LEN: usize = 2048;

    /// Immediate fixed-ratio competitor; its `compress` emits a tiny constant array so the
    /// winner is observable from the output encoding.
    #[derive(Debug)]
    struct FixedRatioScheme {
        ratio: f64,
    }

    impl Scheme for FixedRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.fixed_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(self.ratio))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(ConstantArray::new(
                Scalar::primitive(0i64, Nullability::NonNullable),
                data.array_len(),
            )
            .into_array())
        }
    }

    /// An exact arithmetic sequence, encodable by [`SequenceScheme`].
    fn arithmetic_sequence() -> ArrayRef {
        let values: Buffer<i64> = (0..LEN as i64).map(|i| 10_000 + 7 * i).collect();
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }

    /// With the incumbent below Sequence's best case, the callback must proceed past the
    /// threshold check, trial-encode, and win.
    #[test]
    fn sequence_wins_over_low_threshold() -> VortexResult<()> {
        static COMPETITOR: FixedRatioScheme = FixedRatioScheme { ratio: 2.0 };
        let compressor = CascadingCompressor::new(vec![&COMPETITOR, &SequenceScheme]);

        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&arithmetic_sequence(), &mut exec_ctx)?;

        assert!(compressed.is::<Sequence>());
        Ok(())
    }

    /// With the incumbent at exactly Sequence's best case (`LEN / 2`), Sequence must not be
    /// chosen — the same decision the pre-threshold-handle `max_ratio <= best` skip produced
    /// on this input.
    #[test]
    fn sequence_loses_at_best_case_tie() -> VortexResult<()> {
        static COMPETITOR: FixedRatioScheme = FixedRatioScheme {
            ratio: LEN as f64 / 2.0,
        };
        let compressor = CascadingCompressor::new(vec![&COMPETITOR, &SequenceScheme]);

        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&arithmetic_sequence(), &mut exec_ctx)?;

        assert!(compressed.is::<Constant>());
        Ok(())
    }
}
