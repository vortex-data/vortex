// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pco (pcodec) integer compression.

use std::num::NonZeroUsize;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_pco::Pco;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

/// Pco (pcodec) compression for integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

impl Scheme for PcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.pco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Pco.id()]
    }

    /// The per-chunk tANS tables and bin metadata, written once per Pco chunk of up to
    /// `pco::DEFAULT_MAX_PAGE_N` (262,144) valid values rather than growing with row count inside
    /// a sample.
    fn fixed_nbytes(&self, compressed: &ArrayRef) -> u64 {
        compressed
            .as_opt::<Pco>()
            .map_or(0, |pco| pco.chunk_metas_nbytes())
    }

    /// Pco starts a new chunk every `pco::DEFAULT_MAX_PAGE_N` (262,144) valid values. Scaling by
    /// `full_len.div_ceil(262_144)` is exact for non-null arrays and slightly overestimates when
    /// nulls reduce the valid count below `full_len`.
    fn fixed_nbytes_chunk_len(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(pco::DEFAULT_MAX_PAGE_N)
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        use vortex_array::dtype::PType;

        // Pco does not support I8 or U8.
        if matches!(data.array_as_primitive().ptype(), PType::I8 | PType::U8) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Pco stores the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        Encoded::of_leaf(data.array().clone(), |input, exec_ctx| {
            let input = input
                .as_opt::<Primitive>()
                .vortex_expect("compressing the validity keeps a primitive leaf primitive");
            Ok(
                Pco::from_primitive(input, pco::DEFAULT_COMPRESSION_LEVEL, 8192, exec_ctx)?
                    .into_array(),
            )
        })
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_compressor::scheme::Scheme;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;
    use vortex_session::VortexSession;

    use crate::schemes::integer::IntDictScheme;
    use crate::schemes::integer::PcoScheme;
    use crate::schemes::test_utils::TargetRatioScheme;
    use crate::schemes::test_utils::UnamortizedScheme;
    use crate::schemes::test_utils::assert_pco_fixed_nbytes_and_fallback;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_pco::initialize(&session);
        session
    });

    #[test]
    fn test_fixed_nbytes_reports_chunk_metas_and_zero_for_fallback() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter(0..1024_i32);
        assert_pco_fixed_nbytes_and_fallback(&PcoScheme, primitive, &mut ctx)
    }

    #[test]
    fn test_pco_amortizes_chunk_metas_on_large_skewed_int_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 131_072usize;
        // 160 distinct values across 20 irregularly spaced clusters of 8: IntDictScheme encodes
        // them with 8-bit codes (2,048 B on the 2,048-row production sample plus amortized
        // dictionary bytes), while Pco uses ~7.3 bits/row page data plus a multi-bin chunk_meta
        // whose unamortized cost pushes the sample past IntDictScheme.
        let palette: [i32; 160] = std::array::from_fn(|idx| {
            let group = i32::try_from(idx / 8).expect("group < 20");
            let member = i32::try_from(idx % 8).expect("member < 8");
            let jitter = (group * group * 7 + group * 3) % 19;
            group * 1_000 + jitter * 17 + member
        });
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            values.push(palette[((state >> 32) as usize) % palette.len()]);
        }

        let array = PrimitiveArray::from_iter(values);

        let unamortized =
            CascadingCompressor::new(vec![&UnamortizedScheme(PcoScheme), &IntDictScheme])
                .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            unamortized.is::<Dict>(),
            "without fixed_nbytes amortization, int PcoScheme should lose to IntDictScheme on the production sample, got {}",
            unamortized.encoding_id()
        );

        let compressed = CascadingCompressor::new(vec![&PcoScheme, &IntDictScheme])
            .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            compressed.is::<Pco>(),
            "expected PcoScheme to win once chunk_metas are amortized, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_pco_estimate_never_underestimates_above_chunk_len() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 300_000usize;
        // Generate 300,000 i.i.d. values from a 160-value multi-cluster palette (producing a
        // sizable Pco chunk_meta) without periodic lookback artifacts.
        let palette: [i32; 160] = std::array::from_fn(|idx| {
            let group = i32::try_from(idx / 8).expect("group < 20");
            let member = i32::try_from(idx % 8).expect("member < 8");
            let jitter = (group * group * 7 + group * 3) % 19;
            group * 1_000 + jitter * 17 + member
        });
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let values: Buffer<i32> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                palette[((state >> 32) as usize) % palette.len()]
            })
            .collect();

        // Non-null: 300,000 valid values span 2 Pco chunks (> DEFAULT_MAX_PAGE_N = 262,144).
        let non_null = PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array();
        let actual_non_null =
            CascadingCompressor::new(vec![&PcoScheme]).compress(&non_null, &mut ctx)?;
        assert!(actual_non_null.is::<Pco>());
        let actual_non_null_ratio = non_null.nbytes() as f64 / actual_non_null.nbytes() as f64;

        let target_actual_non_null: &'static dyn Scheme = Box::leak(Box::new(TargetRatioScheme {
            ratio: actual_non_null_ratio,
        }));
        let target_loose_non_null: &'static dyn Scheme = Box::leak(Box::new(TargetRatioScheme {
            ratio: actual_non_null_ratio * 0.90,
        }));

        // With fixed_nbytes_chunk_len, PcoScheme does not underestimate the real 2-chunk
        // compressed size while staying within 10% of the actual ratio.
        let chunk_aware_non_null =
            CascadingCompressor::new(vec![target_actual_non_null, &PcoScheme])
                .compress(&non_null, &mut ctx)?;
        assert!(
            !chunk_aware_non_null.is::<Pco>(),
            "expected chunk-aware PcoScheme not to underestimate non-null 300,000-row array"
        );
        let within_margin_non_null =
            CascadingCompressor::new(vec![target_loose_non_null, &PcoScheme])
                .compress(&non_null, &mut ctx)?;
        assert!(
            within_margin_non_null.is::<Pco>(),
            "expected chunk-aware PcoScheme estimate to stay within 10% of actual non-null ratio"
        );

        // Nullable (50% nulls): 300,000 rows have only 150,000 valid values (1 Pco chunk), so
        // scaling by full_len.div_ceil(262_144) = 2 biases toward a slight overestimate rather
        // than an underestimate.
        let validity = Validity::from(BitBuffer::from_iter((0..len).map(|i| i % 2 == 0)));
        let nullable = PrimitiveArray::new(values, validity).into_array();
        let actual_nullable =
            CascadingCompressor::new(vec![&PcoScheme]).compress(&nullable, &mut ctx)?;
        assert!(actual_nullable.is::<Pco>());
        let actual_nullable_ratio = nullable.nbytes() as f64 / actual_nullable.nbytes() as f64;

        let target_actual_nullable: &'static dyn Scheme = Box::leak(Box::new(TargetRatioScheme {
            ratio: actual_nullable_ratio,
        }));
        let target_loose_nullable: &'static dyn Scheme = Box::leak(Box::new(TargetRatioScheme {
            ratio: actual_nullable_ratio * 0.90,
        }));

        let chunk_aware_nullable =
            CascadingCompressor::new(vec![target_actual_nullable, &PcoScheme])
                .compress(&nullable, &mut ctx)?;
        assert!(
            !chunk_aware_nullable.is::<Pco>(),
            "expected chunk-aware PcoScheme not to underestimate nullable 300,000-row array"
        );
        let within_margin_nullable =
            CascadingCompressor::new(vec![target_loose_nullable, &PcoScheme])
                .compress(&nullable, &mut ctx)?;
        assert!(
            within_margin_nullable.is::<Pco>(),
            "expected chunk-aware PcoScheme estimate to stay within 10% of actual nullable ratio"
        );

        Ok(())
    }
}
