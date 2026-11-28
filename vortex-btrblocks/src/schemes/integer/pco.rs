// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pco (pcodec) integer compression.

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
    ///
    /// Arrays exceeding one Pco chunk pay `ceil(n_valid / 262_144)` chunk metas, so treating this
    /// as a single fixed cost slightly under-charges Pco on very long arrays.
    fn fixed_nbytes(&self, compressed: &ArrayRef) -> u64 {
        compressed
            .as_opt::<Pco>()
            .map_or(0, |pco| pco.chunk_metas_nbytes())
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

    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_compressor::CascadingCompressor;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;
    use vortex_session::VortexSession;

    use crate::ArrayAndStats;
    use crate::CompressorContext;
    use crate::Encoded;
    use crate::Scheme;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::PcoScheme;

    static BITPACKING: BitPackingScheme = BitPackingScheme::v1();

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_pco::initialize(&session);
        vortex_fastlanes::initialize(&session);
        session
    });

    /// Identical to [`PcoScheme`] except it leaves `fixed_nbytes` at the default `0`.
    #[derive(Debug, Copy, Clone)]
    struct UnamortizedPcoScheme;

    impl Scheme for UnamortizedPcoScheme {
        fn scheme_name(&self) -> &'static str {
            PcoScheme.scheme_name()
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            PcoScheme.matches(canonical)
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            PcoScheme.produced_encodings()
        }

        fn expected_compression_ratio(
            &self,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            PcoScheme.expected_compression_ratio(data, compress_ctx, exec_ctx)
        }

        fn encode(
            &self,
            data: &ArrayAndStats,
            compress_ctx: &CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<Encoded> {
            PcoScheme.encode(data, compress_ctx, exec_ctx)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            PcoScheme.compress(compressor, data, compress_ctx, exec_ctx)
        }
    }

    #[test]
    fn test_fixed_nbytes_reports_chunk_metas_and_zero_for_fallback() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter(0..1024_i32);
        let pco = Pco::from_primitive(
            primitive.as_view(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
            &mut ctx,
        )?;
        let expected_fixed = pco.chunk_metas_nbytes();
        assert!(expected_fixed > 0);
        assert_eq!(PcoScheme.fixed_nbytes(&pco.into_array()), expected_fixed);
        assert_eq!(PcoScheme.fixed_nbytes(&primitive.into_array()), 0);
        Ok(())
    }

    #[test]
    fn test_pco_amortizes_chunk_metas_on_large_skewed_int_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 131_072usize;
        // 48 irregularly spaced clusters of width 6 in 0..1023 plus sparse outliers at 1_000_000:
        // BitPacking patches the outliers and packs at 10 bits/row, while Pco must use multi-bin
        // encoding whose chunk_meta pushes the unamortized 2,048-row production sample past
        // BitPacking.
        let centers: [i32; 48] = std::array::from_fn(|idx| {
            let idx = i32::try_from(idx).expect("idx < 48");
            let base = idx * 21;
            let jitter = (idx * idx * 7 + idx * 3) % 7;
            base + jitter
        });
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if i % 256 == 0 {
                values.push(1_000_000);
            } else {
                let cluster = ((state >> 56) as usize) % centers.len();
                let offset = ((state >> 32) % 6) as i32;
                values.push(centers[cluster] + offset);
            }
        }

        let array = PrimitiveArray::from_iter(values);

        let unamortized = CascadingCompressor::new(vec![&UnamortizedPcoScheme, &BITPACKING])
            .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            !unamortized.is::<Pco>(),
            "without fixed_nbytes amortization, int PcoScheme should lose to BitPacking on the production sample"
        );

        let compressed = CascadingCompressor::new(vec![&PcoScheme, &BITPACKING])
            .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            compressed.is::<Pco>(),
            "expected PcoScheme to win once chunk_metas are amortized, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }
}
