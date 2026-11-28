// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pco (pcodec) float compression.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_pco::Pco;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

/// Pco (pcodec) compression for floats.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

impl Scheme for PcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.pco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
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
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
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

    use vortex_alp::ALPRD;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;
    use vortex_session::VortexSession;

    use crate::schemes::float::ALPRDScheme;
    use crate::schemes::float::PcoScheme;
    use crate::schemes::test_utils::UnamortizedScheme;
    use crate::schemes::test_utils::assert_pco_fixed_nbytes_and_fallback;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);
        vortex_pco::initialize(&session);
        session
    });

    #[test]
    fn test_fixed_nbytes_reports_chunk_metas_and_zero_for_fallback() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter((0..1024).map(|idx| idx as f64 * 0.25));
        assert_pco_fixed_nbytes_and_fallback(&PcoScheme, primitive, &mut ctx)
    }

    #[test]
    fn test_pco_amortizes_chunk_metas_on_large_clustered_float_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 131_072usize;
        // 32 clusters across 8 distinct top-16-bit prefixes (4 clusters per prefix) with 44 random
        // low mantissa bits: ALPRD fits all 8 prefixes in its dictionary (~51 bits/row on the
        // 2,048-row production sample), while Pco uses ~50 bits/row page data plus a multi-bin
        // chunk_meta that pushes the unamortized sample past ALPRD.
        let bases: [u64; 32] = std::array::from_fn(|idx| {
            let idx = u64::try_from(idx).expect("idx < 32");
            let exp = (1023 + (idx / 4)) << 52;
            let sub = ((idx % 4) * 5 + (idx % 2)) << 44;
            exp | sub
        });
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let cluster = ((state >> 56) & 0x1f) as usize;
            let low_mantissa = (state >> 11) & ((1_u64 << 44) - 1);
            values.push(f64::from_bits(bases[cluster] | low_mantissa));
        }

        let array = PrimitiveArray::from_iter(values);

        let unamortized =
            CascadingCompressor::new(vec![&UnamortizedScheme(PcoScheme), &ALPRDScheme])
                .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            unamortized.is::<ALPRD>(),
            "without fixed_nbytes amortization, float PcoScheme should lose to ALPRD on the production sample, got {}",
            unamortized.encoding_id()
        );

        let compressed = CascadingCompressor::new(vec![&PcoScheme, &ALPRDScheme])
            .compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            compressed.is::<Pco>(),
            "expected float PcoScheme to win once chunk_metas are amortized, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }
}
