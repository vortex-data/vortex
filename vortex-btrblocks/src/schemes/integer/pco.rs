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

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;
    use vortex_session::VortexSession;

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
        // 64 irregularly spaced narrow clusters in 0..1023 (no common GCD for IntMult):
        // BitPacking needs 10 bits/row (1,280 B on a 1,024-row sample), while Pco uses ~7 bits/row
        // (~896 B page data) plus a multi-bin chunk_meta that exceeds BitPacking until amortized.
        let centers: [i32; 64] = std::array::from_fn(|idx| {
            let idx = i32::try_from(idx).expect("idx < 64");
            let base = idx * 15;
            let jitter = (idx * idx * 7 + idx * 3) % 11;
            base + jitter
        });
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let cluster = ((state >> 56) & 0x3f) as usize;
            let offset = ((state >> 32) & 0x03) as i32;
            values.push(centers[cluster] + offset);
        }

        let array = PrimitiveArray::from_iter(values);

        // Verify the 1,024-row sample has non-trivial chunk_meta overhead relative to bit-packing.
        let sample = PrimitiveArray::new(
            Buffer::from(array.as_slice::<i32>()[..1024].to_vec()),
            Validity::NonNullable,
        );
        let sample_pco = Pco::from_primitive(
            sample.as_view(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
            &mut ctx,
        )?;
        let bp_compressor = CascadingCompressor::new(vec![&BITPACKING]);
        let sample_bp = bp_compressor.compress(&sample.into_array(), &mut ctx)?;
        assert!(
            sample_pco.as_array().nbytes() > sample_bp.nbytes(),
            "without fixed_nbytes amortization, Pco sample ({}) should lose to BitPacking ({})",
            sample_pco.as_array().nbytes(),
            sample_bp.nbytes(),
        );

        let compressor = CascadingCompressor::new(vec![&PcoScheme, &BITPACKING]);
        let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;
        assert!(
            compressed.is::<Pco>(),
            "expected PcoScheme to win once chunk_metas are amortized, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }
}
