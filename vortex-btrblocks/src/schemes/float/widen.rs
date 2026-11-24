// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless floating-point width-narrowing scheme backed by [`WidenArray`].
//!
//! Narrowing an `f64` column whose valid values all round-trip bit-for-bit through `f32` (or
//! narrowing `f64`/`f32` through `f16`) halves or quarters raw storage and lets child float
//! schemes such as `ALP` and `ALPRD` operate at the narrower width without losing a single bit.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::widen::encode::narrow_float_values;
use vortex_array::arrays::widen::encode::narrowest_float_ptype;
use vortex_array::arrays::widen::widen_v2_id;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::float::FloatDictScheme;
use crate::schemes::float::FloatRLEScheme;

/// Cached narrowest fitting width computed during estimation so [`FloatWidenScheme::compress`]
/// avoids re-scanning the full array.
#[derive(Debug, Clone, Copy)]
struct NarrowestFloatTarget(Option<PType>);

/// Lossless floating-point narrowing scheme (`f64 -> f32/f16`, `f32 -> f16`).
///
/// Produces `vortex.widen.v2` wire nodes, which belong to the `UNSTABLE` edition; builders whose
/// allowed encodings do not include `vortex.widen.v2` (or CUDA presets) drop this scheme.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatWidenScheme;

impl Scheme for FloatWidenScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.widen"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![widen_v2_id()]
    }

    /// Children: narrowed values = 0.
    fn num_children(&self) -> usize {
        1
    }

    /// Keep `Dict` and `RLE` at the root rather than under `Widen`.
    ///
    /// Leaving `Dict` at the root preserves DuckDB exporter and dict-layout fast paths; `Dict`
    /// and `RLE` still compress their values children through `FloatWidenScheme` when beneficial.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: FloatDictScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: FloatRLEScheme.id(),
                children: ChildSelection::All,
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let primitive = data.array_as_primitive().into_owned();
        let ptype = primitive.ptype();
        if ptype == PType::F16 || primitive.is_empty() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let Ok(mask) = primitive
            .validity()
            .and_then(|v| v.execute_mask(primitive.len(), exec_ctx))
        else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        if matches!(mask, Mask::AllFalse(_)) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // `narrowest_float_ptype` exits on the first non-fitting 64-element chunk for general
        // columns and confirms 100% full-array bit-exact fit before returning a target.
        let Some(narrowest) = data
            .get_or_insert_with(|| NarrowestFloatTarget(narrowest_float_ptype(&primitive, &mask)))
            .0
        else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };

        // When the cascade budget is exhausted, `compress_child` leaves the narrowed primitive
        // values uncompressed, so the exact ratio is known without sampling.
        if compress_ctx.finished_cascading() {
            let Ok(before) = uncompressed_size_in_bytes(data.array(), exec_ctx) else {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            };
            let before = before as u64;
            let validity_bytes = before.saturating_sub(
                u64::try_from(primitive.len() * ptype.byte_width()).unwrap_or(before),
            );
            let after = u64::try_from(primitive.len() * narrowest.byte_width())
                .unwrap_or(before)
                .saturating_add(validity_bytes);
            if after == 0 {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            }
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(
                before as f64 / after as f64,
            ));
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
        let primitive = data.array_as_primitive().into_owned();
        let mask = primitive
            .validity()?
            .execute_mask(primitive.len(), exec_ctx)?;
        let Some(narrowest) = data
            .get_or_insert_with(|| NarrowestFloatTarget(narrowest_float_ptype(&primitive, &mask)))
            .0
        else {
            return Ok(data.array().clone());
        };

        // When an `f64` array fits both `f16` and `f32`, compare both compressed children because
        // `ALP` and `ALPRD` skip `f16` and may compress `f32` smaller than raw `f16`.
        let candidates: &[PType] = match (primitive.ptype(), narrowest) {
            (PType::F64, PType::F16) if !compress_ctx.finished_cascading() => {
                &[PType::F16, PType::F32]
            }
            _ => &[narrowest],
        };

        let mut best_child: Option<ArrayRef> = None;
        for &target in candidates {
            let narrowed = narrow_float_values(&primitive, target, &mask)?;
            let compressed_child = compressor.compress_child(
                &narrowed.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;
            if best_child
                .as_ref()
                .is_none_or(|best| compressed_child.nbytes() < best.nbytes())
            {
                best_child = Some(compressed_child);
            }
        }

        let Some(best_child) = best_child else {
            return Ok(data.array().clone());
        };
        Ok(WidenArray::try_new(best_child, primitive.dtype().clone())?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::Widen;
    use vortex_array::arrays::widen::WidenArraySlotsExt;
    use vortex_array::arrays::widen::widen_v2_id;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::half::f16;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::FloatWidenScheme;
    use crate::BtrBlocksCompressor;
    use crate::BtrBlocksCompressorBuilder;
    use crate::CompressionSession;
    use crate::SchemeExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<CompressionSession>();
        vortex_array::initialize(&session);
        session
    });

    fn unrestricted_compressor() -> BtrBlocksCompressor {
        BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build()
    }

    /// Deterministic pseudorandom f32-exact values widened to f64 (modeled after OpenAI embeddings
    /// stored as f64) so ALP cannot exploit small decimal scale and ALPRD+Widen wins decisively.
    fn pseudo_random_f32_in_f64(len: usize) -> Vec<f64> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5_f32;
                f64::from(u)
            })
            .collect()
    }

    /// Deterministic diverse finite f16 bit patterns.
    fn pseudo_random_f16(len: usize) -> Vec<f16> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let exp = u16::try_from((state >> 24) % 28 + 1).unwrap_or(1);
                let mant = (state & 0x03ff) as u16;
                let sign = ((state >> 31) as u16) << 15;
                f16::from_bits(sign | (exp << 10) | mant)
            })
            .collect()
    }

    fn assert_bit_exact_f64(
        actual: &ArrayRef,
        expected: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let actual_prim = actual.clone().execute::<PrimitiveArray>(ctx)?;
        assert_arrays_eq!(actual_prim, expected, ctx);
        let mask = expected.validity()?.execute_mask(expected.len(), ctx)?;
        let actual_slice = actual_prim.as_slice::<f64>();
        let expected_slice = expected.as_slice::<f64>();
        for i in 0..expected.len() {
            if mask.value(i) {
                assert_eq!(
                    actual_slice[i].to_bits(),
                    expected_slice[i].to_bits(),
                    "bit mismatch at row {i}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn compresses_f32_exact_f64_with_widen() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = pseudo_random_f32_in_f64(8192);
        let input = PrimitiveArray::from_iter(values);

        let without_widen = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([FloatWidenScheme.id()])
            .build()
            .compress(&input.clone().into_array(), &mut ctx)?;
        let with_widen =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;

        assert!(
            with_widen.is::<Widen>(),
            "expected Widen at root, got {}",
            with_widen.encoding_id()
        );
        assert_eq!(
            with_widen.as_::<Widen>().values().dtype().as_ptype(),
            PType::F32
        );
        assert!(
            with_widen.nbytes() < without_widen.nbytes(),
            "expected Widen to compress smaller, got {} vs {}",
            with_widen.nbytes(),
            without_widen.nbytes()
        );
        assert_bit_exact_f64(&with_widen, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn compresses_f16_exact_f64_choosing_best_narrow_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<f64> = pseudo_random_f16(4096)
            .into_iter()
            .map(f16::to_f64)
            .collect();
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(
            compressed.is::<Widen>(),
            "expected Widen at root, got {}",
            compressed.encoding_id()
        );
        assert_eq!(
            compressed.as_::<Widen>().values().dtype().as_ptype(),
            PType::F16
        );
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn preserves_neg_zero_and_canonical_nan_bits() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(4096);
        values[0] = -0.0_f64;
        values[1] = 0.0_f64;
        values[2] = f64::INFINITY;
        values[3] = f64::NEG_INFINITY;
        values[4] = f64::NAN;
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn skips_non_f32_outlier_and_still_compresses_normally() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(16_384);
        values[100] = 0.1_f64;
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(!compressed.is::<Widen>());
        // Must still compress with ALPRD rather than bailing out to uncompressed Primitive.
        assert!(
            compressed.nbytes() < input.nbytes(),
            "expected fallback float scheme to compress, got {} vs {}",
            compressed.nbytes(),
            input.nbytes()
        );
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn skips_signaling_and_negative_nan() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        for nan in [f64::from_bits(0x7ff0_0000_2000_0000), -f64::NAN] {
            let mut values = pseudo_random_f32_in_f64(4096);
            values[77] = nan;
            let input = PrimitiveArray::from_iter(values);

            let compressed =
                unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
            assert!(!compressed.is::<Widen>());
            assert_bit_exact_f64(&compressed, &input, &mut ctx)?;
        }

        Ok(())
    }

    #[test]
    fn ignores_non_f32_null_garbage() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(4096);
        let mut valid = vec![true; values.len()];
        for i in (0..values.len()).step_by(13) {
            values[i] = 0.1_f64;
            valid[i] = false;
        }
        let input = PrimitiveArray::new(Buffer::from(values), Validity::from_iter(valid));

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn compresses_f16_exact_f32_to_f16_widen() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<f32> = pseudo_random_f16(4096)
            .into_iter()
            .map(f16::to_f32)
            .collect();
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_eq!(
            compressed.as_::<Widen>().values().dtype().as_ptype(),
            PType::F16
        );
        assert_arrays_eq!(compressed, input, &mut ctx);

        Ok(())
    }

    #[test]
    fn wire_round_trip_uses_widen_v2_id() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter(pseudo_random_f32_in_f64(4096));
        let compressed = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FloatWidenScheme)
            .build()
            .compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());

        let array_ctx = ArrayContext::empty();
        let mut blob = ByteBufferMut::empty();
        for buf in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
            blob.extend_from_slice(buf.as_ref());
        }
        let ids = array_ctx.to_ids();
        assert!(ids.contains(&widen_v2_id()), "expected widen.v2 in {ids:?}");

        let serialized = SerializedArray::try_from(blob.freeze())?;
        let decoded = serialized.decode(
            compressed.dtype(),
            compressed.len(),
            &ReadContext::new(ids),
            &SESSION,
        )?;
        assert_bit_exact_f64(&decoded, &input, &mut ctx)?;

        Ok(())
    }
}
