// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cognitive_complexity,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "compute kernel uses short mathematical names"
)]

//! Pushdown comparisons that exploit per-piece bounds.
//!
//! For each piece we precompute `[piece_min, piece_max]` (via `bounds::value_bounds`). For a
//! predicate `x > threshold`:
//!
//! - `piece_max <= threshold` ⇒ zero matches in this piece, skip without decoding.
//! - `piece_min  > threshold` ⇒ every element matches, add piece_len without decoding.
//! - otherwise ⇒ decode the piece's residuals and count element by element.
//!
//! The same idea generalises to `<`, `>=`, `<=`, `=`, `!=`. We only ship `>` here as the
//! representative pushdown and benchmark target; the other operators follow the same
//! piece-skip → decode-straggler pattern.

use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::array::NeaTSArray;
use crate::array::NeaTSArraySlotsExt;
use crate::compute::bounds::model_bounds;
use crate::compute::bounds::value_bounds;
use crate::models::ModelKind;
use crate::models::eval;

/// Count how many decoded values in `array` satisfy `value > threshold`, without ever fully
/// canonicalising the array. Pieces whose bounds prove the predicate are skipped entirely;
/// only pieces whose bounds straddle the threshold are decoded.
///
/// Returns `(count, decoded_pieces, total_pieces)` so callers can see how much skipping
/// happened — useful for benchmarks and selectivity diagnostics.
pub fn count_greater_than(
    array: &NeaTSArray,
    threshold: f64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(usize, usize, usize)> {
    let piece_starts = array
        .piece_starts()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let model_ids = array.model_ids().clone().execute::<PrimitiveArray>(ctx)?;
    let coeff_a = array.coeff_a().clone().execute::<PrimitiveArray>(ctx)?;
    let coeff_b = array.coeff_b().clone().execute::<PrimitiveArray>(ctx)?;
    let coeff_c = array.coeff_c().clone().execute::<PrimitiveArray>(ctx)?;
    let residuals = array.residuals().clone().execute::<PrimitiveArray>(ctx)?;
    let scale = array.data().scale();

    let starts = piece_starts.as_slice::<u32>();
    let kinds = model_ids.as_slice::<u8>();
    let coeff_a = coeff_a.as_slice::<f64>();
    let coeff_b = coeff_b.as_slice::<f64>();
    let coeff_c = coeff_c.as_slice::<f64>();
    let p = kinds.len();

    let DType::Primitive(rptype, _) = residuals.dtype().clone() else {
        vortex_bail!("residuals must be primitive");
    };

    let mut count: usize = 0;
    let mut decoded_pieces: usize = 0;

    // Free over-bound from the residual ptype: any residual is in [-MAX_R, MAX_R]. Using this
    // we can sometimes resolve a piece's predicate without scanning residuals at all.
    let max_abs_r_unscaled = match rptype {
        PType::I8 => i8::MAX as f64,
        PType::I16 => i16::MAX as f64,
        PType::I32 => i32::MAX as f64,
        PType::I64 => i64::MAX as f64,
        _ => vortex_bail!("residuals must be signed int"),
    };
    let max_abs_r = max_abs_r_unscaled * scale;

    macro_rules! sweep_with {
        ($int:ty) => {{
            let r = residuals.as_slice::<$int>();
            for piece in 0..p {
                let s = starts[piece] as usize;
                let e = starts[piece + 1] as usize;
                let kind = ModelKind::from_u8(kinds[piece]).vortex_expect("valid kind");
                let mb = model_bounds(kind, coeff_a[piece], coeff_b[piece], coeff_c[piece], e - s);

                // Cheap-bound phase: use the residual-ptype over-bound. Many pieces resolve
                // here without ever touching residual values.
                if mb.max + max_abs_r <= threshold {
                    continue;
                }
                if mb.min - max_abs_r > threshold {
                    count += e - s;
                    continue;
                }

                // Tight-bound phase: scan piece residuals for their actual min/max. Cheaper
                // than decode because we work in i8/i16/i32 instead of f64.
                let mut r_min = i64::MAX;
                let mut r_max = i64::MIN;
                for v in &r[s..e] {
                    let v = *v as i64;
                    if v < r_min {
                        r_min = v;
                    }
                    if v > r_max {
                        r_max = v;
                    }
                }
                let vb = value_bounds(mb, r_min, r_max, scale);
                if vb.max <= threshold {
                    continue;
                }
                if vb.min > threshold {
                    count += e - s;
                    continue;
                }

                // Straddler: decode and count.
                decoded_pieces += 1;
                let a = coeff_a[piece];
                let b = coeff_b[piece];
                let c = coeff_c[piece];
                for (k, ri) in r[s..e].iter().enumerate() {
                    let t = k as f64;
                    let decoded = eval(kind, a, b, c, t) + (*ri as f64) * scale;
                    if decoded > threshold {
                        count += 1;
                    }
                }
            }
        }};
    }

    match rptype {
        PType::I8 => sweep_with!(i8),
        PType::I16 => sweep_with!(i16),
        PType::I32 => sweep_with!(i32),
        PType::I64 => sweep_with!(i64),
        _ => vortex_bail!("residuals must be signed int"),
    }

    Ok((count, decoded_pieces, p))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::NeaTSOptions;
    use crate::compress::neats_encode;

    #[test]
    fn count_greater_than_matches_naive() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Smooth signal — most pieces should resolve via bounds without decoding.
        let values: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 0.01).sin() + 0.001 * i as f64)
            .collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-6),
                ..NeaTSOptions::default()
            },
            &mut ctx,
        )?;

        for threshold in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 3.5] {
            let naive_count = values.iter().filter(|&&v| v > threshold).count();
            let (pushdown_count, decoded, total) =
                count_greater_than(&encoded, threshold, &mut ctx)?;
            // The pushdown count is exact when no piece straddles, and within the lossy
            // epsilon-bound of the naive count when some pieces straddle. Our test uses ε=1e-6,
            // small enough that integer counts match.
            assert_eq!(
                pushdown_count, naive_count,
                "threshold={threshold}: pushdown={pushdown_count} naive={naive_count} \
                 (decoded {decoded}/{total} pieces)",
            );
        }
        // Pin into the binary so dead-code elimination doesn't trim into_array.
        drop(encoded.into_array());
        Ok(())
    }
}
