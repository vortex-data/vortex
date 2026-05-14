// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "model coefficients and residual quantization use mathematical short names"
)]

//! Compression entry point for NeaTS.
//!
//! The compressor walks the input greedily, extending each piece while a model from any enabled
//! family can fit the span within a residual budget bounded by `2^MAX_RESIDUAL_BITS`. When the
//! budget is exceeded the piece is sealed and a new one starts.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_pco::Pco;

use crate::array::NeaTS;
use crate::array::NeaTSArray;
use crate::array::NeaTSData;
use crate::models::FitResult;
use crate::models::ModelKind;
use crate::models::fit_best;

/// Choice of encoding for the residuals slot.
///
/// The residuals carry 99% of the compressed bytes in practice, so this is the highest-leverage
/// knob.
///
/// - `Pco` (default): hand residuals to `vortex-pco`. PCO's signed-integer pipeline (delta +
///   FSE + chunked statistics) handles noisy residuals well. PCO chunks its own pages, so the
///   bit width is tuned per-page, not per-NeaTS-piece.
/// - `BitPack`: leave residuals as a plain primitive array. The downstream cascading
///   compressor picks FoR + FastLanes bit-pack at chunk granularity. Fastest decode, loosest
///   ratio.
/// - `PerPieceBitPack` (paper-faithful): build one [`ChunkedArray`] chunk per NeaTS piece,
///   ZigZag-encode each piece's residuals, then bit-pack at a piece-specific width. This is
///   the residual encoding the NeaTS paper uses — each piece gets its own bit-width tuned to
///   its own max-abs residual, so a quiet piece pays only its actual residual bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualEncoding {
    /// Hand residuals to PCO. Best compression, slower compress + decompress.
    Pco,
    /// Leave residuals as a plain primitive array. Cascading compressor handles the rest.
    BitPack,
    /// Per-piece ZigZag + bit-pack, wrapped in a chunked array (paper-faithful).
    PerPieceBitPack,
}

/// Options for [`neats_encode`].
#[derive(Clone, Copy, Debug)]
pub struct NeaTSOptions {
    /// Per-value error bound. `None` requests lossless mode; `Some(eps)` requests an absolute
    /// error bound of `eps` per decoded value.
    pub epsilon: Option<f64>,
    /// Soft minimum and maximum piece length. Pieces may shrink below `min_piece_len` if a piece
    /// cannot extend further; they will not exceed `max_piece_len`.
    pub min_piece_len: usize,
    pub max_piece_len: usize,
    /// Encoding to use for the residuals slot. Defaults to `Pco` because it consistently halves
    /// the residual bytes on time-series-shaped signals.
    pub residual_encoding: ResidualEncoding,
}

impl Default for NeaTSOptions {
    fn default() -> Self {
        Self {
            epsilon: None,
            min_piece_len: 8,
            max_piece_len: 4096,
            residual_encoding: ResidualEncoding::Pco,
        }
    }
}

/// Residuals are stored as `i64`. We allow up to 63 bits per residual; that bounds the
/// quantization range of each piece to `(-2^62, 2^62)`.
const MAX_RESIDUAL_BITS: u32 = 62;

/// Encode a `PrimitiveArray` of `f32` or `f64` into a [`NeaTSArray`].
///
/// The execution context is used when the chosen [`ResidualEncoding`] needs to cascade through
/// another encoding (currently only PCO requires this).
pub fn neats_encode(
    parray: ArrayView<'_, Primitive>,
    options: NeaTSOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<NeaTSArray> {
    let values_f64: Vec<f64> = match parray.ptype() {
        PType::F32 => parray.as_slice::<f32>().iter().map(|v| *v as f64).collect(),
        PType::F64 => parray.as_slice::<f64>().to_vec(),
        other => vortex_bail!("NeaTS can only encode f32 or f64, got {other}"),
    };
    let validity = parray.validity()?;
    let logical_dtype = parray.array().dtype().clone();
    let len = values_f64.len();

    let scale = pick_scale(&values_f64, options.epsilon);
    let epsilon = options.epsilon.unwrap_or(0.0);

    let mut piece_starts = BufferMut::<u32>::with_capacity(64);
    let mut model_ids = BufferMut::<u8>::with_capacity(64);
    let mut coeff_a = BufferMut::<f64>::with_capacity(64);
    let mut coeff_b = BufferMut::<f64>::with_capacity(64);
    let mut coeff_c = BufferMut::<f64>::with_capacity(64);
    let mut residuals = BufferMut::<i64>::with_capacity(len);

    // Walk pieces.
    let mut i = 0usize;
    while i < len {
        let (piece_end, fit) = grow_piece(&values_f64, i, scale, &options);
        piece_starts.push(u32::try_from(i).unwrap_or(u32::MAX));
        model_ids.push(fit.kind as u8);
        coeff_a.push(fit.a);
        coeff_b.push(fit.b);
        coeff_c.push(fit.c);
        // Emit residuals.
        for (k, &y) in values_f64[i..piece_end].iter().enumerate() {
            let t = k as f64;
            let pred = crate::models::eval(fit.kind, fit.a, fit.b, fit.c, t);
            let r = ((y - pred) / scale).round() as i64;
            residuals.push(r);
        }
        i = piece_end;
    }
    // Sentinel: piece_starts[P] = len.
    piece_starts.push(u32::try_from(len).unwrap_or(u32::MAX));

    let piece_starts_data = piece_starts.freeze();
    let piece_starts =
        PrimitiveArray::new(piece_starts_data.clone(), Validity::NonNullable).into_array();
    let model_ids = PrimitiveArray::new(model_ids.freeze(), Validity::NonNullable).into_array();
    let coeff_a = PrimitiveArray::new(coeff_a.freeze(), Validity::NonNullable).into_array();
    let coeff_b = PrimitiveArray::new(coeff_b.freeze(), Validity::NonNullable).into_array();
    let coeff_c = PrimitiveArray::new(coeff_c.freeze(), Validity::NonNullable).into_array();

    // Compress the small slots with BtrBlocks. FoR + bit-pack handles `piece_starts`
    // (monotonic u32), Constant compresses `model_ids` when only one family fires, and the
    // coefficient slots get FoR + bit-pack on smooth pieces. Doing the cascade inside the
    // encoder means `NeaTSArray::nbytes()` already reflects the on-disk size without an
    // external pass.
    let btr = BtrBlocksCompressor::default();
    let piece_starts = btr.compress(&piece_starts, ctx)?;
    let model_ids = btr.compress(&model_ids, ctx)?;
    let coeff_a = btr.compress(&coeff_a, ctx)?;
    let coeff_b = btr.compress(&coeff_b, ctx)?;
    let coeff_c = btr.compress(&coeff_c, ctx)?;

    // Carry validity on residuals so downstream sees the same mask as input. Residuals
    // already get their own residual_encoding (PCO by default) so we don't cascade them
    // through BtrBlocks again — that path canonicalises and re-compresses, undoing PCO.
    let residuals_i64 = residuals.freeze();
    let residuals_narrow = narrow_residuals(residuals_i64.clone(), validity.clone());
    let residuals: ArrayRef = match options.residual_encoding {
        ResidualEncoding::BitPack => btr.compress(&residuals_narrow, ctx)?,
        ResidualEncoding::Pco => encode_residuals_pco(residuals_narrow, ctx)?,
        ResidualEncoding::PerPieceBitPack => {
            // Build per-piece chunks with their own bit-width. piece_starts is still a P+1 Vec
            // here (we sentinel-pushed `len` above). We pull start/end pairs from the buffer
            // we still have access to.
            let starts_u32: &[u32] = piece_starts_data.as_slice();
            encode_residuals_per_piece(&residuals_i64, starts_u32, validity, ctx)?
        }
    };

    NeaTS::try_new(
        logical_dtype,
        NeaTSData::new(scale, epsilon),
        piece_starts,
        model_ids,
        coeff_a,
        coeff_b,
        coeff_c,
        residuals,
    )
}

/// Hand the residuals slot to PCO so its signed-integer pipeline (delta + FSE + chunked
/// statistics) can compress it. The other slots (`piece_starts`, `model_ids`, coefficients) are
/// already tiny and BtrBlocks's FoR/bitpack/constant schemes handle them well, so they go
/// through the normal cascade.
///
/// PCO doesn't support `i8`, so we widen i8 residuals to i16 before handing over. The two-byte
/// stride is still tighter than FoR+bitpack on most signals.
/// Per-piece ZigZag + bit-pack residuals, wrapped in a `ChunkedArray<i64>`. Each chunk is
/// `ZigZag(BitPacked(piece's u64 residuals, piece's bit_width))`, so each piece gets exactly the
/// number of bits its own max-abs residual needs.
///
/// This is the residual encoding the NeaTS paper uses (modulo their succinct rank/select tables
/// for per-piece offsets, which Vortex's `ChunkedArray` provides implicitly via chunk boundaries).
fn encode_residuals_per_piece(
    residuals: &[i64],
    piece_starts: &[u32],
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    use vortex_array::arrays::ChunkedArray;

    if piece_starts.len() < 2 {
        // No pieces — return an empty placeholder.
        let empty = PrimitiveArray::new(Buffer::<i64>::empty(), validity).into_array();
        return Ok(empty);
    }

    // Pick the chunk-level signed dtype as the widest needed across all pieces.
    let global_max_abs: u64 = residuals
        .iter()
        .map(|r| r.unsigned_abs())
        .max()
        .unwrap_or(0);
    let signed_ptype = if global_max_abs <= i8::MAX as u64 {
        PType::I8
    } else if global_max_abs <= i16::MAX as u64 {
        PType::I16
    } else if global_max_abs <= i32::MAX as u64 {
        PType::I32
    } else {
        PType::I64
    };
    // ZigZag needs an unsigned ptype one step wider on the unsigned side. For i8, ZigZag values
    // fit in u8 (max 255 → 8 bits, fine). For i16 → u16, etc.
    let unsigned_ptype = match signed_ptype {
        PType::I8 => PType::U8,
        PType::I16 => PType::U16,
        PType::I32 => PType::U32,
        PType::I64 => PType::U64,
        _ => unreachable!(),
    };

    let mut chunks: Vec<ArrayRef> = Vec::with_capacity(piece_starts.len() - 1);
    for w in piece_starts.windows(2) {
        let s = w[0] as usize;
        let e = w[1] as usize;
        let piece = &residuals[s..e];
        let chunk = encode_piece_residuals(piece, unsigned_ptype, ctx)?;
        chunks.push(chunk);
    }

    let dtype = vortex_array::dtype::DType::Primitive(signed_ptype, validity.nullability());
    let chunked = ChunkedArray::try_new(chunks, dtype)?;
    Ok(chunked.into_array())
}

/// Encode a single piece's signed residuals as `ZigZag(BitPacked(unsigned, bit_width))`.
fn encode_piece_residuals(
    piece: &[i64],
    unsigned_ptype: PType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    use vortex_fastlanes::BitPackedData;
    use vortex_zigzag::ZigZag;

    // Step 1: ZigZag-encode each residual.
    let max_abs: u64 = piece.iter().map(|r| r.unsigned_abs()).max().unwrap_or(0);
    let bit_width = crate::bitpack::bits_for_max_abs(max_abs);

    // Step 2: build an unsigned PrimitiveArray of the chunk dtype.
    let unsigned_array = match unsigned_ptype {
        PType::U8 => {
            let buf: BufferMut<u8> = piece
                .iter()
                .map(|r| crate::bitpack::zigzag_encode(*r) as u8)
                .collect();
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array()
        }
        PType::U16 => {
            let buf: BufferMut<u16> = piece
                .iter()
                .map(|r| crate::bitpack::zigzag_encode(*r) as u16)
                .collect();
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array()
        }
        PType::U32 => {
            let buf: BufferMut<u32> = piece
                .iter()
                .map(|r| crate::bitpack::zigzag_encode(*r) as u32)
                .collect();
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array()
        }
        PType::U64 => {
            let buf: BufferMut<u64> = piece
                .iter()
                .map(|r| crate::bitpack::zigzag_encode(*r))
                .collect();
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array()
        }
        _ => vortex_bail!("unexpected unsigned ptype {unsigned_ptype}"),
    };

    // Step 3: bit-pack at the piece's bit_width. If bit_width is the full ptype width, skip the
    // wrap (a BitPacked at max width is identity).
    let bp_capacity = unsigned_ptype.bit_width() as u8;
    let inner = if bit_width >= bp_capacity || bit_width == 0 {
        unsigned_array
    } else {
        BitPackedData::encode(&unsigned_array, bit_width, ctx)?.into_array()
    };

    // Step 4: wrap in ZigZag so the chunk's logical dtype is the signed ptype.
    Ok(ZigZag::try_new(inner)?.into_array())
}

fn encode_residuals_pco(residuals: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let prim = residuals.execute::<PrimitiveArray>(ctx)?;
    let prim = if matches!(prim.ptype(), PType::I8) {
        let mut widened = BufferMut::<i16>::with_capacity(prim.len());
        for v in prim.as_slice::<i8>() {
            widened.push(i16::from(*v));
        }
        PrimitiveArray::new(widened.freeze(), prim.validity()?)
    } else {
        prim
    };
    // Level 8 is PCO's library default, values_per_page=0 picks pco's internal default page size.
    Ok(Pco::from_primitive(prim.as_view(), 8, 0, ctx)?.into_array())
}

/// Downcast the `i64` residual buffer to the narrowest signed integer ptype that fits every
/// residual. This shrinks the residuals slot by 2-8x in the common case (when the chosen scale
/// keeps residuals inside `i8`/`i16`/`i32`), with no precision loss — only the storage width
/// changes; the logical value is unchanged.
fn narrow_residuals(residuals: Buffer<i64>, validity: Validity) -> ArrayRef {
    let max_abs = residuals
        .as_slice()
        .iter()
        .map(|r| r.unsigned_abs())
        .max()
        .unwrap_or(0);

    if max_abs <= i8::MAX as u64 {
        let mut buf = BufferMut::<i8>::with_capacity(residuals.len());
        for &r in residuals.as_slice() {
            buf.push(r as i8);
        }
        PrimitiveArray::new(buf.freeze(), validity).into_array()
    } else if max_abs <= i16::MAX as u64 {
        let mut buf = BufferMut::<i16>::with_capacity(residuals.len());
        for &r in residuals.as_slice() {
            buf.push(r as i16);
        }
        PrimitiveArray::new(buf.freeze(), validity).into_array()
    } else if max_abs <= i32::MAX as u64 {
        let mut buf = BufferMut::<i32>::with_capacity(residuals.len());
        for &r in residuals.as_slice() {
            buf.push(r as i32);
        }
        PrimitiveArray::new(buf.freeze(), validity).into_array()
    } else {
        PrimitiveArray::new(residuals, validity).into_array()
    }
}

/// Grow a piece starting at `start` while the best-fitting model keeps the max residual within
/// the `MAX_RESIDUAL_BITS` budget. Returns the half-open piece end and the chosen fit.
fn grow_piece(
    values: &[f64],
    start: usize,
    scale: f64,
    options: &NeaTSOptions,
) -> (usize, FitResult) {
    let len = values.len();
    let max_end = (start + options.max_piece_len).min(len);
    let max_residual = ((1u64 << MAX_RESIDUAL_BITS) - 1) as f64;

    // Always start with the largest extendable piece and shrink if the fit overshoots the
    // residual budget. We use an exponential probe, then bisect — cheaper than reffit-per-step
    // and still tight enough for v1.
    let mut lo = (start + 1).min(max_end);
    let mut hi = max_end;
    let best_fit_lo = best_fit(values, start, lo, scale);
    if best_fit_lo.is_none() {
        // Degenerate single point: emit a constant of value itself.
        let v = values[start];
        return (
            start + 1,
            FitResult {
                kind: ModelKind::Constant,
                a: v,
                b: 0.0,
                c: 0.0,
                max_abs_residual: 0.0,
            },
        );
    }

    // Probe: try the whole range first.
    if let Some(fit) = best_fit(values, start, hi, scale)
        && fit.max_abs_residual <= max_residual
    {
        return (hi, fit);
    }

    // Bisect between `lo` (fits) and `hi` (doesn't).
    let mut best_fit_lo =
        best_fit_lo.vortex_expect("best_fit_lo is some after the None-check above");
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match best_fit(values, start, mid, scale) {
            Some(f) if f.max_abs_residual <= max_residual => {
                lo = mid;
                best_fit_lo = f;
            }
            _ => {
                hi = mid;
            }
        }
    }
    // Re-fit at exactly `lo` to pick up the best family for the final span (the family that
    // wins on the probe range may differ from the best on the smaller span).
    let final_fit = best_fit(values, start, lo, scale).unwrap_or(best_fit_lo);
    (lo, final_fit)
}

fn best_fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
    if end <= start {
        return None;
    }
    if end == start + 1 {
        let v = values[start];
        if v.is_nan() {
            return None;
        }
        return Some(FitResult {
            kind: ModelKind::Constant,
            a: v,
            b: 0.0,
            c: 0.0,
            max_abs_residual: 0.0,
        });
    }
    fit_best(values, start, end, scale)
}

/// Pick a residual quantization scale.
///
/// For lossy mode (`epsilon = Some(eps)`) we set `scale = 2 * eps` so the worst-case rounding
/// error per value is `eps`.
///
/// For lossless mode we set `scale = 2^-52 * max(|v|)`, which is below the f64 unit-in-the-last-
/// place across the whole range. For data that is exactly integer-valued, callers should set
/// `epsilon = Some(0.5)` for fast lossless residuals; the auto path is conservative.
fn pick_scale(values: &[f64], epsilon: Option<f64>) -> f64 {
    if let Some(eps) = epsilon
        && eps > 0.0
    {
        return 2.0 * eps;
    }
    let max_abs = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_abs == 0.0 {
        return 1.0;
    }
    // 2^-52 * max_abs: every distinct f64 in the range is at least this far apart.
    max_abs * f64::EPSILON
}
