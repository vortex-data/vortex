// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Affine;
use crate::AffineArray;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::slope_term;

/// Which parts of the model [`Affine::encode`] may fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineOptions {
    /// Divide each chunk's residuals by their greatest common divisor.
    pub scale: bool,
    /// Fit a linear trend to each chunk.
    pub slope: bool,
    /// Fit only the least-squares slope, always, as LeCo-fix does, instead of choosing the
    /// narrowest of several slopes and a zero slope. Requires `slope`.
    pub least_squares: bool,
}

impl AffineOptions {
    /// Per-chunk references only, equivalent to chunked FoR.
    pub const FOR: Self = Self {
        scale: false,
        slope: false,
        least_squares: false,
    };
    /// Per-chunk references and common divisors.
    pub const SCALE: Self = Self {
        scale: true,
        slope: false,
        least_squares: false,
    };
    /// Per-chunk references and linear trends.
    pub const SLOPE: Self = Self {
        scale: false,
        slope: true,
        least_squares: false,
    };
    /// Per-chunk references, linear trends and common divisors.
    pub const ALL: Self = Self {
        scale: true,
        slope: true,
        least_squares: false,
    };
    /// LeCo-fix's model: a least-squares line per chunk, with an exact fixed-point slope.
    pub const LECO: Self = Self {
        scale: false,
        slope: true,
        least_squares: true,
    };
}

/// The default number of fractional bits in fitted slopes.
const DEFAULT_SLOPE_SHIFT: u8 = 16;

impl Affine {
    /// Encode `array`, fitting the parts of the model that `options` enables to each chunk.
    ///
    /// For every chunk the encoder tries a zero slope and, when enabled, slopes from the median
    /// delta, the endpoints and a least-squares fit, each rounded to an integer and to fixed point.
    /// It picks the slope whose residuals pack into the fewest bits, preferring a zero slope and an
    /// integer slope on ties. The reference is the smallest residual, and the scale is the greatest
    /// common divisor of the residuals above it.
    pub fn encode(
        array: &PrimitiveArray,
        options: AffineOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<AffineArray> {
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        match_each_integer_ptype!(array.ptype(), |T| {
            encode_typed::<T>(array.as_slice::<T>(), validity, &mask, options)
        })
    }
}

/// The fitted parameters of one chunk.
#[derive(Clone, Copy)]
struct ChunkModel<T> {
    reference: T,
    scale: u64,
    slope: i64,
}

fn encode_typed<T>(
    values: &[T],
    validity: Validity,
    mask: &Mask,
    options: AffineOptions,
) -> VortexResult<AffineArray>
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64> + Into<PValue>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let width_mask = width_mask::<T>();
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    let slope_shift = if options.slope {
        choose_slope_shift(values, mask)
    } else {
        0
    };

    let mut encoded = BufferMut::<u64>::with_capacity(values.len());
    let mut references = BufferMut::<T>::with_capacity(num_chunks);
    let mut scales = BufferMut::<T>::with_capacity(num_chunks);
    let mut slopes = BufferMut::<i64>::with_capacity(num_chunks);
    let mut previous_reference = T::zero();
    let mut valid = Vec::with_capacity(FL_CHUNK_SIZE);

    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = chunk_idx * FL_CHUNK_SIZE;
        valid.clear();
        valid.extend((0..chunk.len()).filter(|&j| mask.value(start + j)));

        let model = if valid.is_empty() {
            // Reuse the previous reference so the references compress into runs.
            ChunkModel {
                reference: previous_reference,
                scale: 1,
                slope: 0,
            }
        } else {
            fit_chunk(chunk, &valid, options, slope_shift)
        };
        previous_reference = model.reference;

        let mut valid_iter = valid.iter().peekable();
        for (j, &v) in chunk.iter().enumerate() {
            if valid_iter.next_if(|&&vj| vj == j).is_some() {
                let term: T = slope_term(model.slope, j, slope_shift).as_();
                let residual: u64 = v
                    .wrapping_sub(&term)
                    .wrapping_sub(&model.reference)
                    .as_()
                    & width_mask;
                encoded.push(residual / model.scale);
            } else {
                encoded.push(0);
            }
        }
        references.push(model.reference);
        scales.push(model.scale.as_());
        slopes.push(model.slope);
    }

    Affine::try_new(
        narrow_residuals::<T>(&encoded, validity),
        constant_or_primitive(references.freeze()),
        constant_or_primitive(scales.freeze()),
        constant_or_primitive(slopes.freeze()),
        0,
        slope_shift,
    )
}

/// Fit the model with the narrowest residuals to the `valid` positions of `chunk`.
fn fit_chunk<T>(
    chunk: &[T],
    valid: &[usize],
    options: AffineOptions,
    slope_shift: u8,
) -> ChunkModel<T>
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let mut best: Option<(u32, ChunkModel<T>)> = None;
    for slope in slope_candidates(chunk, valid, options, slope_shift) {
        let (bits, model) = evaluate(chunk, valid, slope, options.scale, slope_shift);
        if best.is_none_or(|(best_bits, _)| bits < best_bits) {
            best = Some((bits, model));
        }
    }
    best.map(|(_, model)| model)
        .vortex_expect("the zero slope is always a candidate")
}

/// The bit width of the residuals of `slope`, and the model that produces them.
fn evaluate<T>(
    chunk: &[T],
    valid: &[usize],
    slope: i64,
    use_scale: bool,
    slope_shift: u8,
) -> (u32, ChunkModel<T>)
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let width_mask = width_mask::<T>();
    let residual = |j: usize| -> T {
        let term: T = slope_term(slope, j, slope_shift).as_();
        chunk[j].wrapping_sub(&term)
    };
    let reference = valid
        .iter()
        .map(|&j| residual(j))
        .min()
        .unwrap_or_else(T::zero);

    let mut max = 0u64;
    let mut gcd = 0u64;
    for &j in valid {
        let above: u64 = residual(j).wrapping_sub(&reference).as_() & width_mask;
        max = max.max(above);
        if use_scale && gcd != 1 {
            gcd = gcd_u64(gcd, above);
        }
    }
    let scale = if use_scale { gcd.max(1) } else { 1 };
    let bits = u64::BITS - (max / scale).leading_zeros();
    (
        bits,
        ChunkModel {
            reference,
            scale,
            slope,
        },
    )
}

/// Candidate fixed-point slopes for a chunk, starting with zero, then integer slopes, then
/// fractional ones, so that ties keep the simpler model.
fn slope_candidates<T: PrimInt>(
    chunk: &[T],
    valid: &[usize],
    options: AffineOptions,
    slope_shift: u8,
) -> Vec<i64> {
    let mut candidates = vec![0i64];
    if !options.slope || valid.len() < 2 {
        return candidates;
    }

    let points: Vec<(f64, f64)> = valid
        .iter()
        .map(|&j| (j as f64, chunk[j].to_f64().unwrap_or(0.0)))
        .collect();
    let mut deltas: Vec<f64> = points
        .windows(2)
        .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0))
        .collect();
    deltas.sort_by(f64::total_cmp);
    let median = deltas[deltas.len() / 2];
    let (first, last) = (points[0], points[points.len() - 1]);
    let endpoint = (last.1 - first.1) / (last.0 - first.0);
    let least_squares = least_squares_slope(&points);

    let unit = f64::from(1u32 << slope_shift);
    if options.least_squares {
        let slope = (least_squares * unit).round();
        return vec![if slope.is_finite() && slope.abs() * (FL_CHUNK_SIZE as f64) < i64::MAX as f64 {
            slope as i64
        } else {
            0
        }];
    }
    let fits = [median, endpoint, least_squares];
    let integer = fits.iter().map(|s| s.round() * unit);
    let fractional = fits.iter().map(|s| (s * unit).round());
    for slope in integer.chain(fractional) {
        // Slopes whose term could overflow `i64` would only ever give wide residuals.
        if slope.is_finite() && slope.abs() * (FL_CHUNK_SIZE as f64) < i64::MAX as f64 {
            let slope = slope as i64;
            if !candidates.contains(&slope) {
                candidates.push(slope);
            }
        }
    }
    candidates
}

fn least_squares_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &(x, y) in points {
        sxy += (x - mean_x) * (y - mean_y);
        sxx += (x - mean_x) * (x - mean_x);
    }
    if sxx == 0.0 { 0.0 } else { sxy / sxx }
}

/// Pick the number of fractional slope bits, so that the steepest chunk's slope term still fits
/// in `i64` across a whole chunk.
fn choose_slope_shift<T: PrimInt>(values: &[T], mask: &Mask) -> u8 {
    let mut steepest = 0f64;
    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = chunk_idx * FL_CHUNK_SIZE;
        let first = (0..chunk.len()).find(|&j| mask.value(start + j));
        let last = (0..chunk.len()).rev().find(|&j| mask.value(start + j));
        if let (Some(first), Some(last)) = (first, last)
            && last > first
        {
            let rise = chunk[last].to_f64().unwrap_or(0.0) - chunk[first].to_f64().unwrap_or(0.0);
            steepest = steepest.max((rise / (last - first) as f64).abs());
        }
    }
    // Leave one bit of headroom below `i64::MAX` for rounding.
    let headroom = 62.0 - (FL_CHUNK_SIZE as f64).log2() - (steepest + 1.0).log2();
    headroom.clamp(0.0, f64::from(DEFAULT_SLOPE_SHIFT)) as u8
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The mask that keeps the low `T::BITS` bits of a `u64`.
fn width_mask<T>() -> u64 {
    match size_of::<T>() {
        8 => u64::MAX,
        bytes => (1u64 << (bytes * 8)) - 1,
    }
}

/// The residuals as the narrowest unsigned integer array that holds the largest of them, and is no
/// wider than the array's type. Narrow residuals unpack into narrow lanes and multiply as
/// `32 x 32 -> 64` bits, which decodes much faster than full-width residuals.
fn narrow_residuals<T: NativePType>(encoded: &[u64], validity: Validity) -> ArrayRef
where
    u64: AsPrimitive<T>,
{
    let max_bytes = size_of::<T>();
    let max = encoded.iter().copied().max().unwrap_or(0);
    let bytes = match u64::BITS - max.leading_zeros() {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
    .min(max_bytes);
    match if bytes == max_bytes { 0 } else { bytes } {
        1 => PrimitiveArray::new(encoded.iter().map(|&e| e as u8).collect::<Buffer<u8>>(), validity),
        2 => PrimitiveArray::new(encoded.iter().map(|&e| e as u16).collect::<Buffer<u16>>(), validity),
        4 => PrimitiveArray::new(encoded.iter().map(|&e| e as u32).collect::<Buffer<u32>>(), validity),
        // As wide as the array's type: keep that type, so decoding reuses FoR's fused kernels.
        _ => PrimitiveArray::new(encoded.iter().map(|&e| e.as_()).collect::<Buffer<T>>(), validity),
    }
    .into_array()
}

/// A constant array when every value matches, otherwise a primitive array.
fn constant_or_primitive<T: NativePType + Into<PValue>>(values: Buffer<T>) -> ArrayRef {
    match values.first() {
        Some(&first) if values.iter().all(|&v| v == first) => {
            ConstantArray::new(Scalar::primitive(first, Nullability::NonNullable), values.len()).into_array()
        }
        _ => PrimitiveArray::new(values, Validity::NonNullable).into_array(),
    }
}
