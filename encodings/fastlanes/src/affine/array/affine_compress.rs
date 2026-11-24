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
use crate::affine::array::slope_term_split;
use crate::affine::array::split_slope;

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

/// Direct residual bit-width + outlier patch cost estimates (in bits/value) for each candidate
/// [`AffineOptions`] mode over a sample of chunks.
#[derive(Clone, Copy, Debug)]
pub struct AffineModeEstimates {
    /// Estimated bits/value for [`AffineOptions::FOR`].
    pub for_bits: f64,
    /// Estimated bits/value for [`AffineOptions::SCALE`].
    pub scale_bits: f64,
    /// Estimated bits/value for [`AffineOptions::SLOPE`].
    pub slope_bits: f64,
    /// Estimated bits/value for [`AffineOptions::ALL`].
    pub all_bits: f64,
}

/// The default number of fractional bits in fitted slopes.
const DEFAULT_SLOPE_SHIFT: u8 = 16;

/// Maximum number of slope candidates tried per chunk: zero + 3 integer + 3 fractional fits.
const MAX_SLOPE_CANDIDATES: usize = 7;

/// Valid positions within a chunk, avoiding per-chunk index allocations when all rows are valid.
#[derive(Clone, Copy)]
enum ValidPositions<'a> {
    All(usize),
    Indices(&'a [usize]),
}

impl<'a> ValidPositions<'a> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::All(len) => len,
            Self::Indices(indices) => indices.len(),
        }
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Stack-allocated list of candidate fixed-point slopes for one chunk.
struct SlopeCandidates {
    slopes: [i64; MAX_SLOPE_CANDIDATES],
    len: usize,
}

impl SlopeCandidates {
    fn new_zero() -> Self {
        Self {
            slopes: [0; MAX_SLOPE_CANDIDATES],
            len: 1,
        }
    }

    fn single(slope: i64) -> Self {
        let mut slopes = [0; MAX_SLOPE_CANDIDATES];
        slopes[0] = slope;
        Self { slopes, len: 1 }
    }

    fn push_unique(&mut self, slope: i64) {
        if !self.slopes[..self.len].contains(&slope) {
            self.slopes[self.len] = slope;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[i64] {
        &self.slopes[..self.len]
    }
}

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

    /// Estimate compressed bits per value for `FOR`, `SCALE`, `SLOPE`, and `ALL` directly in
    /// memory over up to `sample_chunks` evenly spaced chunks without running child cascades.
    ///
    /// Each mode's estimate models bit-packing + outlier patch cost from the sampled residual
    /// bit-width histogram plus the per-chunk parameter overhead (`references`, `scales`,
    /// `slopes`).
    pub fn estimate_modes(
        array: &PrimitiveArray,
        sample_chunks: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<AffineModeEstimates> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        match_each_integer_ptype!(array.ptype(), |T| {
            Ok(estimate_modes_typed::<T>(
                array.as_slice::<T>(),
                &mask,
                sample_chunks,
            ))
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
    let mut valid_buf = Vec::with_capacity(FL_CHUNK_SIZE);
    let mut residual_scratch = [T::zero(); FL_CHUNK_SIZE];
    let mut delta_scratch = [0f64; FL_CHUNK_SIZE];
    let all_valid = matches!(mask, Mask::AllTrue(_));

    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = chunk_idx * FL_CHUNK_SIZE;
        let valid = if all_valid {
            ValidPositions::All(chunk.len())
        } else {
            valid_buf.clear();
            valid_buf.extend((0..chunk.len()).filter(|&j| mask.value(start + j)));
            ValidPositions::Indices(&valid_buf)
        };

        let model = if valid.is_empty() {
            // Reuse the previous reference so the references compress into runs.
            ChunkModel {
                reference: previous_reference,
                scale: 1,
                slope: 0,
            }
        } else {
            fit_chunk(
                chunk,
                valid,
                options,
                slope_shift,
                &mut residual_scratch,
                &mut delta_scratch,
            )
        };
        previous_reference = model.reference;

        write_chunk_residuals(chunk, valid, model, slope_shift, width_mask, &mut encoded);
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

/// Write the encoded residuals of one chunk, specializing common `(scale, slope)` cases to avoid
/// unnecessary 64-bit divisions and slope evaluations.
fn write_chunk_residuals<T>(
    chunk: &[T],
    valid: ValidPositions<'_>,
    model: ChunkModel<T>,
    slope_shift: u8,
    width_mask: u64,
    encoded: &mut BufferMut<u64>,
) where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let scale = model.scale;
    let is_unit_scale = scale == 1;
    let scale_shift = scale.is_power_of_two().then(|| scale.trailing_zeros());
    let divide = |residual: u64| -> u64 {
        if is_unit_scale {
            residual
        } else if let Some(shift) = scale_shift {
            residual >> shift
        } else {
            residual / scale
        }
    };

    match valid {
        ValidPositions::All(_) => {
            if model.slope == 0 {
                if is_unit_scale {
                    encoded.extend_trusted(
                        chunk
                            .iter()
                            .map(|&v| v.wrapping_sub(&model.reference).as_() & width_mask),
                    );
                } else if let Some(shift) = scale_shift {
                    encoded.extend_trusted(
                        chunk.iter().map(|&v| {
                            (v.wrapping_sub(&model.reference).as_() & width_mask) >> shift
                        }),
                    );
                } else {
                    encoded.extend_trusted(
                        chunk.iter().map(|&v| {
                            (v.wrapping_sub(&model.reference).as_() & width_mask) / scale
                        }),
                    );
                }
            } else {
                let (whole, frac) = split_slope(model.slope, slope_shift);
                let mut whole_j = 0i64;
                let mut frac_j = 0u32;
                for &v in chunk {
                    let term: T = whole_j.wrapping_add(i64::from(frac_j >> slope_shift)).as_();
                    let residual: u64 =
                        v.wrapping_sub(&term).wrapping_sub(&model.reference).as_() & width_mask;
                    encoded.push(divide(residual));
                    whole_j = whole_j.wrapping_add(whole);
                    frac_j = frac_j.wrapping_add(frac);
                }
            }
        }
        ValidPositions::Indices(valid_indices) => {
            let mut valid_iter = valid_indices.iter().peekable();
            if model.slope == 0 {
                for (j, &v) in chunk.iter().enumerate() {
                    if valid_iter.next_if(|&&vj| vj == j).is_some() {
                        let residual: u64 = v.wrapping_sub(&model.reference).as_() & width_mask;
                        encoded.push(divide(residual));
                    } else {
                        encoded.push(0);
                    }
                }
            } else {
                let (whole, frac) = split_slope(model.slope, slope_shift);
                for (j, &v) in chunk.iter().enumerate() {
                    if valid_iter.next_if(|&&vj| vj == j).is_some() {
                        let term: T = slope_term_split(whole, frac, j, slope_shift).as_();
                        let residual: u64 =
                            v.wrapping_sub(&term).wrapping_sub(&model.reference).as_() & width_mask;
                        encoded.push(divide(residual));
                    } else {
                        encoded.push(0);
                    }
                }
            }
        }
    }
}

/// Running histogram and parameter bounds for one candidate mode during [`estimate_modes_typed`].
struct ModeAccumulator<T> {
    hist: [usize; 65],
    max_residual: u64,
    min_ref: T,
    max_ref: T,
    refs_constant: bool,
    min_scale: u64,
    max_scale: u64,
    scales_constant: bool,
    min_slope: i64,
    max_slope: i64,
    slopes_constant: bool,
    chunks: usize,
}

impl<T: NativePType + PrimInt> ModeAccumulator<T> {
    fn new() -> Self {
        Self {
            hist: [0; 65],
            max_residual: 0,
            min_ref: T::zero(),
            max_ref: T::zero(),
            refs_constant: true,
            min_scale: 1,
            max_scale: 1,
            scales_constant: true,
            min_slope: 0,
            max_slope: 0,
            slopes_constant: true,
            chunks: 0,
        }
    }

    fn record_model(&mut self, model: ChunkModel<T>) {
        if self.chunks == 0 {
            self.min_ref = model.reference;
            self.max_ref = model.reference;
            self.min_scale = model.scale;
            self.max_scale = model.scale;
            self.min_slope = model.slope;
            self.max_slope = model.slope;
        } else {
            if model.reference != self.min_ref {
                self.refs_constant = false;
            }
            if model.reference < self.min_ref {
                self.min_ref = model.reference;
            }
            if model.reference > self.max_ref {
                self.max_ref = model.reference;
            }
            if model.scale != self.min_scale {
                self.scales_constant = false;
            }
            self.min_scale = self.min_scale.min(model.scale);
            self.max_scale = self.max_scale.max(model.scale);
            if model.slope != self.min_slope {
                self.slopes_constant = false;
            }
            self.min_slope = self.min_slope.min(model.slope);
            self.max_slope = self.max_slope.max(model.slope);
        }
        self.chunks += 1;
    }

    fn add_chunk_hist(&mut self, chunk_hist: &[usize; 65], chunk_max: u64) {
        for (dst, &count) in self.hist.iter_mut().zip(chunk_hist) {
            *dst += count;
        }
        self.max_residual = self.max_residual.max(chunk_max);
    }

    fn estimated_bits_per_value(&self, total_len: usize) -> f64
    where
        T: WrappingSub + AsPrimitive<u64>,
    {
        if total_len == 0 {
            return 0.0;
        }
        let max_bytes = size_of::<T>();
        let res_bytes = match u64::BITS - self.max_residual.leading_zeros() {
            0..=8 => 1,
            9..=16 => 2,
            17..=32 => 4,
            _ => 8,
        }
        .min(max_bytes);
        let bytes_per_exception = res_bytes + 4;
        let max_bw = res_bytes * 8;

        let mut num_packed = 0usize;
        let mut best_bytes = total_len * bytes_per_exception;
        for (bw, &freq) in self.hist[..=max_bw].iter().enumerate() {
            let packed_bytes = (bw * total_len).div_ceil(8);
            num_packed += freq;
            let exception_bytes = (total_len - num_packed) * bytes_per_exception;
            best_bytes = best_bytes.min(packed_bytes + exception_bytes);
        }

        let mut total_bits = (best_bytes * 8) as f64;
        let chunks = self.chunks as f64;
        if !self.refs_constant {
            let span = self.max_ref.wrapping_sub(&self.min_ref).as_() & width_mask::<T>();
            total_bits += chunks * f64::from(u64::BITS - span.leading_zeros());
        }
        if !self.scales_constant {
            let span = self.max_scale.wrapping_sub(self.min_scale);
            total_bits += chunks * f64::from(u64::BITS - span.leading_zeros());
        }
        if !self.slopes_constant {
            let span = self.max_slope.wrapping_sub(self.min_slope) as u64;
            total_bits += chunks * f64::from(u64::BITS - span.leading_zeros());
        }
        total_bits / total_len as f64
    }
}

/// Evaluate all four modes (`FOR`, `SCALE`, `SLOPE`, `ALL`) in one pass over up to
/// `sample_chunks` evenly spaced chunks.
fn estimate_modes_typed<T>(values: &[T], mask: &Mask, sample_chunks: usize) -> AffineModeEstimates
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    if num_chunks == 0 || sample_chunks == 0 {
        return AffineModeEstimates {
            for_bits: 0.0,
            scale_bits: 0.0,
            slope_bits: 0.0,
            all_bits: 0.0,
        };
    }

    let k = num_chunks.min(sample_chunks);
    let sampled_chunk_idx = |i: usize| -> usize {
        if num_chunks <= sample_chunks {
            i
        } else {
            i * num_chunks / sample_chunks
        }
    };

    let all_valid = matches!(mask, Mask::AllTrue(_));
    let slope_shift = choose_slope_shift_sampled(values, mask, num_chunks, sample_chunks);
    let width_mask = width_mask::<T>();

    let mut acc_for = ModeAccumulator::<T>::new();
    let mut acc_scale = ModeAccumulator::<T>::new();
    let mut acc_slope = ModeAccumulator::<T>::new();
    let mut acc_all = ModeAccumulator::<T>::new();

    let mut valid_buf = Vec::with_capacity(FL_CHUNK_SIZE);
    let mut residuals = [T::zero(); FL_CHUNK_SIZE];
    let mut deltas = [0f64; FL_CHUNK_SIZE];
    let mut total_len = 0usize;
    let mut prev_refs = [T::zero(); 4];

    for i in 0..k {
        let chunk_idx = sampled_chunk_idx(i);
        let start = chunk_idx * FL_CHUNK_SIZE;
        let end = (start + FL_CHUNK_SIZE).min(values.len());
        let chunk = &values[start..end];
        total_len += chunk.len();

        let valid = if all_valid {
            ValidPositions::All(chunk.len())
        } else {
            valid_buf.clear();
            valid_buf.extend((0..chunk.len()).filter(|&j| mask.value(start + j)));
            ValidPositions::Indices(&valid_buf)
        };

        if valid.is_empty() {
            let mut empty_hist = [0usize; 65];
            empty_hist[0] = chunk.len();
            for (acc, prev_ref) in [
                (&mut acc_for, prev_refs[0]),
                (&mut acc_scale, prev_refs[1]),
                (&mut acc_slope, prev_refs[2]),
                (&mut acc_all, prev_refs[3]),
            ] {
                acc.record_model(ChunkModel {
                    reference: prev_ref,
                    scale: 1,
                    slope: 0,
                });
                acc.add_chunk_hist(&empty_hist, 0);
            }
            continue;
        }

        let invalid_count = chunk.len() - valid.len();

        // 1. Evaluate zero-slope models (`FOR` and `SCALE`) together in one pass.
        let (for_bits_c, for_model, scale_bits_c, scale_model) =
            evaluate_pair(chunk, valid, 0, slope_shift, &mut residuals);
        let (for_hist, for_max) = residuals_histogram(
            &residuals[..valid.len()],
            for_model.reference,
            1,
            width_mask,
            invalid_count,
        );
        let (scale_hist, scale_max) = if scale_model.scale == 1 {
            (for_hist, for_max)
        } else {
            residuals_histogram(
                &residuals[..valid.len()],
                scale_model.reference,
                scale_model.scale,
                width_mask,
                invalid_count,
            )
        };

        acc_for.record_model(for_model);
        acc_for.add_chunk_hist(&for_hist, for_max);
        prev_refs[0] = for_model.reference;

        acc_scale.record_model(scale_model);
        acc_scale.add_chunk_hist(&scale_hist, scale_max);
        prev_refs[1] = scale_model.reference;

        // 2. Evaluate non-zero slope candidates once for both `SLOPE` and `ALL`.
        let mut best_slope = (for_bits_c, for_model);
        let mut best_all = (scale_bits_c, scale_model);
        let candidates =
            slope_candidates(chunk, valid, AffineOptions::SLOPE, slope_shift, &mut deltas);
        for &slope in &candidates.as_slice()[1..] {
            let (s_bits, s_model, a_bits, a_model) =
                evaluate_pair(chunk, valid, slope, slope_shift, &mut residuals);
            if s_bits < best_slope.0 {
                best_slope = (s_bits, s_model);
            }
            if a_bits < best_all.0 {
                best_all = (a_bits, a_model);
            }
        }

        let (slope_hist, slope_max) = if best_slope.1.slope == 0 {
            (for_hist, for_max)
        } else {
            fill_slope_residuals(
                chunk,
                valid,
                best_slope.1.slope,
                slope_shift,
                &mut residuals,
            );
            residuals_histogram(
                &residuals[..valid.len()],
                best_slope.1.reference,
                1,
                width_mask,
                invalid_count,
            )
        };
        acc_slope.record_model(best_slope.1);
        acc_slope.add_chunk_hist(&slope_hist, slope_max);
        prev_refs[2] = best_slope.1.reference;

        let (all_hist, all_max) = if best_all.1.slope == 0 {
            (scale_hist, scale_max)
        } else if best_all.1.slope == best_slope.1.slope && best_all.1.scale == 1 {
            (slope_hist, slope_max)
        } else {
            if best_all.1.slope != best_slope.1.slope {
                fill_slope_residuals(chunk, valid, best_all.1.slope, slope_shift, &mut residuals);
            }
            residuals_histogram(
                &residuals[..valid.len()],
                best_all.1.reference,
                best_all.1.scale,
                width_mask,
                invalid_count,
            )
        };
        acc_all.record_model(best_all.1);
        acc_all.add_chunk_hist(&all_hist, all_max);
        prev_refs[3] = best_all.1.reference;
    }

    AffineModeEstimates {
        for_bits: acc_for.estimated_bits_per_value(total_len),
        scale_bits: acc_scale.estimated_bits_per_value(total_len),
        slope_bits: acc_slope.estimated_bits_per_value(total_len),
        all_bits: acc_all.estimated_bits_per_value(total_len),
    }
}

/// Build the 65-bin bit-width histogram and max residual for one chunk's slope-removed values.
fn residuals_histogram<T>(
    residuals: &[T],
    reference: T,
    scale: u64,
    width_mask: u64,
    invalid_count: usize,
) -> ([usize; 65], u64)
where
    T: NativePType + WrappingSub + AsPrimitive<u64>,
{
    let mut hist = [0usize; 65];
    hist[0] = invalid_count;
    let mut max_val = 0u64;
    if scale == 1 {
        for &r in residuals {
            let above = r.wrapping_sub(&reference).as_() & width_mask;
            max_val = max_val.max(above);
            hist[(u64::BITS - above.leading_zeros()) as usize] += 1;
        }
    } else if scale.is_power_of_two() {
        let shift = scale.trailing_zeros();
        for &r in residuals {
            let above = (r.wrapping_sub(&reference).as_() & width_mask) >> shift;
            max_val = max_val.max(above);
            hist[(u64::BITS - above.leading_zeros()) as usize] += 1;
        }
    } else {
        for &r in residuals {
            let above = (r.wrapping_sub(&reference).as_() & width_mask) / scale;
            max_val = max_val.max(above);
            hist[(u64::BITS - above.leading_zeros()) as usize] += 1;
        }
    }
    (hist, max_val)
}

/// Fill `residuals[..valid.len()]` with `chunk[j] - slope_term(slope, j, shift)` and return its
/// minimum (`reference`).
fn fill_slope_residuals<T>(
    chunk: &[T],
    valid: ValidPositions<'_>,
    slope: i64,
    slope_shift: u8,
    residuals: &mut [T; FL_CHUNK_SIZE],
) -> T
where
    T: NativePType + PrimInt + WrappingSub,
    i64: AsPrimitive<T>,
{
    match valid {
        ValidPositions::All(len) => {
            if slope == 0 {
                let dst = &mut residuals[..len];
                dst.copy_from_slice(&chunk[..len]);
                dst.iter().copied().min().unwrap_or_else(T::zero)
            } else {
                let (whole, frac) = split_slope(slope, slope_shift);
                let mut whole_j = 0i64;
                let mut frac_j = 0u32;
                let mut min_val = T::max_value();
                for (dst, &v) in residuals[..len].iter_mut().zip(&chunk[..len]) {
                    let term: T = whole_j.wrapping_add(i64::from(frac_j >> slope_shift)).as_();
                    let r = v.wrapping_sub(&term);
                    *dst = r;
                    if r < min_val {
                        min_val = r;
                    }
                    whole_j = whole_j.wrapping_add(whole);
                    frac_j = frac_j.wrapping_add(frac);
                }
                min_val
            }
        }
        ValidPositions::Indices(indices) => {
            if slope == 0 {
                let mut min_val = T::max_value();
                for (dst, &j) in residuals[..indices.len()].iter_mut().zip(indices) {
                    let r = chunk[j];
                    *dst = r;
                    if r < min_val {
                        min_val = r;
                    }
                }
                min_val
            } else {
                let (whole, frac) = split_slope(slope, slope_shift);
                let mut min_val = T::max_value();
                for (dst, &j) in residuals[..indices.len()].iter_mut().zip(indices) {
                    let term: T = slope_term_split(whole, frac, j, slope_shift).as_();
                    let r = chunk[j].wrapping_sub(&term);
                    *dst = r;
                    if r < min_val {
                        min_val = r;
                    }
                }
                min_val
            }
        }
    }
}

/// Evaluate both unscaled (`scale = 1`) and scaled (`scale = gcd`) models for `slope` in one pass.
fn evaluate_pair<T>(
    chunk: &[T],
    valid: ValidPositions<'_>,
    slope: i64,
    slope_shift: u8,
    residuals: &mut [T; FL_CHUNK_SIZE],
) -> (u32, ChunkModel<T>, u32, ChunkModel<T>)
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let width_mask = width_mask::<T>();
    let reference = fill_slope_residuals(chunk, valid, slope, slope_shift, residuals);

    let mut max = 0u64;
    let mut gcd = 0u64;
    for &r in &residuals[..valid.len()] {
        let above: u64 = r.wrapping_sub(&reference).as_() & width_mask;
        max = max.max(above);
        if gcd != 1 {
            gcd = gcd_u64(gcd, above);
        }
    }
    let scale = gcd.max(1);
    let unscaled_bits = u64::BITS - max.leading_zeros();
    let scaled_bits = u64::BITS - (max / scale).leading_zeros();
    (
        unscaled_bits,
        ChunkModel {
            reference,
            scale: 1,
            slope,
        },
        scaled_bits,
        ChunkModel {
            reference,
            scale,
            slope,
        },
    )
}

/// Fit the model with the narrowest residuals to the `valid` positions of `chunk`.
fn fit_chunk<T>(
    chunk: &[T],
    valid: ValidPositions<'_>,
    options: AffineOptions,
    slope_shift: u8,
    residuals: &mut [T; FL_CHUNK_SIZE],
    deltas: &mut [f64; FL_CHUNK_SIZE],
) -> ChunkModel<T>
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let mut best: Option<(u32, ChunkModel<T>)> = None;
    let candidates = slope_candidates(chunk, valid, options, slope_shift, deltas);
    for &slope in candidates.as_slice() {
        let (bits, model) = evaluate(chunk, valid, slope, options.scale, slope_shift, residuals);
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
    valid: ValidPositions<'_>,
    slope: i64,
    use_scale: bool,
    slope_shift: u8,
    residuals: &mut [T; FL_CHUNK_SIZE],
) -> (u32, ChunkModel<T>)
where
    T: NativePType + PrimInt + WrappingSub + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
{
    let width_mask = width_mask::<T>();
    let reference = fill_slope_residuals(chunk, valid, slope, slope_shift, residuals);

    let mut max = 0u64;
    let mut gcd = 0u64;
    if use_scale {
        for &r in &residuals[..valid.len()] {
            let above: u64 = r.wrapping_sub(&reference).as_() & width_mask;
            max = max.max(above);
            if gcd != 1 {
                gcd = gcd_u64(gcd, above);
            }
        }
    } else {
        for &r in &residuals[..valid.len()] {
            let above: u64 = r.wrapping_sub(&reference).as_() & width_mask;
            max = max.max(above);
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
    valid: ValidPositions<'_>,
    options: AffineOptions,
    slope_shift: u8,
    deltas: &mut [f64; FL_CHUNK_SIZE],
) -> SlopeCandidates {
    let mut candidates = SlopeCandidates::new_zero();
    if !options.slope || valid.len() < 2 {
        return candidates;
    }

    let (median, endpoint, least_squares) = match valid {
        ValidPositions::All(len) => {
            let mut prev = chunk[0].to_f64().unwrap_or(0.0);
            let first = prev;
            let mut sum_y = first;
            for (k, &v) in chunk[1..len].iter().enumerate() {
                let y = v.to_f64().unwrap_or(0.0);
                deltas[k] = y - prev;
                sum_y += y;
                prev = y;
            }
            let delta_count = len - 1;
            let mid = delta_count / 2;
            deltas[..delta_count].select_nth_unstable_by(mid, f64::total_cmp);
            let median = deltas[mid];
            let endpoint = (prev - first) / (delta_count as f64);

            let n = len as f64;
            let mean_x = (delta_count as f64) * 0.5;
            let mean_y = sum_y / n;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (j, &v) in chunk[..len].iter().enumerate() {
                let dx = j as f64 - mean_x;
                let dy = v.to_f64().unwrap_or(0.0) - mean_y;
                sxy += dx * dy;
                sxx += dx * dx;
            }
            let least_squares = if sxx == 0.0 { 0.0 } else { sxy / sxx };
            (median, endpoint, least_squares)
        }
        ValidPositions::Indices(indices) => {
            let count = indices.len();
            let mut prev_x = indices[0] as f64;
            let mut prev_y = chunk[indices[0]].to_f64().unwrap_or(0.0);
            let (first_x, first_y) = (prev_x, prev_y);
            let mut sum_x = prev_x;
            let mut sum_y = prev_y;
            for (slot, &pos) in indices[1..].iter().enumerate() {
                let x_val = pos as f64;
                let y_val = chunk[pos].to_f64().unwrap_or(0.0);
                deltas[slot] = (y_val - prev_y) / (x_val - prev_x);
                sum_x += x_val;
                sum_y += y_val;
                prev_x = x_val;
                prev_y = y_val;
            }
            let delta_count = count - 1;
            let mid = delta_count / 2;
            deltas[..delta_count].select_nth_unstable_by(mid, f64::total_cmp);
            let median = deltas[mid];
            let endpoint = (prev_y - first_y) / (prev_x - first_x);

            let mean_x = sum_x / (count as f64);
            let mean_y = sum_y / (count as f64);
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for &pos in indices {
                let dx = pos as f64 - mean_x;
                let dy = chunk[pos].to_f64().unwrap_or(0.0) - mean_y;
                sxy += dx * dy;
                sxx += dx * dx;
            }
            let least_squares = if sxx == 0.0 { 0.0 } else { sxy / sxx };
            (median, endpoint, least_squares)
        }
    };

    let unit = f64::from(1u32 << slope_shift);
    if options.least_squares {
        let slope = (least_squares * unit).round();
        return SlopeCandidates::single(
            if slope.is_finite() && slope.abs() * (FL_CHUNK_SIZE as f64) < i64::MAX as f64 {
                slope as i64
            } else {
                0
            },
        );
    }
    let fits = [median, endpoint, least_squares];
    let integer = fits.iter().map(|s| s.round() * unit);
    let fractional = fits.iter().map(|s| (s * unit).round());
    for slope in integer.chain(fractional) {
        // Slopes whose term could overflow `i64` would only ever give wide residuals.
        if slope.is_finite() && slope.abs() * (FL_CHUNK_SIZE as f64) < i64::MAX as f64 {
            candidates.push_unique(slope as i64);
        }
    }
    candidates
}

/// Pick the number of fractional slope bits, so that the steepest chunk's slope term still fits
/// in `i64` across a whole chunk.
fn choose_slope_shift<T: PrimInt>(values: &[T], mask: &Mask) -> u8 {
    let all_valid = matches!(mask, Mask::AllTrue(_));
    let mut steepest = 0f64;
    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = chunk_idx * FL_CHUNK_SIZE;
        let (first, last) = if all_valid {
            ((!chunk.is_empty()).then_some(0), chunk.len().checked_sub(1))
        } else {
            (
                (0..chunk.len()).find(|&j| mask.value(start + j)),
                (0..chunk.len()).rev().find(|&j| mask.value(start + j)),
            )
        };
        if let (Some(first), Some(last)) = (first, last)
            && last > first
        {
            let rise = chunk[last].to_f64().unwrap_or(0.0) - chunk[first].to_f64().unwrap_or(0.0);
            steepest = steepest.max((rise / (last - first) as f64).abs());
        }
    }
    slope_shift_from_steepest(steepest)
}

fn choose_slope_shift_sampled<T: PrimInt>(
    values: &[T],
    mask: &Mask,
    num_chunks: usize,
    sample_chunks: usize,
) -> u8 {
    let all_valid = matches!(mask, Mask::AllTrue(_));
    let k = num_chunks.min(sample_chunks);
    let mut steepest = 0f64;
    for i in 0..k {
        let chunk_idx = if num_chunks <= sample_chunks {
            i
        } else {
            i * num_chunks / sample_chunks
        };
        let start = chunk_idx * FL_CHUNK_SIZE;
        let end = (start + FL_CHUNK_SIZE).min(values.len());
        let chunk = &values[start..end];
        let (first, last) = if all_valid {
            ((!chunk.is_empty()).then_some(0), chunk.len().checked_sub(1))
        } else {
            (
                (0..chunk.len()).find(|&j| mask.value(start + j)),
                (0..chunk.len()).rev().find(|&j| mask.value(start + j)),
            )
        };
        if let (Some(first), Some(last)) = (first, last)
            && last > first
        {
            let rise = chunk[last].to_f64().unwrap_or(0.0) - chunk[first].to_f64().unwrap_or(0.0);
            steepest = steepest.max((rise / (last - first) as f64).abs());
        }
    }
    slope_shift_from_steepest(steepest)
}

#[inline]
fn slope_shift_from_steepest(steepest: f64) -> u8 {
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
        1 => PrimitiveArray::new(
            encoded.iter().map(|&e| e as u8).collect::<Buffer<u8>>(),
            validity,
        ),
        2 => PrimitiveArray::new(
            encoded.iter().map(|&e| e as u16).collect::<Buffer<u16>>(),
            validity,
        ),
        4 => PrimitiveArray::new(
            encoded.iter().map(|&e| e as u32).collect::<Buffer<u32>>(),
            validity,
        ),
        // As wide as the array's type: keep that type, so decoding reuses FoR's fused kernels.
        _ => PrimitiveArray::new(
            encoded.iter().map(|&e| e.as_()).collect::<Buffer<T>>(),
            validity,
        ),
    }
    .into_array()
}

/// A constant array when every value matches, otherwise a primitive array.
fn constant_or_primitive<T: NativePType + Into<PValue>>(values: Buffer<T>) -> ArrayRef {
    match values.first() {
        Some(&first) if values.iter().all(|&v| v == first) => ConstantArray::new(
            Scalar::primitive(first, Nullability::NonNullable),
            values.len(),
        )
        .into_array(),
        _ => PrimitiveArray::new(values, Validity::NonNullable).into_array(),
    }
}
