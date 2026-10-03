// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elementwise transforms that split numbers into a primary and an optional secondary latent
//! stream, each coded by its own [`Stream`].
//!
//! These play the role of Pco's modes. Every transform is elementwise, so both streams share the
//! block layout and random access still decodes one block of each. Candidates come from cheap
//! detectors (several adapted from Pco's `float_mult`), and the winner is whichever has the
//! smallest estimated encoded size on a sample, so a weak detector costs time but never ratio.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::latent::Latent;
use crate::lookback;
use crate::latent::Number;
use crate::stream::BLOCK_SIZE;
use crate::stream::Config;
use crate::stream::Stream;
use crate::stream::StreamDecoder;
use crate::stream::compress_latents;
use crate::stream::estimate_bits;
use crate::stream::sample;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Classic,
    /// `latent = primary * base + secondary`, with `secondary < base`.
    IntMult(u64),
    /// `x = from_latent(to_latent(primary as float * base) + secondary - MID)`.
    FloatMult(f64),
    /// `latent = primary << k | secondary`, with `secondary < 2^k`.
    FloatQuant(u8),
}

/// Prefer the simpler, faster mode unless a transform saves at least this fraction of size.
const MIN_RELATIVE_SAVINGS: f64 = 0.01;
/// Lookback decoding is slower than consecutive deltas, so it must save at least this much.
const MIN_LOOKBACK_SAVINGS: f64 = 0.03;

/// Numbers with the per-type pieces the transforms need.
pub trait Numeric: Number {
    const PRECISION_BITS: u32;
    const IS_FLOAT: bool;

    fn to_f64(self) -> f64;
    /// Splits `self` against `base`, rounding `self * inv_base` to the nearest multiple.
    fn float_mult_split(self, base: f64, inv_base: f64) -> (Self::L, Self::L);
    fn float_mult_join(primary: Self::L, secondary: Self::L, base: f64) -> Self;
}

macro_rules! impl_int_numeric {
    ($t:ty) => {
        impl Numeric for $t {
            const PRECISION_BITS: u32 = <$t>::BITS;
            const IS_FLOAT: bool = false;

            #[allow(clippy::cast_precision_loss)]
            fn to_f64(self) -> f64 {
                self as f64
            }

            fn float_mult_split(self, _: f64, _: f64) -> (Self::L, Self::L) {
                (self.to_latent(), Latent::ZERO)
            }

            fn float_mult_join(primary: Self::L, _: Self::L, _: f64) -> Self {
                Self::from_latent(primary)
            }
        }
    };
}

impl_int_numeric!(u8);
impl_int_numeric!(u16);
impl_int_numeric!(u32);
impl_int_numeric!(u64);
impl_int_numeric!(i8);
impl_int_numeric!(i16);
impl_int_numeric!(i32);
impl_int_numeric!(i64);

impl Numeric for half::f16 {
    const PRECISION_BITS: u32 = 10;
    // Too narrow for the float transforms to pay for their second stream.
    const IS_FLOAT: bool = false;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn float_mult_split(self, _: f64, _: f64) -> (u16, u16) {
        (self.to_latent(), 0)
    }

    fn float_mult_join(primary: u16, _: u16, _: f64) -> Self {
        Self::from_latent(primary)
    }
}

/// Exact `i64 -> f64` for `|q| < 2^51` using only integer and float adds, so it vectorizes
/// without AVX-512.
#[inline]
#[allow(clippy::cast_sign_loss)]
fn small_i64_to_f64(q: i64) -> f64 {
    const MAGIC: f64 = 6_755_399_441_055_744.0; // 2^52 + 2^51
    f64::from_bits((q as u64).wrapping_add(MAGIC.to_bits())) - MAGIC
}

#[inline]
fn i32_to_f32(q: i32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let f = q as f32;
    f
}

macro_rules! impl_float_numeric {
    ($f:ty, $i:ty, $u:ty, $precision:expr, $limit_log:expr, $to_float:ident) => {
        impl Numeric for $f {
            const PRECISION_BITS: u32 = $precision;
            const IS_FLOAT: bool = true;

            fn to_f64(self) -> f64 {
                f64::from(self)
            }

            #[inline]
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            fn float_mult_split(self, base: f64, inv_base: f64) -> ($u, $u) {
                let (base, inv_base) = (base as $f, inv_base as $f);
                let mult = (self * inv_base).round();
                // Out-of-range multiples fall back to 0, leaving the whole value in the
                // adjustment; the limit keeps the int-to-float conversion exact and vectorizable.
                const LIMIT: $f = (1u64 << $limit_log) as $f;
                let q: $i = if mult.abs() < LIMIT { mult as $i } else { 0 };
                let approx = $to_float(q) * base;
                let primary = (q as $u) ^ <$u as Latent>::MID;
                let secondary = self
                    .to_latent()
                    .wrapping_sub(approx.to_latent())
                    .wrapping_add(<$u as Latent>::MID);
                (primary, secondary)
            }

            #[inline]
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            fn float_mult_join(primary: $u, secondary: $u, base: f64) -> Self {
                let q = (primary ^ <$u as Latent>::MID) as $i;
                let approx = $to_float(q) * (base as $f);
                <$f>::from_latent(
                    approx
                        .to_latent()
                        .wrapping_add(secondary)
                        .wrapping_sub(<$u as Latent>::MID),
                )
            }
        }
    };
}

impl_float_numeric!(f32, i32, u32, 23, 30, i32_to_f32);
impl_float_numeric!(f64, i64, u64, 52, 51, small_i64_to_f64);

fn split<T: Numeric>(values: &[T], mode: Mode) -> (Vec<T::L>, Option<Vec<T::L>>) {
    match mode {
        Mode::Classic => (values.iter().map(|v| v.to_latent()).collect(), None),
        Mode::IntMult(base) => {
            let (p, s) = values
                .iter()
                .map(|v| {
                    let l = v.to_latent().to_u64();
                    (T::L::from_u64(l / base), T::L::from_u64(l % base))
                })
                .unzip();
            (p, Some(s))
        }
        Mode::FloatQuant(k) => {
            let mask = (1u64 << k) - 1;
            let (p, s) = values
                .iter()
                .map(|v| {
                    let l = v.to_latent().to_u64();
                    (T::L::from_u64(l >> k), T::L::from_u64(l & mask))
                })
                .unzip();
            (p, Some(s))
        }
        Mode::FloatMult(base) => {
            let inv_base = 1.0 / base;
            let (p, s) = values
                .iter()
                .map(|v| v.float_mult_split(base, inv_base))
                .unzip();
            (p, Some(s))
        }
    }
}

/// Writes `n` values produced by `f(i)` after `out`'s current end.
#[inline(always)]
#[allow(clippy::inline_always)]
fn extend_with<T>(out: &mut Vec<T>, n: usize, f: impl Fn(usize) -> T) {
    out.reserve(n);
    let dst = &mut out.spare_capacity_mut()[..n];
    for (i, d) in dst.iter_mut().enumerate() {
        d.write(f(i));
    }
    // SAFETY: the loop initialized the `n` values after the current length.
    unsafe { out.set_len(out.len() + n) };
}

#[inline(always)]
#[allow(clippy::inline_always)]
fn join_impl<T: Numeric>(mode: Mode, primary: &[T::L], secondary: &[T::L], out: &mut Vec<T>) {
    let n = primary.len();
    match mode {
        Mode::Classic => extend_with(out, n, |i| T::from_latent(primary[i])),
        Mode::IntMult(base) => {
            let secondary = &secondary[..n];
            extend_with(out, n, |i| {
                T::from_latent(T::L::from_u64(
                    primary[i]
                        .to_u64()
                        .wrapping_mul(base)
                        .wrapping_add(secondary[i].to_u64()),
                ))
            })
        }
        Mode::FloatQuant(k) => {
            let secondary = &secondary[..n];
            extend_with(out, n, |i| {
                T::from_latent(T::L::from_u64(
                    (primary[i].to_u64() << k) | secondary[i].to_u64(),
                ))
            })
        }
        Mode::FloatMult(base) => {
            let secondary = &secondary[..n];
            extend_with(out, n, |i| T::float_mult_join(primary[i], secondary[i], base))
        }
    }
}

/// [`join_impl`] compiled for AVX2.
///
/// # Safety
///
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn join_avx2<T: Numeric>(mode: Mode, primary: &[T::L], secondary: &[T::L], out: &mut Vec<T>) {
    join_impl(mode, primary, secondary, out);
}

#[inline]
fn join<T: Numeric>(
    avx2: bool,
    mode: Mode,
    primary: &[T::L],
    secondary: &[T::L],
    out: &mut Vec<T>,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: `avx2` is only set when the CPU supports it.
        unsafe { join_avx2(mode, primary, secondary, out) };
        return;
    }
    let _ = avx2;
    join_impl(mode, primary, secondary, out);
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn int_mult_candidates<T: Numeric>(sample: &[T]) -> Vec<Mode> {
    let latents: Vec<u64> = sample.iter().map(|v| v.to_latent().to_u64()).collect();
    let Some(&first) = latents.first() else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    let exact = latents.iter().fold(0, |g, &l| gcd(g, l.abs_diff(first)));
    if exact > 1 {
        candidates.push(Mode::IntMult(exact));
    }
    // A base most adjacent pairs share, tolerating a few values that are not multiples.
    let mut pair_gcds: Vec<u64> = latents
        .chunks_exact(2)
        .map(|p| gcd(p[0], p[1]))
        .filter(|&g| g > 1)
        .collect();
    pair_gcds.sort_unstable();
    let mut best = (0, 0);
    for run in pair_gcds.chunk_by(|a, b| a == b) {
        if run.len() > best.1 {
            best = (run[0], run.len());
        }
    }
    if best.1 * 10 >= latents.len() / 2 && best.0 != exact {
        candidates.push(Mode::IntMult(best.0));
    }
    candidates
}

fn float_quant_candidates<T: Numeric>(sample: &[T]) -> Vec<Mode> {
    let mut tz: Vec<u32> = sample
        .iter()
        .map(|v| {
            // Trailing zeros of the raw bits; the latent of a negative float has trailing ones.
            let l = v.to_latent().to_u64();
            let raw = if l & T::L::MID.to_u64() != 0 { l } else { !l };
            raw.trailing_zeros().min(T::PRECISION_BITS)
        })
        .collect();
    tz.sort_unstable();
    let mut candidates = Vec::new();
    for quantile in [0.1, 0.5] {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = tz[((tz.len() as f64 * quantile) as usize).min(tz.len() - 1)];
        #[allow(clippy::cast_possible_truncation)]
        let mode = Mode::FloatQuant(k as u8);
        if k >= 4 && !candidates.contains(&mode) {
            candidates.push(mode);
        }
    }
    candidates
}

/// Approximate GCD of two floats by a tolerant Euclidean algorithm, adapted from Pco.
fn approx_pair_gcd(greater: f64, lesser: f64, precision_bits: u32) -> Option<f64> {
    let p = f64::from(precision_bits);
    let insignificant = |x: f64| x * (-(p - 6.0)).exp2();
    if lesser <= insignificant(greater) || lesser == greater {
        return None;
    }
    let machine_eps = (-p).exp2();
    let (mut g, mut g_err) = (greater, 0.0f64);
    let (mut l, mut l_err) = (lesser, 0.0f64);
    loop {
        let prev = g;
        let ratio = (g / l).round();
        g_err += ratio * l_err + g * machine_eps;
        g = (g - ratio * l).abs();
        if g <= prev * (-16.0f64).exp2() || g <= g_err {
            return Some(l);
        }
        if g <= insignificant(greater) || g <= g_err * 64.0 {
            return None;
        }
        std::mem::swap(&mut g, &mut l);
        std::mem::swap(&mut g_err, &mut l_err);
    }
}

fn float_mult_candidates<T: Numeric>(sample: &[T]) -> Vec<Mode> {
    let floats: Vec<f64> = sample
        .iter()
        .map(|v| v.to_f64())
        .filter(|f| f.is_finite())
        .collect();
    let mut bases = Vec::new();
    // Decimal-like data: multiples of 10^k, exact or close (the secondary stream absorbs the
    // remainder).
    for k in -12..=6 {
        bases.push(10f64.powi(k));
    }
    // Multiples of a power of two, from trailing mantissa zeros.
    let units: Vec<i32> = floats
        .iter()
        .filter(|f| **f != 0.0)
        .filter_map(|&f| {
            let bits = f.to_bits();
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
            let tz = (bits & ((1u64 << 52) - 1)).trailing_zeros().min(52);
            #[allow(clippy::cast_possible_wrap)]
            (tz >= 5).then_some(exponent - (52 - tz as i32))
        })
        .collect();
    if units.len() * 2 >= floats.len()
        && let Some(&unit) = units.iter().min()
    {
        bases.push(f64::from(unit).exp2());
    }
    // A shared approximate GCD of adjacent pairs.
    let mut gcds: Vec<f64> = floats
        .chunks_exact(2)
        .filter_map(|p| {
            approx_pair_gcd(p[0].abs().max(p[1].abs()), p[0].abs().min(p[1].abs()), 52)
        })
        .collect();
    gcds.sort_by(f64::total_cmp);
    let required = 1 + floats.len() / 1000;
    if gcds.len() >= required {
        for quantile in [0.1, 0.3, 0.5] {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let candidate = gcds[(gcds.len() as f64 * quantile) as usize];
            let similar = gcds
                .iter()
                .filter(|&&g| (g - candidate).abs() < 0.01 * candidate)
                .count();
            if similar >= required {
                // Snap to an integer or decimal reciprocal when close, as Pco does.
                let inv = 1.0 / candidate;
                let decimal = 10f64.powf(inv.log10().round());
                let base = if (inv - inv.round()).abs() < 0.02 && inv.round() != 0.0 {
                    1.0 / inv.round()
                } else if ((inv - decimal) / inv).abs() < 0.01 {
                    1.0 / decimal
                } else {
                    candidate
                };
                bases.push(base);
                break;
            }
        }
    }
    bases
        .into_iter()
        .filter(|b| b.is_finite() && *b > 0.0)
        .map(Mode::FloatMult)
        .collect()
}

fn estimated_bits<T: Numeric>(sample: &[T], mode: Mode, config: &Config) -> f64 {
    let (primary, secondary) = split(sample, mode);
    estimate_bits(&primary, config) + secondary.map_or(0.0, |s| estimate_bits(&s, config))
}

pub fn choose_mode<T: Numeric>(values: &[T], config: &Config) -> Mode {
    let sample = sample(values);
    if sample.len() < 64 {
        return Mode::Classic;
    }
    let mut candidates = Vec::new();
    if T::IS_FLOAT {
        candidates.extend(float_mult_candidates(&sample));
        candidates.extend(float_quant_candidates(&sample));
    } else if T::PRECISION_BITS > 8 {
        candidates.extend(int_mult_candidates(&sample));
    }
    // Screen every candidate on two blocks of the sample, then score the most promising few on
    // all of it.
    if sample.len() > 2 * BLOCK_SIZE && candidates.len() > SHORTLIST {
        let mid = sample.len() / 2;
        let screen: Vec<T> = sample[..BLOCK_SIZE]
            .iter()
            .chain(&sample[mid..mid + BLOCK_SIZE])
            .copied()
            .collect();
        let mut scored: Vec<(f64, Mode)> = candidates
            .iter()
            .map(|&mode| (estimated_bits(&screen, mode, config), mode))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidates = scored.into_iter().take(SHORTLIST).map(|(_, m)| m).collect();
    }
    let classic = estimated_bits(&sample, Mode::Classic, config);
    let mut best = (Mode::Classic, classic * (1.0 - MIN_RELATIVE_SAVINGS));
    for mode in candidates {
        let bits = estimated_bits(&sample, mode, config);
        if bits < best.1 {
            best = (mode, bits);
        }
    }
    best.0
}

/// Candidates scored on the whole sample after screening.
const SHORTLIST: usize = 3;

#[derive(Clone, Debug)]
pub struct Encoded<T: Number> {
    pub n: usize,
    pub mode: Mode,
    /// With `lookbacks`, the primary stream holds lookback deltas rather than primary latents.
    pub primary: Stream<T::L>,
    pub secondary: Option<Stream<T::L>>,
    /// Per-value lookback distances for the primary latents, when lookback is used.
    pub lookbacks: Option<Stream<u16>>,
}

impl<T: Number> Encoded<T> {
    pub fn nbytes(&self) -> usize {
        10 + self.primary.nbytes()
            + self.secondary.as_ref().map_or(0, Stream::nbytes)
            + self.lookbacks.as_ref().map_or(0, Stream::nbytes)
    }
}

fn no_delta(config: &Config) -> Config {
    Config {
        max_delta_order: 0,
        ..config.clone()
    }
}

/// Whether block-local lookback beats consecutive deltas on the primary latents of a sample.
pub fn choose_lookback<T: Numeric>(values: &[T], mode: Mode, config: &Config) -> bool {
    let sample = sample(values);
    if sample.len() < 64 {
        return false;
    }
    let (primary, _) = split(&sample, mode);
    let consecutive = estimate_bits(&primary, config);
    let lookbacks = lookback::choose(&primary);
    let deltas = lookback::encode(&primary, &lookbacks);
    let with_lookback =
        estimate_bits(&deltas, &no_delta(config)) + estimate_bits(&lookbacks, config);
    with_lookback < consecutive * (1.0 - MIN_LOOKBACK_SAVINGS)
}

pub fn compress<T: Numeric>(values: &[T], config: &Config) -> VortexResult<Encoded<T>> {
    let mode = choose_mode(values, config);
    let lookback = choose_lookback(values, mode, config);
    compress_with_mode(values, mode, lookback, config)
}

pub fn compress_with_mode<T: Numeric>(
    values: &[T],
    mode: Mode,
    lookback: bool,
    config: &Config,
) -> VortexResult<Encoded<T>> {
    let (primary, secondary) = split(values, mode);
    let (primary, lookbacks) = if lookback {
        let lookbacks = lookback::choose(&primary);
        let deltas = lookback::encode(&primary, &lookbacks);
        (
            compress_latents(&deltas, &no_delta(config))?,
            Some(compress_latents(&lookbacks, config)?),
        )
    } else {
        (compress_latents(&primary, config)?, None)
    };
    Ok(Encoded {
        n: values.len(),
        mode,
        lookbacks,
        primary,
        secondary: secondary
            .map(|s| compress_latents(&s, config))
            .transpose()?,
    })
}

/// Decode tables for every stream of an [`Encoded`], built once and reused by every read.
pub struct Decoder<T: Number> {
    enc: Arc<Encoded<T>>,
    mode: Mode,
    avx2: bool,
    primary: StreamDecoder<T::L>,
    secondary: Option<StreamDecoder<T::L>>,
    lookbacks: Option<StreamDecoder<u16>>,
}

impl<T: Numeric> Decoder<T> {
    pub fn new(enc: Arc<Encoded<T>>) -> VortexResult<Self> {
        let primary = StreamDecoder::new(&enc.primary)?;
        let secondary = enc
            .secondary
            .as_ref()
            .map(StreamDecoder::new)
            .transpose()?;
        vortex_ensure!(primary.len() == enc.n, "primary stream length mismatch");
        vortex_ensure!(
            secondary.is_some() == (enc.mode != Mode::Classic),
            "mode {:?} and secondary stream disagree",
            enc.mode
        );
        if let Some(s) = &secondary {
            vortex_ensure!(s.len() == enc.n, "secondary stream length mismatch");
        }
        let lookbacks = enc
            .lookbacks
            .as_ref()
            .map(StreamDecoder::new)
            .transpose()?;
        if let Some(l) = &lookbacks {
            vortex_ensure!(l.len() == enc.n, "lookback stream length mismatch");
        }
        if let Mode::IntMult(base) = enc.mode {
            vortex_ensure!(base > 0, "IntMult base must be positive");
        }
        if let Mode::FloatQuant(k) = enc.mode {
            vortex_ensure!(u32::from(k) < T::L::BITS, "FloatQuant k {k} too large");
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Ok(Self {
            mode: enc.mode,
            enc,
            avx2,
            primary,
            secondary,
            lookbacks,
        })
    }

    pub fn len(&self) -> usize {
        self.primary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }

    pub fn decode(&self) -> Vec<T> {
        self.decode_range(0, self.len())
    }

    /// Decodes values `start..stop`, touching only the blocks they span.
    pub fn decode_range(&self, start: usize, stop: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(stop - start);
        let mut p_buf = [T::L::ZERO; BLOCK_SIZE];
        let mut s_buf = [T::L::ZERO; BLOCK_SIZE];
        let mut lb_buf = [0u16; BLOCK_SIZE];
        let mut pos = start;
        while pos < stop {
            let block = pos / BLOCK_SIZE;
            let block_start = block * BLOCK_SIZE;
            let local_stop = (stop - block_start).min(BLOCK_SIZE);
            let enc = &*self.enc;
            self.primary
                .decode_block_prefix(&enc.primary, block, local_stop, &mut p_buf);
            if let (Some(l), Some(ls)) = (&self.lookbacks, &enc.lookbacks) {
                l.decode_block_prefix(ls, block, local_stop, &mut lb_buf);
                lookback::decode_in_place(&mut p_buf[..local_stop], &lb_buf[..local_stop]);
            }
            if let (Some(s), Some(ss)) = (&self.secondary, &enc.secondary) {
                s.decode_block_prefix(ss, block, local_stop, &mut s_buf);
            }
            let range = pos - block_start..local_stop;
            join(
                self.avx2,
                self.mode,
                &p_buf[range.clone()],
                &s_buf[range],
                &mut out,
            );
            pos = block_start + local_stop;
        }
        out
    }

    /// Decodes the single value at `index`.
    pub fn get(&self, index: usize) -> T {
        let mut out = self.decode_range(index, index + 1);
        out.swap_remove(0)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn lcg(seed: u64) -> impl Iterator<Item = u64> {
        let mut x = seed;
        std::iter::repeat_with(move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 11
        })
    }

    fn roundtrip<T: Numeric + PartialEq + std::fmt::Debug>(
        values: &[T],
        mode: Mode,
    ) -> VortexResult<usize> {
        roundtrip_with(values, mode, false)?;
        roundtrip_with(values, mode, true)
    }

    fn roundtrip_with<T: Numeric + PartialEq + std::fmt::Debug>(
        values: &[T],
        mode: Mode,
        lookback: bool,
    ) -> VortexResult<usize> {
        let enc = Arc::new(compress_with_mode(values, mode, lookback, &Config::default())?);
        let dec = Decoder::new(Arc::clone(&enc))?;
        let all = dec.decode();
        assert_eq!(all.len(), values.len());
        for (a, b) in all.iter().zip(values) {
            assert_eq!(a.to_latent(), b.to_latent());
        }
        for i in (0..values.len()).step_by(101) {
            assert_eq!(dec.get(i).to_latent(), values[i].to_latent(), "index {i}");
        }
        Ok(enc.nbytes())
    }

    #[rstest]
    #[case(Mode::Classic)]
    #[case(Mode::FloatMult(0.01))]
    #[case(Mode::FloatMult(0.1))]
    #[case(Mode::FloatMult(1.0 / 3.0))]
    #[case(Mode::FloatQuant(20))]
    fn float_modes_are_lossless(#[case] mode: Mode) -> VortexResult<()> {
        let mut values: Vec<f64> = lcg(1)
            .take(5000)
            .map(|x| (x % 100_000) as f64 / 100.0)
            .collect();
        values.extend([f64::NAN, f64::INFINITY, -0.0, 1e300, -1e-300, f64::MIN_POSITIVE]);
        roundtrip(&values, mode)?;
        let values32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        roundtrip(&values32, mode)?;
        Ok(())
    }

    #[rstest]
    #[case(Mode::IntMult(1000))]
    #[case(Mode::IntMult(7))]
    fn int_modes_are_lossless(#[case] mode: Mode) -> VortexResult<()> {
        let values: Vec<i64> = lcg(2)
            .take(5000)
            .map(|x| (x % 1000) as i64 * 1000 - 77)
            .chain([i64::MIN, i64::MAX, 0])
            .collect();
        roundtrip(&values, mode)?;
        Ok(())
    }

    #[test]
    fn decimals_choose_float_mult() -> VortexResult<()> {
        let values: Vec<f64> = lcg(3)
            .take(20_000)
            .map(|x| (x % 1_000_000) as f64 / 100.0)
            .collect();
        let mode = choose_mode(&values, &Config::default());
        assert!(matches!(mode, Mode::FloatMult(_)), "{mode:?}");
        let classic = roundtrip(&values, Mode::Classic)?;
        let chosen = roundtrip(&values, mode)?;
        assert!(chosen * 2 < classic, "{chosen} vs {classic}");
        Ok(())
    }

    #[test]
    fn multiples_choose_int_mult() {
        let values: Vec<u32> = lcg(4).take(20_000).map(|x| (x % 5000) as u32 * 60).collect();
        assert_eq!(choose_mode(&values, &Config::default()), Mode::IntMult(60));
    }
}
