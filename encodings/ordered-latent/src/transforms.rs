// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bijective, order-preserving casts between a primitive value and an unsigned
//! latent representation. The mapping matches pco's `to_latent_ordered` /
//! `from_latent_ordered` so that sorting the latent bytes yields the same
//! ordering (using IEEE-754 total order for floats) as sorting the originals.

use vortex_array::dtype::NativePType;
use vortex_array::dtype::half::f16;

/// Pairing of a primitive `Self` with its order-preserving unsigned latent
/// representation `Latent`, plus the conversion functions in both directions.
///
/// The conversions form a bijection: `from_latent_ordered(to_latent_ordered(x))
/// == x` for every `Self`, including non-finite floats (`+0.0`, `-0.0`, `NaN`,
/// `±Infinity`), and they preserve IEEE-754 total order for floats.
pub(crate) trait OrderedLatentNumber: NativePType {
    /// The unsigned integer type the value is recast to.
    type Latent: NativePType;

    /// Map `x: Self` to its order-preserving unsigned latent.
    fn to_latent_ordered(x: Self) -> Self::Latent;

    /// Inverse of [`Self::to_latent_ordered`].
    fn from_latent_ordered(l: Self::Latent) -> Self;
}

macro_rules! unsigned_identity {
    ($T:ty) => {
        impl OrderedLatentNumber for $T {
            type Latent = $T;

            #[inline]
            fn to_latent_ordered(x: Self) -> Self::Latent {
                x
            }

            #[inline]
            fn from_latent_ordered(l: Self::Latent) -> Self {
                l
            }
        }
    };
}

unsigned_identity!(u8);
unsigned_identity!(u16);
unsigned_identity!(u32);
unsigned_identity!(u64);

macro_rules! signed_to_unsigned {
    ($T:ty, $U:ty) => {
        impl OrderedLatentNumber for $T {
            type Latent = $U;

            #[inline]
            fn to_latent_ordered(x: Self) -> Self::Latent {
                x.wrapping_sub(<$T>::MIN) as $U
            }

            #[inline]
            fn from_latent_ordered(l: Self::Latent) -> Self {
                (l as $T).wrapping_add(<$T>::MIN)
            }
        }
    };
}

signed_to_unsigned!(i8, u8);
signed_to_unsigned!(i16, u16);
signed_to_unsigned!(i32, u32);
signed_to_unsigned!(i64, u64);

macro_rules! float_to_unsigned {
    ($T:ty, $U:ty) => {
        impl OrderedLatentNumber for $T {
            type Latent = $U;

            #[inline]
            fn to_latent_ordered(x: Self) -> Self::Latent {
                const SIGN: $U = 1 << (<$U>::BITS - 1);
                let b = x.to_bits();
                if b & SIGN != 0 { !b } else { b ^ SIGN }
            }

            #[inline]
            fn from_latent_ordered(l: Self::Latent) -> Self {
                const SIGN: $U = 1 << (<$U>::BITS - 1);
                if l & SIGN != 0 {
                    <$T>::from_bits(l ^ SIGN)
                } else {
                    <$T>::from_bits(!l)
                }
            }
        }
    };
}

float_to_unsigned!(f16, u16);
float_to_unsigned!(f32, u32);
float_to_unsigned!(f64, u64);

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;

    fn roundtrip<T: OrderedLatentNumber + std::fmt::Debug>(x: T) {
        let latent = T::to_latent_ordered(x);
        let back = T::from_latent_ordered(latent);
        // Use NativePType bitwise equality so signed zeros and NaNs match.
        assert!(
            T::is_eq(x, back),
            "round-trip mismatch: {x:?} -> {latent:?} -> {back:?}"
        );
    }

    #[test]
    fn roundtrip_unsigned_extremes() -> VortexResult<()> {
        roundtrip(0u8);
        roundtrip(u8::MAX);
        roundtrip(0u16);
        roundtrip(u16::MAX);
        roundtrip(0u32);
        roundtrip(u32::MAX);
        roundtrip(0u64);
        roundtrip(u64::MAX);
        Ok(())
    }

    #[test]
    fn roundtrip_signed_extremes() -> VortexResult<()> {
        roundtrip(i8::MIN);
        roundtrip(0i8);
        roundtrip(i8::MAX);
        roundtrip(i16::MIN);
        roundtrip(i16::MAX);
        roundtrip(i32::MIN);
        roundtrip(0i32);
        roundtrip(i32::MAX);
        roundtrip(i64::MIN);
        roundtrip(i64::MAX);
        Ok(())
    }

    #[test]
    fn roundtrip_floats() -> VortexResult<()> {
        roundtrip(f16::from_f32(0.0));
        roundtrip(f16::from_f32(-0.0));
        roundtrip(f16::from_f32(1.0));
        roundtrip(f16::from_f32(-1.0));
        roundtrip(f16::INFINITY);
        roundtrip(f16::NEG_INFINITY);

        roundtrip(0.0f32);
        roundtrip(-0.0f32);
        roundtrip(1.0f32);
        roundtrip(-1.0f32);
        roundtrip(f32::INFINITY);
        roundtrip(f32::NEG_INFINITY);

        roundtrip(0.0f64);
        roundtrip(-0.0f64);
        roundtrip(1.0f64);
        roundtrip(-1.0f64);
        roundtrip(f64::INFINITY);
        roundtrip(f64::NEG_INFINITY);
        Ok(())
    }

    /// For floats the latent encoding must reflect IEEE-754 total order: every
    /// negative finite must produce a smaller latent than every non-negative
    /// finite, more-negative values produce smaller latents than less-negative
    /// ones, and so on.
    #[test]
    fn float_order_preserving() -> VortexResult<()> {
        let xs = [
            f64::NEG_INFINITY,
            -1e308,
            -1.0,
            -0.5,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            0.5,
            1.0,
            1e308,
            f64::INFINITY,
        ];
        let latents: Vec<u64> = xs.iter().copied().map(f64::to_latent_ordered).collect();
        for w in latents.windows(2) {
            assert!(w[0] <= w[1], "non-monotone: {} > {}", w[0], w[1]);
        }
        Ok(())
    }

    #[test]
    fn signed_order_preserving() -> VortexResult<()> {
        let xs = [i32::MIN, -1_000, -1, 0, 1, 1_000, i32::MAX];
        let latents: Vec<u32> = xs.iter().copied().map(i32::to_latent_ordered).collect();
        for w in latents.windows(2) {
            assert!(w[0] < w[1]);
        }
        Ok(())
    }
}
