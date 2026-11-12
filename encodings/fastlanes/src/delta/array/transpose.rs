// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FastLanes transposes dispatched to the best instruction set the CPU supports.

use std::mem::MaybeUninit;

use fastlanes::FastLanes;
#[cfg(target_arch = "x86_64")]
use fastlanes::Transpose;
use fearless_simd::Level;

use crate::FL_CHUNK_SIZE;

type Kernel<T> = unsafe fn(&[T; FL_CHUNK_SIZE], &mut [T; FL_CHUNK_SIZE]);

/// A transpose or untranspose of one 1024-value chunk, for an instruction set that `level`
/// proved this CPU supports.
#[derive(Clone, Copy)]
pub(crate) struct TransposeKernel<T>(Kernel<T>);

impl<T: FastLanes> TransposeKernel<T> {
    /// The [`fastlanes::Transpose::transpose`] variant for `level`.
    pub(crate) fn transpose(level: Level) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if level.as_avx512().is_some() {
                return Self(avx512_or_dispatch(
                    fastlanes::transpose_avx512::<T>,
                    <T as Transpose>::transpose,
                ));
            }
            if level.as_avx2().is_some() {
                return Self(fastlanes::transpose_avx2::<T>);
            }
            if level.as_sse2().is_some() {
                return Self(fastlanes::transpose_sse2::<T>);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if level.as_neon().is_some() {
            return Self(fastlanes::transpose_neon::<T>);
        }
        let _ = level;
        Self(fastlanes::transpose_scalar::<T>)
    }

    /// The [`fastlanes::Transpose::untranspose`] variant for `level`.
    pub(crate) fn untranspose(level: Level) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if level.as_avx512().is_some() {
                return Self(avx512_or_dispatch(
                    fastlanes::untranspose_avx512::<T>,
                    <T as Transpose>::untranspose,
                ));
            }
            if level.as_avx2().is_some() {
                return Self(fastlanes::untranspose_avx2::<T>);
            }
            if level.as_sse2().is_some() {
                return Self(fastlanes::untranspose_sse2::<T>);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if level.as_neon().is_some() {
            return Self(fastlanes::untranspose_neon::<T>);
        }
        let _ = level;
        Self(fastlanes::untranspose_scalar::<T>)
    }

    #[inline]
    pub(crate) fn apply(self, input: &[T; FL_CHUNK_SIZE], output: &mut [T; FL_CHUNK_SIZE]) {
        // SAFETY: the constructors only pick kernels for instruction sets their level proved.
        unsafe { (self.0)(input, output) }
    }

    /// Like [`Self::apply`], initializing every value of `output`.
    #[inline]
    pub(crate) fn apply_uninit(
        self,
        input: &[T; FL_CHUNK_SIZE],
        output: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) {
        // SAFETY: the kernel only writes `output`, every value of it, and `MaybeUninit<T>` has the
        // layout of `T`.
        self.apply(input, unsafe {
            &mut *output.as_mut_ptr().cast::<[T; FL_CHUNK_SIZE]>()
        });
    }
}

/// The AVX-512 kernel for `u8` and `u16`, and fastlanes' own dispatch for wider types.
///
/// AVX-512 is slower than AVX2 for `u32` and `u64` on Zen 4, which runs 512-bit shuffles at half
/// rate. A fearless_simd level cannot tell Zen 4 apart, but fastlanes' dispatch can.
#[cfg(target_arch = "x86_64")]
fn avx512_or_dispatch<T: FastLanes>(avx512: Kernel<T>, dispatch: Kernel<T>) -> Kernel<T> {
    if size_of::<T>() <= 2 {
        avx512
    } else {
        dispatch
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use fastlanes::FastLanes;
    use fearless_simd::Level;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand::distr::StandardUniform;
    use rand::rngs::StdRng;
    use rstest::rstest;

    use super::TransposeKernel;
    use crate::FL_CHUNK_SIZE;

    fn check<T: FastLanes + std::fmt::Debug>(level: Level)
    where
        StandardUniform: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(0);
        let input: [T; FL_CHUNK_SIZE] = array::from_fn(|_| rng.random());
        let mut expected = [T::zero(); FL_CHUNK_SIZE];
        let mut actual = [T::zero(); FL_CHUNK_SIZE];

        fastlanes::transpose_scalar(&input, &mut expected);
        TransposeKernel::transpose(level).apply(&input, &mut actual);
        assert_eq!(actual, expected);

        fastlanes::untranspose_scalar(&input, &mut expected);
        TransposeKernel::untranspose(level).apply(&input, &mut actual);
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::detected(Level::new())]
    #[case::baseline(Level::baseline())]
    fn kernels_match_scalar(#[case] level: Level) {
        check::<u8>(level);
        check::<u16>(level);
        check::<u32>(level);
        check::<u64>(level);
    }
}
