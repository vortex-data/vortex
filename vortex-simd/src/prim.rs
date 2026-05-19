//! Primitive compute: typed front-end on top of the [`kernels`](crate::kernels)
//! registry.
//!
//! The [`Prim`] trait gives each primitive element type a small ergonomic
//! API: `i32::add(a, b, out)`, `i32::eq(a, b, out)`. The free helpers
//! [`add`] and [`eq`] dispatch through the trait so generic code can write
//! `add::<T>(a, b, out)` without naming the kernel slot.
//!
//! There is **no extra dispatch cost** over [`kernels`](crate::kernels):
//! every `Prim::add` is a direct call to the corresponding registry slot
//! and inlines away under `-O`. The trait exists for ergonomics, not for
//! a second layer of indirection.
//!
//! ```
//! use vortex_simd::prim::{Prim, add, eq};
//!
//! let lhs = [1_i32, 2, 3, 4, 5, 6, 7, 8];
//! let rhs = [1_i32, 0, 3, 0, 5, 0, 7, 0];
//! let mut sums = [0_i32; 8];
//! let mut mask = [0_u8; 1];
//! add::<i32>(&lhs, &rhs, &mut sums);
//! eq::<i32>(&lhs, &rhs, &mut mask);
//! assert_eq!(mask[0], 0b0101_0101);
//! # // Use the trait directly:
//! # i32::add(&lhs, &rhs, &mut sums);
//! ```

use crate::kernels::kernels;

/// A primitive element type with vectorized compute kernels.
///
/// Implementations are intentionally thin: they read the active
/// [`kernels`](crate::kernels) table and call the slot for this type. Adding
/// a new primitive is one `impl Prim for <T>` block once the corresponding
/// fields exist on [`crate::Kernels`].
pub trait Prim: Sized + Copy + 'static {
    /// Element-wise wrapping add: `out[i] = a[i].wrapping_add(b[i])`.
    fn add(a: &[Self], b: &[Self], out: &mut [Self]);

    /// Element-wise equality, written as a packed bitmap into `out`
    /// (LSB-first; `out.len() == a.len().div_ceil(8)`).
    fn eq(a: &[Self], b: &[Self], out: &mut [u8]);
}

impl Prim for i32 {
    #[inline]
    fn add(a: &[i32], b: &[i32], out: &mut [i32]) {
        (kernels().i32_add)(a, b, out)
    }

    #[inline]
    fn eq(a: &[i32], b: &[i32], out: &mut [u8]) {
        (kernels().i32_eq)(a, b, out)
    }
}

/// Generic front for [`Prim::add`].
///
/// Lets generic code dispatch by type parameter without naming the kernel
/// slot directly.
#[inline]
pub fn add<T: Prim>(a: &[T], b: &[T], out: &mut [T]) {
    T::add(a, b, out)
}

/// Generic front for [`Prim::eq`].
#[inline]
pub fn eq<T: Prim>(a: &[T], b: &[T], out: &mut [u8]) {
    T::eq(a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prim_i32_add_matches_kernels() {
        let lhs = [10_i32, 20, 30, 40];
        let rhs = [1_i32, 2, 3, 4];
        let mut got = [0_i32; 4];
        let mut want = [0_i32; 4];
        add::<i32>(&lhs, &rhs, &mut got);
        (kernels().i32_add)(&lhs, &rhs, &mut want);
        assert_eq!(got, want);
        assert_eq!(got, [11, 22, 33, 44]);
    }

    #[test]
    fn prim_i32_eq_matches_kernels() {
        let lhs = [1_i32, 2, 3, 4, 5, 6, 7, 8];
        let rhs = [1_i32, 0, 3, 0, 5, 0, 7, 0];
        let mut got = [0_u8; 1];
        let mut want = [0_u8; 1];
        eq::<i32>(&lhs, &rhs, &mut got);
        (kernels().i32_eq)(&lhs, &rhs, &mut want);
        assert_eq!(got, want);
        assert_eq!(got[0], 0b0101_0101);
    }
}
