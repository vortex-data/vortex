//! Shared kernel bodies that the autovectorizer handles well across
//! architectures.
//!
//! These functions are marked `#[inline]` and contain no architecture-specific
//! intrinsics. The arch-specific wrappers in [`crate::arch`] re-emit each body
//! under `#[target_feature(enable = "...")]`, and LLVM autovectorizes against
//! the wrapper's ISA. The result: one source file produces SSE2, AVX2, AVX-512
//! and NEON variants of the same kernel.
//!
//! Use this module for kernels with a simple element-wise shape (add, sub,
//! mul, and ordered compares with byte-per-element output). For kernels with
//! lane permutations or mask packing (bitmap-output eq/lt/gt, fastlanes
//! bit-packing) the autovectorizer is not reliable; those live in
//! [`crate::arch`] with hand-written intrinsics.

#[inline]
pub fn add_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for idx in 0..a.len() {
        out[idx] = a[idx].wrapping_add(b[idx]);
    }
}
