//! Codegen subjects for `verify-simd.sh`. Each `#[unsafe(no_mangle)]` function
//! is a concrete monomorphization of the public API whose assembly is checked
//! for wide vector instructions and the absence of `memcpy`.

use vortex_iter::BatchIter;
use vortex_iter::batches;
use vortex_iter::zip;

/// `reduce_lanes` over u32, N=8 (256-bit AVX2).
#[unsafe(no_mangle)]
pub fn subject_sum_u32(data: &[u32]) -> u32 {
    batches::<u32, 8>(data).reduce_lanes(0, |a, b| a.wrapping_add(b))
}

/// `map` + `write_to` over u32, N=8.
#[unsafe(no_mangle)]
pub fn subject_mul3_u32(src: &[u32], out: &mut [u32]) {
    batches::<u32, 8>(src)
        .map(|x| x.wrapping_mul(3))
        .write_to(out);
}

/// Planar `zip` dot product over f32 with fused multiply-add, N=8.
#[unsafe(no_mangle)]
pub fn subject_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    zip::<f32, f32, 8>(a, b).fold_lanes(0.0, |acc, x, y| x.mul_add(y, acc), |x, y| x + y)
}

/// Planar `zip` elementwise add over i32, N=8.
#[unsafe(no_mangle)]
pub fn subject_add_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    zip::<i32, i32, 8>(a, b).map_to(out, |x, y| x.wrapping_add(y));
}

fn main() {}
