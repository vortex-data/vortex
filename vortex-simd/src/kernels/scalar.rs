//! Scalar reference implementations. Always available; serve as the
//! correctness oracle every vectorized kernel is fuzz-checked against.

/// `out[i] = a[i].wrapping_add(b[i])` for `i in 0..len`.
///
/// All slices must be the same length; this is checked once at the boundary.
pub fn add_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for idx in 0..a.len() {
        out[idx] = a[idx].wrapping_add(b[idx]);
    }
}

/// `out` is a packed bitmap (LSB-first within each byte) of `a[i] == b[i]`.
/// `out` must have length `(a.len() + 7) / 8`.
pub fn eq_i32(a: &[i32], b: &[i32], out: &mut [u8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(out.len(), a.len().div_ceil(8));
    out.fill(0);
    for idx in 0..a.len() {
        if a[idx] == b[idx] {
            out[idx / 8] |= 1 << (idx % 8);
        }
    }
}
