#![allow(incomplete_features)]
#![allow(clippy::cast_possible_truncation)]
#![feature(generic_const_exprs)]
#![feature(vec_into_raw_parts)]
#![feature(iter_array_chunks)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;

mod bitpacking;
mod delta;
mod r#for;

/// FastLanes is built around the idea of 1024-bit virtual SIMD registers, therefore we enforce
/// an alignment of 128 bytes.
pub(crate) const FASTLANES_ALIGNMENT: usize = 128;
