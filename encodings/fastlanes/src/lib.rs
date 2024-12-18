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
