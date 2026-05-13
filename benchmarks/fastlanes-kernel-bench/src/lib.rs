// SPDX-FileCopyrightText: Copyright the FastLanes Authors
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the upstream `fastlanes` crate v0.5.0 (https://github.com/spiraldb/fastlanes).
// Trimmed to the modules required by the 1024-element unpack and fused FoR+unpack kernels.

#![allow(clippy::all, clippy::pedantic)]

use core::mem::size_of;

use num_traits::PrimInt;
use num_traits::Unsigned;

mod bitpacking;
mod ffor;
mod macros;

pub use bitpacking::*;
pub use ffor::*;

pub const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

pub trait FastLanes: Sized + Unsigned + PrimInt {
    const T: usize = size_of::<Self>() * 8;
    const LANES: usize = 1024 / Self::T;
}

impl FastLanes for u8 {}
impl FastLanes for u16 {}
impl FastLanes for u32 {}
impl FastLanes for u64 {}

// Macro for repeating a code block bit_size_of::<T> times.
#[macro_export]
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {
        seq_macro::seq!($ident in 0..8 $body)
    };
    ($ident:ident in u16 $body:tt) => {
        seq_macro::seq!($ident in 0..16 $body)
    };
    ($ident:ident in u32 $body:tt) => {
        seq_macro::seq!($ident in 0..32 $body)
    };
    ($ident:ident in u64 $body:tt) => {
        seq_macro::seq!($ident in 0..64 $body)
    };
}

pub(crate) const fn supported_bit_width(width: usize, type_width: usize) -> bool {
    match type_width {
        8 => width <= 8,
        16 => width <= 16,
        32 => width <= 32,
        64 => width <= 64,
        _ => unreachable!(),
    }
}
