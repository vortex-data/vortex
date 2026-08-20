// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared structural operations for fixed-width canonical arrays.

mod array;
pub(crate) mod filter;
pub(crate) mod take;
pub(crate) mod vtable;

pub(crate) use self::array::FixedWidthArray;
pub(crate) use self::array::with_values;

/// Dispatches a runtime byte width to a compile-time `const $W: usize` for every record width
/// with a dedicated fixed-width kernel, falling back to `$fallback` for any other width.
macro_rules! match_each_record_width {
    ($byte_width:expr, | $W:ident | $body:block,_ => $fallback:block) => {
        match $byte_width {
            1 => {
                const $W: usize = 1;
                $body
            }
            2 => {
                const $W: usize = 2;
                $body
            }
            4 => {
                const $W: usize = 4;
                $body
            }
            8 => {
                const $W: usize = 8;
                $body
            }
            16 => {
                const $W: usize = 16;
                $body
            }
            32 => {
                const $W: usize = 32;
                $body
            }
            _ => $fallback,
        }
    };
}
pub(crate) use match_each_record_width;
