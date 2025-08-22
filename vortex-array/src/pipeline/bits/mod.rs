// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vector;
mod view;
mod view_mut;

pub use vector::*;
pub use view::*;
pub use view_mut::*;

use crate::pipeline::N;

// Number of usize words needed to store SC bits
#[cfg(target_pointer_width = "32")]
const N_BITS: usize = N / 32; // 32 bits per usize
#[cfg(target_pointer_width = "64")]
const N_BITS: usize = N / 64; // 64 bits per usize
