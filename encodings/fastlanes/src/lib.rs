// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;

mod bitpacking;
mod delta;
mod r#for;
mod rle;

#[cfg(test)]
mod test_order;

pub(crate) const FL_CHUNK_SIZE: usize = 1024;
