// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffer;
mod iter;
mod vector;
mod view;

pub use buffer::*;
pub use iter::*;
pub use vector::*;
pub use view::*;

use crate::experiment::N;

pub trait BitMask {
    fn true_count(&self) -> usize;
    fn as_raw(&self) -> &[u64; N / 64];
    fn to_owned(&self) -> BitVector;
}

impl dyn BitMask + '_ {
    /// Runs the provided function `f` for each index of a `true` bit in the view.
    pub fn iter_ones<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self.true_count() {
            0 => {}
            N => (0..N).for_each(&mut f),
            _ => {
                let mut bit_idx = 0;
                for raw in self.as_raw().iter() {
                    let mut raw = *raw;
                    if raw == 0 {
                        bit_idx += 64;
                        continue;
                    }
                    while raw != 0 {
                        let bit_pos = raw.trailing_zeros();
                        raw ^= 1 << bit_pos;

                        f(bit_idx + bit_pos as usize);
                    }
                    bit_idx += 64;
                }
            }
        }
    }
}
