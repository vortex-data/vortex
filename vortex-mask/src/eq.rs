// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;

use crate::Mask;

impl PartialEq for Mask {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        if mem::discriminant(self) == mem::discriminant(other) && !matches!(self, Mask::Values(_)) {
            return true;
        }
        if self.true_count() != other.true_count() {
            return false;
        }

        // TODO(ngates): we could compare by indices if density is low enough
        self.bit_buffer() == other.bit_buffer()
    }
}

impl Eq for Mask {}
