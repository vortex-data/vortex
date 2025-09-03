// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;

use crate::arrays::varbin::VarBinArray;

pub struct Iter<'a> {
    index: usize,
    array: &'a VarBinArray,
}

impl<'a> Iterator for Iter<'a> {
    type Item = Option<ByteBuffer>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.len() {
            return None;
        }

        let result = self
            .array
            .validity
            .is_valid(self.index)
            .then(|| self.array.bytes_at(self.index));

        self.index += 1;
        Some(result)
    }
}

impl VarBinArray {
    /// Get an iterator over the byte values inside the array.
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            index: 0,
            array: self,
        }
    }
}

pub struct IntoIter {
    index: usize,
    array: VarBinArray,
}

impl Iterator for IntoIter {
    type Item = Option<ByteBuffer>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.len() {
            return None;
        }

        let result = self
            .array
            .validity
            .is_valid(self.index)
            .then(|| self.array.bytes_at(self.index));

        self.index += 1;
        Some(result)
    }
}

impl IntoIterator for VarBinArray {
    type Item = Option<ByteBuffer>;
    type IntoIter = IntoIter;

    fn into_iter(self) -> IntoIter {
        IntoIter {
            index: 0,
            array: self,
        }
    }
}
