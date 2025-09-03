// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;

use crate::arrays::varbinview::VarBinViewArray;

pub struct Iter<'a> {
    index: usize,
    array: &'a VarBinViewArray,
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
            .is_valid(self.index)
            .then(|| self.array.bytes_at(self.index));

        self.index += 1;

        Some(result)
    }
}

impl VarBinViewArray {
    /// Get an iterator over the byte values inside the array.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            index: 0,
            array: self,
        }
    }
}

pub struct IntoIter {
    index: usize,
    array: VarBinViewArray,
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

impl IntoIterator for VarBinViewArray {
    type Item = Option<ByteBuffer>;
    type IntoIter = IntoIter;

    fn into_iter(self) -> IntoIter {
        IntoIter {
            index: 0,
            array: self,
        }
    }
}
