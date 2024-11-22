use arrow_buffer::NullBuffer;
use vortex_dtype::NativePType;

use crate::ArrayLen;

pub struct VarBinIter<'a, I> {
    bytes: &'a [u8],
    indices: &'a [I],
    validity: NullBuffer,
    idx: usize,
}

impl<'a, I: NativePType> Iterator for VarBinIter<'a, I> {
    type Item = Option<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.indices.len() - 1 {
            return None;
        }

        if self.validity.is_valid(self.idx) {
            let start = self.indices[self.idx];
            let end = self.indices[self.idx + 1];
            let value = Some(&self.bytes[start..end]);
            self.idx += 1;
            Some(value)
        } else {
            self.idx += 1;
            Some(None)
        }
    }
}
