// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impl for [`VarBinArray`].
//!
//! Decoded offsets are normalized to `Buffer<usize>` *once* in
//! [`IterArray::iter`] and held on the iterator alongside a reference to
//! the byte buffer. Per-element work is two `usize` reads, one slice
//! bounds check, and (when nullable) one bit lookup. There is no per-tick
//! ptype dispatch.
//!
//! The conversion to `usize` is a one-time `O(n)` cost; for the common
//! case where offsets are already `u32`/`u64` it's a tight cast loop. We
//! pay this once per `iter()` call (not per element), and in exchange the
//! per-element path is fully monomorphized.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::VarBinArray;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::arrays::varbin::VarBinArrayExt;
use crate::iter_array::IterArray;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Iterator over a [`VarBinArray`]'s byte slices.
pub struct VarBinIter<'a> {
    // Owned buffer of `usize` offsets, normalized from the array's
    // physical offset ptype. We hold this so the iterator can return
    // `&'a [u8]` slices for the iterator's full lifetime.
    offsets: Buffer<usize>,
    bytes: &'a [u8],
    pos: usize,
    len: usize,
    validity: ValidityState,
}

enum ValidityState {
    AllValid,
    AllInvalid,
    WithValidity(ValidityCursor),
}

impl<'a> Iterator for VarBinIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let offsets = self.offsets.as_slice();
        let start = offsets[self.pos];
        let end = offsets[self.pos + 1];
        self.pos += 1;
        let slice = &self.bytes[start..end];
        match &mut self.validity {
            ValidityState::AllValid => Some(Some(slice)),
            ValidityState::AllInvalid => Some(None),
            ValidityState::WithValidity(v) => {
                let valid = v.next_bit().unwrap_or(false);
                Some(valid.then_some(slice))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.pos;
        (n, Some(n))
    }
}

impl ExactSizeIterator for VarBinIter<'_> {}

impl IterArray<[u8]> for VarBinArray {
    type Iter<'a> = VarBinIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        #[expect(deprecated)]
        let offsets_arr = self.offsets().to_primitive();
        let bytes: &[u8] = self.bytes().as_slice();
        let len = self.len();
        // Normalize offsets to usize once. For u32/i32/u64/i64/etc the
        // conversion is a single tight cast loop; the resulting Buffer is
        // owned by the iterator.
        let offsets: Buffer<usize> = match_each_integer_ptype!(offsets_arr.ptype(), |O| {
            let src = offsets_arr.as_slice::<O>();
            Buffer::<usize>::from_iter(src.iter().map(|v| {
                // varbin offsets are stored as integer ptypes (i8..i64,
                // u8..u64) and represent indices into a byte buffer, so
                // they always fit in usize on any supported target.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    *v as usize
                }
            }))
        });
        let validity = self
            .validity()
            .vortex_expect("varbin validity should be derivable");
        let validity = match validity {
            Validity::NonNullable | Validity::AllValid => ValidityState::AllValid,
            Validity::AllInvalid => ValidityState::AllInvalid,
            Validity::Array(v) => {
                #[expect(deprecated)]
                let bits: BitBuffer = v.to_bool().into_bit_buffer();
                ValidityState::WithValidity(ValidityCursor::new(bits))
            }
        };
        VarBinIter {
            offsets,
            bytes,
            pos: 0,
            len,
            validity,
        }
    }
}
