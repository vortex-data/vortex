// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impl for [`BoolArray`].
//!
//! Yields owned `bool` values (not references), since bit-packed booleans
//! are not byte-addressable. Both the value bits and the validity bits are
//! resolved once in [`IterArrayValue::iter_value`] and held on the iterator.

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::iter_array::IterArrayValue;
use crate::validity::Validity;

/// Iterator over a [`BoolArray`]'s elements.
pub struct BoolIter {
    values: ValueCursor,
    inner: BoolIterInner,
}

struct ValueCursor {
    bits: BitBuffer,
    pos: usize,
}

impl ValueCursor {
    #[inline]
    fn next_value(&mut self) -> Option<bool> {
        if self.pos >= self.bits.len() {
            return None;
        }
        // SAFETY: pos < len was just checked.
        let v = unsafe { self.bits.value_unchecked(self.pos) };
        self.pos += 1;
        Some(v)
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.bits.len().saturating_sub(self.pos)
    }
}

enum BoolIterInner {
    AllValid,
    AllInvalid,
    WithValidity(ValidityCursor),
}

impl Iterator for BoolIter {
    type Item = Option<bool>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.values.next_value()?;
        match &mut self.inner {
            BoolIterInner::AllValid => Some(Some(v)),
            BoolIterInner::AllInvalid => Some(None),
            BoolIterInner::WithValidity(vc) => {
                let valid = vc.next_bit().unwrap_or(false);
                Some(valid.then_some(v))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.values.remaining();
        (n, Some(n))
    }
}

impl ExactSizeIterator for BoolIter {}

impl IterArrayValue<bool> for BoolArray {
    type Iter<'a> = BoolIter;

    fn iter_value(&self) -> Self::Iter<'_> {
        let bits = self.to_bit_buffer();
        let values = ValueCursor { bits, pos: 0 };
        let validity = self
            .validity()
            .vortex_expect("bool validity should be derivable");
        let inner = match validity {
            Validity::NonNullable | Validity::AllValid => BoolIterInner::AllValid,
            Validity::AllInvalid => BoolIterInner::AllInvalid,
            Validity::Array(v) => {
                #[expect(deprecated)]
                let bits: BitBuffer = v.to_bool().into_bit_buffer();
                BoolIterInner::WithValidity(ValidityCursor::new(bits))
            }
        };
        BoolIter { values, inner }
    }
}
