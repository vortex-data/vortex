// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impl for [`PrimitiveArray`].
//!
//! The iterator state captures the value slice and (when nullable) an
//! owned validity [`BitBuffer`] plus a cursor, so the per-element cost is
//! a single slice load and (in the nullable case) one bit lookup.

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::iter_array::IterArray;
use crate::validity::Validity;

/// Owned cursor over a validity bit buffer.
///
/// Avoids a self-referential iterator by storing the [`BitBuffer`] (cheap
/// to clone — it's a refcounted allocation) and an index, looking up one
/// bit per call. Used by the per-array `IterArray` / `IterArrayValue`
/// implementations to thread validity through their inner iterators.
pub struct ValidityCursor {
    bits: BitBuffer,
    pos: usize,
}

impl ValidityCursor {
    /// Construct a cursor that walks `bits` from position 0.
    #[inline]
    pub fn new(bits: BitBuffer) -> Self {
        Self { bits, pos: 0 }
    }

    /// Return the next bit, or `None` when the cursor is exhausted.
    #[inline]
    pub fn next_bit(&mut self) -> Option<bool> {
        if self.pos >= self.bits.len() {
            return None;
        }
        // SAFETY: pos < len was just checked.
        let v = unsafe { self.bits.value_unchecked(self.pos) };
        self.pos += 1;
        Some(v)
    }
}

/// Iterator over a primitive array's elements.
///
/// Constructed by [`IterArray::iter`] on a [`PrimitiveArray`].
pub enum PrimitiveIter<'a, T: NativePType> {
    /// All elements are valid. Yields `Some(&value)` for every slot.
    AllValid(std::slice::Iter<'a, T>),
    /// All elements are null. Yields `None` for every slot.
    AllInvalid { remaining: usize },
    /// Per-element validity is determined by a bit buffer captured at
    /// construction time.
    WithValidity {
        values: std::slice::Iter<'a, T>,
        validity: ValidityCursor,
    },
}

impl<'a, T: NativePType> Iterator for PrimitiveIter<'a, T> {
    type Item = Option<&'a T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PrimitiveIter::AllValid(it) => it.next().map(Some),
            PrimitiveIter::AllInvalid { remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(None)
                }
            }
            PrimitiveIter::WithValidity { values, validity } => {
                let v = values.next()?;
                let valid = validity.next_bit().unwrap_or(false);
                Some(valid.then_some(v))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            PrimitiveIter::AllValid(it) => it.len(),
            PrimitiveIter::AllInvalid { remaining } => *remaining,
            PrimitiveIter::WithValidity { values, .. } => values.len(),
        };
        (n, Some(n))
    }
}

impl<T: NativePType> ExactSizeIterator for PrimitiveIter<'_, T> {}

impl<T: NativePType> IterArray<T> for PrimitiveArray {
    type Iter<'a> = PrimitiveIter<'a, T>;

    fn iter(&self) -> Self::Iter<'_> {
        let values = self.as_slice::<T>().iter();
        match self
            .validity()
            .vortex_expect("primitive validity should be derivable")
        {
            Validity::NonNullable | Validity::AllValid => PrimitiveIter::AllValid(values),
            Validity::AllInvalid => PrimitiveIter::AllInvalid {
                remaining: self.len(),
            },
            Validity::Array(v) => {
                #[expect(deprecated)]
                let bits: BitBuffer = v.to_bool().into_bit_buffer();
                PrimitiveIter::WithValidity {
                    values,
                    validity: ValidityCursor::new(bits),
                }
            }
        }
    }
}
