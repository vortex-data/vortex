// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impls for [`ListArray`].
//!
//! Two entry points are offered, mirroring [`crate::arrays::varbin::iter`]:
//!
//! * [`IterArrayValue<(usize, usize)>::iter_value`] returns a polymorphic
//!   iterator that handles any offset ptype by normalizing offsets to
//!   `Buffer<usize>` once at construction.
//!
//! * [`iter_offsets`] returns a [`ListTypedIter<O>`] parameterized over
//!   the offset type. Use it under `match_each_integer_ptype!` when you
//!   want zero per-tick dispatch and zero upfront conversion.
//!
//! Items are `(start, end)` byte-style ranges into the elements
//! child array. The caller materializes per-row sub-arrays via
//! `arr.elements().slice(start..end)` only when needed — the iterator
//! itself does no array construction.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::ListArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::arrays::varbin::VarBinOffset;
use crate::iter_array::IterArrayValue;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Marker for offset ptypes that may back a [`ListArray`]. Same set as
/// [`VarBinOffset`].
pub trait ListOffset: VarBinOffset {}
impl<T: VarBinOffset> ListOffset for T {}

enum ValidityState {
    AllValid,
    AllInvalid,
    WithValidity(ValidityCursor),
}

/// Polymorphic iterator over a [`ListArray`]'s per-row `(start, end)`
/// ranges into the elements array.
///
/// Internally specializes on the common offset ptype (`u32`) so the
/// common path is zero-overhead: refcount-clone of the existing
/// `Buffer<u32>`, per-tick u32 read + cast to usize. Rare ptypes
/// (i8/u8/i16/u16/i32/i64/u64) are normalized to `Buffer<usize>` once at
/// construction.
pub struct ListIter {
    inner: ListIterInner,
    pos: usize,
    len: usize,
    validity: ValidityState,
}

enum ListIterInner {
    /// Fast path: refcount-clone of an existing `Buffer<u32>`.
    U32 { _buf: Buffer<u32>, ptr: *const u32 },
    /// Fallback path: offsets normalized to `Buffer<usize>` once.
    Usize { offsets: Buffer<usize> },
}

// SAFETY: the raw pointer is owned via the adjacent `_buf`.
unsafe impl Send for ListIterInner {}

/// Typed iterator over `(start, end)` ranges, parameterized over the
/// physical offset type. No upfront conversion; per tick reads two `O`
/// values and casts to `usize`.
pub struct ListTypedIter<O: ListOffset> {
    _offsets_owner: Buffer<O>,
    offsets_ptr: *const O,
    pos: usize,
    len: usize,
    validity: ValidityState,
}

// SAFETY: the raw pointer is owned via the adjacent `_offsets_owner`.
unsafe impl<O: ListOffset + Send> Send for ListTypedIter<O> {}

impl Iterator for ListIter {
    type Item = Option<(usize, usize)>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // The variant tag never changes during iteration, so the branch
        // predictor pins on one arm and the body inlines as if it were
        // monomorphic.
        let (start, end) = match &self.inner {
            ListIterInner::U32 { ptr, .. } => unsafe {
                (*ptr.add(self.pos) as usize, *ptr.add(self.pos + 1) as usize)
            },
            ListIterInner::Usize { offsets } => {
                let s = offsets.as_slice();
                (s[self.pos], s[self.pos + 1])
            }
        };
        self.pos += 1;
        let range = (start, end);
        match &mut self.validity {
            ValidityState::AllValid => Some(Some(range)),
            ValidityState::AllInvalid => Some(None),
            ValidityState::WithValidity(v) => {
                let valid = v.next_bit().unwrap_or(false);
                Some(valid.then_some(range))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.pos;
        (n, Some(n))
    }
}

impl ExactSizeIterator for ListIter {}

impl<O: ListOffset> Iterator for ListTypedIter<O> {
    type Item = Option<(usize, usize)>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // SAFETY: pos < len, offsets buffer has length len + 1.
        let (start, end) = unsafe {
            (
                (*self.offsets_ptr.add(self.pos)).to_byte_index(),
                (*self.offsets_ptr.add(self.pos + 1)).to_byte_index(),
            )
        };
        self.pos += 1;
        let range = (start, end);
        match &mut self.validity {
            ValidityState::AllValid => Some(Some(range)),
            ValidityState::AllInvalid => Some(None),
            ValidityState::WithValidity(v) => {
                let valid = v.next_bit().unwrap_or(false);
                Some(valid.then_some(range))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.pos;
        (n, Some(n))
    }
}

impl<O: ListOffset> ExactSizeIterator for ListTypedIter<O> {}

fn make_validity(v: Validity) -> ValidityState {
    match v {
        Validity::NonNullable | Validity::AllValid => ValidityState::AllValid,
        Validity::AllInvalid => ValidityState::AllInvalid,
        Validity::Array(v) => {
            #[expect(deprecated)]
            let bits: BitBuffer = v.to_bool().into_bit_buffer();
            ValidityState::WithValidity(ValidityCursor::new(bits))
        }
    }
}

impl IterArrayValue<(usize, usize)> for ListArray {
    type Iter<'a> = ListIter;

    fn iter_value(&self) -> Self::Iter<'_> {
        #[expect(deprecated)]
        let offsets_arr = self.offsets().to_primitive();
        let len = self.len();
        let inner = if offsets_arr.ptype() == crate::dtype::PType::U32 {
            // Fast path: refcount-clone the existing Buffer<u32>. No
            // conversion, no allocation.
            let buf = offsets_arr.into_buffer::<u32>();
            let ptr = buf.as_slice().as_ptr();
            ListIterInner::U32 { _buf: buf, ptr }
        } else {
            // Fallback: normalize to Buffer<usize> once.
            let offsets: Buffer<usize> = match_each_integer_ptype!(offsets_arr.ptype(), |O| {
                let src = offsets_arr.as_slice::<O>();
                Buffer::<usize>::from_iter(src.iter().map(|v| (*v).to_byte_index()))
            });
            ListIterInner::Usize { offsets }
        };
        let validity = self
            .validity()
            .vortex_expect("list validity should be derivable");
        ListIter {
            inner,
            pos: 0,
            len,
            validity: make_validity(validity),
        }
    }
}

/// Construct a typed list iterator with no upfront offset conversion.
///
/// `O` must match the array's offsets ptype. Panics otherwise.
pub fn iter_offsets<O: ListOffset>(arr: &ListArray) -> ListTypedIter<O> {
    #[expect(deprecated)]
    let offsets_arr = arr.offsets().to_primitive();
    assert_eq!(
        offsets_arr.ptype(),
        O::PTYPE,
        "ListArray::iter_offsets called with O = {} but offsets ptype is {}",
        O::PTYPE,
        offsets_arr.ptype()
    );
    let len = arr.len();
    let buf: Buffer<O> = offsets_arr.into_buffer::<O>();
    let offsets_ptr = buf.as_slice().as_ptr();
    let validity = arr
        .validity()
        .vortex_expect("list validity should be derivable");
    ListTypedIter {
        _offsets_owner: buf,
        offsets_ptr,
        pos: 0,
        len,
        validity: make_validity(validity),
    }
}
