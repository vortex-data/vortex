// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impl for [`VarBinViewArray`].
//!
//! The data-buffer references and (if nullable) the validity bitmap are
//! captured once in [`IterArray::iter`]. Per-element cost is a view read
//! plus an inlined/borrowed branch — no per-tick allocation.

use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::VarBinViewArray;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::arrays::varbinview::view::BinaryView;
use crate::iter_array::IterArray;
use crate::validity::Validity;

/// Iterator over a [`VarBinViewArray`]'s byte slices.
pub struct VarBinViewIter<'a> {
    inner: VarBinViewIterInner<'a>,
}

enum VarBinViewIterInner<'a> {
    AllValid(Cursor<'a>),
    AllInvalid { remaining: usize },
    WithValidity(Cursor<'a>, ValidityCursor),
}

struct Cursor<'a> {
    views: &'a [BinaryView],
    // Slices into the array's data buffers. Cached once in `iter()` so we
    // don't rebuild a Vec on each iter() call (or per-element).
    buffers: Vec<&'a [u8]>,
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    fn next_slice(&mut self) -> Option<&'a [u8]> {
        let view = self.views.get(self.pos)?;
        self.pos += 1;
        Some(view_to_slice(view, &self.buffers))
    }
}

#[inline]
fn view_to_slice<'a>(view: &'a BinaryView, buffers: &[&'a [u8]]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let v = view.as_view();
        &buffers[v.buffer_index as usize][v.as_range()]
    }
}

impl<'a> Iterator for VarBinViewIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            VarBinViewIterInner::AllValid(c) => c.next_slice().map(Some),
            VarBinViewIterInner::AllInvalid { remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(None)
                }
            }
            VarBinViewIterInner::WithValidity(cursor, validity) => {
                let slice = cursor.next_slice()?;
                let valid = validity.next_bit().unwrap_or(false);
                Some(valid.then_some(slice))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.inner {
            VarBinViewIterInner::AllValid(c) => c.views.len() - c.pos,
            VarBinViewIterInner::AllInvalid { remaining } => *remaining,
            VarBinViewIterInner::WithValidity(c, _) => c.views.len() - c.pos,
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for VarBinViewIter<'_> {}

impl IterArray<[u8]> for VarBinViewArray {
    type Iter<'a> = VarBinViewIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        let views: &[BinaryView] = self.views();
        // Collect buffer slice references once. Each `buffer(i)` returns
        // a `&ByteBuffer`; we narrow to `&[u8]` so the per-tick branch is
        // a single slice index.
        let buffers: Vec<&[u8]> = (0..self.data_buffers().len())
            .map(|i| self.buffer(i).as_slice())
            .collect();
        let cursor = Cursor {
            views,
            buffers,
            pos: 0,
        };
        let validity = self
            .validity()
            .vortex_expect("varbinview validity should be derivable");
        let inner = match validity {
            Validity::NonNullable | Validity::AllValid => VarBinViewIterInner::AllValid(cursor),
            Validity::AllInvalid => VarBinViewIterInner::AllInvalid {
                remaining: views.len(),
            },
            Validity::Array(v) => {
                #[expect(deprecated)]
                let bits: BitBuffer = v.to_bool().into_bit_buffer();
                VarBinViewIterInner::WithValidity(cursor, ValidityCursor::new(bits))
            }
        };
        VarBinViewIter { inner }
    }
}

// keep ByteBuffer reachable in scope to silence rustc on some toolchains
const _: fn(&ByteBuffer) = |_b| {};
