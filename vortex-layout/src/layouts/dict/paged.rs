// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Page-at-a-time decoding of string dictionary values.
//!
//! A dictionary's values are shared by every chunk of codes that references them, so the
//! reader normally decodes all of them once. When a selective filter leaves only a few codes in
//! a chunk, that decodes values no row needs. Instead, the values are split into fixed-size pages
//! that are decoded the first time any code hits them and cached for later chunks.
//!
//! A projection then gathers each code's view straight out of its page in a single pass, sharing
//! the pages' byte buffers rather than copying them. That pass costs the same as gathering from
//! fully decoded values, and every page is decoded at most once, so even codes that hit every page
//! do no more work than decoding the dictionary whole.

use std::sync::LazyLock;
use std::sync::OnceLock;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// Values per page when `VORTEX_DICT_PAGE_SIZE` is unset.
const DEFAULT_PAGE_SIZE: usize = 1024;

/// Number of values per page, a power of two, or `0` to always decode the values whole.
///
/// Read from `VORTEX_DICT_PAGE_SIZE`, rounded up to a power of two.
pub(super) static PAGE_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("VORTEX_DICT_PAGE_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map_or(DEFAULT_PAGE_SIZE, |size| {
            if size == 0 {
                0
            } else {
                size.next_power_of_two()
            }
        })
});

/// Marks a page that no code of the current take has hit yet.
const UNSEEN: u32 = u32::MAX;

/// One decoded page of values.
struct Page {
    values: VarBinViewArray,
    /// The values' validity, or `None` when every value is valid.
    validity: Option<Mask>,
}

/// Lazily decoded pages of one dictionary's string or binary values.
pub(super) struct DictPages {
    page_size: usize,
    len: usize,
    pages: Box<[OnceLock<Page>]>,
}

impl Drop for DictPages {
    fn drop(&mut self) {
        tracing::debug!(
            pages = self.pages.len(),
            decoded = self.decoded_pages(),
            len = self.len,
            "dict pages dropped"
        );
    }
}

impl DictPages {
    pub(super) fn new(len: usize, page_size: usize) -> std::sync::Arc<Self> {
        debug_assert!(page_size.is_power_of_two());
        std::sync::Arc::new(Self {
            page_size,
            len,
            pages: (0..len.div_ceil(page_size))
                .map(|_| OnceLock::new())
                .collect(),
        })
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.pages.len()
    }

    /// How many pages have been decoded so far.
    pub(super) fn decoded_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.get().is_some()).count()
    }

    fn page(&self, idx: usize, source: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<&Page> {
        if let Some(page) = self.pages[idx].get() {
            return Ok(page);
        }
        let start = idx * self.page_size;
        let end = (start + self.page_size).min(self.len);
        let values = source.slice(start..end)?.execute::<VarBinViewArray>(ctx)?;
        let validity = values.validity()?.execute_mask(values.len(), ctx)?;
        let page = Page {
            values,
            validity: (!validity.all_true()).then_some(validity),
        };
        // A racing decode of the same page is discarded; both decoded the same values.
        Ok(self.pages[idx].get_or_init(|| page))
    }

    /// Looks up each code in the dictionary `source`, decoding only the pages the codes hit.
    pub(super) fn take(
        &self,
        source: &ArrayRef,
        codes: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let codes_validity = codes.validity()?;
        let codes_mask = codes_validity.execute_mask(codes.len(), ctx)?;

        let mut gather = Gather {
            pages: self,
            source,
            shift: self.page_size.trailing_zeros(),
            slot: vec![UNSEEN; self.pages.len()],
            hit: Vec::new(),
            buffers: Vec::new(),
        };
        let views = match_each_integer_ptype!(codes.ptype(), |C| {
            gather.views(codes.as_slice::<C>(), &codes_mask, ctx)?
        });

        let validity = if source.dtype().is_nullable() {
            match_each_integer_ptype!(codes.ptype(), |C| {
                gather.validity(codes.as_slice::<C>(), &codes_mask)
            })
        } else {
            codes_validity
        };
        tracing::debug!(
            codes = codes.len(),
            hit = gather.hit.len(),
            pages = self.pages.len(),
            "dict paged take"
        );

        let dtype = source
            .dtype()
            .union_nullability(codes.dtype().nullability());
        // SAFETY: every view is either inlined, or a view of a decoded page re-pointed at that
        // page's buffers, which are copied into `buffers` in the same order at the recorded base.
        Ok(
            unsafe {
                VarBinViewArray::new_unchecked(views, gather.buffers.into(), dtype, validity)
            }
            .into_array(),
        )
    }
}

/// The state of one take: which pages its codes have hit so far, and the buffers of those pages.
struct Gather<'a> {
    pages: &'a DictPages,
    source: &'a ArrayRef,
    shift: u32,
    /// For each page, its index into `hit`, or [`UNSEEN`].
    slot: Vec<u32>,
    /// The hit pages, each with the index of its first buffer in `buffers`.
    hit: Vec<(&'a Page, u32)>,
    buffers: Vec<ByteBuffer>,
}

impl<'a> Gather<'a> {
    /// The hit page holding `code`, decoding it the first time any code of this take hits it.
    #[inline]
    fn page_of(&mut self, code: usize, ctx: &mut ExecutionCtx) -> VortexResult<(&'a Page, u32)> {
        let idx = code >> self.shift;
        let slot = self.slot[idx];
        if slot != UNSEEN {
            return Ok(self.hit[slot as usize]);
        }
        let page = self.pages.page(idx, self.source, ctx)?;
        let base = u32::try_from(self.buffers.len())?;
        self.buffers.extend(
            page.values
                .data_buffers()
                .iter()
                .map(|buffer| buffer.as_host().clone()),
        );
        self.slot[idx] = u32::try_from(self.hit.len())?;
        self.hit.push((page, base));
        Ok((page, base))
    }

    fn views<C: NativePType + AsPrimitive<usize>>(
        &mut self,
        codes: &[C],
        codes_mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Buffer<BinaryView>> {
        let low = (1usize << self.shift) - 1;
        let mut views = BufferMut::<BinaryView>::with_capacity(codes.len());
        for (row, &code) in codes.iter().enumerate() {
            if !codes_mask.value(row) {
                views.push(BinaryView::empty_view());
                continue;
            }
            let code: usize = code.as_();
            let (page, base) = self.page_of(code, ctx)?;
            let view = page.values.views()[code & low];
            views.push(if view.is_inlined() {
                view
            } else {
                let view = view.as_view();
                view.with_buffer_and_offset(view.buffer_index + base, view.offset)
                    .into()
            });
        }
        Ok(views.freeze())
    }

    /// The validity of the gathered values: a row is valid when its code and its value are.
    fn validity<C: NativePType + AsPrimitive<usize>>(
        &self,
        codes: &[C],
        codes_mask: &Mask,
    ) -> Validity {
        let low = (1usize << self.shift) - 1;
        let mut valid = BitBufferMut::with_capacity(codes.len());
        for (row, &code) in codes.iter().enumerate() {
            let code: usize = code.as_();
            valid.append(
                codes_mask.value(row) && {
                    let (page, _) = self.hit[self.slot[code >> self.shift] as usize];
                    page.validity
                        .as_ref()
                        .is_none_or(|validity| validity.value(code & low))
                },
            );
        }
        Validity::from(valid.freeze())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DictPages;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn values(len: usize) -> VortexResult<vortex_array::ArrayRef> {
        Ok(VarBinViewArray::from_iter_nullable_str(
            (0..len).map(|i| (i % 7 != 3).then(|| format!("value-{i}-padded-past-inline"))),
        )
        .into_array())
    }

    #[rstest]
    #[case::one_page(PrimitiveArray::new(buffer![5u16, 5, 9], Validity::NonNullable))]
    #[case::two_pages(PrimitiveArray::new(buffer![1u16, 40, 1, 99], Validity::NonNullable))]
    #[case::null_codes(PrimitiveArray::new(
        buffer![70u16, 0, 3, 65],
        Validity::from_iter([true, false, true, true]),
    ))]
    #[case::all_null(PrimitiveArray::new(
        buffer![70u16, 0],
        Validity::from_iter([false, false]),
    ))]
    fn paged_take_matches_dict(#[case] codes: PrimitiveArray) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(130)?;
        let pages = DictPages::new(values.len(), 16);
        let expected = DictArray::try_new(codes.clone().into_array(), values.clone())?;
        // Twice, so the second take reads the cached pages.
        for _ in 0..2 {
            let actual = pages.take(&values, &codes, &mut ctx)?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn dense_codes_decode_every_page_once() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(64)?;
        let pages = DictPages::new(values.len(), 16);
        let codes = PrimitiveArray::new(buffer![0u8, 17, 33, 49, 63, 0], Validity::NonNullable);
        let expected = DictArray::try_new(codes.clone().into_array(), values.clone())?;
        let actual = pages.take(&values, &codes, &mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        assert_eq!(pages.decoded_pages(), pages.len());
        Ok(())
    }

    #[test]
    fn sparse_codes_decode_only_hit_pages() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(130)?;
        let pages = DictPages::new(values.len(), 16);
        let codes = PrimitiveArray::new(buffer![100u8, 101, 3], Validity::NonNullable);
        pages.take(&values, &codes, &mut ctx)?;
        assert_eq!(pages.decoded_pages(), 2);
        Ok(())
    }
}
