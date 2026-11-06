// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Page-at-a-time decoding of string dictionary values.
//!
//! A dictionary's values are shared by every chunk of codes that references them, so the
//! reader normally decodes all of them once. When a selective filter leaves only a few codes in
//! a chunk, that decodes values no row needs. Instead, the values are split into fixed-size pages
//! that are decoded the first time any code hits them and cached for later chunks.
//!
//! A projection first checks which pages its codes hit, in one pass over the codes. If they hit
//! only a few, it decodes just those pages and gathers each code's view straight out of its page,
//! sharing the pages' byte buffers rather than copying them. If they hit most pages, it decodes
//! the values whole, through the same shared decode as without paging, so dense codes cost only
//! that one extra pass over the codes.

use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

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
use vortex_error::VortexResult;
use vortex_mask::AllOr;
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
    /// Set once a take hit most pages, after which every take uses the whole values.
    whole: AtomicBool,
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
            whole: AtomicBool::new(false),
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
    ///
    /// Returns [`PagedTake::Whole`] instead once a take hits more than half the pages: decoding
    /// the values whole then costs no more than decoding those pages, and the caller can keep the
    /// codes over the whole values lazily, exactly as without paging. From then on every take of
    /// this dictionary uses the whole values without checking its codes.
    pub(super) fn take(
        &self,
        source: &ArrayRef,
        codes: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PagedTake> {
        if self.whole.load(Ordering::Relaxed) {
            return Ok(PagedTake::Whole);
        }
        let codes_validity = codes.validity()?;
        let codes_mask = codes_validity.execute_mask(codes.len(), ctx)?;
        let shift = self.page_size.trailing_zeros();

        let mut hit = vec![false; self.pages.len()];
        match_each_integer_ptype!(codes.ptype(), |C| {
            mark_hits(codes.as_slice::<C>(), &codes_mask, shift, &mut hit)
        });
        let hits = hit.iter().filter(|&&h| h).count();
        tracing::debug!(
            codes = codes.len(),
            hit = hits,
            pages = self.pages.len(),
            "dict paged take"
        );
        if hits * 2 > self.pages.len() {
            self.whole.store(true, Ordering::Relaxed);
            return Ok(PagedTake::Whole);
        }

        // Decode the hit pages, and lay out their buffers one after another.
        let mut table = vec![PageRef::default(); self.pages.len()];
        let mut buffers = Vec::new();
        for (idx, _) in hit.iter().enumerate().filter(|(_, h)| **h) {
            let page = self.page(idx, source, ctx)?;
            table[idx] = PageRef {
                views: page.values.views(),
                base: u32::try_from(buffers.len())?,
                validity: page.validity.as_ref(),
            };
            buffers.extend(
                page.values
                    .data_buffers()
                    .iter()
                    .map(|buffer| buffer.as_host().clone()),
            );
        }

        let gather = Gather { table, shift };
        let views = match_each_integer_ptype!(codes.ptype(), |C| {
            gather.views(codes.as_slice::<C>(), &codes_mask)
        });
        let validity = if source.dtype().is_nullable() {
            match_each_integer_ptype!(codes.ptype(), |C| {
                gather.validity(codes.as_slice::<C>(), &codes_mask)
            })
        } else {
            codes_validity
        };

        let dtype = source
            .dtype()
            .union_nullability(codes.dtype().nullability());
        // SAFETY: every view is either inlined, or a view of a decoded page re-pointed at that
        // page's buffers, which are laid out in `buffers` from the page's recorded base.
        Ok(PagedTake::Gathered(
            unsafe { VarBinViewArray::new_unchecked(views, buffers.into(), dtype, validity) }
                .into_array(),
        ))
    }
}

/// The result of [`DictPages::take`].
pub(super) enum PagedTake {
    /// The codes' values, gathered from the pages they hit.
    Gathered(ArrayRef),
    /// The codes hit most pages; look them up in the whole decoded values instead.
    Whole,
}

/// Marks the page of every valid code as hit.
fn mark_hits<C: NativePType + AsPrimitive<usize>>(
    codes: &[C],
    codes_mask: &Mask,
    shift: u32,
    hit: &mut [bool],
) {
    match codes_mask.indices() {
        AllOr::All => codes.iter().for_each(|&c| hit[c.as_() >> shift] = true),
        AllOr::None => {}
        AllOr::Some(valid) => valid
            .iter()
            .for_each(|&i| hit[codes[i].as_() >> shift] = true),
    }
}

/// A decoded page as one take sees it. Pages the take does not hit stay empty.
#[derive(Clone, Copy, Default)]
struct PageRef<'a> {
    views: &'a [BinaryView],
    /// Index in the take's buffers of this page's first buffer.
    base: u32,
    /// The page's validity, or `None` when every value is valid.
    validity: Option<&'a Mask>,
}

/// Gathers values from the pages a take hits.
struct Gather<'a> {
    table: Vec<PageRef<'a>>,
    shift: u32,
}

impl Gather<'_> {
    #[inline]
    fn view(&self, code: usize) -> BinaryView {
        let page = &self.table[code >> self.shift];
        let view = page.views[code & ((1 << self.shift) - 1)];
        if view.is_inlined() {
            view
        } else {
            let view = view.as_view();
            view.with_buffer_and_offset(view.buffer_index + page.base, view.offset)
                .into()
        }
    }

    fn views<C: NativePType + AsPrimitive<usize>>(
        &self,
        codes: &[C],
        codes_mask: &Mask,
    ) -> Buffer<BinaryView> {
        match codes_mask.bit_buffer() {
            AllOr::All => codes.iter().map(|&code| self.view(code.as_())).collect(),
            AllOr::None => Buffer::full(BinaryView::empty_view(), codes.len()),
            AllOr::Some(valid) => {
                let mut views = BufferMut::full(BinaryView::empty_view(), codes.len());
                valid.for_each_set_index(|row| views[row] = self.view(codes[row].as_()));
                views.freeze()
            }
        }
    }

    /// The validity of the gathered values: a row is valid when its code and its value are.
    fn validity<C: NativePType + AsPrimitive<usize>>(
        &self,
        codes: &[C],
        codes_mask: &Mask,
    ) -> Validity {
        let low = (1usize << self.shift) - 1;
        let value_valid = |code: usize| {
            self.table[code >> self.shift]
                .validity
                .is_none_or(|validity| validity.value(code & low))
        };
        let mut valid = BitBufferMut::new_unset(codes.len());
        match codes_mask.bit_buffer() {
            AllOr::All => codes.iter().enumerate().for_each(|(row, &code)| {
                if value_valid(code.as_()) {
                    valid.set(row);
                }
            }),
            AllOr::None => {}
            AllOr::Some(codes_valid) => codes_valid.for_each_set_index(|row| {
                if value_valid(codes[row].as_()) {
                    valid.set(row);
                }
            }),
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
    use super::PagedTake;

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
            let PagedTake::Gathered(actual) = pages.take(&values, &codes, &mut ctx)? else {
                panic!("sparse codes should be gathered from pages");
            };
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn dense_codes_use_whole_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(64)?;
        let pages = DictPages::new(values.len(), 16);
        let dense = PrimitiveArray::new(buffer![0u8, 17, 33, 49], Validity::NonNullable);
        assert!(matches!(
            pages.take(&values, &dense, &mut ctx)?,
            PagedTake::Whole
        ));
        assert_eq!(pages.decoded_pages(), 0);
        // Once whole, sparse codes use the whole values too.
        let sparse = PrimitiveArray::new(buffer![1u8], Validity::NonNullable);
        assert!(matches!(
            pages.take(&values, &sparse, &mut ctx)?,
            PagedTake::Whole
        ));
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
