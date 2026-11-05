// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Page-at-a-time decoding of dictionary values.
//!
//! A dictionary's values are shared by every chunk of codes that references them, so the
//! reader normally decodes all of them once. When a selective filter leaves only a few codes in
//! a chunk, that decodes values no row needs. Instead, the values are split into fixed-size pages
//! that are decoded the first time any code hits them and cached for later chunks. A projection
//! then builds a dictionary over just the hit pages, with its codes renumbered to match.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

/// Number of values per page, or `0` to always decode the values whole.
pub(super) static PAGE_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("VORTEX_DICT_PAGE_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&size| size > 0)
        .map_or(0, usize::next_power_of_two)
});

/// Paged decoding is used only when at most `1 / MAX_HIT_DENOMINATOR` of the pages are hit;
/// otherwise decoding the values whole is as cheap and avoids renumbering the codes.
static MAX_HIT_DENOMINATOR: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("VORTEX_DICT_PAGE_MAX_HIT_DENOM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
});

/// Lazily decoded pages of one dictionary's values.
pub(super) struct DictPages {
    page_size: usize,
    len: usize,
    pages: Box<[OnceLock<ArrayRef>]>,
}

impl Drop for DictPages {
    fn drop(&mut self) {
        let decoded = self.pages.iter().filter(|p| p.get().is_some()).count();
        tracing::debug!(
            pages = self.pages.len(),
            decoded,
            len = self.len,
            "dict pages dropped"
        );
    }
}

impl DictPages {
    pub(super) fn new(len: usize, page_size: usize) -> Arc<Self> {
        Arc::new(Self {
            page_size,
            len,
            pages: (0..len.div_ceil(page_size)).map(|_| OnceLock::new()).collect(),
        })
    }

    fn page(
        &self,
        idx: usize,
        source: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(page) = self.pages[idx].get() {
            return Ok(page.clone());
        }
        let start = idx * self.page_size;
        let end = (start + self.page_size).min(self.len);
        let decoded = source
            .slice(start..end)?
            .execute::<Canonical>(ctx)?
            .into_array();
        Ok(self.pages[idx].get_or_init(|| decoded).clone())
    }

    /// Builds a dictionary over only the pages `codes` hits, or returns `None` when too many
    /// pages are hit for paging to pay off.
    pub(super) fn take(
        &self,
        source: &ArrayRef,
        codes: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = codes.validity()?;
        let mask = validity.execute_mask(codes.len(), ctx)?;

        // Rank of each hit page among the hit pages, or `u32::MAX` for a page no code hits.
        let mut rank = vec![u32::MAX; self.pages.len()];
        let shift = self.page_size.trailing_zeros();
        debug_assert!(self.page_size.is_power_of_two());
        match_each_integer_ptype!(codes.ptype(), |C| {
            mark_hits(codes.as_slice::<C>(), &mask, shift, &mut rank)
        });

        let mut hit = Vec::new();
        for (page, r) in rank.iter_mut().enumerate() {
            if *r != u32::MAX {
                *r = u32::try_from(hit.len())?;
                hit.push(page);
            }
        }
        let fallback = hit.len() * *MAX_HIT_DENOMINATOR > self.pages.len();
        tracing::debug!(
            codes = codes.len(),
            hit = hit.len(),
            pages = self.pages.len(),
            fallback,
            "dict paged take"
        );
        if fallback {
            return Ok(None);
        }

        let new_codes = match_each_integer_ptype!(codes.ptype(), |C| {
            PrimitiveArray::new(
                remap(codes.as_slice::<C>(), shift, &rank),
                validity,
            )
        });

        let pages = hit
            .iter()
            .map(|&idx| self.page(idx, source, ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let values = if pages.len() == 1 {
            pages.into_iter().next().unwrap_or_else(|| unreachable!())
        } else {
            ChunkedArray::try_new(pages, source.dtype().clone())?.into_array()
        };

        // SAFETY: every valid code is renumbered into a hit page, and null codes to 0, which is in
        // bounds whenever any page is hit. With no page hit, every code is null.
        let values = if values.is_empty() {
            source.slice(0..source.len().min(1))?
        } else {
            values
        };
        Ok(Some(
            unsafe { DictArray::new_unchecked(new_codes.into_array(), values) }.into_array(),
        ))
    }
}

fn mark_hits<C: NativePType + AsPrimitive<usize>>(
    codes: &[C],
    mask: &vortex_mask::Mask,
    shift: u32,
    rank: &mut [u32],
) {
    match mask.indices() {
        AllOr::All => codes.iter().for_each(|&c| rank[c.as_() >> shift] = 0),
        AllOr::None => {}
        AllOr::Some(valid) => valid
            .iter()
            .for_each(|&i| rank[codes[i].as_() >> shift] = 0),
    }
}

fn remap<C>(codes: &[C], shift: u32, rank: &[u32]) -> Buffer<C>
where
    C: NativePType + AsPrimitive<usize>,
    usize: AsPrimitive<C>,
{
    let low = (1usize << shift) - 1;
    codes
        .iter()
        .map(|&c| {
            let c: usize = c.as_();
            let r = rank.get(c >> shift).copied().unwrap_or(u32::MAX);
            if r == u32::MAX {
                C::default()
            } else {
                (((r as usize) << shift) | (c & low)).as_()
            }
        })
        .collect()
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
            let actual = pages
                .take(&values, &codes, &mut ctx)?
                .expect("sparse codes should be paged");
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn dense_codes_fall_back() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(64)?;
        let pages = DictPages::new(values.len(), 16);
        let codes = PrimitiveArray::new(buffer![0u8, 17, 33, 49], Validity::NonNullable);
        assert!(pages.take(&values, &codes, &mut ctx)?.is_none());
        Ok(())
    }
}
