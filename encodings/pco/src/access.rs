// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::OnceLock;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArrayExt;
use crate::page_index::LogicalPageIndex;

const RANK_STRIDE: usize = 512;
struct RankMask {
    mask: Mask,
    ranks: OnceLock<Vec<usize>>,
}

impl RankMask {
    fn new(mask: Mask) -> Self {
        Self {
            mask,
            ranks: OnceLock::new(),
        }
    }

    fn boundary_rank(&self, row: usize) -> usize {
        match &self.mask {
            Mask::AllTrue(_) => row,
            Mask::AllFalse(_) => 0,
            Mask::Values(values) => {
                let bits = values.bit_buffer();
                if row <= bits.len() - row {
                    bits.count_range(0, row)
                } else {
                    self.mask.true_count() - bits.count_range(row, bits.len())
                }
            }
        }
    }

    fn rank(&self, row: usize) -> usize {
        match &self.mask {
            Mask::AllTrue(_) => row,
            Mask::AllFalse(_) => 0,
            Mask::Values(values) => {
                if row == self.mask.len() {
                    return self.mask.true_count();
                }
                let ranks = self.ranks.get_or_init(|| {
                    let bits = values.bit_buffer();
                    let mut ranks = Vec::with_capacity(bits.len().div_ceil(RANK_STRIDE));
                    let mut count = 0;
                    for start in (0..bits.len()).step_by(RANK_STRIDE) {
                        ranks.push(count);
                        count += bits.count_range(start, (start + RANK_STRIDE).min(bits.len()));
                    }
                    ranks
                });
                let block = row / RANK_STRIDE;
                ranks[block] + values.bit_buffer().count_range(block * RANK_STRIDE, row)
            }
        }
    }

    fn value_index<const INDEXED: bool>(&self, row: usize) -> Option<usize> {
        match &self.mask {
            Mask::AllTrue(_) => Some(row),
            Mask::AllFalse(_) => None,
            Mask::Values(values) => {
                let bits = values.bit_buffer();
                bits.value(row).then(|| {
                    if INDEXED {
                        self.rank(row)
                    } else {
                        self.boundary_rank(row)
                    }
                })
            }
        }
    }
}

/// Only new-prefix arrays use this operation-owned last-page validity preparation.
#[derive(Default)]
pub(crate) struct AccessState {
    last: Option<(usize, PreparedPage)>,
}
struct PreparedPage {
    rows: Range<usize>,
    mask: RankMask,
}
impl AccessState {
    fn page(
        &mut self,
        view: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(usize, &PreparedPage)> {
        if !self
            .last
            .as_ref()
            .is_some_and(|(_, page)| page.rows.contains(&row))
        {
            let page_index = view.find_row(row);
            let rows = view.rows(page_index);
            let rows = usize::try_from(rows.start)
                .vortex_expect("page row start must fit the validity array")
                ..usize::try_from(rows.end)
                    .vortex_expect("page row end must fit the validity array");
            let mask = array
                .unsliced_validity()
                .slice(rows.clone())?
                .execute_mask(rows.len(), ctx)?;
            self.last = Some((
                page_index,
                PreparedPage {
                    rows,
                    mask: RankMask::new(mask),
                },
            ));
        }
        let (page_index, page) = self.last.as_ref().vortex_expect("page prepared above");
        Ok((*page_index, page))
    }
    pub fn resolve<const RANKED: bool>(
        &mut self,
        view: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(usize, usize)>> {
        let (page_index, page) = self.page(view, array, row, ctx)?;
        Ok(page
            .mask
            .value_index::<RANKED>(row - page.rows.start)
            .map(|value| (page_index, value)))
    }
    pub fn span(
        &mut self,
        view: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        row: usize,
        stop: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PageSpan> {
        let (page_index, page) = self.page(view, array, row, ctx)?;
        let end = stop.min(page.rows.end);
        let start_in_page = row - page.rows.start;
        let end_in_page = end - page.rows.start;
        Ok(PageSpan {
            page_index,
            stop: end,
            values: page.mask.boundary_rank(start_in_page)..page.mask.boundary_rank(end_in_page),
            mask: page.mask.mask.slice(start_in_page..end_in_page),
        })
    }
    pub fn page_id(&self) -> Option<usize> {
        self.last.as_ref().map(|(page_index, _)| *page_index)
    }
    pub fn cached_position(&self, row: usize) -> Option<Option<(usize, usize)>> {
        let (page_index, page) = self.last.as_ref()?;
        if !page.rows.contains(&row) {
            return None;
        }
        Some(
            page.mask
                .value_index::<true>(row - page.rows.start)
                .map(|value| (*page_index, value)),
        )
    }
}
pub(crate) struct PageSpan {
    pub page_index: usize,
    pub stop: usize,
    /// Dense-value positions within this physical page, never global value ordinals.
    pub values: Range<usize>,
    pub mask: Mask,
}
