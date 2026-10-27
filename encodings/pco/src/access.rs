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
            .is_some_and(|(_, cached)| cached.rows.contains(&row))
        {
            let page = view.find_row(row);
            let rows = view.rows(page);
            let rows = usize::try_from(rows.start)
                .vortex_expect("page row start must fit the validity array")
                ..usize::try_from(rows.end)
                    .vortex_expect("page row end must fit the validity array");
            let mask = array
                .unsliced_validity()
                .slice(rows.clone())?
                .execute_mask(rows.len(), ctx)?;
            self.last = Some((
                page,
                PreparedPage {
                    rows,
                    mask: RankMask::new(mask),
                },
            ));
        }
        let (page, cached) = self.last.as_ref().vortex_expect("page prepared above");
        Ok((*page, cached))
    }
    pub fn resolve<const RANKED: bool>(
        &mut self,
        view: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(usize, usize)>> {
        let (page, cached) = self.page(view, array, row, ctx)?;
        Ok(cached
            .mask
            .value_index::<RANKED>(row - cached.rows.start)
            .map(|value| (page, value)))
    }
}
