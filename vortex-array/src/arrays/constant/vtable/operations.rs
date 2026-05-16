// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Constant;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

impl OperationsVTable<Constant> for Constant {
    fn scalar_at(
        array: ArrayView<'_, Constant>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }

    /// `search_sorted` on a constant array is O(1): compare the search value
    /// against the constant once and decide.
    fn point_search_sorted(
        array: ArrayView<'_, Constant>,
        value: &Scalar,
        side: SearchSortedSide,
        _d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let len = array.as_ref().len();
        Ok(match array.scalar.partial_cmp(value) {
            Some(Ordering::Equal) => match side {
                SearchSortedSide::Left => SearchResult::Found(0),
                SearchSortedSide::Right => SearchResult::Found(len),
            },
            Some(Ordering::Less) => SearchResult::NotFound(len),
            // Greater or None (incomparable: e.g. nulls) → insert before all.
            _ => SearchResult::NotFound(0),
        })
    }
}
