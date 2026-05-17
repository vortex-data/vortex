// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::dict::DictArraySlotsExt;
use crate::match_each_integer_ptype;
use crate::point_fn::PointDispatch;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

impl OperationsVTable<Dict> for Dict {
    fn scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let Some(dict_index) = array
            .codes()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        Ok(array
            .values()
            .execute_scalar(dict_index, ctx)?
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }

    // TODO(point-fn migration): port these point_scalar_at / point_search_sorted
    // overrides to ScalarAtKernel / SearchSortedKernel impls registered via
    // `point_kernels()`. Coexists with the kernel-per-op pattern; no
    // behavioural change blocking this.
    /// Recurse via the dispatch: read the code at `index` from `codes`, then look
    /// up the corresponding dict value. Both child calls hit the session's caches.
    fn point_scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let Some(dict_index) = d
            .scalar_at(array.codes(), index)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(array.dtype().clone()));
        };
        Ok(d.scalar_at(array.values(), dict_index)?
            .cast(array.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }

    /// `search_sorted` on a Dict whose dict (values) **and** codes are both sorted:
    /// search the (typically tiny) dict first, then translate to codes-space.
    ///
    /// Precondition (caller's responsibility): both `dict.values()` and
    /// `dict.codes()` are individually sorted, so the logical array is sorted.
    /// The default `OperationsVTable::point_search_sorted` (generic binary
    /// search) still works correctly for any Dict shape; this override is a
    /// strict speedup only when the precondition holds.
    fn point_search_sorted(
        array: ArrayView<'_, Dict>,
        value: &Scalar,
        side: SearchSortedSide,
        dispatch: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let dict = array.values();
        let codes = array.codes();
        // Always search dict with Left side to get the canonical matching dict
        // index (or insertion point). Apply the original `side` on the codes
        // search to pick the correct boundary in codes-space.
        let dict_search = dispatch.search_sorted(dict, value, SearchSortedSide::Left)?;

        match dict_search {
            SearchResult::Found(matching_dict_idx) => {
                // Target value is in the dict at this index. Find the run of
                // rows that map to this code: codes.search_sorted(_, side).
                let code_scalar = usize_as_scalar(codes, matching_dict_idx)?;
                dispatch.search_sorted(codes, &code_scalar, side)
            }
            SearchResult::NotFound(insertion_dict_idx) => {
                // Target value is missing from the dict. Insertion would be at
                // dict[insertion_dict_idx]. In codes-space the insertion point
                // is the first row whose code is ≥ insertion_dict_idx — i.e.
                // codes.search_sorted(insertion_dict_idx, Left).
                if insertion_dict_idx == 0 {
                    return Ok(SearchResult::NotFound(0));
                }
                let code_scalar = usize_as_scalar(codes, insertion_dict_idx)?;
                let result = dispatch.search_sorted(codes, &code_scalar, SearchSortedSide::Left)?;
                Ok(SearchResult::NotFound(result.to_index()))
            }
        }
    }
}

/// Construct a Scalar of the codes array's dtype from a usize.
///
/// Dict codes are always an integer ptype.
fn usize_as_scalar(codes: &ArrayRef, value: usize) -> VortexResult<Scalar> {
    let ptype = codes.dtype().as_ptype();
    let pvalue = match_each_integer_ptype!(ptype, |P| {
        let v: P = <P as num_traits::FromPrimitive>::from_usize(value).ok_or_else(|| {
            vortex_error::vortex_err!("usize {} out of range for {:?}", value, ptype)
        })?;
        PValue::from(v)
    });
    Scalar::try_new(codes.dtype().clone(), Some(ScalarValue::Primitive(pvalue)))
}
