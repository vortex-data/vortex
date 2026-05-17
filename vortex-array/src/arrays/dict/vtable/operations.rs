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

    fn point_kernels() -> Option<&'static crate::point_fn::PointKernels<Dict>> {
        Some(&POINT_KERNELS)
    }
}

const POINT_KERNELS: crate::point_fn::PointKernels<Dict> = crate::point_fn::PointKernels::empty()
    .with_scalar_at(crate::point_fn::PointKernels::lift_scalar_at(
        &DictScalarAtKernel,
    ))
    .with_search_sorted(crate::point_fn::PointKernels::lift_search_sorted(
        &DictSearchSortedKernel,
    ));

/// Recurse via the dispatch: read the code at `index` from `codes`, then look
/// up the corresponding dict value. Both child calls hit the session's caches.
struct DictScalarAtKernel;

impl crate::point_fn::ScalarAtKernel<Dict> for DictScalarAtKernel {
    fn execute(
        view: ArrayView<'_, Dict>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let Some(dict_index) = d
            .scalar_at(view.codes(), index)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(view.dtype().clone()));
        };
        Ok(d.scalar_at(view.values(), dict_index)?
            .cast(view.dtype())
            .vortex_expect("Array dtype will only differ by nullability"))
    }
}

/// `search_sorted` on a Dict whose dict (values) **and** codes are both sorted:
/// search the (typically tiny) dict first, then translate to codes-space.
///
/// Precondition (caller's responsibility): both `dict.values()` and
/// `dict.codes()` are individually sorted, so the logical array is sorted.
/// The default generic binary search still works correctly for any Dict
/// shape; this override is a strict speedup only when the precondition holds.
struct DictSearchSortedKernel;

impl crate::point_fn::SearchSortedKernel<Dict> for DictSearchSortedKernel {
    fn execute(
        view: ArrayView<'_, Dict>,
        value: &Scalar,
        side: SearchSortedSide,
        dispatch: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let dict = view.values();
        let codes = view.codes();
        // Always search dict with Left side to get the canonical matching dict
        // index (or insertion point). Apply the original `side` on the codes
        // search to pick the correct boundary in codes-space.
        let dict_search = dispatch.search_sorted(dict, value, SearchSortedSide::Left)?;

        match dict_search {
            SearchResult::Found(matching_dict_idx) => {
                let code_scalar = usize_as_scalar(codes, matching_dict_idx)?;
                dispatch.search_sorted(codes, &code_scalar, side)
            }
            SearchResult::NotFound(insertion_dict_idx) => {
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
