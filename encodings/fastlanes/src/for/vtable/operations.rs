// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::point_fn::PointDispatch;
use vortex_array::point_fn::algorithms::generic_search_sorted;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::FoR;
use crate::r#for::array::FoRArrayExt;
impl OperationsVTable<FoR> for FoR {
    fn scalar_at(
        array: ArrayView<'_, FoR>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let encoded_pvalue = array.encoded().execute_scalar(index, ctx)?;
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = array.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(array.ptype(), |P| {
            encoded_pvalue
                .typed_value::<P>()
                .map(|v| {
                    v.wrapping_add(
                        reference
                            .typed_value::<P>()
                            .vortex_expect("FoRArray Reference value cannot be null"),
                    )
                })
                .map(|v| Scalar::primitive::<P>(v, array.reference_scalar().dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.reference_scalar().dtype().clone()))
        }))
    }

    // TODO(point-fn migration): port these point_scalar_at / point_search_sorted
    // overrides to ScalarAtKernel / SearchSortedKernel impls registered via
    // `point_kernels()`. Coexists with the kernel-per-op pattern; no
    // behavioural change blocking this.
    /// Recurse: read the encoded delta via the dispatch, then add the
    /// reference value.
    fn point_scalar_at(
        array: ArrayView<'_, FoR>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let encoded_pvalue = d.scalar_at(array.encoded(), index)?;
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = array.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(array.ptype(), |P| {
            encoded_pvalue
                .typed_value::<P>()
                .map(|v| {
                    v.wrapping_add(
                        reference
                            .typed_value::<P>()
                            .vortex_expect("FoRArray Reference value cannot be null"),
                    )
                })
                .map(|v| Scalar::primitive::<P>(v, array.reference_scalar().dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.reference_scalar().dtype().clone()))
        }))
    }

    /// Push search into encoded-delta space: subtract reference from target
    /// once, then run search_sorted on `encoded`. Strictly equivalent to the
    /// generic search-via-scalar_at, but avoids the per-probe add and lets
    /// the encoded child run its own (possibly faster) search.
    fn point_search_sorted(
        array: ArrayView<'_, FoR>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let reference = array.reference_scalar().as_primitive();
        let target = value.as_primitive();
        let ptype = array.ptype();

        // If the target is null, defer to the generic default which knows how
        // to treat unordered values during binary search.
        let delta_pvalue = match_each_integer_ptype!(ptype, |P| {
            let Some(target_v) = target.typed_value::<P>() else {
                return generic_search_sorted(array.as_ref(), value, side, d);
            };
            let ref_v = reference
                .typed_value::<P>()
                .vortex_expect("FoRArray reference cannot be null");
            PValue::from(target_v.wrapping_sub(ref_v))
        });

        let delta_scalar = Scalar::try_new(
            array.encoded().dtype().clone(),
            Some(ScalarValue::Primitive(delta_pvalue)),
        )?;
        d.search_sorted(array.encoded(), &delta_scalar, side)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::FoRData;

    #[test]
    fn for_scalar_at() {
        let for_arr = FoRData::encode(PrimitiveArray::from_iter([-100, 1100, 1500, 1900])).unwrap();
        let expected = PrimitiveArray::from_iter([-100, 1100, 1500, 1900]);
        assert_arrays_eq!(for_arr, expected);
    }
}
