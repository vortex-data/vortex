// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::slice::SliceArray;
use crate::arrays::slice::SliceVTable;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

pub(super) const RULES: ReduceRuleSet<SliceVTable> = ReduceRuleSet::new(&[
    &SliceSliceRule,
    // Try the generic VTable::slice first for compressed encodings
    &SliceVTableRule,
    &SliceNullRule,
    &SliceBoolRule,
    &SlicePrimitiveRule,
    &SliceDecimalRule,
    &SliceConstantRule,
    &SliceDictRule,
    &SliceFilterRule,
    &SliceMaskedRule,
    &SliceVarBinRule,
    &SliceVarBinViewRule,
    &SliceScalarFnRule,
    &SliceListRule,
    &SliceListViewRule,
    &SliceFixedSizeListRule,
    &SliceStructRule,
    &SliceExtensionRule,
]);

/// Reduce rule for Slice(Slice(child)) -> Slice(child) with combined ranges
#[derive(Debug)]
struct SliceSliceRule;

impl ArrayReduceRule<SliceVTable> for SliceSliceRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(inner_slice) = array.child().as_opt::<SliceVTable>() else {
            return Ok(None);
        };

        // Combine the ranges: outer range is relative to inner slice
        let outer_range = array.slice_range();
        let inner_range = inner_slice.slice_range();

        let combined_start = inner_range.start + outer_range.start;
        let combined_end = inner_range.start + outer_range.end;

        Ok(Some(
            SliceArray::new(inner_slice.child().clone(), combined_start..combined_end).into_array(),
        ))
    }
}

/// Generic reduce rule that calls VTable::slice on the child.
/// This allows compressed encodings to implement their own slice logic.
#[derive(Debug)]
struct SliceVTableRule;

impl ArrayReduceRule<SliceVTable> for SliceVTableRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        // Try the child's VTable::slice implementation
        array
            .child()
            .encoding()
            .slice(array.child(), array.slice_range().clone())
    }
}

/// Reduce rule for Slice(Null) -> Null
#[derive(Debug)]
struct SliceNullRule;

impl ArrayReduceRule<SliceVTable> for SliceNullRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(_null) = array.child().as_opt::<NullVTable>() else {
            return Ok(None);
        };
        Ok(Some(NullArray::new(array.slice_range().len()).into_array()))
    }
}

/// Reduce rule for Slice(Bool) -> Bool
#[derive(Debug)]
struct SliceBoolRule;

impl ArrayReduceRule<SliceVTable> for SliceBoolRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(bool_arr) = array.child().as_opt::<BoolVTable>() else {
            return Ok(None);
        };
        let range = array.slice_range().clone();
        let result = BoolArray::from_bit_buffer(
            bool_arr.bit_buffer().slice(range.clone()),
            bool_arr.validity().slice(range),
        )
        .into_array();
        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Primitive) -> Primitive
#[derive(Debug)]
struct SlicePrimitiveRule;

impl ArrayReduceRule<SliceVTable> for SlicePrimitiveRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(primitive) = array.child().as_opt::<PrimitiveVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();

        let result = match_each_native_ptype!(primitive.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                primitive.buffer_handle().slice_typed::<T>(range.clone()),
                T::PTYPE,
                primitive.validity().slice(range),
            )
            .into_array()
        });

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Decimal) -> Decimal
#[derive(Debug)]
struct SliceDecimalRule;

impl ArrayReduceRule<SliceVTable> for SliceDecimalRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(decimal) = array.child().as_opt::<DecimalVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();

        let validity = match decimal.validity() {
            v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => v.clone(),
            Validity::Array(arr) => {
                Validity::Array(SliceArray::new(arr.clone(), range.clone()).into_array())
            }
        };

        let result = match_each_decimal_value_type!(decimal.values_type(), |D| {
            let sliced = decimal.buffer::<D>().slice(range);
            // SAFETY: Slicing preserves all DecimalArray invariants
            unsafe { DecimalArray::new_unchecked(sliced, decimal.decimal_dtype(), validity) }
                .into_array()
        });

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(VarBinView) -> VarBinView
#[derive(Debug)]
struct SliceVarBinViewRule;

impl ArrayReduceRule<SliceVTable> for SliceVarBinViewRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(varbinview) = array.child().as_opt::<VarBinViewVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let views = varbinview.views().slice(range.clone());

        let result = VarBinViewArray::new(
            views,
            varbinview.buffers().clone(),
            varbinview.dtype().clone(),
            varbinview.validity().slice(range),
        )
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(List) -> List
#[derive(Debug)]
struct SliceListRule;

impl ArrayReduceRule<SliceVTable> for SliceListRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(list) = array.child().as_opt::<ListVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();

        // List slice keeps elements unchanged, slices offsets with +1 for the extra offset
        let result = ListArray::new(
            list.elements().clone(),
            list.offsets().slice(range.start..range.end + 1),
            list.validity().slice(range),
        )
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(ListView) -> ListView
#[derive(Debug)]
struct SliceListViewRule;

impl ArrayReduceRule<SliceVTable> for SliceListViewRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(listview) = array.child().as_opt::<ListViewVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();

        // SAFETY: Slicing the components of an existing valid array is still valid.
        let result = unsafe {
            ListViewArray::new_unchecked(
                listview.elements().clone(),
                listview.offsets().slice(range.clone()),
                listview.sizes().slice(range.clone()),
                listview.validity().slice(range),
            )
            .with_zero_copy_to_list(listview.is_zero_copy_to_list())
        }
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(FixedSizeList) -> FixedSizeList
#[derive(Debug)]
struct SliceFixedSizeListRule;

impl ArrayReduceRule<SliceVTable> for SliceFixedSizeListRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(fsl) = array.child().as_opt::<FixedSizeListVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let new_len = range.len();
        let list_size = fsl.list_size() as usize;

        // SAFETY: Slicing preserves FixedSizeListArray invariants
        let result = unsafe {
            FixedSizeListArray::new_unchecked(
                fsl.elements()
                    .slice(range.start * list_size..range.end * list_size),
                fsl.list_size(),
                fsl.validity().slice(range),
                new_len,
            )
        }
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Struct) -> Struct
#[derive(Debug)]
struct SliceStructRule;

impl ArrayReduceRule<SliceVTable> for SliceStructRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(struct_arr) = array.child().as_opt::<StructVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let fields = struct_arr
            .fields()
            .iter()
            .map(|field| field.slice(range.clone()))
            .collect_vec();

        // SAFETY: Slicing preserves all StructArray invariants
        let result = unsafe {
            StructArray::new_unchecked(
                fields,
                struct_arr.struct_fields().clone(),
                range.len(),
                struct_arr.validity().slice(range),
            )
        }
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Extension) -> Extension
#[derive(Debug)]
struct SliceExtensionRule;

impl ArrayReduceRule<SliceVTable> for SliceExtensionRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(ext) = array.child().as_opt::<ExtensionVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let result =
            ExtensionArray::new(ext.ext_dtype().clone(), ext.storage().slice(range)).into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Constant) -> Constant
#[derive(Debug)]
struct SliceConstantRule;

impl ArrayReduceRule<SliceVTable> for SliceConstantRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = array.child().as_opt::<ConstantVTable>() else {
            return Ok(None);
        };

        let result =
            ConstantArray::new(constant.scalar().clone(), array.slice_range().len()).into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Dict) -> Dict (or Constant if codes become constant)
#[derive(Debug)]
struct SliceDictRule;

impl ArrayReduceRule<SliceVTable> for SliceDictRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(dict) = array.child().as_opt::<DictVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let sliced_codes = dict.codes().slice(range);

        // If sliced codes become constant, resolve to the actual value
        if sliced_codes.is::<ConstantVTable>() {
            let code = sliced_codes.scalar_at(0).as_primitive().as_::<usize>();
            return Ok(Some(if let Some(code) = code {
                ConstantArray::new(dict.values().scalar_at(code), sliced_codes.len()).into_array()
            } else {
                ConstantArray::new(
                    vortex_scalar::Scalar::null(array.dtype().clone()),
                    sliced_codes.len(),
                )
                .into_array()
            }));
        }

        // SAFETY: slicing the codes preserves invariants
        let result =
            unsafe { DictArray::new_unchecked(sliced_codes, dict.values().clone()).into_array() };

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Filter) -> Filter
#[derive(Debug)]
struct SliceFilterRule;

// TODO(joe): review this rule maybe only have a execute parent.
impl ArrayReduceRule<SliceVTable> for SliceFilterRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(filter) = array.child().as_opt::<FilterVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let mask = filter.filter_mask();

        // range refers to filtered positions (0..true_count), not raw positions
        // We need to find the raw positions corresponding to these filtered positions
        assert!(
            range.end <= mask.true_count(),
            "slice end {} exceeds filter true_count {}",
            range.end,
            mask.true_count()
        );
        let start_raw = mask.rank(range.start);
        let end_raw = if range.end == mask.true_count() {
            mask.len()
        } else {
            mask.rank(range.end)
        };

        let result = FilterArray::new(
            filter.child().slice(start_raw..end_raw),
            mask.slice(start_raw..end_raw),
        )
        .into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(Masked) -> Masked
#[derive(Debug)]
struct SliceMaskedRule;

impl ArrayReduceRule<SliceVTable> for SliceMaskedRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(masked) = array.child().as_opt::<MaskedVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let child = masked.child().slice(range.clone());
        let validity = masked.validity().slice(range);

        let result = MaskedArray::try_new(child, validity)?.into_array();

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(VarBin) -> VarBin
#[derive(Debug)]
struct SliceVarBinRule;

impl ArrayReduceRule<SliceVTable> for SliceVarBinRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(varbin) = array.child().as_opt::<VarBinVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();

        // SAFETY: slicing preserves VarBinArray invariants
        let result = unsafe {
            VarBinArray::new_unchecked(
                varbin.offsets().slice(range.start..range.end + 1),
                varbin.bytes().clone(),
                varbin.dtype().clone(),
                varbin.validity().slice(range),
            )
            .into_array()
        };

        Ok(Some(result))
    }
}

/// Reduce rule for Slice(ScalarFn) -> ScalarFn
#[derive(Debug)]
struct SliceScalarFnRule;

impl ArrayReduceRule<SliceVTable> for SliceScalarFnRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        let Some(scalar_fn_arr) = array.child().as_opt::<ScalarFnVTable>() else {
            return Ok(None);
        };

        let range = array.slice_range().clone();
        let children: Vec<_> = scalar_fn_arr
            .children()
            .iter()
            .map(|c| c.slice(range.clone()))
            .collect();

        let result =
            ScalarFnArray::try_new(scalar_fn_arr.scalar_fn().clone(), children, range.len())?
                .into_array();

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::SliceFilterRule;
    use super::SliceSliceRule;
    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::arrays::SliceVTable;
    use crate::assert_arrays_eq;
    use crate::optimizer::rules::ArrayReduceRule;

    #[test]
    fn test_slice_slice() -> VortexResult<()> {
        // Slice(1..4, Slice(2..8, base)) combines to Slice(3..6, base)
        let arr = PrimitiveArray::from_iter(0i32..10).into_array();
        let inner_slice = SliceArray::new(arr, 2..8).into_array();
        let outer_slice = SliceArray::new(inner_slice, 1..4);

        let result = SliceSliceRule.reduce(&outer_slice)?;
        assert!(result.is_some());

        let reduced = result.unwrap();
        assert_eq!(reduced.as_::<SliceVTable>().slice_range(), &(3..6));
        assert_arrays_eq!(reduced, PrimitiveArray::from_iter([3i32, 4, 5]));

        Ok(())
    }

    #[test]
    fn test_slice_filter_basic() -> VortexResult<()> {
        // Tests rank() conversion from filtered positions to raw positions
        let child = PrimitiveArray::from_iter(0i32..6).into_array();
        let mask = Mask::from_iter([true, false, true, true, false, true]);
        let filter = FilterArray::new(child, mask).into_array();

        let slice = SliceArray::new(filter, 1..3);
        let result = SliceFilterRule.reduce(&slice)?;

        assert!(result.is_some());
        assert_arrays_eq!(result.unwrap(), PrimitiveArray::from_iter([2i32, 3]));

        Ok(())
    }

    #[test]
    fn test_slice_filter_to_end() -> VortexResult<()> {
        // Tests end boundary: when range.end == true_count, uses mask.len() not rank()
        let child = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]); // true_count = 3
        let filter = FilterArray::new(child, mask).into_array();

        let slice = SliceArray::new(filter, 1..3); // end == true_count
        let result = SliceFilterRule.reduce(&slice)?;

        assert!(result.is_some());
        assert_arrays_eq!(result.unwrap(), PrimitiveArray::from_iter([30i32, 50]));

        Ok(())
    }

    #[test]
    fn test_slice_filter_sparse_mask() -> VortexResult<()> {
        // Tests rank() on sparse mask where it must walk many false values
        let child = PrimitiveArray::from_iter(0i32..20).into_array();
        let mask = Mask::from_iter([
            false, false, false, true, // pos 3
            false, false, false, false, false, false, true, // pos 10
            false, false, false, false, true, // pos 15
            false, false, false, true, // pos 19
        ]);
        let filter = FilterArray::new(child, mask).into_array();

        let slice = SliceArray::new(filter, 1..3);
        let result = SliceFilterRule.reduce(&slice)?;

        assert!(result.is_some());
        assert_arrays_eq!(result.unwrap(), PrimitiveArray::from_iter([10i32, 15]));

        Ok(())
    }
}
