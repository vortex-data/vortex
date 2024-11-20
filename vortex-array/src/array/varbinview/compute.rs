use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ByteViewType;
use arrow_array::{Datum, GenericByteViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_ord::cmp;
use arrow_schema::DataType;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, PType};
use vortex_error::{vortex_bail, VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;

use crate::array::varbin::varbin_scalar;
use crate::array::varbinview::{VarBinViewArray, VIEW_SIZE_BYTES};
use crate::array::{ConstantArray, PrimitiveArray, VarBinViewEncoding};
use crate::arrow::FromArrowArray;
use crate::compute::unary::ScalarAtFn;
use crate::compute::{
    slice, ArrayCompute, ComputeVTable, MaybeCompareFn, Operator, SliceFn, TakeFn, TakeOptions,
};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, IntoCanonical};

impl ArrayCompute for VarBinViewArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ComputeVTable for VarBinViewEncoding {
    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn for VarBinViewArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.bytes_at(index)
            .map(|bytes| varbin_scalar(Buffer::from(bytes), self.dtype()))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        <Self as ScalarAtFn>::scalar_at(self, index).vortex_unwrap()
    }
}

impl SliceFn<VarBinViewArray> for VarBinViewEncoding {
    fn slice(&self, array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(VarBinViewArray::try_new(
            slice(
                array.views(),
                start * VIEW_SIZE_BYTES,
                stop * VIEW_SIZE_BYTES,
            )?,
            (0..array.metadata().buffer_lens.len())
                .map(|i| array.buffer(i))
                .collect::<Vec<_>>(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeFn for VarBinViewArray {
    fn take(&self, indices: &ArrayData, options: TakeOptions) -> VortexResult<ArrayData> {
        // Compute the new validity
        let validity = self.validity().take(indices, options)?;

        // Convert our views array into an Arrow u128 ScalarBuffer (16 bytes per view)
        let views_buffer =
            ScalarBuffer::<u128>::from(self.views().into_primitive()?.into_buffer().into_arrow());

        let indices = indices.clone().into_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            if options.skip_bounds_check {
                take_views_unchecked(views_buffer, indices.maybe_null_slice::<$I>())
            } else {
                take_views(views_buffer, indices.maybe_null_slice::<$I>())
            }
        });

        // Cast views back to u8
        let views_array = PrimitiveArray::new(
            views_buffer.into_inner().into(),
            PType::U8,
            Validity::NonNullable,
        );

        Ok(Self::try_new(
            views_array.into_array(),
            self.buffers().collect_vec(),
            self.dtype().clone(),
            validity,
        )?
        .into_array())
    }
}

fn take_views<I: AsPrimitive<usize>>(
    views: ScalarBuffer<u128>,
    indices: &[I],
) -> ScalarBuffer<u128> {
    ScalarBuffer::<u128>::from_iter(indices.iter().map(|i| views[i.as_()]))
}

fn take_views_unchecked<I: AsPrimitive<usize>>(
    views: ScalarBuffer<u128>,
    indices: &[I],
) -> ScalarBuffer<u128> {
    ScalarBuffer::<u128>::from_iter(
        indices
            .iter()
            .map(|i| unsafe { *views.get_unchecked(i.as_()) }),
    )
}

impl MaybeCompareFn for VarBinViewArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        other.as_constant().map(|rhs_const| {
            compare_constant(self, &ConstantArray::new(rhs_const, self.len()), operator)
        })
    }
}

fn compare_constant(
    lhs: &VarBinViewArray,
    rhs: &ConstantArray,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let arrow_lhs = lhs.clone().into_canonical()?.into_arrow()?;
    let constant = Arc::<dyn Datum>::try_from(&rhs.owned_scalar())?;

    match arrow_lhs.data_type() {
        DataType::BinaryView => {
            compare_constant_arrow(arrow_lhs.as_binary_view(), constant, operator)
        }
        DataType::Utf8View => {
            compare_constant_arrow(arrow_lhs.as_string_view(), constant, operator)
        }
        _ => {
            vortex_bail!("Cannot compare VarBinViewArray with non-binary type");
        }
    }
}

fn compare_constant_arrow<T: ByteViewType>(
    lhs: &GenericByteViewArray<T>,
    rhs: Arc<dyn Datum>,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let rhs = rhs.as_ref();
    let array = match operator {
        Operator::Eq => cmp::eq(lhs, rhs)?,
        Operator::NotEq => cmp::neq(lhs, rhs)?,
        Operator::Gt => cmp::gt(lhs, rhs)?,
        Operator::Gte => cmp::gt_eq(lhs, rhs)?,
        Operator::Lt => cmp::lt(lhs, rhs)?,
        Operator::Lte => cmp::lt_eq(lhs, rhs)?,
    };
    Ok(ArrayData::from_arrow(&array, true))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::accessor::ArrayAccessor;
    use crate::array::varbinview::compute::compare_constant;
    use crate::array::{ConstantArray, PrimitiveArray, VarBinViewArray};
    use crate::compute::{take, Operator, TakeOptions};
    use crate::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant};

    #[test]
    fn basic_test() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            Some("two"),
            Some("three"),
            Some("four"),
            Some("five"),
            Some("six"),
        ]);

        let s = Scalar::utf8("seven".to_string(), Nullability::Nullable);

        let constant_array = ConstantArray::new(s, arr.len());

        let r = compare_constant(&arr, &constant_array, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        assert!(r.boolean_buffer().iter().all(|v| !v));
    }

    #[test]
    fn take_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = take(
            arr,
            PrimitiveArray::from(vec![0, 3]).into_array(),
            TakeOptions::default(),
        )
        .unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .into_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }
}
