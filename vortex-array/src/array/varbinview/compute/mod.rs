use arrow_buffer::ScalarBuffer;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, PType};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::varbin::varbin_scalar;
use crate::array::varbinview::{VarBinViewArray, VIEW_SIZE_BYTES};
use crate::array::{PrimitiveArray, VarBinViewEncoding};
use crate::compute::unary::ScalarAtFn;
use crate::compute::{slice, ComputeVTable, SliceFn, TakeFn, TakeOptions};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

impl ComputeVTable for VarBinViewEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<VarBinViewArray> for VarBinViewEncoding {
    fn scalar_at(&self, array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        array
            .bytes_at(index)
            .map(|bytes| varbin_scalar(Buffer::from(bytes), array.dtype()))
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
impl TakeFn<VarBinViewArray> for VarBinViewEncoding {
    fn take(
        &self,
        array: &VarBinViewArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        // Compute the new validity
        let validity = array.validity().take(indices, options)?;

        // Convert our views array into an Arrow u128 ScalarBuffer (16 bytes per view)
        let views_buffer =
            ScalarBuffer::<u128>::from(array.views().into_primitive()?.into_buffer().into_arrow());

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

        Ok(VarBinViewArray::try_new(
            views_array.into_array(),
            array.buffers().collect_vec(),
            array.dtype().clone(),
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

#[cfg(test)]
mod tests {
    use crate::accessor::ArrayAccessor;
    use crate::array::{PrimitiveArray, VarBinViewArray};
    use crate::compute::{take, TakeOptions};
    use crate::{ArrayDType, IntoArrayData, IntoArrayVariant};

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
