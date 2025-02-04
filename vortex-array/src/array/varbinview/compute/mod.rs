mod min_max;
mod to_arrow;

use std::ops::Deref;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, DType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use super::BinaryView;
use crate::array::varbin::varbin_scalar;
use crate::array::varbinview::VarBinViewArray;
use crate::array::VarBinViewEncoding;
use crate::compute::{CastFn, MaskFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::variants::PrimitiveArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray, IntoArrayVariant};

impl ComputeVTable for VarBinViewEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<VarBinViewArray> for VarBinViewEncoding {
    fn scalar_at(&self, array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }
}

impl SliceFn<VarBinViewArray> for VarBinViewEncoding {
    fn slice(&self, array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<Array> {
        let views = array.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            (0..array.nbuffers())
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
    fn take(&self, array: &VarBinViewArray, indices: &Array) -> VortexResult<Array> {
        // Compute the new validity
        let validity = array.validity().take(indices)?;
        let indices = indices.clone().into_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            take_views(array.views(), indices.as_slice::<$I>())
        });

        Ok(VarBinViewArray::try_new(
            views_buffer,
            array.buffers().collect(),
            array.dtype().clone(),
            validity,
        )?
        .into_array())
    }

    unsafe fn take_unchecked(
        &self,
        array: &VarBinViewArray,
        indices: &Array,
    ) -> VortexResult<Array> {
        // Compute the new validity
        let validity = array.validity().take(indices)?;
        let indices = indices.clone().into_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            take_views_unchecked(array.views(), indices.as_slice::<$I>())
        });

        Ok(VarBinViewArray::try_new(
            views_buffer,
            array.buffers().collect(),
            array.dtype().clone(),
            validity,
        )?
        .into_array())
    }
}

fn take_views<I: AsPrimitive<usize>>(
    views: Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}

fn take_views_unchecked<I: AsPrimitive<usize>>(
    views: Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::from_iter(
        indices
            .iter()
            .map(|i| unsafe { *views_ref.get_unchecked(i.as_()) }),
    )
}

impl CastFn<VarBinViewArray> for VarBinViewEncoding {
    fn cast(&self, array: &VarBinViewArray, dtype: &DType) -> VortexResult<Array> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().cast_nullability(new_nullability)?;
        let new_dtype = array.dtype().with_nullability(new_nullability);
        VarBinViewArray::try_new(
            array.views(),
            array.buffers().collect(),
            new_dtype,
            new_validity,
        )
        .map(IntoArray::into_array)
    }
}

impl MaskFn<VarBinViewArray> for VarBinViewEncoding {
    fn mask(&self, array: &VarBinViewArray, mask: Mask) -> VortexResult<Array> {
        VarBinViewArray::try_new(
            array.views(),
            array.buffers().collect(),
            array.dtype().as_nullable(),
            array.validity().mask(&mask)?,
        )
        .map(IntoArray::into_array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::accessor::ArrayAccessor;
    use crate::array::VarBinViewArray;
    use crate::compute::take;
    use crate::compute::test_harness::test_mask;
    use crate::{IntoArray, IntoArrayVariant};

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

        let taken = take(arr, buffer![0, 3].into_array()).unwrap();

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

    #[test]
    fn take_mask_var_bin_view_array() {
        test_mask(
            VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
        );

        test_mask(
            VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .into_array(),
        );
    }
}
