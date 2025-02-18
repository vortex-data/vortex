mod cast;
mod min_max;
mod take;
mod to_arrow;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::varbin::varbin_scalar;
use crate::array::varbinview::VarBinViewArray;
use crate::array::VarBinViewEncoding;
use crate::compute::{CastFn, MaskFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

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
    use crate::builders::{ArrayBuilder, VarBinViewBuilder};
    use crate::compute::test_harness::test_mask;
    use crate::compute::{take, take_into};
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

    #[test]
    fn take_into_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let mut builder = VarBinViewBuilder::with_capacity(arr.dtype().clone(), arr.len());

        take_into(arr, buffer![0, 3].into_array(), &mut builder).unwrap();

        let taken = builder.finish();
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
