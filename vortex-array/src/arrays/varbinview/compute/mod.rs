mod cast;
mod is_constant;
mod is_sorted;
mod min_max;
mod take;
mod to_arrow;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use super::BinaryView;
use crate::arrays::VarBinViewEncoding;
use crate::arrays::varbin::varbin_scalar;
use crate::arrays::varbinview::VarBinViewArray;
use crate::compute::{
    CastFn, IsConstantFn, IsSortedFn, MaskFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn,
    UncompressedSizeFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef};

impl ComputeVTable for VarBinViewEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&VarBinViewArray> for VarBinViewEncoding {
    fn scalar_at(&self, array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }
}

impl SliceFn<&VarBinViewArray> for VarBinViewEncoding {
    fn slice(&self, array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let views = array.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            array.buffers().to_vec(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

impl MaskFn<&VarBinViewArray> for VarBinViewEncoding {
    fn mask(&self, array: &VarBinViewArray, mask: Mask) -> VortexResult<ArrayRef> {
        Ok(VarBinViewArray::try_new(
            array.views().clone(),
            array.buffers().to_vec(),
            array.dtype().as_nullable(),
            array.validity().mask(&mask)?,
        )?
        .into_array())
    }
}

impl UncompressedSizeFn<&VarBinViewArray> for VarBinViewEncoding {
    fn uncompressed_size(&self, array: &VarBinViewArray) -> VortexResult<usize> {
        let views = array.views().len() * size_of::<BinaryView>();
        let mut buffers_size = 0;
        for buffer in array.buffers() {
            buffers_size += buffer.len();
        }

        Ok(views + buffers_size + array.validity().uncompressed_size())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::VarBinViewArray;
    use crate::builders::{ArrayBuilder, VarBinViewBuilder};
    use crate::canonical::ToCanonical;
    use crate::compute::test_harness::test_mask;
    use crate::compute::{take, take_into};

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

        let taken = take(&arr, &buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .to_varbinview()
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
        test_mask(&VarBinViewArray::from_iter_str([
            "one", "two", "three", "four", "five",
        ]));

        test_mask(&VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            Some("five"),
        ]));
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

        take_into(&arr, &buffer![0, 3].into_array(), &mut builder).unwrap();

        let taken = builder.finish();
        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .to_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }
}
