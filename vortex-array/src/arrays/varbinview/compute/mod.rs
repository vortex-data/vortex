mod cast;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod take;

use vortex_error::VortexResult;

use super::BinaryView;
use crate::Array;
use crate::arrays::VarBinViewEncoding;
use crate::arrays::varbinview::VarBinViewArray;
use crate::compute::{TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for VarBinViewEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
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
    use crate::compute::conformance::mask::test_mask;
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
