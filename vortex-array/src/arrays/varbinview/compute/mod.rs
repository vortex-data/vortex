mod cast;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod take;

use crate::arrays::VarBinViewEncoding;
use crate::vtable::ComputeVTable;

impl ComputeVTable for VarBinViewEncoding {}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::VarBinViewArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::mask::test_mask;
    use crate::compute::take;

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
}
