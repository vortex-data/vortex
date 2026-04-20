// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;
mod zip;

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::VarBinViewArray;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
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

        let taken = arr.take(buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        #[expect(deprecated)]
        let result = taken.to_varbinview().with_iterator(|it| {
            it.map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                .collect::<Vec<_>>()
        });
        assert_eq!(result, [Some("one".to_string()), Some("four".to_string())]);
    }
    // Consistency tests
    use rstest::rstest;

    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    // From test_all_consistency
    #[case::varbinview_str(VarBinViewArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::varbinview_nullable_str(VarBinViewArray::from_iter_nullable_str([
        Some("hello"),
        None,
        Some("test"),
        Some("data"),
        None,
    ]))]
    #[case::varbinview_binary(VarBinViewArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    // Additional test cases
    #[case::varbinview_empty_strings(VarBinViewArray::from_iter(
        ["", "non-empty", "", "another", ""].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::varbinview_single(VarBinViewArray::from_iter(
        ["single"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::varbinview_large_strings(VarBinViewArray::from_iter(
        ["a".repeat(100), "b".repeat(200), "c".repeat(150)].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::varbinview_all_null(VarBinViewArray::from_iter_nullable_str([
        None::<&str>, None, None, None
    ]))]
    fn test_varbinview_consistency(#[case] array: VarBinViewArray) {
        test_array_consistency(&array.into_array());
    }
}
