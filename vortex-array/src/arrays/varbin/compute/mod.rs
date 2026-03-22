// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod rules;
mod slice;

mod cast;
mod compare;
mod filter;
mod mask;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::VarBinArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    // UTF-8 strings
    #[case::str_non_nullable(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::str_nullable(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    // Binary data
    #[case::binary_non_nullable(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case::binary_nullable(VarBinArray::from_iter(
        [Some(b"hello".as_slice()), None, Some(b"test"), Some(b"data"), None],
        DType::Binary(Nullability::Nullable),
    ))]
    // Edge cases
    #[case::single_str(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    #[case::empty_strings(VarBinArray::from_iter(
        ["", "non-empty", "", "another", ""].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::all_null(VarBinArray::from_iter(
        [None::<&str>, None, None, None].into_iter(),
        DType::Utf8(Nullability::Nullable),
    ))]
    // Large strings
    #[case::large_strings(VarBinArray::from_iter(
        ["a".repeat(100), "b".repeat(200), "c".repeat(150), "d".repeat(50), "e".repeat(300)].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    // Mixed sizes
    #[case::mixed_sizes(VarBinArray::from_iter(
        ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    fn test_varbin_consistency(#[case] array: VarBinArray) {
        test_array_consistency(&array.into_array());
    }
}
