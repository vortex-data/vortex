use rstest::{fixture, rstest};
use vortex_buffer::buffer;
use vortex_dtype::Nullability;
use vortex_scalar::Scalar;

use crate::array::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::compute::{SearchResult, SearchSortedSide, search_sorted};
use crate::validity::Validity;
use crate::{Array, ArrayRef};

fn sparse_high_null_fill() -> ArrayRef {
    SparseArray::try_new(
        buffer![17u64, 18, 19].into_array(),
        PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
        20,
        Scalar::null_typed::<i32>(),
    )
    .unwrap()
    .into_array()
}

fn sparse_high_non_null_fill() -> ArrayRef {
    SparseArray::try_new(
        buffer![17u64, 18, 19].into_array(),
        buffer![33_i32, 44, 55].into_array(),
        20,
        Scalar::primitive(22, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_low() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 2].into_array(),
        buffer![33_i32, 44, 55].into_array(),
        20,
        Scalar::primitive(60, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_low_high() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 17, 18, 19].into_array(),
        buffer![11i32, 22, 33, 44, 55].into_array(),
        20,
        Scalar::primitive(30, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_high_fill_in_patches() -> ArrayRef {
    SparseArray::try_new(
        buffer![17u64, 18, 19].into_array(),
        buffer![33_i32, 44, 55].into_array(),
        20,
        Scalar::primitive(33, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_low_fill_in_patches() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 2].into_array(),
        buffer![33_i32, 44, 55].into_array(),
        20,
        Scalar::primitive(55, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_low_high_fill_in_patches_low() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 17, 18, 19].into_array(),
        buffer![11i32, 22, 33, 44, 55].into_array(),
        20,
        Scalar::primitive(22, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

fn sparse_low_high_fill_in_patches_high() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 17, 18, 19].into_array(),
        buffer![11i32, 22, 33, 44, 55].into_array(),
        20,
        Scalar::primitive(33, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

#[fixture]
fn sparse_edge_patch_high() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 1, 2, 19].into_array(),
        buffer![11i32, 22, 23, 55].into_array(),
        20,
        Scalar::primitive(33, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

#[fixture]
fn sparse_edge_patch_low() -> ArrayRef {
    SparseArray::try_new(
        buffer![0u64, 17, 18, 19].into_array(),
        buffer![11i32, 33, 44, 55].into_array(),
        20,
        Scalar::primitive(22, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
#[case(sparse_low(), SearchResult::NotFound(20))]
#[case(sparse_low_high(), SearchResult::NotFound(20))]
fn search_larger_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 66, SearchSortedSide::Left).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(20))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(20))]
#[case(sparse_low(), SearchResult::NotFound(20))]
#[case(sparse_low_high(), SearchResult::NotFound(20))]
fn search_larger_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 66, SearchSortedSide::Right).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
#[case(sparse_low(), SearchResult::NotFound(0))]
#[case(sparse_low_high(), SearchResult::NotFound(1))]
fn search_less_than_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 21, SearchSortedSide::Left).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(17))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(0))]
#[case(sparse_low(), SearchResult::NotFound(0))]
#[case(sparse_low_high(), SearchResult::NotFound(1))]
fn search_less_than_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 21, SearchSortedSide::Right).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::Found(18))]
#[case(sparse_high_non_null_fill(), SearchResult::Found(18))]
#[case(sparse_low(), SearchResult::Found(1))]
#[case(sparse_low_high(), SearchResult::Found(18))]
fn search_patches_found_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 44, SearchSortedSide::Left).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::Found(19))]
#[case(sparse_high_non_null_fill(), SearchResult::Found(19))]
#[case(sparse_low(), SearchResult::Found(2))]
#[case(sparse_low_high(), SearchResult::Found(19))]
fn search_patches_found_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 44, SearchSortedSide::Right).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(19))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(19))]
#[case(sparse_low(), SearchResult::NotFound(2))]
#[case(sparse_low_high(), SearchResult::NotFound(19))]
fn search_mid_patches_not_found_left(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 45, SearchSortedSide::Left).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[case(sparse_high_null_fill(), SearchResult::NotFound(19))]
#[case(sparse_high_non_null_fill(), SearchResult::NotFound(19))]
#[case(sparse_low(), SearchResult::NotFound(2))]
#[case(sparse_low_high(), SearchResult::NotFound(19))]
fn search_mid_patches_not_found_right(#[case] array: ArrayRef, #[case] expected: SearchResult) {
    let res = search_sorted(&array, 45, SearchSortedSide::Right).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[should_panic]
#[case(sparse_high_null_fill(), Scalar::null_typed::<i32>(), SearchResult::Found(18))]
#[case(
    sparse_high_non_null_fill(),
    Scalar::primitive(22, Nullability::NonNullable),
    SearchResult::Found(0)
)]
#[case(
    sparse_low(),
    Scalar::primitive(60, Nullability::NonNullable),
    SearchResult::Found(3)
)]
#[case(
    sparse_low_high(),
    Scalar::primitive(30, Nullability::NonNullable),
    SearchResult::Found(2)
)]
#[case(
    sparse_high_fill_in_patches(),
    Scalar::primitive(33, Nullability::NonNullable),
    SearchResult::Found(0)
)]
#[case(
    sparse_low_fill_in_patches(),
    Scalar::primitive(55, Nullability::NonNullable),
    SearchResult::Found(2)
)]
#[case(
    sparse_low_high_fill_in_patches_low(),
    Scalar::primitive(22, Nullability::NonNullable),
    SearchResult::Found(1)
)]
#[case(
    sparse_low_high_fill_in_patches_high(),
    Scalar::primitive(33, Nullability::NonNullable),
    SearchResult::Found(17)
)]
fn search_fill_left(
    #[case] array: ArrayRef,
    #[case] search: Scalar,
    #[case] expected: SearchResult,
) {
    let res = search_sorted(&array, search, SearchSortedSide::Left).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
#[should_panic]
#[case(sparse_high_null_fill(), Scalar::null_typed::<i32>(), SearchResult::Found(18))]
#[case(
    sparse_high_non_null_fill(),
    Scalar::primitive(22, Nullability::NonNullable),
    SearchResult::Found(17)
)]
#[case(
    sparse_low(),
    Scalar::primitive(60, Nullability::NonNullable),
    SearchResult::Found(20)
)]
#[case(
    sparse_low_high(),
    Scalar::primitive(30, Nullability::NonNullable),
    SearchResult::Found(17)
)]
#[case(
    sparse_high_fill_in_patches(),
    Scalar::primitive(33, Nullability::NonNullable),
    SearchResult::Found(18)
)]
#[case(
    sparse_low_fill_in_patches(),
    Scalar::primitive(55, Nullability::NonNullable),
    SearchResult::Found(20)
)]
#[case(
    sparse_low_high_fill_in_patches_low(),
    Scalar::primitive(22, Nullability::NonNullable),
    SearchResult::Found(17)
)]
#[case(
    sparse_low_high_fill_in_patches_high(),
    Scalar::primitive(33, Nullability::NonNullable),
    SearchResult::Found(18)
)]
fn search_fill_right(
    #[case] array: ArrayRef,
    #[case] search: Scalar,
    #[case] expected: SearchResult,
) {
    let res = search_sorted(&array, search, SearchSortedSide::Right).unwrap();
    assert_eq!(res, expected);
}

#[rstest]
fn search_between_fill_and_patch_high_left(#[from(sparse_edge_patch_high)] array: ArrayRef) {
    assert_eq!(
        search_sorted(&array, 25, SearchSortedSide::Left).unwrap(),
        SearchResult::NotFound(3)
    );
    assert_eq!(
        search_sorted(&array, 44, SearchSortedSide::Left).unwrap(),
        SearchResult::NotFound(19)
    );
}

#[rstest]
fn search_between_fill_and_patch_high_right(#[from(sparse_edge_patch_high)] array: ArrayRef) {
    assert_eq!(
        search_sorted(&array, 25, SearchSortedSide::Right).unwrap(),
        SearchResult::NotFound(3)
    );
    assert_eq!(
        search_sorted(&array, 44, SearchSortedSide::Right).unwrap(),
        SearchResult::NotFound(19)
    );
}

#[rstest]
fn search_between_fill_and_patch_low_left(#[from(sparse_edge_patch_low)] array: ArrayRef) {
    assert_eq!(
        search_sorted(&array, 20, SearchSortedSide::Left).unwrap(),
        SearchResult::NotFound(1)
    );
    assert_eq!(
        search_sorted(&array, 28, SearchSortedSide::Left).unwrap(),
        SearchResult::NotFound(17)
    );
}

#[rstest]
fn search_between_fill_and_patch_low_right(#[from(sparse_edge_patch_low)] array: ArrayRef) {
    assert_eq!(
        search_sorted(&array, 20, SearchSortedSide::Right).unwrap(),
        SearchResult::NotFound(1)
    );
    assert_eq!(
        search_sorted(&array, 28, SearchSortedSide::Right).unwrap(),
        SearchResult::NotFound(17)
    );
}
