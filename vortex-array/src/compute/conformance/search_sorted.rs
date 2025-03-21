pub use rstest_reuse;
use rstest_reuse::template;
use vortex_buffer::buffer;
use vortex_error::VortexUnwrap;

use crate::array::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::{Array, ArrayRef};

pub fn sparse_high_null_fill() -> ArrayRef {
    PrimitiveArray::new(buffer![0; 20], Validity::AllInvalid)
        .patch(&Patches::new(
            20,
            0,
            buffer![17u64, 18, 19].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

pub fn sparse_high_non_null_fill() -> ArrayRef {
    PrimitiveArray::new(buffer![22; 20], Validity::NonNullable)
        .patch(&Patches::new(
            20,
            0,
            buffer![17u64, 18, 19].into_array(),
            buffer![33_i32, 44, 55].into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

pub fn sparse_low() -> ArrayRef {
    PrimitiveArray::new(buffer![60; 20], Validity::NonNullable)
        .patch(&Patches::new(
            20,
            0,
            buffer![0u64, 1, 2].into_array(),
            buffer![33i32, 44, 55].into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

pub fn sparse_low_high() -> ArrayRef {
    PrimitiveArray::new(buffer![30; 20], Validity::NonNullable)
        .patch(&Patches::new(
            20,
            0,
            buffer![0u64, 1, 17, 18, 19].into_array(),
            buffer![11i32, 22, 33, 44, 55].into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

pub fn sparse_edge_patch_high() -> ArrayRef {
    PrimitiveArray::new(buffer![33; 20], Validity::NonNullable)
        .patch(&Patches::new(
            20,
            0,
            buffer![0u64, 1, 2, 19].into_array(),
            buffer![11i32, 22, 23, 55].into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

pub fn sparse_edge_patch_low() -> ArrayRef {
    PrimitiveArray::new(buffer![22; 20], Validity::NonNullable)
        .patch(&Patches::new(
            20,
            0,
            buffer![0u64, 17, 18, 19].into_array(),
            buffer![11i32, 33, 44, 55].into_array(),
        ))
        .vortex_unwrap()
        .into_array()
}

#[template]
#[export]
#[rstest]
#[case::larger_than_left_sparse_high_null_fill(
    sparse_high_null_fill(),
    66,
    SearchSortedSide::Left,
    SearchResult::NotFound(20)
)]
#[case::larger_than_left_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    66,
    SearchSortedSide::Left,
    SearchResult::NotFound(20)
)]
#[case::larger_than_left_sparse_low(
    sparse_low(),
    66,
    SearchSortedSide::Left,
    SearchResult::NotFound(20)
)]
#[case::larger_than_left_sparse_low_high(
    sparse_low_high(),
    66,
    SearchSortedSide::Left,
    SearchResult::NotFound(20)
)]
#[case::larger_than_right_sparse_high_null_fill(
    sparse_high_null_fill(),
    66,
    SearchSortedSide::Right,
    SearchResult::NotFound(20)
)]
#[case::larger_than_right_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    66,
    SearchSortedSide::Right,
    SearchResult::NotFound(20)
)]
#[case::larger_than_right_sparse_low(
    sparse_low(),
    66,
    SearchSortedSide::Right,
    SearchResult::NotFound(20)
)]
#[case::larger_than_right_sparse_low_high(
    sparse_low_high(),
    66,
    SearchSortedSide::Right,
    SearchResult::NotFound(20)
)]
#[case::less_than_left_sparse_high_null_fill(
    sparse_high_null_fill(),
    21,
    SearchSortedSide::Left,
    SearchResult::NotFound(17)
)]
#[case::less_than_left_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    21,
    SearchSortedSide::Left,
    SearchResult::NotFound(0)
)]
#[case::less_than_left_sparse_low(
    sparse_low(),
    21,
    SearchSortedSide::Left,
    SearchResult::NotFound(0)
)]
#[case::less_than_left_sparse_low_high(
    sparse_low_high(),
    21,
    SearchSortedSide::Left,
    SearchResult::NotFound(1)
)]
#[case::less_than_right_sparse_high_null_fill(
    sparse_high_null_fill(),
    21,
    SearchSortedSide::Right,
    SearchResult::NotFound(17)
)]
#[case::less_than_right_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    21,
    SearchSortedSide::Right,
    SearchResult::NotFound(0)
)]
#[case::less_than_right_sparse_low(
    sparse_low(),
    21,
    SearchSortedSide::Right,
    SearchResult::NotFound(0)
)]
#[case::less_than_right_sparse_low_high(
    sparse_low_high(),
    21,
    SearchSortedSide::Right,
    SearchResult::NotFound(1)
)]
#[case::patches_found_left_sparse_high_null_fill(
    sparse_high_null_fill(),
    44,
    SearchSortedSide::Left,
    SearchResult::Found(18)
)]
#[case::patches_found_left_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    44,
    SearchSortedSide::Left,
    SearchResult::Found(18)
)]
#[case::patches_found_left_sparse_low(
    sparse_low(),
    44,
    SearchSortedSide::Left,
    SearchResult::Found(1)
)]
#[case::patches_found_left_sparse_low_high(
    sparse_low_high(),
    44,
    SearchSortedSide::Left,
    SearchResult::Found(18)
)]
#[case::patches_found_right_sparse_high_null_fill(
    sparse_high_null_fill(),
    44,
    SearchSortedSide::Right,
    SearchResult::Found(19)
)]
#[case::patches_found_right_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    44,
    SearchSortedSide::Right,
    SearchResult::Found(19)
)]
#[case::patches_found_right_sparse_low(
    sparse_low(),
    44,
    SearchSortedSide::Right,
    SearchResult::Found(2)
)]
#[case::patches_found_right_sparse_low_high(
    sparse_low_high(),
    44,
    SearchSortedSide::Right,
    SearchResult::Found(19)
)]
#[case::mid_patches_not_found_left_sparse_high_null_fill(
    sparse_high_null_fill(),
    45,
    SearchSortedSide::Left,
    SearchResult::NotFound(19)
)]
#[case::mid_patches_not_found_left_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    45,
    SearchSortedSide::Left,
    SearchResult::NotFound(19)
)]
#[case::mid_patches_not_found_left_sparse_low(
    sparse_low(),
    45,
    SearchSortedSide::Left,
    SearchResult::NotFound(2)
)]
#[case::mid_patches_not_found_left_sparse_low_high(
    sparse_low_high(),
    45,
    SearchSortedSide::Left,
    SearchResult::NotFound(19)
)]
#[case::mid_patches_not_found_right_sparse_high_null_fill(
    sparse_high_null_fill(),
    45,
    SearchSortedSide::Right,
    SearchResult::NotFound(19)
)]
#[case::mid_patches_not_found_right_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    45,
    SearchSortedSide::Right,
    SearchResult::NotFound(19)
)]
#[case::mid_patches_not_found_right_sparse_low(
    sparse_low(),
    45,
    SearchSortedSide::Right,
    SearchResult::NotFound(2)
)]
#[case::mid_patches_not_found_right_sparse_low_high(
    sparse_low_high(),
    45,
    SearchSortedSide::Right,
    SearchResult::NotFound(19)
)]
#[case::fill_left_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    22,
    SearchSortedSide::Left,
    SearchResult::Found(0)
)]
#[case::fill_left_sparse_low(sparse_low(), 60, SearchSortedSide::Left, SearchResult::Found(3))]
#[case::fill_left_sparse_low_high(
    sparse_low_high(),
    30,
    SearchSortedSide::Left,
    SearchResult::Found(2)
)]
#[case::fill_right_sparse_high_non_null_fill(
    sparse_high_non_null_fill(),
    22,
    SearchSortedSide::Right,
    SearchResult::Found(17)
)]
#[case::fill_right_sparse_low(sparse_low(), 60, SearchSortedSide::Right, SearchResult::Found(20))]
#[case::fill_right_sparse_low_high(
    sparse_low_high(),
    30,
    SearchSortedSide::Right,
    SearchResult::Found(17)
)]
#[case::between_fill_and_patch_high_left_smaller(
    sparse_edge_patch_high(),
    25,
    SearchSortedSide::Left,
    SearchResult::NotFound(3)
)]
#[case::between_fill_and_patch_high_left_larger(
    sparse_edge_patch_high(),
    44,
    SearchSortedSide::Left,
    SearchResult::NotFound(19)
)]
#[case::between_fill_and_patch_high_right_smaller(
    sparse_edge_patch_high(),
    25,
    SearchSortedSide::Right,
    SearchResult::NotFound(3)
)]
#[case::between_fill_and_patch_high_right_larger(
    sparse_edge_patch_high(),
    44,
    SearchSortedSide::Right,
    SearchResult::NotFound(19)
)]
#[case::between_fill_and_patch_low_left_smaller(
    sparse_edge_patch_low(),
    20,
    SearchSortedSide::Left,
    SearchResult::NotFound(1)
)]
#[case::between_fill_and_patch_low_left_larger(
    sparse_edge_patch_low(),
    28,
    SearchSortedSide::Left,
    SearchResult::NotFound(17)
)]
#[case::between_fill_and_patch_low_right_smaller(
    sparse_edge_patch_low(),
    20,
    SearchSortedSide::Right,
    SearchResult::NotFound(1)
)]
#[case::between_fill_and_patch_low_right_larger(
    sparse_edge_patch_low(),
    28,
    SearchSortedSide::Right,
    SearchResult::NotFound(17)
)]
pub fn search_sorted_conformance(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) {
}
