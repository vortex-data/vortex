// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::fixture;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::offsets_tile_utf8;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;

#[fixture]
fn binary_array() -> ArrayRef {
    let values = Buffer::copy_from("hello worldhello world this is a long string".as_bytes());
    let offsets = buffer![0, 11, 44].into_array();

    VarBinArray::try_new(
        offsets.into_array(),
        values,
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

#[rstest]
pub fn test_scalar_at(binary_array: ArrayRef) {
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(
        binary_array,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
        &mut ctx
    );
}

#[rstest]
pub fn slice_array(binary_array: ArrayRef) {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr = binary_array.slice(1..2).unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"]),
        &mut ctx
    );
}

#[rstest]
#[case::tiles_multibyte_strings(        vec![0, 1, 3, 6], "héllo".as_bytes(),  true)]
#[case::ignores_bytes_outside_the_range(vec![1, 2, 3],    &[0xff, b'a', b'b'], true)]
#[case::splits_a_char(                  vec![0, 2, 6],    "héllo".as_bytes(),  false)]
#[case::invalid_byte(                   vec![0, 1, 2],    &[b'a', 0xff],       false)]
#[case::decreasing(                     vec![0, 3, 1, 6], "héllo".as_bytes(),  false)]
#[case::past_the_end(                   vec![0, 7],       "héllo".as_bytes(),  false)]
#[case::no_offsets(                     vec![],           b"",                 false)]
fn test_offsets_tile_utf8(#[case] offsets: Vec<u32>, #[case] bytes: &[u8], #[case] expected: bool) {
    assert_eq!(offsets_tile_utf8(&offsets, bytes), expected);
}

#[test]
fn test_offsets_tile_utf8_rejects_negative_offsets() {
    assert!(!offsets_tile_utf8(&[-1i32, 0], b"a"));
}
