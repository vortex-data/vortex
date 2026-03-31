// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::fixture;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;

#[fixture]
fn binary_array() -> ArrayRef {
    let values =
        Buffer::copy_from("bet periodtbet periodt this string hits different on god fr".as_bytes());
    let offsets = buffer![0, 11, 59].into_array();

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
    assert_arrays_eq!(
        binary_array,
        VarBinViewArray::from_iter_str([
            "bet periodt",
            "bet periodt this string hits different on god fr"
        ])
    );
}

#[rstest]
pub fn slice_array(binary_array: ArrayRef) {
    let binary_arr = binary_array.slice(1..2).unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["bet periodt this string hits different on god fr"])
    );
}
