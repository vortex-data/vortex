// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::{
    fixture,
    rstest,
};
use vortex_buffer::{
    Buffer,
    buffer,
};
use vortex_dtype::{
    DType,
    Nullability,
};

use crate::arrays::varbin::VarBinArray;
use crate::validity::Validity;
use crate::{
    Array,
    ArrayRef,
    IntoArray,
};

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
    assert_eq!(binary_array.len(), 2);
    assert_eq!(binary_array.scalar_at(0), "hello world".into());
    assert_eq!(
        binary_array.scalar_at(1),
        "hello world this is a long string".into()
    )
}

#[rstest]
pub fn slice_array(binary_array: ArrayRef) {
    let binary_arr = binary_array.slice(1..2);
    assert_eq!(
        binary_arr.scalar_at(0),
        "hello world this is a long string".into()
    );
}
