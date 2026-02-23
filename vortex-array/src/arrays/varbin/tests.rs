// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::fixture;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArray;
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
    assert_arrays_eq!(
        binary_array,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
    );
}

#[rstest]
pub fn slice_array(binary_array: ArrayRef) {
    let binary_arr = binary_array.slice(1..2).unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"])
    );
}

#[test]
fn test_zero_offsets() -> vortex_error::VortexResult<()> {
    use crate::arrays::VarBinVTable;
    use crate::dtype::Nullability::NonNullable;

    let items = VarBinArray::from_iter_nonnull(["abc", "def", "ghi"], DType::Utf8(NonNullable));
    let sliced = items.slice(1..3)?.as_::<VarBinVTable>().clone();

    // After slicing, there is some unused data at the front of the bytes.
    assert_eq!(sliced.offset_at(0), 3);

    // But after zeroing the offsets, the extraneous data is gone.
    let truncated = sliced.zero_offsets();
    assert_eq!(truncated.offset_at(0), 0);
    assert_eq!(truncated.len(), 2);
    Ok(())
}
