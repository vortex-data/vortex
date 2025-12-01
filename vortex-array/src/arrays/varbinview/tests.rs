// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;
use vortex_vector::binaryview::BinaryView;

use crate::Array;
use crate::ToCanonical;
use crate::arrays::VarBinViewArray;

#[test]
pub fn varbin_view() {
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
    assert_eq!(binary_arr.len(), 2);
    assert_eq!(binary_arr.scalar_at(0), Scalar::from("hello world"));
    assert_eq!(
        binary_arr.scalar_at(1),
        Scalar::from("hello world this is a long string")
    );
}

#[test]
pub fn slice_array() {
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
            .slice(1..2);
    assert_eq!(
        binary_arr.scalar_at(0),
        Scalar::from("hello world this is a long string")
    );
}

#[test]
pub fn flatten_array() {
    let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);
    let var_bin = binary_arr.to_varbinview().unwrap();
    assert_eq!(var_bin.scalar_at(0), Scalar::from("string1"));
    assert_eq!(var_bin.scalar_at(1), Scalar::from("string2"));
}

#[test]
pub fn binary_view_size_and_alignment() {
    assert_eq!(size_of::<BinaryView>(), 16);
    assert_eq!(align_of::<BinaryView>(), 16);
}
