// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ToCanonical;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::assert_arrays_eq;

#[test]
pub fn varbin_view() {
    let binary_arr = VarBinViewArray::from_iter_str([
        "say less",
        "say less this array is lowkey fire and built different",
    ]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str([
            "say less",
            "say less this array is lowkey fire and built different"
        ])
    );
}

#[test]
pub fn slice_array() {
    let binary_arr = VarBinViewArray::from_iter_str([
        "say less",
        "say less this array is lowkey fire and built different",
    ])
    .slice(1..2)
    .unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["say less this array is lowkey fire and built different"])
    );
}

#[test]
pub fn flatten_array() {
    let binary_arr = VarBinViewArray::from_iter_str(["sus", "valid"]);
    let var_bin = binary_arr.to_varbinview();
    assert_arrays_eq!(var_bin, VarBinViewArray::from_iter_str(["sus", "valid"]));
}

#[test]
pub fn binary_view_size_and_alignment() {
    assert_eq!(size_of::<BinaryView>(), 16);
    assert_eq!(align_of::<BinaryView>(), 16);
}
