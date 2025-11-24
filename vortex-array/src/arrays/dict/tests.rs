// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_scalar::Scalar;

use super::DictArray;
use crate::arrays::{BoolArray, ListArray, PrimitiveArray, VarBinArray};
use crate::validity::Validity;
use crate::{Array, IntoArray, assert_arrays_eq};

#[test]
fn test_slice_into_const_dict() {
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![Some(0u32), None, Some(1)]).to_array(),
        PrimitiveArray::from_option_iter(vec![Some(0i32), Some(1), Some(2)]).to_array(),
    )
    .unwrap();

    assert_eq!(
        Some(Scalar::new(dict.dtype().clone(), 0i32.into())),
        dict.slice(0..1).as_constant()
    );

    assert_eq!(
        Some(Scalar::null(dict.dtype().clone())),
        dict.slice(1..2).as_constant()
    );
}

#[test]
fn test_scalar_at_null_code() {
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![None, Some(0u32), None]).to_array(),
        buffer![1i32].into_array(),
    )
    .unwrap();

    let expected = PrimitiveArray::from_option_iter(vec![None, Some(1i32), None]).into_array();
    assert_arrays_eq!(dict, expected);
}

#[test]
fn test_dict_display() {
    let x = DictArray::try_new(
        buffer![0u8, 0, 0, 1, 0, 3].into_array(),
        VarBinArray::from(vec!["Hello", "你好", "Bonjour", "Hola"]).into_array(),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        x.display_values().to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Hola\"]"
    )
}

#[test]
fn test_dict_list_dict_display() {
    let elements = DictArray::try_new(
        buffer![0u8, 0, 0, 1, 0, 3, 3, 2].into_array(),
        <VarBinArray as FromIterator<_>>::from_iter([
            Some("Hello"),
            Some("你好"),
            None,
            Some("Bonjour"),
            Some("Hola"),
        ])
        .into_array(),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        elements.display_values().to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Bonjour\", \"Bonjour\", null]"
    );

    let lists = ListArray::try_new(
        elements,
        buffer![0, 1, 1, 1, 3, 3, 5, 8].into_array(),
        Validity::Array(
            BoolArray::from_iter([true, true, false, true, false, true, true]).into_array(),
        ),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        lists.display_values().to_string(),
        "[[\"Hello\"], [], null, [\"Hello\", \"Hello\"], null, [\"你好\", \"Hello\"], [\"Bonjour\", \"Bonjour\", null]]"
    );

    let x = DictArray::try_new(buffer![6u8, 5, 2, 3, 2, 1].into_array(), lists)
        .unwrap()
        .into_array();

    assert_eq!(
        x.display_values().to_string(),
        "[[\"Bonjour\", \"Bonjour\", null], [\"你好\", \"Hello\"], null, [\"Hello\", \"Hello\"], null, []]"
    )
}
