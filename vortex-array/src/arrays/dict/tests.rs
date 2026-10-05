// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use super::DictArray;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::assert_arrays_eq;
use crate::builders::builder_with_capacity_in;
use crate::validity::Validity;

#[test]
fn test_scalar_at_null_code() {
    let mut ctx = array_session().create_execution_ctx();
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![None, Some(0u32), None]).into_array(),
        buffer![1i32].into_array(),
    )
    .unwrap();

    let expected = PrimitiveArray::from_option_iter(vec![None, Some(1i32), None]).into_array();
    assert_arrays_eq!(dict, expected, &mut ctx);
}

#[test]
fn test_canonicalize_dict_with_all_null_array_validity_codes() -> vortex_error::VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let codes = PrimitiveArray::new(
        buffer![0u32, 1],
        Validity::Array(BoolArray::from_iter([false, false]).into_array()),
    )
    .into_array();

    let primitive = DictArray::try_new(
        codes.clone(),
        PrimitiveArray::from_option_iter([Some(10i32), Some(20)]).into_array(),
    )?
    .into_array();
    let mut primitive_builder = builder_with_capacity_in(primitive.dtype(), 2, ctx.allocator());
    primitive.append_to_builder(primitive_builder.as_mut(), &mut ctx)?;
    let primitive = primitive_builder.finish();
    let expected_primitive = PrimitiveArray::from_option_iter([None::<i32>, None]);
    assert_arrays_eq!(primitive, expected_primitive, &mut ctx);

    let lists = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4].into_array(),
        2,
        Validity::AllValid,
        2,
    )
    .into_array();
    let dict_lists = DictArray::try_new(codes, lists)?.into_array();
    let mut list_builder = builder_with_capacity_in(dict_lists.dtype(), 2, ctx.allocator());
    dict_lists.append_to_builder(list_builder.as_mut(), &mut ctx)?;
    let decoded_lists = list_builder.finish();
    let expected_lists = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4].into_array(),
        2,
        Validity::AllInvalid,
        2,
    );
    assert_arrays_eq!(decoded_lists, expected_lists, &mut ctx);
    Ok(())
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
