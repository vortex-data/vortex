// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::DictArray;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::assert_arrays_eq;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn repeated_dictionary_codes_cross_blocks_and_skip_empty_runs() -> VortexResult<()> {
    let codes = PiecewiseSequenceArray::try_new(
        buffer![u64::MAX, 0, 1, u64::MAX].into_array(),
        buffer![0u64, 1025, 1047, 0].into_array(),
        ConstantArray::new(0u64, 4).into_array(),
        2072,
    )?;
    let table = buffer![5u32, 9].into_array();
    let dict = DictArray::try_new(codes.into_array(), table)?.into_array();
    let mut ctx = SESSION.create_execution_ctx().with_block_fusion(true);
    let actual = dict.execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_iter(
        std::iter::repeat_n(5u32, 1025).chain(std::iter::repeat_n(9u32, 1047)),
    );
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn repeated_dictionary_codes_check_declared_length() -> VortexResult<()> {
    for declared_len in [2, 4] {
        let codes = PiecewiseSequenceArray::try_new(
            buffer![0u64].into_array(),
            buffer![3u64].into_array(),
            ConstantArray::new(0u64, 1).into_array(),
            declared_len,
        )?;
        let dict = DictArray::try_new(codes.into_array(), buffer![99u32].into_array())?;
        let mut ctx = SESSION.create_execution_ctx().with_block_fusion(true);
        assert!(
            dict.into_array()
                .execute::<PrimitiveArray>(&mut ctx)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn test_scalar_at_null_code() {
    let mut ctx = SESSION.create_execution_ctx();
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![None, Some(0u32), None]).into_array(),
        buffer![1i32].into_array(),
    )
    .unwrap();

    let expected = PrimitiveArray::from_option_iter(vec![None, Some(1i32), None]).into_array();
    assert_arrays_eq!(dict, expected, &mut ctx);
}

#[test]
fn test_canonicalize_dict_with_all_null_array_validity_codes() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
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
    let ctx = SESSION.create_execution_ctx();
    let x = DictArray::try_new(
        buffer![0u8, 0, 0, 1, 0, 3].into_array(),
        VarBinArray::from(vec!["Hello", "你好", "Bonjour", "Hola"]).into_array(),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        x.display_values(&ctx).to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Hola\"]"
    )
}

#[test]
fn test_dict_list_dict_display() {
    let mut ctx = SESSION.create_execution_ctx();
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
        elements.display_values(&ctx).to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Bonjour\", \"Bonjour\", null]"
    );

    let lists = ListArray::try_new(
        elements,
        buffer![0, 1, 1, 1, 3, 3, 5, 8].into_array(),
        Validity::Array(
            BoolArray::from_iter([true, true, false, true, false, true, true]).into_array(),
        ),
        &mut ctx,
    )
    .unwrap()
    .into_array();

    assert_eq!(
        lists.display_values(&ctx).to_string(),
        "[[\"Hello\"], [], null, [\"Hello\", \"Hello\"], null, [\"你好\", \"Hello\"], [\"Bonjour\", \"Bonjour\", null]]"
    );

    let x = DictArray::try_new(buffer![6u8, 5, 2, 3, 2, 1].into_array(), lists)
        .unwrap()
        .into_array();

    assert_eq!(
        x.display_values(&ctx).to_string(),
        "[[\"Bonjour\", \"Bonjour\", null], [\"你好\", \"Hello\"], null, [\"Hello\", \"Hello\"], null, []]"
    )
}

#[test]
fn canonicalize_constant_nullable_code_over_non_nullable_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let code = Scalar::primitive(1u8, Nullability::Nullable);
    let dict = DictArray::try_new(
        ConstantArray::new(code, 3).into_array(),
        VarBinArray::from_iter(
            ["a", "long enough to live out of line"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array(),
    )?;
    let expected = VarBinArray::from_iter(
        [Some("long enough to live out of line"); 3],
        DType::Utf8(Nullability::Nullable),
    );
    assert_arrays_eq!(dict, expected, &mut ctx);
    Ok(())
}

#[test]
fn canonicalize_constant_null_code() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dict = DictArray::try_new(
        ConstantArray::new(
            Scalar::null(DType::Primitive(PType::U8, Nullability::Nullable)),
            3,
        )
        .into_array(),
        buffer![10i32, 20, 30].into_array(),
    )?;
    assert_arrays_eq!(
        dict,
        PrimitiveArray::from_option_iter([None::<i32>, None, None]),
        &mut ctx
    );
    Ok(())
}
