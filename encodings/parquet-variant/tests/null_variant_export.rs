// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(test)]

use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantBuilder;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_parquet_variant::ParquetVariant;
use vortex_session::VortexSession;

fn session() -> VortexSession {
    let session = VortexSession::empty();
    vortex_parquet_variant::initialize(&session);
    session
}

fn export(session: &VortexSession, array: ArrayRef) -> VortexResult<ArrowVariantArray> {
    let mut ctx = session.create_execution_ctx();
    let field = session.arrow().to_arrow_field("variant", array.dtype())?;
    let result = session
        .arrow()
        .execute_arrow(array, Some(&field), &mut ctx)?;
    Ok(ArrowVariantArray::try_new(result.as_ref())?)
}

#[test]
fn object_shredding_null_core() -> VortexResult<()> {
    let session = session();
    let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let shredded = StructArray::try_new(
        FieldNames::from_iter(["a"]),
        vec![a],
        3,
        Validity::from(BitBuffer::from_iter([true, true, false])),
    )?
    .into_array();
    let core = ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), 3).into_array();
    let array = VariantArray::try_new(core, Some(shredded))?.into_array();
    let result = export(&session, array)?;
    assert_eq!(result.len(), 3);
    assert_eq!(
        result.try_value(0)?.as_object().unwrap().get("a"),
        Some(PqVariant::Int32(1))
    );
    assert!(result.try_value(1)?.as_object().unwrap().get("a").is_none());
    assert_eq!(result.try_value(2)?, PqVariant::Null);
    Ok(())
}

#[test]
fn reject_nested_residual_object_with_null_core() -> VortexResult<()> {
    let session = session();
    let mut b = VariantBuilder::new();
    {
        let mut obj = b.new_object();
        obj.insert("x", 1i32);
        obj.finish();
    }
    let (meta, val) = b.finish();
    let nested_core = ParquetVariant::try_new(
        Validity::NonNullable,
        VarBinViewArray::from_iter_bin([meta.as_slice(), meta.as_slice()]).into_array(),
        Some(VarBinViewArray::from_iter_bin([val.as_slice(), val.as_slice()]).into_array()),
        None,
    )?
    .into_array();
    let nested = VariantArray::try_new(nested_core, None)?.into_array();
    let shredded = StructArray::try_new(
        FieldNames::from_iter(["a"]),
        vec![nested],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    let core = ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), 2).into_array();
    let array = VariantArray::try_new(core, Some(shredded))?.into_array();
    let error = export(&session, array).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("nested residual values over a null-constant Variant core are unsupported"),
        "{error}"
    );
    Ok(())
}

#[test]
fn reject_nested_shredded_override() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let (meta, val) = VariantBuilder::new().with_value("residual").finish();
    let inner_core = ParquetVariant::try_new(
        Validity::NonNullable,
        VarBinViewArray::from_iter_bin([meta.as_slice(); 3]).into_array(),
        Some(
            VarBinViewArray::from_iter_nullable_bin([None, None, Some(val.as_slice())])
                .into_array(),
        ),
        None,
    )?
    .into_array();
    let inner_shredded = PrimitiveArray::from_option_iter([Some(1i32), Some(2), None]).into_array();
    let inner = VariantArray::try_new(inner_core, Some(inner_shredded))?.into_array();
    let masked = MaskedArray::try_new(
        inner,
        Validity::from(BitBuffer::from_iter([true, true, true])),
        &mut ctx,
    )?
    .into_array();
    let outer_shredded = PrimitiveArray::from_option_iter([Some(100i32), None, None]).into_array();
    let array = VariantArray::try_new(masked, Some(outer_shredded))?.into_array();
    let error = export(&session, array).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("multiple levels of Variant shredding are unsupported"),
        "{error}"
    );
    Ok(())
}

#[test]
fn nested_object_names_in_list_are_in_metadata() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let elements = StructArray::try_new(
        FieldNames::from_iter(["typed_value"]),
        vec![PrimitiveArray::from_iter([42i32]).into_array()],
        1,
        Validity::NonNullable,
    )?
    .into_array();
    let list = ListArray::try_new(
        elements,
        PrimitiveArray::from_iter([0u32, 1]).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let shredded = StructArray::try_new(
        FieldNames::from_iter(["value"]),
        vec![list],
        1,
        Validity::NonNullable,
    )?
    .into_array();
    let core = ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), 1).into_array();
    let array = VariantArray::try_new(core, Some(shredded))?.into_array();
    let result = export(&session, array)?;
    let row = result.try_value(0)?;
    let list = row.as_object().unwrap().get("value").unwrap();
    let item = list.as_list().unwrap().get(0).unwrap();
    assert_eq!(
        item.as_object().unwrap().get("typed_value"),
        Some(PqVariant::Int32(42))
    );
    Ok(())
}
