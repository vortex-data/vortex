// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Field and array assertions shared by property tests and fuzz targets.
//!
//! Explicit targets must agree with their exported Arrow types. Default exports can choose a
//! different physical representation, so imports retain the field metadata with that actual type.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrowSessionExt;
use crate::ExecuteArrowExt;
use crate::test_harness::DefaultExport;

/// Checks dtype and values while retaining field metadata for a default export.
#[track_caller]
fn check_field_round_trip(
    array: ArrayRef,
    field: &Field,
    target: Option<&Field>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    assert_eq!(arrow.from_arrow_field(field)?, *array.dtype());

    let exported = array.clone().execute_arrow(target, ctx)?;
    exported.to_data().validate_full()?;
    if target.is_some() {
        assert_eq!(exported.data_type(), field.data_type());
    }

    let actual_field = field.clone().with_data_type(exported.data_type().clone());
    let imported = arrow.from_arrow_array(exported, &actual_field)?;
    assert_eq!(imported.dtype(), array.dtype());
    assert_arrays_eq!(imported, array, ctx);

    Ok(())
}

/// Checks an array at the top level and in every required container.
///
/// `default_export` specifies whether the top-level dtype supports field inference. Containers
/// with validity are checked with both parent nullabilities, including a masked second row when
/// one exists. Empty arrays retain all placements, including dictionary and run-end fields.
#[track_caller]
pub fn check_containers(
    array: ArrayRef,
    default_export: DefaultExport,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    let values_field = arrow.to_arrow_field("value", array.dtype())?;
    check_field_round_trip(array.clone(), &values_field, Some(&values_field), ctx)?;

    match default_export {
        DefaultExport::Supported => {
            check_field_round_trip(array.clone(), &values_field, None, ctx)?;
        }
        DefaultExport::Unsupported => {
            assert!(arrow.default_arrow_export_field(array.dtype()).is_err());
            assert!(array.clone().execute_arrow(None, ctx).is_err());
        }
    }

    let len = array.len();

    for nullable in [false, true] {
        let validity = if nullable {
            Validity::from_iter((0..len).map(|i| i != 1))
        } else {
            Validity::NonNullable
        };
        let structure = StructArray::try_new(
            FieldNames::from(["child"]),
            vec![array.clone()],
            len,
            validity.clone(),
        )?
        .into_array();
        let list = ListArray::try_new(
            array.clone(),
            PrimitiveArray::from_iter(0..=u64::try_from(len)?).into_array(),
            validity.clone(),
            ctx,
        )?
        .into_array();
        let fixed =
            FixedSizeListArray::try_new(array.clone(), 1, validity.clone(), len)?.into_array();
        let entries = StructArray::try_new(
            FieldNames::from(["key", "value"]),
            vec![
                PrimitiveArray::from_iter(0..u32::try_from(len)?).into_array(),
                array.clone(),
            ],
            len,
            Validity::NonNullable,
        )?
        .into_array();
        let entries = ListViewArray::try_new(
            entries,
            PrimitiveArray::from_iter(0..u64::try_from(len)?).into_array(),
            PrimitiveArray::from_iter(std::iter::repeat_n(1u64, len)).into_array(),
            validity,
            ctx,
        )?;
        let map_dtype = DType::map(
            DType::Primitive(PType::U32, Nullability::NonNullable),
            array.dtype().clone(),
            true,
            nullable.into(),
        )?;
        let map = MapArray::try_new(
            map_dtype
                .into_map_opt()
                .vortex_expect("DType::map constructed a map dtype"),
            entries,
        )?
        .into_array();

        for array in [structure, list, fixed, map] {
            let field = arrow.to_arrow_field("container", array.dtype())?;
            check_field_round_trip(array.clone(), &field, Some(&field), ctx)?;
            check_field_round_trip(array, &field, None, ctx)?;
        }
    }

    let dictionary = DictArray::try_new(
        PrimitiveArray::from_iter(0..u32::try_from(len)?).into_array(),
        array.clone(),
    )?
    .into_array();
    let field = values_field.clone().with_data_type(DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(values_field.data_type().clone()),
    ));
    check_field_round_trip(dictionary, &field, Some(&field), ctx)?;

    let run_end = RunEnd::try_new(
        PrimitiveArray::from_iter((1..=u32::try_from(len)?).map(|i| i * 2)).into_array(),
        array,
        ctx,
    )?
    .into_array();
    let field = Field::new(
        "runs",
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(values_field),
        ),
        run_end.dtype().is_nullable(),
    );

    check_field_round_trip(run_end, &field, Some(&field), ctx)
}
