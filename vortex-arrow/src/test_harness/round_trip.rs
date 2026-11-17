// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Field and array assertions shared by property tests and fuzz targets.
//!
//! Explicit targets must agree with their exported Arrow types. Default exports can choose a
//! different physical representation, so imports retain the field metadata with that actual type.

use std::sync::Arc;

use arrow_array::Array;
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

/// Checks extension import, explicit export, and batch selections against an Arrow reference.
///
/// The caller supplies both representations from source values. The expected field must not come
/// from the Vortex exporter. The Vortex array must have an extension dtype. Arrow data equality
/// ignores values masked by nulls.
#[track_caller]
pub fn check_arrow(
    array: ArrayRef,
    field: &Field,
    expected: arrow_array::ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    check_arrow_values(array.clone(), field, Arc::clone(&expected), ctx)?;
    check_nested_arrow(array.clone(), field, Arc::clone(&expected), ctx)?;
    let len = array.len();
    let indices: Vec<_> = (0..u32::try_from(len)?)
        .rev()
        .flat_map(|i| [i, i])
        .collect();
    let mask: Vec<_> = (0..len).map(|i| i % 2 == 0).collect();
    let selected = [
        (
            "slice",
            array.slice(len / 2..len)?,
            expected.slice(len / 2, len - len / 2),
        ),
        (
            "take",
            array.take(PrimitiveArray::from_iter(indices.clone()).into_array())?,
            reference_take(expected.as_ref(), &indices)?,
        ),
        (
            "filter",
            array.filter(vortex_mask::Mask::from_iter(mask.iter().copied()))?,
            reference_take(
                expected.as_ref(),
                &(0..u32::try_from(len)?)
                    .filter(|&i| mask[usize::try_from(i).vortex_expect("row index fits usize")])
                    .collect::<Vec<_>>(),
            )?,
        ),
    ];
    for (operation, selected, expected) in selected {
        assert_eq!(selected.dtype(), array.dtype(), "{operation} dtype");
        let selected = selected
            .execute::<vortex_array::arrays::ExtensionArray>(ctx)?
            .into_array();
        let exported = selected.execute_arrow(Some(field), ctx)?;
        assert_eq!(exported.to_data(), expected.to_data(), "{operation} values");
    }

    Ok(())
}

#[track_caller]
fn check_arrow_values(
    array: ArrayRef,
    field: &Field,
    expected: arrow_array::ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    assert_eq!(expected.data_type(), field.data_type());
    expected.to_data().validate_full()?;
    assert_eq!(arrow.from_arrow_field(field)?, *array.dtype());
    let imported = arrow.from_arrow_array(Arc::clone(&expected), field)?;
    assert_arrays_eq!(imported, array, ctx);
    let exported = array.execute_arrow(Some(field), ctx)?;
    exported.to_data().validate_full()?;
    assert_eq!(exported.to_data(), expected.to_data());

    Ok(())
}

/// Builds parents independently around the two source representations.
#[track_caller]
fn check_nested_arrow(
    array: ArrayRef,
    field: &Field,
    expected: arrow_array::ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = array.len();
    let validity = Validity::from_iter((0..len).map(|i| i != 1));
    let nulls = Some(arrow_buffer::NullBuffer::from(
        (0..len).map(|i| i != 1).collect::<Vec<_>>(),
    ));
    let child_field = Arc::new(field.clone().with_name("child"));
    let structure = StructArray::try_new(
        FieldNames::from(["child"]),
        vec![array.clone()],
        len,
        validity.clone(),
    )?
    .into_array();
    let expected_structure = arrow_array::StructArray::try_new_with_length(
        vec![Arc::clone(&child_field)].into(),
        vec![Arc::clone(&expected)],
        nulls.clone(),
        len,
    )?;
    let structure_field = Field::new("container", expected_structure.data_type().clone(), true);
    check_arrow_values(
        structure,
        &structure_field,
        Arc::new(expected_structure),
        ctx,
    )?;

    let fixed = FixedSizeListArray::try_new(array.clone(), 1, validity.clone(), len)?.into_array();
    let expected_fixed = arrow_array::FixedSizeListArray::try_new_with_length(
        Arc::clone(&child_field),
        1,
        Arc::clone(&expected),
        nulls.clone(),
        len,
    )?;
    let fixed_field = Field::new("container", expected_fixed.data_type().clone(), true);
    check_arrow_values(fixed, &fixed_field, Arc::new(expected_fixed), ctx)?;

    let mut offsets: Vec<_> = (0..=i32::try_from(len)?).map(|i| i - i % 2).collect();
    *offsets
        .last_mut()
        .vortex_expect("list offsets include the initial zero") = i32::try_from(len)?;
    let list = ListArray::try_new(
        array.clone(),
        PrimitiveArray::from_iter(offsets.clone()).into_array(),
        validity,
        ctx,
    )?
    .into_array();
    let expected_list = arrow_array::ListArray::try_new(
        child_field,
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        Arc::clone(&expected),
        nulls,
    )?;
    let list_field = Field::new("container", expected_list.data_type().clone(), true);
    check_arrow_values(list, &list_field, Arc::new(expected_list), ctx)?;

    let keys = (0..u32::try_from(len)?).collect::<Vec<_>>();
    let entries = StructArray::try_new(
        FieldNames::from(["key", "value"]),
        vec![
            PrimitiveArray::from_iter(keys.clone()).into_array(),
            array.clone(),
        ],
        len,
        Validity::NonNullable,
    )?
    .into_array();
    let entries = ListViewArray::try_new(
        entries,
        PrimitiveArray::from_iter(0..u32::try_from(len)?).into_array(),
        PrimitiveArray::from_iter(std::iter::repeat_n(1u32, len)).into_array(),
        Validity::from_iter((0..len).map(|i| i != 1)),
        ctx,
    )?;
    let map_dtype = DType::map(
        DType::Primitive(PType::U32, Nullability::NonNullable),
        array.dtype().clone(),
        true,
        Nullability::Nullable,
    )?;
    let map = MapArray::try_new(
        map_dtype
            .into_map_opt()
            .vortex_expect("DType::map constructed a map"),
        entries,
    )?
    .into_array();
    let entry_fields = vec![
        Field::new("key", DataType::UInt32, false),
        field.clone().with_name("value"),
    ];
    let expected_entries = arrow_array::StructArray::try_new_with_length(
        entry_fields.clone().into(),
        vec![
            Arc::new(arrow_array::UInt32Array::from(keys)),
            Arc::clone(&expected),
        ],
        None,
        len,
    )?;
    let expected_map = arrow_array::MapArray::try_new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(entry_fields.into()),
            false,
        )),
        arrow_buffer::OffsetBuffer::new((0..=i32::try_from(len)?).collect::<Vec<_>>().into()),
        expected_entries,
        Some(arrow_buffer::NullBuffer::from(
            (0..len).map(|i| i != 1).collect::<Vec<_>>(),
        )),
        true,
    )?;
    let map_field = Field::new("container", expected_map.data_type().clone(), true);
    check_arrow_values(map, &map_field, Arc::new(expected_map), ctx)?;

    let codes: Vec<_> = (0..i32::try_from(len)?)
        .rev()
        .flat_map(|i| [i, i])
        .collect();
    let dict = DictArray::try_new(
        PrimitiveArray::from_iter(
            codes
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .into_array(),
        array.clone(),
    )?
    .into_array();
    let expected_dict = arrow_array::DictionaryArray::<arrow_array::types::Int32Type>::try_new(
        arrow_array::Int32Array::from(codes),
        Arc::clone(&expected),
    )?;
    let dict_field = field
        .clone()
        .with_data_type(expected_dict.data_type().clone());
    check_arrow_values(dict, &dict_field, Arc::new(expected_dict), ctx)?;

    let ends: Vec<_> = (1..=i32::try_from(len)?).map(|i| i * (i + 1) / 2).collect();
    let runs = RunEnd::try_new(
        PrimitiveArray::from_iter(
            ends.iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .into_array(),
        array,
        ctx,
    )?
    .into_array();
    let expected_runs = arrow_array::RunArray::<arrow_array::types::Int32Type>::try_new(
        &arrow_array::Int32Array::from(ends),
        expected.as_ref(),
    )?;
    let DataType::RunEndEncoded(ends_field, _) = expected_runs.data_type() else {
        unreachable!("RunArray has run-end dtype")
    };
    let run_field = Field::new(
        "runs",
        DataType::RunEndEncoded(Arc::clone(ends_field), Arc::new(field.clone())),
        field.is_nullable(),
    );
    let expected_runs = arrow_array::make_array(
        expected_runs
            .to_data()
            .into_builder()
            .data_type(run_field.data_type().clone())
            .build()?,
    );
    check_arrow_values(runs, &run_field, expected_runs, ctx)
}

/// Selects source rows while retaining the row count of zero-width lists.
fn reference_take(array: &dyn Array, indices: &[u32]) -> VortexResult<arrow_array::ArrayRef> {
    if let DataType::FixedSizeList(field, 0) = array.data_type() {
        let values = array
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .vortex_expect("fixed-size-list dtype");

        // Arrow's take kernel derives this length from the empty child, losing nonempty rows.
        let nulls = array.nulls().map(|_| {
            arrow_buffer::NullBuffer::from(
                indices
                    .iter()
                    .map(|&i| {
                        array.is_valid(usize::try_from(i).vortex_expect("row index fits usize"))
                    })
                    .collect::<Vec<_>>(),
            )
        });
        return Ok(Arc::new(
            arrow_array::FixedSizeListArray::try_new_with_length(
                Arc::clone(field),
                0,
                Arc::clone(values.values()),
                nulls,
                indices.len(),
            )?,
        ));
    }

    Ok(arrow_select::take::take(
        array,
        &arrow_array::UInt32Array::from(indices.to_vec()),
        None,
    )?)
}

/// Checks canonical field and schema inference against an independent expected field.
#[track_caller]
pub fn check_canonical_field(
    array: &ArrayRef,
    field: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    assert_eq!(arrow.to_arrow_field(field.name(), array.dtype())?, *field);
    let dtype = DType::struct_(
        [(field.name().as_str(), array.dtype().clone())],
        Nullability::NonNullable,
    );
    let schema = arrow_schema::Schema::new(vec![field.clone()]);
    assert_eq!(arrow.to_arrow_schema(&dtype)?, schema);
    assert_eq!(arrow.from_arrow_schema(&schema)?, dtype);

    Ok(())
}

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
        PrimitiveArray::from_iter((0..u32::try_from(len)?).rev().flat_map(|i| [i, i])).into_array(),
        array.clone(),
    )?
    .into_array();
    let field = values_field.clone().with_data_type(DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(values_field.data_type().clone()),
    ));
    check_field_round_trip(dictionary, &field, Some(&field), ctx)?;

    let run_end = RunEnd::try_new(
        PrimitiveArray::from_iter((1..=u32::try_from(len)?).map(|i| i * (i + 1) / 2)).into_array(),
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
