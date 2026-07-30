// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use crate::Array;
use crate::ArrayContext;
use crate::ArrayParts;
use crate::ArrayVTable;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::FilterArray;
use crate::arrays::ListViewArray;
use crate::arrays::Map;
use crate::arrays::MapArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapData;
use crate::arrays::map::MapDataParts;
use crate::assert_arrays_eq;
use crate::builders::ArrayBuilder;
use crate::builders::MapBuilder;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::session::ArraySessionExt;
use crate::validity::Validity;

fn map_dtype() -> VortexResult<MapDType> {
    MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        true,
    )
}

fn key(value: i32) -> Scalar {
    Scalar::primitive(value, Nullability::NonNullable)
}

fn value(value: Option<&str>) -> Scalar {
    match value {
        Some(value) => Scalar::utf8(value, Nullability::Nullable),
        None => Scalar::null(DType::Utf8(Nullability::Nullable)),
    }
}

fn sample_scalar(dtype: DType) -> VortexResult<Scalar> {
    Scalar::try_map(
        dtype,
        [
            (key(2), value(Some("two"))),
            (key(1), value(None)),
            (key(2), value(Some("duplicate"))),
        ],
    )
}

fn map_array_from_rows(
    map_dtype: MapDType,
    nullability: Nullability,
    rows: impl IntoIterator<Item = Option<Vec<(Scalar, Scalar)>>>,
) -> VortexResult<MapArray> {
    let rows = rows.into_iter().collect::<Vec<_>>();
    let dtype = DType::Map(map_dtype.clone(), nullability);
    let mut builder = MapBuilder::<u64, u64>::with_capacity(map_dtype, nullability, rows.len());

    for row in rows {
        let scalar = match row {
            Some(entries) => Scalar::try_map(dtype.clone(), entries)?,
            None => Scalar::null(dtype.clone()),
        };
        builder.append_scalar(&scalar)?;
    }

    Ok(builder.finish_into_map())
}

fn sample_array() -> VortexResult<MapArray> {
    let map_dtype = map_dtype()?;
    map_array_from_rows(
        map_dtype,
        Nullability::Nullable,
        [
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            Some(vec![]),
            None,
        ],
    )
}

#[test]
fn constructs_map_with_listview_entries() -> VortexResult<()> {
    let MapDataParts { map_dtype, entries } = sample_array()?.into_data_parts();
    let array = MapArray::try_new(map_dtype.clone(), entries)?;

    assert_eq!(
        array.dtype(),
        &DType::Map(map_dtype.clone(), Nullability::Nullable)
    );
    assert!(array.keys_sorted());
    assert_eq!(array.entry_count_at(0), 3);
    assert_eq!(array.entry_count_at(1), 0);
    assert_eq!(array.entries_at(0)?.dtype(), &map_dtype.entries_dtype());

    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(
        array
            .map_validity()
            .execute_mask(array.len(), &mut ctx)?
            .true_count(),
        2
    );

    Ok(())
}

#[test]
fn accepts_duplicate_and_unsorted_keys() -> VortexResult<()> {
    let array = sample_array()?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        sample_scalar(array.dtype().clone())?
    );

    Ok(())
}

#[test]
fn rejects_malformed_entry_storage() -> VortexResult<()> {
    let map_dtype = map_dtype()?;
    let offsets = PrimitiveArray::from_iter([0u64]).into_array();
    let sizes = PrimitiveArray::from_iter([1u64]).into_array();
    let non_struct_entries = ListViewArray::try_new(
        PrimitiveArray::from_iter([1i32]).into_array(),
        offsets,
        sizes,
        Validity::NonNullable,
    )?;
    assert!(MapArray::try_new(map_dtype.clone(), non_struct_entries).is_err());

    let nullable_entry_struct =
        ConstantArray::new(Scalar::null(map_dtype.entries_dtype().as_nullable()), 1).into_array();
    let null_entries = ListViewArray::try_new(
        nullable_entry_struct,
        PrimitiveArray::from_iter([0u64]).into_array(),
        PrimitiveArray::from_iter([1u64]).into_array(),
        Validity::NonNullable,
    )?;
    assert!(MapArray::try_new(map_dtype.clone(), null_entries).is_err());

    let MapDataParts { entries, .. } = sample_array()?.into_data_parts();
    let parts = ArrayParts::new(
        Map,
        DType::Map(map_dtype, Nullability::NonNullable),
        entries.len(),
        MapData,
    )
    .with_slots(smallvec![Some(entries.into_array())]);
    assert!(Array::<Map>::try_from_parts(parts).is_err());

    assert!(
        MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Utf8(Nullability::Nullable),
            false,
        )
        .is_err()
    );

    Ok(())
}

#[test]
fn scalar_access_preserves_null_and_empty_maps() -> VortexResult<()> {
    let array = sample_array()?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        sample_scalar(array.dtype().clone())?
    );
    assert!(array.execute_scalar(1, &mut ctx)?.as_map().is_empty());
    assert!(array.execute_scalar(2, &mut ctx)?.is_null());

    Ok(())
}

#[test]
fn scalar_access_preserves_variable_entry_counts_and_utf8_pairs() -> VortexResult<()> {
    let map_dtype = MapDType::try_new(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
    )?;
    let dtype = DType::Map(map_dtype.clone(), Nullability::Nullable);
    let mut builder = MapBuilder::<u64, u64>::with_capacity(map_dtype, Nullability::Nullable, 4);
    let rows = [
        Some(vec![
            (
                Scalar::utf8("alpha", Nullability::NonNullable),
                Scalar::utf8("one", Nullability::Nullable),
            ),
            (
                Scalar::utf8("longer-key", Nullability::NonNullable),
                Scalar::utf8("longer-value", Nullability::Nullable),
            ),
        ]),
        Some(vec![(
            Scalar::utf8("z", Nullability::NonNullable),
            Scalar::utf8("last", Nullability::Nullable),
        )]),
        None,
        Some(vec![
            (
                Scalar::utf8("repeated", Nullability::NonNullable),
                Scalar::utf8("first", Nullability::Nullable),
            ),
            (
                Scalar::utf8("repeated", Nullability::NonNullable),
                Scalar::null(DType::Utf8(Nullability::Nullable)),
            ),
            (
                Scalar::utf8("tail", Nullability::NonNullable),
                Scalar::utf8("", Nullability::Nullable),
            ),
        ]),
    ];
    let expected = rows
        .into_iter()
        .map(|row| {
            row.map_or_else(
                || Ok(Scalar::null(dtype.clone())),
                |entries| Scalar::try_map(dtype.clone(), entries),
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;

    for scalar in &expected {
        builder.append_scalar(scalar)?;
    }
    let array = builder.finish_into_map();
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(array.entry_count_at(0), 2);
    assert_eq!(array.entry_count_at(1), 1);
    assert_eq!(array.entry_count_at(2), 0);
    assert_eq!(array.entry_count_at(3), 3);
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(array.execute_scalar(index, &mut ctx)?, expected);
    }

    Ok(())
}

#[test]
fn slice_preserves_map_rows() -> VortexResult<()> {
    let source = sample_array()?.into_array();
    let expected = map_array_from_rows(
        map_dtype()?,
        Nullability::Nullable,
        [
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            Some(vec![]),
        ],
    )?;
    let sliced = source.slice(0..2)?;
    let mut ctx = array_session().create_execution_ctx();

    assert!(sliced.is::<Map>());
    assert_arrays_eq!(sliced, expected, &mut ctx);

    Ok(())
}

#[test]
fn filter_handles_all_none_and_mixed_maps() -> VortexResult<()> {
    let source = sample_array()?.into_array();
    let map_dtype = map_dtype()?;
    let mut ctx = array_session().create_execution_ctx();

    let all = source.filter(Mask::from_iter([true, true, true]))?;
    assert!(all.is::<Map>());
    assert_arrays_eq!(all, sample_array()?, &mut ctx);

    let none = source.filter(Mask::from_iter([false, false, false]))?;
    let expected_none = map_array_from_rows(
        map_dtype.clone(),
        Nullability::Nullable,
        Vec::<Option<Vec<(Scalar, Scalar)>>>::new(),
    )?;
    assert!(none.is::<Map>());
    assert_arrays_eq!(none, expected_none, &mut ctx);

    let mixed = source.filter(Mask::from_iter([true, false, true]))?;
    let expected_mixed = map_array_from_rows(
        map_dtype,
        Nullability::Nullable,
        [
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            None,
        ],
    )?;
    assert!(mixed.is::<Map>());
    assert_arrays_eq!(mixed, expected_mixed, &mut ctx);

    Ok(())
}

#[test]
fn filter_dropping_nonempty_middle_row_clears_zero_copy_flag() -> VortexResult<()> {
    let source = map_array_from_rows(
        map_dtype()?,
        Nullability::Nullable,
        [
            Some(vec![
                (key(1), value(Some("one"))),
                (key(2), value(Some("two"))),
            ]),
            Some(vec![(key(3), value(Some("three")))]),
            Some(vec![(key(4), value(Some("four")))]),
        ],
    )?
    .into_array();
    let expected = map_array_from_rows(
        map_dtype()?,
        Nullability::Nullable,
        [
            Some(vec![
                (key(1), value(Some("one"))),
                (key(2), value(Some("two"))),
            ]),
            Some(vec![(key(4), value(Some("four")))]),
        ],
    )?;
    let mask = Mask::from_iter([true, false, true]);
    let mut ctx = array_session().create_execution_ctx();

    let reduced = source.filter(mask.clone())?;
    assert!(reduced.is::<Map>());
    // The dropped middle row leaves a gap in the referenced entry elements, so the filtered
    // entries must not claim zero-copy-to-list convertibility.
    assert!(
        !reduced
            .as_::<Map>()
            .entries()
            .into_owned()
            .is_zero_copy_to_list()
    );
    assert_arrays_eq!(reduced, expected, &mut ctx);

    let executed = FilterArray::new(source, mask)
        .into_array()
        .execute::<MapArray>(&mut ctx)?;
    assert!(!executed.entries().into_owned().is_zero_copy_to_list());
    assert_arrays_eq!(executed, expected, &mut ctx);

    Ok(())
}

#[test]
fn take_supports_reordered_duplicate_and_nullable_indices() -> VortexResult<()> {
    let source = sample_array()?.into_array();
    let map_dtype = map_dtype()?;
    let mut ctx = array_session().create_execution_ctx();

    let taken = source.take(PrimitiveArray::from_iter([2u64, 0, 0]).into_array())?;
    let expected_taken = map_array_from_rows(
        map_dtype.clone(),
        Nullability::Nullable,
        [
            None,
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
        ],
    )?;
    assert!(taken.is::<Map>());
    assert_arrays_eq!(taken, expected_taken, &mut ctx);

    let nonnullable_source = map_array_from_rows(
        map_dtype.clone(),
        Nullability::NonNullable,
        [
            Some(vec![(key(1), value(Some("one")))]),
            Some(vec![]),
            Some(vec![(key(3), value(None))]),
        ],
    )?
    .into_array();
    let nullable_taken = nonnullable_source
        .take(PrimitiveArray::from_option_iter([Some(1u64), None, Some(0)]).into_array())?;
    let expected_nullable_taken = map_array_from_rows(
        map_dtype,
        Nullability::Nullable,
        [Some(vec![]), None, Some(vec![(key(1), value(Some("one")))])],
    )?;
    assert!(nullable_taken.is::<Map>());
    assert_eq!(nullable_taken.dtype().nullability(), Nullability::Nullable);
    assert_arrays_eq!(nullable_taken, expected_nullable_taken, &mut ctx);

    Ok(())
}

#[test]
fn mask_combines_with_existing_map_validity() -> VortexResult<()> {
    let source = sample_array()?.into_array();
    let mask = BoolArray::from_iter([true, false, true]).into_array();
    let masked = source.mask(mask)?;
    let expected = map_array_from_rows(
        map_dtype()?,
        Nullability::Nullable,
        [
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            None,
            None,
        ],
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert!(masked.is::<Map>());
    assert_arrays_eq!(masked, expected, &mut ctx);

    Ok(())
}

#[test]
fn cast_widens_map_key_value_and_outer_nullability() -> VortexResult<()> {
    let source_map_dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::NonNullable),
        true,
    )?;
    let source = map_array_from_rows(
        source_map_dtype,
        Nullability::NonNullable,
        [
            Some(vec![
                (
                    Scalar::primitive(1i32, Nullability::NonNullable),
                    Scalar::utf8("one", Nullability::NonNullable),
                ),
                (
                    Scalar::primitive(2i32, Nullability::NonNullable),
                    Scalar::utf8("two", Nullability::NonNullable),
                ),
            ]),
            Some(vec![]),
        ],
    )?
    .into_array();
    let target_dtype = DType::map(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        true,
        Nullability::Nullable,
    )?;
    let target_map_dtype = target_dtype
        .as_map_opt()
        .vortex_expect("target dtype is map")
        .clone();
    let cast = source.cast(target_dtype)?;
    let expected = map_array_from_rows(
        target_map_dtype,
        Nullability::Nullable,
        [
            Some(vec![
                (
                    Scalar::primitive(1i64, Nullability::NonNullable),
                    Scalar::utf8("one", Nullability::Nullable),
                ),
                (
                    Scalar::primitive(2i64, Nullability::NonNullable),
                    Scalar::utf8("two", Nullability::Nullable),
                ),
            ]),
            Some(vec![]),
        ],
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert!(cast.is::<Map>());
    assert_arrays_eq!(cast, expected, &mut ctx);

    Ok(())
}

#[test]
fn cast_can_drop_but_not_create_sortedness_assertion() -> VortexResult<()> {
    let sorted_source = sample_array()?.into_array();
    let unsorted_target = DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
        Nullability::Nullable,
    )?;
    let cast = sorted_source.cast(unsorted_target)?;
    let expected = map_array_from_rows(
        MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
        )?,
        Nullability::Nullable,
        [
            Some(vec![
                (key(2), value(Some("two"))),
                (key(1), value(None)),
                (key(2), value(Some("duplicate"))),
            ]),
            Some(vec![]),
            None,
        ],
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(cast, expected, &mut ctx);

    let unsorted_map_dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
    )?;
    let unsorted_source = map_array_from_rows(
        unsorted_map_dtype,
        Nullability::Nullable,
        [Some(vec![(key(1), value(Some("one")))])],
    )?
    .into_array();
    let sorted_target = DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        true,
        Nullability::Nullable,
    )?;
    assert!(unsorted_source.cast(sorted_target).is_err());

    Ok(())
}

#[test]
fn null_map_cast_cannot_create_sortedness_assertion() -> VortexResult<()> {
    let unsorted_map_dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
    )?;
    let unsorted_dtype = DType::Map(unsorted_map_dtype.clone(), Nullability::Nullable);
    let sorted_dtype = DType::map(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        true,
        Nullability::Nullable,
    )?;
    let scalar = Scalar::null(unsorted_dtype);

    assert!(scalar.cast(&sorted_dtype).is_err());
    let constant_cast = ConstantArray::new(scalar, 2)
        .into_array()
        .cast(sorted_dtype.clone())?;
    let mut ctx = array_session().create_execution_ctx();
    assert!(constant_cast.execute::<Canonical>(&mut ctx).is_err());

    let all_null =
        map_array_from_rows(unsorted_map_dtype, Nullability::Nullable, [None, None])?.into_array();
    assert!(all_null.cast(sorted_dtype).is_err());

    Ok(())
}

#[test]
fn filter_preserves_duplicate_map_keys() -> VortexResult<()> {
    let source = sample_array()?.into_array();
    let filtered = source.filter(Mask::from_iter([true, false, false]))?;
    let expected = map_array_from_rows(
        map_dtype()?,
        Nullability::Nullable,
        [Some(vec![
            (key(2), value(Some("two"))),
            (key(1), value(None)),
            (key(2), value(Some("duplicate"))),
        ])],
    )?;
    let mut ctx = array_session().create_execution_ctx();

    assert_arrays_eq!(filtered, expected, &mut ctx);

    Ok(())
}

#[test]
fn builder_appends_existing_map_arrays() -> VortexResult<()> {
    let source = sample_array()?;
    let mut builder = MapBuilder::<u64, u64>::with_capacity(
        source.map_dtype().clone(),
        source.dtype().nullability(),
        0,
    );
    let mut ctx = array_session().create_execution_ctx();
    builder.append_map_array(source.as_view(), &mut ctx)?;
    builder.append_map_array(source.as_view(), &mut ctx)?;
    let array = builder.finish_into_map();

    assert_eq!(array.len(), 6);
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        source.execute_scalar(0, &mut ctx)?
    );
    assert!(array.execute_scalar(2, &mut ctx)?.is_null());
    assert_eq!(
        array.execute_scalar(3, &mut ctx)?,
        source.execute_scalar(0, &mut ctx)?
    );

    Ok(())
}

#[test]
fn canonicalizes_empty_constant_and_chunked_maps() -> VortexResult<()> {
    let map_dtype = map_dtype()?;
    let dtype = DType::Map(map_dtype, Nullability::Nullable);
    let empty = Canonical::empty(&dtype);
    assert!(empty.as_map().is_empty());

    let mut ctx = array_session().create_execution_ctx();
    let constant = ConstantArray::new(sample_scalar(dtype.clone())?, 2)
        .into_array()
        .execute::<Canonical>(&mut ctx)?;
    assert_eq!(constant.as_map().len(), 2);

    let first = sample_array()?.into_array();
    let second = sample_array()?.into_array();
    let chunked = ChunkedArray::try_new(vec![first, second], dtype)?.into_array();
    let canonical = chunked.execute::<Canonical>(&mut ctx)?;
    assert_eq!(canonical.as_map().len(), 6);

    Ok(())
}

#[test]
fn serde_roundtrip_uses_registered_map_vtable() -> VortexResult<()> {
    let session = array_session();
    assert!(session.arrays().registry().contains_key(&Map.id()));

    let array = sample_array()?.into_array();
    let dtype = array.dtype().clone();
    let len = array.len();
    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buffer in serialized {
        concat.extend_from_slice(buffer.as_ref());
    }

    let serialized = SerializedArray::try_from(concat.freeze())?;
    let decoded =
        serialized.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;
    assert!(decoded.is::<Map>());

    let mut ctx = session.create_execution_ctx();
    for index in 0..len {
        assert_eq!(
            decoded.execute_scalar(index, &mut ctx)?,
            array.execute_scalar(index, &mut ctx)?
        );
    }

    Ok(())
}
