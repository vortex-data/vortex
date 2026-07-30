// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_session::registry::ReadContext;

use crate::Array;
use crate::ArrayContext;
use crate::ArrayParts;
use crate::ArrayVTable;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::Map;
use crate::arrays::MapArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapData;
use crate::arrays::map::MapDataParts;
use crate::builders::ArrayBuilder;
use crate::builders::MapBuilder;
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

fn sample_array() -> VortexResult<MapArray> {
    let map_dtype = map_dtype()?;
    let dtype = DType::Map(map_dtype.clone(), Nullability::Nullable);
    let mut builder = MapBuilder::<u64, u64>::with_capacity(map_dtype, Nullability::Nullable, 3);
    builder.append_scalar(&sample_scalar(dtype.clone())?)?;
    builder.append_scalar(&Scalar::try_map(dtype.clone(), [])?)?;
    builder.append_scalar(&Scalar::null(dtype))?;
    Ok(builder.finish_into_map())
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
