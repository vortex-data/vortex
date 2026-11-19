// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Snapshots of the storage dtypes that the default mapping options derive.

use insta::assert_snapshot;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;

use super::LEGACY_PROTO;
use super::all_types_dtype;
use super::describe_fields;
use super::protobuf_dtype;

#[test]
fn test_storage_dtype_proto3() -> VortexResult<()> {
    let dtype = all_types_dtype(Nullability::NonNullable)?;
    assert_snapshot!(describe_fields(dtype.storage_dtype()), @r"
    f_double: f64
    f_float: f32
    f_int32: i32
    f_int64: i64
    f_uint32: u32
    f_uint64: u64
    f_sint32: i32
    f_sint64: i64
    f_fixed32: u32
    f_fixed64: u64
    f_sfixed32: i32
    f_sfixed64: i64
    f_bool: bool
    f_string: utf8
    f_bytes: binary
    f_enum: i32
    inner: {name=utf8, count=i64}?
    outer: {inner={name=utf8, count=i64}?, items=list({name=utf8, count=i64})}?
    r_int32: list(i32)
    r_string: list(utf8)
    r_inner: list({name=utf8, count=i64})
    m_inner: list({key=utf8, value={name=utf8, count=i64}})
    m_labels: list({key=i32, value=utf8})
    c_text: utf8?
    c_inner: {name=utf8, count=i64}?
    o_int32: i32?
    o_string: utf8?
    created: {seconds=i64, nanos=i32}?
    ");
    Ok(())
}

#[test]
fn test_storage_dtype_proto2() -> VortexResult<()> {
    let dtype = protobuf_dtype(&[LEGACY_PROTO], "legacy.Legacy", Nullability::NonNullable)?;
    assert_snapshot!(describe_fields(dtype.storage_dtype()), @r"
    a: i32?
    b: utf8?
    c: i64?
    d: list(u32)
    settings: {flag=bool?}?
    ");
    Ok(())
}

#[test]
fn test_storage_dtype_top_level_nullability() -> VortexResult<()> {
    let non_nullable = all_types_dtype(Nullability::NonNullable)?;
    let nullable = all_types_dtype(Nullability::Nullable)?;
    assert!(!non_nullable.storage_dtype().is_nullable());
    assert!(nullable.storage_dtype().is_nullable());
    assert_eq!(
        nullable.storage_dtype(),
        &non_nullable.storage_dtype().as_nullable()
    );
    Ok(())
}

#[test]
fn test_dtype_display() -> VortexResult<()> {
    let dtype = DType::Extension(all_types_dtype(Nullability::Nullable)?.erased());
    assert!(
        dtype
            .to_string()
            .starts_with("vortex.protobuf[test.AllTypes]({"),
        "{dtype}"
    );
    Ok(())
}
