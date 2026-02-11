// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type mapping between Vortex DType and ClickHouse types.
//!
//! # ClickHouse Type System
//!
//! ClickHouse has a rich type system that includes:
//! - Numeric types: Int8-Int256, UInt8-UInt256, Float32, Float64
//! - String types: String, FixedString(N)
//! - Date/Time: Date, Date32, DateTime, DateTime64
//! - Compound: Array(T), Tuple(T1, T2, ...), Map(K, V), Nested
//! - Special: Nullable(T), LowCardinality(T), Enum
//!
//! # Mapping Strategy
//!
//! We map Vortex types to the most appropriate ClickHouse types:
//! - Primitive types map directly to corresponding ClickHouse numeric types
//! - Vortex `Utf8` maps to ClickHouse `String`
//! - Vortex `List` maps to ClickHouse `Array`
//! - Vortex `Struct` maps to ClickHouse `Tuple`
//! - Temporal extensions map to DateTime64 with appropriate precision

use std::sync::Arc;

use vortex::dtype::Nullability::{NonNullable, Nullable};
use vortex::dtype::{DType, DecimalDType, FieldName, FieldNames, PType, StructFields};
use vortex::error::{VortexResult, vortex_bail};

use crate::ext_types::UUID;
use crate::ext_types::{BigInt, BigIntType};
use crate::ext_types::{
    ClickHouseDate, ClickHouseDateTime, ClickHouseEnum, ClickHouseFixedString,
    ClickHouseLowCardinality, DateTimeMetadata, LowCardinalityMetadata,
};
use crate::ext_types::{Geo, GeoType};
use crate::ext_types::{IPAddress, IPAddressType};

/// Convert a ClickHouse type string to Vortex DType.
///
/// # Arguments
/// * `ch_type` - ClickHouse type string (e.g., "Int32", "String", "Array(UInt64)")
///
/// # Returns
/// The corresponding Vortex DType.
pub fn clickhouse_type_to_vortex(ch_type: &str) -> VortexResult<DType> {
    let ch_type = ch_type.trim();

    // Handle Nullable wrapper
    if let Some(inner) = ch_type
        .strip_prefix("Nullable(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let inner_dtype = clickhouse_type_to_vortex(inner)?;
        return Ok(inner_dtype.with_nullability(Nullable));
    }

    // Handle LowCardinality wrapper — preserve as extension type
    if let Some(inner) = ch_type
        .strip_prefix("LowCardinality(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let inner_dtype = clickhouse_type_to_vortex(inner)?;
        let storage = match &inner_dtype {
            DType::Extension(ext) => ext.storage_dtype().clone(),
            other => other.clone(),
        };
        return Ok(ClickHouseLowCardinality::dtype(
            inner.trim().to_string(),
            storage,
            inner_dtype.nullability(),
        ));
    }

    match ch_type {
        // Boolean
        "Bool" => Ok(DType::Bool(NonNullable)),

        // Signed integers
        "Int8" => Ok(DType::Primitive(PType::I8, NonNullable)),
        "Int16" => Ok(DType::Primitive(PType::I16, NonNullable)),
        "Int32" => Ok(DType::Primitive(PType::I32, NonNullable)),
        "Int64" => Ok(DType::Primitive(PType::I64, NonNullable)),
        // Large signed integers - use Extension types for disambiguation
        "Int128" => Ok(BigInt::dtype(BigIntType::Int128, NonNullable)),
        "Int256" => Ok(BigInt::dtype(BigIntType::Int256, NonNullable)),

        // Unsigned integers
        "UInt8" => Ok(DType::Primitive(PType::U8, NonNullable)),
        "UInt16" => Ok(DType::Primitive(PType::U16, NonNullable)),
        "UInt32" => Ok(DType::Primitive(PType::U32, NonNullable)),
        "UInt64" => Ok(DType::Primitive(PType::U64, NonNullable)),
        // Large unsigned integers - use Extension types for disambiguation
        "UInt128" => Ok(BigInt::dtype(BigIntType::UInt128, NonNullable)),
        "UInt256" => Ok(BigInt::dtype(BigIntType::UInt256, NonNullable)),

        // Floating point
        "Float32" => Ok(DType::Primitive(PType::F32, NonNullable)),
        "Float64" => Ok(DType::Primitive(PType::F64, NonNullable)),

        // IP Address types
        // IPv4 is stored as UInt32 (4 bytes), same as Parquet/Arrow
        "IPv4" => Ok(DType::Primitive(PType::U32, NonNullable)),
        // IPv6 is stored as 16-byte fixed binary, using Extension type for disambiguation
        "IPv6" => Ok(IPAddress::dtype(IPAddressType::IPv6, NonNullable)),

        // UUID type - using Extension type for disambiguation from Int128
        "UUID" => Ok(UUID::dtype(NonNullable)),

        // String types
        "String" => Ok(DType::Utf8(NonNullable)),

        // Date/Time types - use extension types to preserve Date semantics
        "Date" => {
            // Date is days since 1970-01-01, stored as UInt16
            Ok(ClickHouseDate::dtype(false, NonNullable))
        }
        "Date32" => {
            // Date32 is days since 1970-01-01, stored as Int32
            Ok(ClickHouseDate::dtype(true, NonNullable))
        }

        // GEO types - stored as WKB-encoded String in Vortex using Extension type.
        // The C++ side converts GEO columns to/from WKB binary strings.
        // The Geo Extension type preserves the type name so the C++ read side
        // can reconstruct the GEO column.
        "Point" => Ok(Geo::dtype(GeoType::Point, NonNullable)),
        "LineString" => Ok(Geo::dtype(GeoType::LineString, NonNullable)),
        "Ring" => Ok(Geo::dtype(GeoType::Ring, NonNullable)),
        "Polygon" => Ok(Geo::dtype(GeoType::Polygon, NonNullable)),
        "MultiLineString" => Ok(Geo::dtype(GeoType::MultiLineString, NonNullable)),
        "MultiPolygon" => Ok(Geo::dtype(GeoType::MultiPolygon, NonNullable)),

        // Handle complex types
        _ => parse_complex_clickhouse_type(ch_type),
    }
}

/// Parse an Enum8 or Enum16 type string (e.g., `Enum8('a' = 1, 'b' = 2)`).
///
/// In native mode the enum names are discarded and the underlying integer type is returned.
fn parse_enum_type(ch_type: &str) -> VortexResult<DType> {
    if ch_type.starts_with("Enum8(") {
        Ok(DType::Primitive(PType::I8, NonNullable))
    } else if ch_type.starts_with("Enum16(") {
        Ok(DType::Primitive(PType::I16, NonNullable))
    } else {
        vortex_bail!("Not an Enum type: {}", ch_type)
    }
}

/// Parse a `Map(K, V)` type string into `List(Struct([key K, value V]))`.
///
/// This follows the standard Arrow/Parquet convention for representing Maps.
fn parse_map_type(inner: &str) -> VortexResult<DType> {
    let parts = split_balanced_commas(inner);
    if parts.len() != 2 {
        vortex_bail!(
            "Map type expects exactly 2 type arguments, got {}",
            parts.len()
        );
    }
    let key_dtype = clickhouse_type_to_vortex(parts[0].trim())?;
    let value_dtype = clickhouse_type_to_vortex(parts[1].trim())?;

    // Build Struct([key K, value V])
    let field_names = FieldNames::from(vec![FieldName::from("key"), FieldName::from("value")]);
    let entry_struct = DType::Struct(
        StructFields::new(field_names, vec![key_dtype, value_dtype]),
        NonNullable,
    );
    // Wrap in List
    Ok(DType::List(Arc::new(entry_struct), NonNullable))
}

/// Parse complex ClickHouse types like Array, Tuple, DateTime64, Decimal, Enum, Map, etc.
fn parse_complex_clickhouse_type(ch_type: &str) -> VortexResult<DType> {
    // Array(T)
    if let Some(inner) = ch_type
        .strip_prefix("Array(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let element_dtype = clickhouse_type_to_vortex(inner)?;
        return Ok(DType::List(Arc::new(element_dtype), NonNullable));
    }

    // Map(K, V) → List(Struct([key K, value V]))
    if let Some(inner) = ch_type
        .strip_prefix("Map(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return parse_map_type(inner);
    }

    // FixedString(N)
    if ch_type.starts_with("FixedString(") {
        // Treat FixedString as String for now
        return Ok(DType::Utf8(NonNullable));
    }

    // DateTime (seconds precision) - use extension type to preserve DateTime semantics
    if ch_type == "DateTime" || ch_type.starts_with("DateTime(") {
        let timezone = if ch_type.starts_with("DateTime('") {
            ch_type
                .strip_prefix("DateTime('")
                .and_then(|s| s.strip_suffix("')"))
                .map(|s| s.to_string())
        } else {
            None
        };
        let metadata = DateTimeMetadata {
            precision: 0,
            timezone,
        };
        return Ok(ClickHouseDateTime::dtype(metadata, NonNullable));
    }

    // DateTime64(precision, [timezone]) - use extension type to preserve DateTime64 semantics
    if ch_type.starts_with("DateTime64(") {
        let inner = ch_type
            .strip_prefix("DateTime64(")
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or("3");
        let parts: Vec<&str> = inner.splitn(2, ',').collect();
        let precision: u8 = parts[0].trim().parse().unwrap_or(3);
        let timezone = if parts.len() > 1 {
            let tz = parts[1].trim();
            tz.strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
                .map(|s| s.to_string())
        } else {
            None
        };
        let metadata = DateTimeMetadata {
            precision,
            timezone,
        };
        return Ok(ClickHouseDateTime::dtype(metadata, NonNullable));
    }

    // Decimal(P, S) or Decimal32/64/128/256
    if ch_type.starts_with("Decimal") {
        return parse_decimal_type(ch_type);
    }

    // Enum8/Enum16 → Primitive(I8/I16) in native mode
    if ch_type.starts_with("Enum8(") || ch_type.starts_with("Enum16(") {
        return parse_enum_type(ch_type);
    }

    // Tuple(T1, T2, ...)
    if let Some(inner) = ch_type
        .strip_prefix("Tuple(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let fields = parse_tuple_fields(inner)?;
        return Ok(DType::Struct(fields, NonNullable));
    }

    vortex_bail!("Unsupported ClickHouse type: {}", ch_type)
}

/// Split a string by commas, respecting nested parentheses.
///
/// Only splits at commas where the parenthesis depth is zero.
/// For example, `"a Array(Int32), b String"` splits into `["a Array(Int32)", " b String"]`.
fn split_balanced_commas(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}

/// Parse tuple field definitions.
fn parse_tuple_fields(fields_str: &str) -> VortexResult<StructFields> {
    let mut dtypes = Vec::new();
    let mut names = Vec::new();

    for (i, field) in split_balanced_commas(fields_str).into_iter().enumerate() {
        let field = field.trim();
        // Check if field has name: "name Type" or just "Type"
        if let Some((name, type_str)) = field.split_once(' ') {
            names.push(FieldName::from(name.trim()));
            dtypes.push(clickhouse_type_to_vortex(type_str.trim())?);
        } else {
            names.push(FieldName::from(format!("_{}", i)));
            dtypes.push(clickhouse_type_to_vortex(field)?);
        }
    }

    let field_names = FieldNames::from(names);
    Ok(StructFields::new(field_names, dtypes))
}

/// Parse ClickHouse Decimal types into Vortex Decimal DType.
///
/// ClickHouse supports these Decimal types:
/// - Decimal(P, S) - Generic decimal with precision P and scale S
/// - Decimal32(S) - Decimal with precision 9 and scale S (stored as Int32)
/// - Decimal64(S) - Decimal with precision 18 and scale S (stored as Int64)
/// - Decimal128(S) - Decimal with precision 38 and scale S (stored as Int128)
/// - Decimal256(S) - Decimal with precision 76 and scale S (stored as Int256)
fn parse_decimal_type(ch_type: &str) -> VortexResult<DType> {
    // Decimal(P, S)
    if let Some(inner) = ch_type
        .strip_prefix("Decimal(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            vortex_bail!(
                "Invalid Decimal type '{}': expected Decimal(precision, scale)",
                ch_type
            );
        }
        let precision: u8 = parts[0].parse().map_err(|_| {
            vortex::error::vortex_err!(
                "Invalid precision '{}' in Decimal type '{}'",
                parts[0],
                ch_type
            )
        })?;
        let scale: i8 = parts[1].parse().map_err(|_| {
            vortex::error::vortex_err!("Invalid scale '{}' in Decimal type '{}'", parts[1], ch_type)
        })?;
        let decimal_dtype = DecimalDType::try_new(precision, scale)?;
        return Ok(DType::Decimal(decimal_dtype, NonNullable));
    }

    // Decimal32(S) - precision 9
    if let Some(inner) = ch_type
        .strip_prefix("Decimal32(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let scale: i8 = inner.trim().parse().map_err(|_| {
            vortex::error::vortex_err!("Invalid scale '{}' in Decimal32 type '{}'", inner, ch_type)
        })?;
        let decimal_dtype = DecimalDType::try_new(9, scale)?;
        return Ok(DType::Decimal(decimal_dtype, NonNullable));
    }

    // Decimal64(S) - precision 18
    if let Some(inner) = ch_type
        .strip_prefix("Decimal64(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let scale: i8 = inner.trim().parse().map_err(|_| {
            vortex::error::vortex_err!("Invalid scale '{}' in Decimal64 type '{}'", inner, ch_type)
        })?;
        let decimal_dtype = DecimalDType::try_new(18, scale)?;
        return Ok(DType::Decimal(decimal_dtype, NonNullable));
    }

    // Decimal128(S) - precision 38
    if let Some(inner) = ch_type
        .strip_prefix("Decimal128(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let scale: i8 = inner.trim().parse().map_err(|_| {
            vortex::error::vortex_err!("Invalid scale '{}' in Decimal128 type '{}'", inner, ch_type)
        })?;
        let decimal_dtype = DecimalDType::try_new(38, scale)?;
        return Ok(DType::Decimal(decimal_dtype, NonNullable));
    }

    // Decimal256(S) - precision 76
    if let Some(inner) = ch_type
        .strip_prefix("Decimal256(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let scale: i8 = inner.trim().parse().map_err(|_| {
            vortex::error::vortex_err!("Invalid scale '{}' in Decimal256 type '{}'", inner, ch_type)
        })?;
        let decimal_dtype = DecimalDType::try_new(76, scale)?;
        return Ok(DType::Decimal(decimal_dtype, NonNullable));
    }

    vortex_bail!("Unsupported Decimal type: {}", ch_type)
}

/// Convert a Vortex DType to ClickHouse type string.
///
/// # Arguments
/// * `dtype` - The Vortex DType to convert
///
/// # Returns
/// The corresponding ClickHouse type string.
pub fn vortex_to_clickhouse_type(dtype: &DType) -> VortexResult<String> {
    let base_type = match dtype {
        DType::Null => return Ok("Nothing".to_string()),
        DType::Bool(_) => "Bool".to_string(),
        DType::Primitive(ptype, _) => ptype_to_clickhouse(*ptype),
        DType::Utf8(_) => "String".to_string(),
        DType::Binary(_) => "String".to_string(), // ClickHouse uses String for binary
        DType::Struct(fields, _) => {
            let mut field_strs = Vec::new();
            for (name, dtype) in fields.names().iter().zip(fields.fields()) {
                let ch_type = vortex_to_clickhouse_type(&dtype)?;
                field_strs.push(format!("{} {}", name, ch_type));
            }
            format!("Tuple({})", field_strs.join(", "))
        }
        DType::List(elem, _) => {
            let elem_type = vortex_to_clickhouse_type(elem)?;
            format!("Array({})", elem_type)
        }
        DType::FixedSizeList(elem, size, _) => {
            // Check if this is a big integer type (FixedSizeList<u8, 16/32>)
            if matches!(elem.as_ref(), DType::Primitive(PType::U8, _)) {
                match *size {
                    16 => "Int128".to_string(), // Default to signed for 128-bit
                    32 => "Int256".to_string(), // Default to signed for 256-bit
                    _ => {
                        let elem_type = vortex_to_clickhouse_type(elem)?;
                        format!("Array({})", elem_type)
                    }
                }
            } else {
                let elem_type = vortex_to_clickhouse_type(elem)?;
                format!("Array({})", elem_type) // ClickHouse doesn't have FixedSizeArray
            }
        }
        DType::Extension(_ext) => {
            // Check for BigInt extension type
            if let Some(bigint_type) = BigInt::try_get_type(dtype) {
                bigint_type.clickhouse_type_name().to_string()
            }
            // Check for Geo extension type
            else if let Some(geo_type) = Geo::try_get_type(dtype) {
                geo_type.clickhouse_type_name().to_string()
            }
            // Check for UUID extension type
            else if UUID::is_uuid(dtype) {
                UUID::clickhouse_type_name().to_string()
            }
            // Check for IPAddress extension type
            else if let Some(ip_type) = IPAddress::try_get_type(dtype) {
                ip_type.clickhouse_type_name().to_string()
            }
            // Check for ClickHouse Enum extension type
            else if let Some(metadata) = ClickHouseEnum::try_get_metadata(dtype) {
                ClickHouseEnum::to_clickhouse_type(&metadata)
            }
            // Check for ClickHouse DateTime extension type
            else if let Some(metadata) = ClickHouseDateTime::try_get_metadata(dtype) {
                ClickHouseDateTime::to_clickhouse_type(&metadata)
            }
            // Check for ClickHouse Date extension type
            else if let Some(metadata) = ClickHouseDate::try_get_metadata(dtype) {
                ClickHouseDate::to_clickhouse_type(&metadata)
            }
            // Check for ClickHouse LowCardinality extension type
            else if let Some(metadata) = ClickHouseLowCardinality::try_get_metadata(dtype) {
                ClickHouseLowCardinality::to_clickhouse_type(&metadata)
            }
            // Check for ClickHouse FixedString extension type
            else if let Some(metadata) = ClickHouseFixedString::try_get_metadata(dtype) {
                ClickHouseFixedString::to_clickhouse_type(&metadata)
            } else {
                // For other extension types, return String as a fallback
                "String".to_string()
            }
        }
        DType::Decimal(decimal_dtype, _) => {
            // Map Vortex Decimal to ClickHouse Decimal
            format!(
                "Decimal({}, {})",
                decimal_dtype.precision(),
                decimal_dtype.scale()
            )
        }
    };

    // Wrap in Nullable if needed
    if dtype.is_nullable() && !matches!(dtype, DType::Null) {
        Ok(format!("Nullable({})", base_type))
    } else {
        Ok(base_type)
    }
}

/// Convert Vortex PType to ClickHouse type string.
fn ptype_to_clickhouse(ptype: PType) -> String {
    match ptype {
        PType::I8 => "Int8",
        PType::I16 => "Int16",
        PType::I32 => "Int32",
        PType::I64 => "Int64",
        PType::U8 => "UInt8",
        PType::U16 => "UInt16",
        PType::U32 => "UInt32",
        PType::U64 => "UInt64",
        PType::F16 => "Float32", // ClickHouse doesn't have Float16, upcast
        PType::F32 => "Float32",
        PType::F64 => "Float64",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use vortex::dtype::Nullability::NonNullable;

    use super::*;
    use crate::ext_types::{IPAddressType, UUID};

    // ==========================================================================
    // ClickHouse -> Vortex conversion tests
    // ==========================================================================

    #[test]
    fn test_primitive_type_conversion() {
        // ClickHouse -> Vortex (bare types are NonNullable)
        assert!(matches!(
            clickhouse_type_to_vortex("Int32").unwrap(),
            DType::Primitive(PType::I32, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("UInt64").unwrap(),
            DType::Primitive(PType::U64, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Float64").unwrap(),
            DType::Primitive(PType::F64, NonNullable)
        ));

        // Vortex -> ClickHouse
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I32, NonNullable)).unwrap(),
            "Int32"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I32, Nullable)).unwrap(),
            "Nullable(Int32)"
        );
    }

    #[test]
    fn test_all_signed_integers() {
        // ClickHouse -> Vortex (bare types are NonNullable)
        assert!(matches!(
            clickhouse_type_to_vortex("Int8").unwrap(),
            DType::Primitive(PType::I8, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Int16").unwrap(),
            DType::Primitive(PType::I16, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Int32").unwrap(),
            DType::Primitive(PType::I32, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Int64").unwrap(),
            DType::Primitive(PType::I64, NonNullable)
        ));

        // Vortex -> ClickHouse
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I8, NonNullable)).unwrap(),
            "Int8"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I16, NonNullable)).unwrap(),
            "Int16"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I32, NonNullable)).unwrap(),
            "Int32"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I64, NonNullable)).unwrap(),
            "Int64"
        );
    }

    #[test]
    fn test_all_unsigned_integers() {
        // ClickHouse -> Vortex (bare types are NonNullable)
        assert!(matches!(
            clickhouse_type_to_vortex("UInt8").unwrap(),
            DType::Primitive(PType::U8, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("UInt16").unwrap(),
            DType::Primitive(PType::U16, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("UInt32").unwrap(),
            DType::Primitive(PType::U32, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("UInt64").unwrap(),
            DType::Primitive(PType::U64, NonNullable)
        ));

        // Vortex -> ClickHouse
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::U8, NonNullable)).unwrap(),
            "UInt8"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::U16, NonNullable)).unwrap(),
            "UInt16"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::U32, NonNullable)).unwrap(),
            "UInt32"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::U64, NonNullable)).unwrap(),
            "UInt64"
        );
    }

    #[test]
    fn test_floating_point_types() {
        // ClickHouse -> Vortex (bare types are NonNullable)
        assert!(matches!(
            clickhouse_type_to_vortex("Float32").unwrap(),
            DType::Primitive(PType::F32, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Float64").unwrap(),
            DType::Primitive(PType::F64, NonNullable)
        ));

        // Vortex -> ClickHouse
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::F32, NonNullable)).unwrap(),
            "Float32"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::F64, NonNullable)).unwrap(),
            "Float64"
        );
        // F16 should upcast to Float32
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::F16, NonNullable)).unwrap(),
            "Float32"
        );
    }

    #[test]
    fn test_boolean_type() {
        // ClickHouse -> Vortex (bare type is NonNullable)
        assert!(matches!(
            clickhouse_type_to_vortex("Bool").unwrap(),
            DType::Bool(NonNullable)
        ));

        // Vortex -> ClickHouse
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Bool(NonNullable)).unwrap(),
            "Bool"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Bool(Nullable)).unwrap(),
            "Nullable(Bool)"
        );
    }

    #[test]
    fn test_string_type_conversion() {
        assert!(matches!(
            clickhouse_type_to_vortex("String").unwrap(),
            DType::Utf8(NonNullable)
        ));

        assert_eq!(
            vortex_to_clickhouse_type(&DType::Utf8(NonNullable)).unwrap(),
            "String"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Utf8(Nullable)).unwrap(),
            "Nullable(String)"
        );
    }

    #[test]
    fn test_fixed_string_type() {
        // FixedString(N) should map to Utf8 NonNullable (bare type)
        assert!(matches!(
            clickhouse_type_to_vortex("FixedString(10)").unwrap(),
            DType::Utf8(NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("FixedString(256)").unwrap(),
            DType::Utf8(NonNullable)
        ));
    }

    #[test]
    fn test_binary_type() {
        // Vortex Binary -> ClickHouse String
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Binary(NonNullable)).unwrap(),
            "String"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Binary(Nullable)).unwrap(),
            "Nullable(String)"
        );
    }

    // ==========================================================================
    // Date/Time type tests
    // ==========================================================================

    #[test]
    fn test_date_types() {
        // Date -> Extension(clickhouse.date)
        let dtype = clickhouse_type_to_vortex("Date").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseDate::try_get_metadata(&dtype).unwrap();
        assert!(!metadata.is_date32);

        // Date32 -> Extension(clickhouse.date)
        let dtype = clickhouse_type_to_vortex("Date32").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseDate::try_get_metadata(&dtype).unwrap();
        assert!(metadata.is_date32);
    }

    #[test]
    fn test_datetime_types() {
        // DateTime -> Extension(clickhouse.datetime)
        let dtype = clickhouse_type_to_vortex("DateTime").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.precision, 0);
        assert_eq!(metadata.timezone, None);

        // DateTime with timezone
        let dtype = clickhouse_type_to_vortex("DateTime('UTC')").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.precision, 0);
        assert_eq!(metadata.timezone, Some("UTC".to_string()));

        let dtype = clickhouse_type_to_vortex("DateTime('Asia/Shanghai')").unwrap();
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.timezone, Some("Asia/Shanghai".to_string()));
    }

    #[test]
    fn test_datetime64_types() {
        // DateTime64(3) -> Extension(clickhouse.datetime)
        let dtype = clickhouse_type_to_vortex("DateTime64(3)").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.precision, 3);
        assert_eq!(metadata.timezone, None);

        let dtype = clickhouse_type_to_vortex("DateTime64(6, 'UTC')").unwrap();
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.precision, 6);
        assert_eq!(metadata.timezone, Some("UTC".to_string()));

        let dtype = clickhouse_type_to_vortex("DateTime64(9, 'America/New_York')").unwrap();
        let metadata = ClickHouseDateTime::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.precision, 9);
        assert_eq!(metadata.timezone, Some("America/New_York".to_string()));
    }

    // ==========================================================================
    // Complex type tests
    // ==========================================================================

    #[test]
    fn test_array_type_conversion() {
        let dtype = clickhouse_type_to_vortex("Array(Int32)").unwrap();
        assert!(matches!(dtype, DType::List(_, NonNullable)));

        let list_dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            NonNullable,
        );
        assert_eq!(
            vortex_to_clickhouse_type(&list_dtype).unwrap(),
            "Array(Int32)"
        );
    }

    #[test]
    fn test_array_of_various_types() {
        // Array of String
        let dtype = clickhouse_type_to_vortex("Array(String)").unwrap();
        if let DType::List(elem, _) = dtype {
            assert!(matches!(elem.as_ref(), DType::Utf8(NonNullable)));
        } else {
            panic!("Expected List type");
        }

        // Array of Float64
        let dtype = clickhouse_type_to_vortex("Array(Float64)").unwrap();
        if let DType::List(elem, _) = dtype {
            assert!(matches!(
                elem.as_ref(),
                DType::Primitive(PType::F64, NonNullable)
            ));
        } else {
            panic!("Expected List type");
        }

        // Array of Bool
        let dtype = clickhouse_type_to_vortex("Array(Bool)").unwrap();
        if let DType::List(elem, _) = dtype {
            assert!(matches!(elem.as_ref(), DType::Bool(NonNullable)));
        } else {
            panic!("Expected List type");
        }
    }

    #[test]
    fn test_nested_array() {
        // Array(Array(Int32))
        let dtype = clickhouse_type_to_vortex("Array(Array(Int32))").unwrap();
        if let DType::List(outer_elem, _) = dtype {
            if let DType::List(inner_elem, _) = outer_elem.as_ref() {
                assert!(matches!(
                    inner_elem.as_ref(),
                    DType::Primitive(PType::I32, NonNullable)
                ));
            } else {
                panic!("Expected nested List type");
            }
        } else {
            panic!("Expected List type");
        }
    }

    #[test]
    fn test_tuple_unnamed_fields() {
        // Tuple(Int32, String)
        let dtype = clickhouse_type_to_vortex("Tuple(Int32, String)").unwrap();
        if let DType::Struct(fields, _) = dtype {
            assert_eq!(fields.nfields(), 2);
            // Unnamed fields should get default names
            assert_eq!(fields.field_name(0).unwrap().as_ref(), "_0");
            assert_eq!(fields.field_name(1).unwrap().as_ref(), "_1");
            assert!(matches!(
                fields.field_by_index(0).unwrap(),
                DType::Primitive(PType::I32, NonNullable)
            ));
            assert!(matches!(
                fields.field_by_index(1).unwrap(),
                DType::Utf8(NonNullable)
            ));
        } else {
            panic!("Expected Struct type");
        }
    }

    #[test]
    fn test_tuple_named_fields() {
        // Tuple(id Int32, name String)
        let dtype = clickhouse_type_to_vortex("Tuple(id Int32, name String)").unwrap();
        if let DType::Struct(fields, _) = dtype {
            assert_eq!(fields.nfields(), 2);
            assert_eq!(fields.field_name(0).unwrap().as_ref(), "id");
            assert_eq!(fields.field_name(1).unwrap().as_ref(), "name");
            assert!(matches!(
                fields.field_by_index(0).unwrap(),
                DType::Primitive(PType::I32, NonNullable)
            ));
            assert!(matches!(
                fields.field_by_index(1).unwrap(),
                DType::Utf8(NonNullable)
            ));
        } else {
            panic!("Expected Struct type");
        }
    }

    #[test]
    fn test_struct_to_tuple_conversion() {
        // Build a Vortex struct
        let names = FieldNames::from(vec![
            FieldName::from("col_a"),
            FieldName::from("col_b"),
            FieldName::from("col_c"),
        ]);
        let dtypes = vec![
            DType::Primitive(PType::I64, NonNullable),
            DType::Utf8(NonNullable),
            DType::Bool(NonNullable),
        ];
        let struct_dtype = DType::Struct(StructFields::new(names, dtypes), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&struct_dtype).unwrap();
        assert_eq!(ch_type, "Tuple(col_a Int64, col_b String, col_c Bool)");
    }

    #[test]
    fn test_fixed_size_list() {
        // FixedSizeList should map to Array
        let fsl_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, NonNullable)),
            4,
            NonNullable,
        );
        assert_eq!(
            vortex_to_clickhouse_type(&fsl_dtype).unwrap(),
            "Array(Float32)"
        );
    }

    // ==========================================================================
    // Nullable wrapper tests
    // ==========================================================================

    #[test]
    fn test_nullable_wrapper() {
        let dtype = clickhouse_type_to_vortex("Nullable(Int32)").unwrap();
        assert!(dtype.is_nullable());
        assert!(matches!(dtype, DType::Primitive(PType::I32, Nullable)));
    }

    #[test]
    fn test_nullable_string() {
        let dtype = clickhouse_type_to_vortex("Nullable(String)").unwrap();
        assert!(dtype.is_nullable());
        assert!(matches!(dtype, DType::Utf8(Nullable)));
    }

    #[test]
    fn test_nullable_float() {
        let dtype = clickhouse_type_to_vortex("Nullable(Float64)").unwrap();
        assert!(dtype.is_nullable());
        assert!(matches!(dtype, DType::Primitive(PType::F64, Nullable)));
    }

    #[test]
    fn test_low_cardinality_wrapper() {
        // LowCardinality should be preserved as Extension type
        let dtype = clickhouse_type_to_vortex("LowCardinality(String)").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseLowCardinality::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.inner_type, "String");

        // Roundtrip
        let ch_type = vortex_to_clickhouse_type(&dtype).unwrap();
        assert_eq!(ch_type, "LowCardinality(String)");

        // LowCardinality with Nullable
        let dtype = clickhouse_type_to_vortex("LowCardinality(Nullable(String))").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseLowCardinality::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.inner_type, "Nullable(String)");
    }

    // ==========================================================================
    // Special type tests
    // ==========================================================================

    #[test]
    fn test_uuid_type() {
        // UUID -> Extension(clickhouse.uuid)
        let dtype = clickhouse_type_to_vortex("UUID").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        assert!(UUID::is_uuid(&dtype));

        // Roundtrip
        let ch_type = vortex_to_clickhouse_type(&dtype).unwrap();
        assert_eq!(ch_type, "UUID");

        // Nullable UUID
        let dtype = clickhouse_type_to_vortex("Nullable(UUID)").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        assert!(UUID::is_uuid(&dtype));
    }

    #[test]
    fn test_decimal_types() {
        // Decimal(P, S) should map to Vortex Decimal
        let dtype = clickhouse_type_to_vortex("Decimal(10, 2)").unwrap();
        assert!(matches!(dtype, DType::Decimal(_, NonNullable)));
        if let DType::Decimal(decimal_dtype, _) = dtype {
            assert_eq!(decimal_dtype.precision(), 10);
            assert_eq!(decimal_dtype.scale(), 2);
        }

        // Decimal32(S) - precision 9
        let dtype = clickhouse_type_to_vortex("Decimal32(4)").unwrap();
        assert!(matches!(dtype, DType::Decimal(_, NonNullable)));
        if let DType::Decimal(decimal_dtype, _) = dtype {
            assert_eq!(decimal_dtype.precision(), 9);
            assert_eq!(decimal_dtype.scale(), 4);
        }

        // Decimal64(S) - precision 18
        let dtype = clickhouse_type_to_vortex("Decimal64(8)").unwrap();
        assert!(matches!(dtype, DType::Decimal(_, NonNullable)));
        if let DType::Decimal(decimal_dtype, _) = dtype {
            assert_eq!(decimal_dtype.precision(), 18);
            assert_eq!(decimal_dtype.scale(), 8);
        }

        // Decimal128(S) - precision 38
        let dtype = clickhouse_type_to_vortex("Decimal128(18)").unwrap();
        assert!(matches!(dtype, DType::Decimal(_, NonNullable)));
        if let DType::Decimal(decimal_dtype, _) = dtype {
            assert_eq!(decimal_dtype.precision(), 38);
            assert_eq!(decimal_dtype.scale(), 18);
        }

        // Decimal256(S) - precision 76
        let dtype = clickhouse_type_to_vortex("Decimal256(30)").unwrap();
        assert!(matches!(dtype, DType::Decimal(_, NonNullable)));
        if let DType::Decimal(decimal_dtype, _) = dtype {
            assert_eq!(decimal_dtype.precision(), 76);
            assert_eq!(decimal_dtype.scale(), 30);
        }
    }

    #[test]
    fn test_decimal_roundtrip() {
        // Test Vortex Decimal -> ClickHouse Decimal -> Vortex Decimal
        use crate::convert::dtype::DecimalDType;

        let decimal_dtype = DecimalDType::new(10, 2);
        let vortex_dtype = DType::Decimal(decimal_dtype, NonNullable);
        let ch_type = vortex_to_clickhouse_type(&vortex_dtype).unwrap();
        assert_eq!(ch_type, "Decimal(10, 2)");

        let roundtrip = clickhouse_type_to_vortex(&ch_type).unwrap();
        if let DType::Decimal(rt_decimal, _) = roundtrip {
            assert_eq!(rt_decimal.precision(), 10);
            assert_eq!(rt_decimal.scale(), 2);
        } else {
            panic!("Expected Decimal type");
        }
    }

    #[test]
    fn test_nullable_decimal() {
        let dtype = clickhouse_type_to_vortex("Nullable(Decimal(10, 2))").unwrap();
        assert!(dtype.is_nullable());
        assert!(matches!(dtype, DType::Decimal(_, Nullable)));
    }

    #[test]
    fn test_null_type() {
        // Vortex Null -> ClickHouse Nothing
        assert_eq!(vortex_to_clickhouse_type(&DType::Null).unwrap(), "Nothing");
    }

    // ==========================================================================
    // Edge cases and error handling tests
    // ==========================================================================

    #[test]
    fn test_whitespace_handling() {
        // Type strings with extra whitespace
        assert!(matches!(
            clickhouse_type_to_vortex("  Int32  ").unwrap(),
            DType::Primitive(PType::I32, NonNullable)
        ));
        assert!(matches!(
            clickhouse_type_to_vortex("Nullable( Int32 )").unwrap(),
            DType::Primitive(PType::I32, Nullable)
        ));
    }

    #[test]
    fn test_unsupported_type_error() {
        // Unknown type should return error
        let result = clickhouse_type_to_vortex("SomeUnknownType(1, 2)");
        assert!(result.is_err());
    }

    #[test]
    fn test_enum8_type() {
        // Enum8 should map to Primitive(I8)
        let dtype = clickhouse_type_to_vortex("Enum8('a' = 1, 'b' = 2)").unwrap();
        assert!(matches!(dtype, DType::Primitive(PType::I8, NonNullable)));
    }

    #[test]
    fn test_enum16_type() {
        // Enum16 should map to Primitive(I16)
        let dtype = clickhouse_type_to_vortex("Enum16('x' = 100, 'y' = 200)").unwrap();
        assert!(matches!(dtype, DType::Primitive(PType::I16, NonNullable)));
    }

    #[test]
    fn test_map_type() {
        // Map(String, Int32) should map to List(Struct([key String, value Int32]))
        let dtype = clickhouse_type_to_vortex("Map(String, Int32)").unwrap();
        if let DType::List(elem, _) = &dtype {
            if let DType::Struct(fields, _) = elem.as_ref() {
                assert_eq!(fields.nfields(), 2);
                assert_eq!(fields.field_name(0).unwrap().as_ref(), "key");
                assert_eq!(fields.field_name(1).unwrap().as_ref(), "value");
                assert!(matches!(fields.field_by_index(0).unwrap(), DType::Utf8(..)));
                assert!(matches!(
                    fields.field_by_index(1).unwrap(),
                    DType::Primitive(PType::I32, ..)
                ));
            } else {
                panic!("Expected Struct element type");
            }
        } else {
            panic!("Expected List type for Map");
        }
    }

    #[test]
    fn test_map_nested_type() {
        // Map(String, Array(Int32))
        let dtype = clickhouse_type_to_vortex("Map(String, Array(Int32))").unwrap();
        if let DType::List(elem, _) = &dtype {
            if let DType::Struct(fields, _) = elem.as_ref() {
                assert_eq!(fields.nfields(), 2);
                assert!(matches!(fields.field_by_index(1).unwrap(), DType::List(..)));
            } else {
                panic!("Expected Struct element type");
            }
        } else {
            panic!("Expected List type for Map");
        }
    }

    #[test]
    fn test_nullable_enum8() {
        let dtype = clickhouse_type_to_vortex("Nullable(Enum8('a' = 1))").unwrap();
        assert!(dtype.is_nullable());
        assert!(matches!(dtype, DType::Primitive(PType::I8, Nullable)));
    }

    #[test]
    fn test_lowcardinality_enum() {
        // LowCardinality(Enum8(...)) should preserve LowCardinality wrapper
        let dtype = clickhouse_type_to_vortex("LowCardinality(Enum8('a' = 1, 'b' = 2))").unwrap();
        assert!(matches!(dtype, DType::Extension(_)));
        let metadata = ClickHouseLowCardinality::try_get_metadata(&dtype).unwrap();
        assert_eq!(metadata.inner_type, "Enum8('a' = 1, 'b' = 2)");
    }

    #[test]
    fn test_nested_tuple_parsing() {
        // Tuple(a Array(Int32), b String)
        let dtype = clickhouse_type_to_vortex("Tuple(a Array(Int32), b String)").unwrap();
        if let DType::Struct(fields, _) = dtype {
            assert_eq!(fields.nfields(), 2);
            assert_eq!(fields.field_name(0).unwrap().as_ref(), "a");
            assert_eq!(fields.field_name(1).unwrap().as_ref(), "b");
            assert!(matches!(fields.field_by_index(0).unwrap(), DType::List(..)));
            assert!(matches!(fields.field_by_index(1).unwrap(), DType::Utf8(..)));
        } else {
            panic!("Expected Struct type for nested Tuple");
        }
    }

    #[test]
    fn test_deeply_nested_tuple() {
        // Tuple(x Tuple(a Int32, b Float64), y Array(String))
        let dtype =
            clickhouse_type_to_vortex("Tuple(x Tuple(a Int32, b Float64), y Array(String))")
                .unwrap();
        if let DType::Struct(fields, _) = dtype {
            assert_eq!(fields.nfields(), 2);
            assert_eq!(fields.field_name(0).unwrap().as_ref(), "x");
            assert_eq!(fields.field_name(1).unwrap().as_ref(), "y");
            assert!(matches!(
                fields.field_by_index(0).unwrap(),
                DType::Struct(..)
            ));
            assert!(matches!(fields.field_by_index(1).unwrap(), DType::List(..)));
        } else {
            panic!("Expected Struct type");
        }
    }

    #[test]
    fn test_split_balanced_commas() {
        let result = split_balanced_commas("a Array(Int32), b String");
        assert_eq!(result, vec!["a Array(Int32)", " b String"]);

        let result = split_balanced_commas("x Tuple(a Int32, b Float64), y Array(String)");
        assert_eq!(
            result,
            vec!["x Tuple(a Int32, b Float64)", " y Array(String)"]
        );

        let result = split_balanced_commas("Int32, String");
        assert_eq!(result, vec!["Int32", " String"]);

        let result = split_balanced_commas("single");
        assert_eq!(result, vec!["single"]);
    }

    #[test]
    fn test_ip_type_conversion() {
        // IPv4 should map to Primitive(U32)
        let result = clickhouse_type_to_vortex("IPv4").unwrap();
        assert!(matches!(result, DType::Primitive(PType::U32, _)));

        // IPv6 should map to Extension(clickhouse.ip)
        let result = clickhouse_type_to_vortex("IPv6").unwrap();
        assert!(matches!(result, DType::Extension(_)));
        assert_eq!(IPAddress::try_get_type(&result), Some(IPAddressType::IPv6));

        // Roundtrip
        let ch_type = vortex_to_clickhouse_type(&result).unwrap();
        assert_eq!(ch_type, "IPv6");

        // Nullable IPv4
        let result = clickhouse_type_to_vortex("Nullable(IPv4)").unwrap();
        assert!(matches!(result, DType::Primitive(PType::U32, Nullable)));

        // Nullable IPv6
        let result = clickhouse_type_to_vortex("Nullable(IPv6)").unwrap();
        assert!(matches!(result, DType::Extension(_)));
        assert_eq!(IPAddress::try_get_type(&result), Some(IPAddressType::IPv6));
    }

    // ==========================================================================
    // Roundtrip tests
    // ==========================================================================

    #[test]
    fn test_primitive_roundtrip() {
        // Non-nullable types
        for ptype in [
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::F32,
            PType::F64,
        ] {
            let vortex_dtype = DType::Primitive(ptype, NonNullable);
            let ch_type = vortex_to_clickhouse_type(&vortex_dtype).unwrap();
            let roundtrip = clickhouse_type_to_vortex(&ch_type).unwrap();
            // Roundtrip preserves nullability: NonNullable CH type -> NonNullable Vortex type
            assert!(matches!(roundtrip, DType::Primitive(..)));
        }
    }

    #[test]
    fn test_list_roundtrip() {
        let vortex_dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            NonNullable,
        );
        let ch_type = vortex_to_clickhouse_type(&vortex_dtype).unwrap();
        assert_eq!(ch_type, "Array(Int32)");

        let roundtrip = clickhouse_type_to_vortex(&ch_type).unwrap();
        assert!(matches!(roundtrip, DType::List(..)));
    }

    #[test]
    fn test_complex_nested_struct() {
        // Create a complex nested struct: Tuple(id Int64, data Tuple(x Float64, y Float64), tags Array(String))
        let inner_names = FieldNames::from(vec![FieldName::from("x"), FieldName::from("y")]);
        let inner_dtypes = vec![
            DType::Primitive(PType::F64, NonNullable),
            DType::Primitive(PType::F64, NonNullable),
        ];
        let inner_struct = DType::Struct(StructFields::new(inner_names, inner_dtypes), NonNullable);

        let outer_names = FieldNames::from(vec![
            FieldName::from("id"),
            FieldName::from("data"),
            FieldName::from("tags"),
        ]);
        let outer_dtypes = vec![
            DType::Primitive(PType::I64, NonNullable),
            inner_struct,
            DType::List(Arc::new(DType::Utf8(NonNullable)), NonNullable),
        ];
        let outer_struct = DType::Struct(StructFields::new(outer_names, outer_dtypes), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&outer_struct).unwrap();
        assert_eq!(
            ch_type,
            "Tuple(id Int64, data Tuple(x Float64, y Float64), tags Array(String))"
        );
    }

    #[test]
    fn test_uuid_roundtrip() {
        let ch_type = "UUID";
        let dtype = clickhouse_type_to_vortex(ch_type).unwrap();
        let back = vortex_to_clickhouse_type(&dtype).unwrap();
        assert_eq!(back, "UUID");
    }

    #[test]
    fn test_ipv6_roundtrip() {
        let ch_type = "IPv6";
        let dtype = clickhouse_type_to_vortex(ch_type).unwrap();
        let back = vortex_to_clickhouse_type(&dtype).unwrap();
        assert_eq!(back, "IPv6");
    }

    #[test]
    fn test_bigint_roundtrip() {
        for (ch_type, expected_back) in [
            ("Int128", "Int128"),
            ("UInt128", "UInt128"),
            ("Int256", "Int256"),
            ("UInt256", "UInt256"),
        ] {
            let dtype = clickhouse_type_to_vortex(ch_type).unwrap();
            let back = vortex_to_clickhouse_type(&dtype).unwrap();
            assert_eq!(back, expected_back, "Roundtrip failed for {}", ch_type);
        }
    }

    #[test]
    fn test_lowcardinality_roundtrip() {
        let ch_type = "LowCardinality(String)";
        let dtype = clickhouse_type_to_vortex(ch_type).unwrap();
        let back = vortex_to_clickhouse_type(&dtype).unwrap();
        assert_eq!(back, "LowCardinality(String)");
    }
}
