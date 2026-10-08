// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-path queries that both the current reader and an old reader run against a file.
//!
//! A full scan exercises one path through the layout tree. Filters go through zone-map pruning,
//! projections through the struct layout, and row indices through the random-access path, and a
//! mistake on any of them is a silently wrong result rather than an error. The sweep writes a
//! `queries.json` beside its files; the old reader runs every query and dumps each result as
//! `<file>.q<id>.arrow`, and `check-reader` compares them with the current reader's results.
//!
//! The query language is deliberately tiny so every released Python binding can express it:
//! column-list projections, a single comparison against a typed literal, a sorted list of row
//! indices, or a limit.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use arbitrary::Unstructured;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::not_eq;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// The sidecar file name, written next to the `.vortex` files.
pub const QUERIES_FILE: &str = "queries.json";

/// Queries per file name.
pub type QueryFile = BTreeMap<String, Vec<Query>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query {
    pub id: u32,
    /// Top-level field names to project, in order. `None` reads every field.
    pub projection: Option<Vec<String>>,
    /// A single comparison filter.
    pub filter: Option<Filter>,
    /// Strictly increasing row indices to read.
    pub indices: Option<Vec<u64>>,
    /// Row limit.
    pub limit: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter {
    pub column: String,
    /// One of `eq`, `not_eq`, `gt`, `gt_eq`, `lt`, `lt_eq`.
    pub op: String,
    pub value: serde_json::Value,
    pub dtype: LiteralDType,
}

/// The literal's type, in terms every binding can construct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiteralDType {
    /// One of `int`, `uint`, `float`, `bool`, `utf8`.
    pub kind: String,
    pub width: Option<u8>,
    pub nullable: bool,
}

const OPS: [&str; 6] = ["eq", "not_eq", "gt", "gt_eq", "lt", "lt_eq"];

/// The literal type for a field, when the field can be filtered on.
fn literal_dtype(dtype: &DType) -> Option<LiteralDType> {
    let nullable = dtype.is_nullable();
    match dtype {
        DType::Bool(_) => Some(LiteralDType {
            kind: "bool".into(),
            width: None,
            nullable,
        }),
        DType::Utf8(_) => Some(LiteralDType {
            kind: "utf8".into(),
            width: None,
            nullable,
        }),
        DType::Primitive(ptype, _) => {
            let width = u8::try_from(ptype.bit_width()).ok()?;
            let kind = if ptype.is_unsigned_int() {
                "uint"
            } else if ptype.is_signed_int() {
                "int"
            } else if matches!(ptype, PType::F32 | PType::F64) {
                "float"
            } else {
                return None;
            };
            Some(LiteralDType {
                kind: kind.into(),
                width: Some(width),
                nullable,
            })
        }
        _ => None,
    }
}

/// A JSON literal for the scalar, or `None` when the value cannot be expressed (null, NaN).
fn json_value(scalar: &Scalar, kind: &str) -> Option<serde_json::Value> {
    if scalar.is_null() {
        return None;
    }
    match kind {
        "bool" => scalar.as_bool().value().map(serde_json::Value::Bool),
        "utf8" => scalar
            .as_utf8()
            .value()
            .map(|s| serde_json::Value::String(s.to_string())),
        // Python bindings convert literals through a C long, so stay within i64.
        "uint" => scalar
            .as_primitive()
            .as_::<u64>()
            .filter(|v| *v <= i64::MAX as u64)
            .map(Into::into),
        "int" => scalar.as_primitive().as_::<i64>().map(Into::into),
        "float" => scalar
            .as_primitive()
            .as_::<f64>()
            .filter(|f| f.is_finite())
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number),
        _ => None,
    }
}

fn random_projection(u: &mut Unstructured<'_>, names: &[String]) -> arbitrary::Result<Vec<String>> {
    let mut picked: BTreeSet<usize> = BTreeSet::new();
    let count = u.int_in_range(1..=names.len())?;
    while picked.len() < count {
        picked.insert(u.choose_index(names.len())?);
    }
    // Keep a random order rather than field order: the reader must honour the request.
    let mut picked: Vec<String> = picked.into_iter().map(|i| names[i].clone()).collect();
    if u.arbitrary()? {
        picked.reverse();
    }
    Ok(picked)
}

/// Generate queries for a top-level struct array.
#[allow(deprecated)]
pub fn generate(
    array: &ArrayRef,
    u: &mut Unstructured<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Query>> {
    let DType::Struct(fields, _) = array.dtype() else {
        return Ok(vec![]);
    };
    // Use the raw name: `Display` for a field name escapes control characters.
    let names: Vec<String> = fields.names().iter().map(|n| n.as_ref().to_string()).collect();
    if names.is_empty() {
        return Ok(vec![]);
    }
    let len = array.len();
    let Canonical::Struct(canonical) = array.clone().execute::<Canonical>(ctx)? else {
        return Ok(vec![]);
    };
    let filterable: Vec<(usize, LiteralDType)> = fields
        .fields()
        .enumerate()
        .filter_map(|(i, dtype)| literal_dtype(&dtype).map(|lit| (i, lit)))
        .collect();

    let mut queries = Vec::new();
    let mut next_id = 1u32;
    let bad = |e: arbitrary::Error| vortex_err!("arbitrary: {e}");

    // Comparison filters, with literals taken from the data so that zones are actually pruned.
    if !filterable.is_empty() && len > 0 {
        for _ in 0..u.int_in_range(1..=3).map_err(bad)? {
            let (idx, dtype) = u.choose(&filterable).map_err(bad)?.clone();
            let field = canonical.unmasked_field(idx);
            let row = u.int_in_range(0..=len - 1).map_err(bad)?;
            let Some(value) = json_value(&field.scalar_at(row)?, &dtype.kind) else {
                continue;
            };
            let projection = if u.arbitrary().map_err(bad)? {
                Some(random_projection(u, &names).map_err(bad)?)
            } else {
                None
            };
            queries.push(Query {
                id: next_id,
                projection,
                filter: Some(Filter {
                    column: names[idx].clone(),
                    op: (*u.choose(&OPS).map_err(bad)?).to_string(),
                    value,
                    dtype,
                }),
                indices: None,
                limit: None,
            });
            next_id += 1;
        }
    }

    // A projection on its own.
    queries.push(Query {
        id: next_id,
        projection: Some(random_projection(u, &names).map_err(bad)?),
        filter: None,
        indices: None,
        limit: None,
    });
    next_id += 1;

    // Random access by row index.
    if len > 0 {
        let count = u.int_in_range(1..=len.min(64)).map_err(bad)?;
        let mut indices = BTreeSet::new();
        while indices.len() < count {
            indices.insert(u.int_in_range(0..=len - 1).map_err(bad)? as u64);
        }
        let projection = if u.arbitrary().map_err(bad)? {
            Some(random_projection(u, &names).map_err(bad)?)
        } else {
            None
        };
        queries.push(Query {
            id: next_id,
            projection,
            filter: None,
            indices: Some(indices.into_iter().collect()),
            limit: None,
        });
        next_id += 1;
    }

    // A limit.
    queries.push(Query {
        id: next_id,
        projection: None,
        filter: None,
        indices: None,
        limit: Some(u.int_in_range(0..=len as u64).map_err(bad)?),
    });

    Ok(queries)
}

fn scalar_from_json(value: &serde_json::Value, dtype: &LiteralDType) -> VortexResult<Scalar> {
    let n = if dtype.nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let bad = || vortex_err!("literal {value} does not match {dtype:?}");
    Ok(match (dtype.kind.as_str(), dtype.width) {
        ("bool", _) => Scalar::bool(value.as_bool().ok_or_else(bad)?, n),
        ("utf8", _) => Scalar::utf8(value.as_str().ok_or_else(bad)?.to_string(), n),
        ("int", Some(8)) => Scalar::primitive(i8::try_from(value.as_i64().ok_or_else(bad)?)?, n),
        ("int", Some(16)) => Scalar::primitive(i16::try_from(value.as_i64().ok_or_else(bad)?)?, n),
        ("int", Some(32)) => Scalar::primitive(i32::try_from(value.as_i64().ok_or_else(bad)?)?, n),
        ("int", Some(64)) => Scalar::primitive(value.as_i64().ok_or_else(bad)?, n),
        ("uint", Some(8)) => Scalar::primitive(u8::try_from(value.as_u64().ok_or_else(bad)?)?, n),
        ("uint", Some(16)) => {
            Scalar::primitive(u16::try_from(value.as_u64().ok_or_else(bad)?)?, n)
        }
        ("uint", Some(32)) => {
            Scalar::primitive(u32::try_from(value.as_u64().ok_or_else(bad)?)?, n)
        }
        ("uint", Some(64)) => Scalar::primitive(value.as_u64().ok_or_else(bad)?, n),
        ("float", Some(32)) => Scalar::primitive(value.as_f64().ok_or_else(bad)? as f32, n),
        ("float", Some(64)) => Scalar::primitive(value.as_f64().ok_or_else(bad)?, n),
        _ => vortex_bail!("unsupported literal type {dtype:?}"),
    })
}

/// The filter as an unbound expression.
pub fn filter_expression(filter: &Filter) -> VortexResult<Expression> {
    let lhs = col(FieldName::from(filter.column.as_str()));
    let rhs = lit(scalar_from_json(&filter.value, &filter.dtype)?);
    Ok(match filter.op.as_str() {
        "eq" => eq(lhs, rhs),
        "not_eq" => not_eq(lhs, rhs),
        "gt" => gt(lhs, rhs),
        "gt_eq" => gt_eq(lhs, rhs),
        "lt" => lt(lhs, rhs),
        "lt_eq" => lt_eq(lhs, rhs),
        other => vortex_bail!("unknown filter op {other}"),
    })
}
