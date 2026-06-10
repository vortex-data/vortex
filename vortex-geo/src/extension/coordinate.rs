// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Coordinate building blocks for geometry extension types: the `Struct<x, y, z?, m?>` storage,
//! its [`Dimension`], and the decoded [`Coordinate`] value.
//!
//! The coordinate fields, where `?` marks an optional field, are:
//! - `x` — longitude or easting
//! - `y` — latitude or northing
//! - `z?` — elevation
//! - `m?` — measure: an arbitrary per-point value such as distance along a route or a timestamp

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// Coordinate dimensions, matching GeoArrow. Field order is fixed: `x`, `y`, then `z` before `m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dimension {
    /// 2D: `x`, `y`.
    Xy,
    /// 3D with elevation: `x`, `y`, `z`.
    Xyz,
    /// 3D with a measure: `x`, `y`, `m`.
    Xym,
    /// 4D: `x`, `y`, `z`, `m`.
    Xyzm,
}

impl Dimension {
    /// Recover the dimension from a coordinate's field names, in GeoArrow order.
    pub(crate) fn from_field_names(names: &[&str]) -> VortexResult<Dimension> {
        Ok(match names {
            ["x", "y"] => Dimension::Xy,
            ["x", "y", "z"] => Dimension::Xyz,
            ["x", "y", "m"] => Dimension::Xym,
            ["x", "y", "z", "m"] => Dimension::Xyzm,
            _ => vortex_bail!("not a valid GeoArrow coordinate dimension: {names:?}"),
        })
    }
}

/// A decoded coordinate. `z?`/`m?` are `Some` iff the storage dimension includes them.
///
/// This is the native value produced when unpacking a [`Point`](crate::extension::Point) scalar;
/// the rest of the coordinate machinery is crate-internal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    /// The x (longitude/easting) ordinate.
    x: f64,
    /// The y (latitude/northing) ordinate.
    y: f64,
    /// The optional `z?` (elevation) ordinate.
    z: Option<f64>,
    /// The optional `m?` (measure) ordinate.
    m: Option<f64>,
}

impl Coordinate {
    /// A 2D coordinate (`z?`/`m?` unset).
    pub fn xy(x: f64, y: f64) -> Self {
        Coordinate {
            x,
            y,
            z: None,
            m: None,
        }
    }

    /// The x (longitude/easting) ordinate.
    pub fn x(&self) -> f64 {
        self.x
    }

    /// The y (latitude/northing) ordinate.
    pub fn y(&self) -> f64 {
        self.y
    }

    /// The `z?` (elevation) ordinate, if the dimension includes one.
    pub fn z(&self) -> Option<f64> {
        self.z
    }

    /// The `m?` (measure) ordinate, if the dimension includes one.
    pub fn m(&self) -> Option<f64> {
        self.m
    }
}

impl Display for Coordinate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "POINT({} {})", self.x, self.y)
    }
}

/// Validate that `dtype` is a coordinate struct of non-nullable `f64` fields, returning its
/// [`Dimension`]. Any of the four GeoArrow dimensions validates.
pub(crate) fn coordinate_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("coordinate storage must be a Struct, was {dtype}");
    };
    let names: Vec<&str> = fields.names().iter().map(|n| n.as_ref()).collect();
    for (i, field) in fields.fields().enumerate() {
        if !matches!(
            field,
            DType::Primitive(PType::F64, Nullability::NonNullable)
        ) {
            vortex_bail!(
                "coordinate field {} must be non-nullable f64, was {field}",
                names[i]
            );
        }
    }
    Dimension::from_field_names(&names)
}

/// Decode a [`Coordinate`] from a coordinate `Struct<x, y, z?, m?>` scalar (`z?`/`m?` read iff
/// present, so the same decoder serves every dimension).
pub(crate) fn coordinate_from_struct(scalar: &Scalar) -> VortexResult<Coordinate> {
    let fields = scalar.as_struct();
    let required = |name: &str| -> VortexResult<f64> {
        f64::try_from(
            &fields
                .field(name)
                .ok_or_else(|| vortex_err!("coordinate missing {name}"))?,
        )
    };
    let optional = |name: &str| -> VortexResult<Option<f64>> {
        fields
            .field(name)
            .map(|value| f64::try_from(&value))
            .transpose()
    };
    Ok(Coordinate {
        x: required("x")?,
        y: required("y")?,
        z: optional("z")?,
        m: optional("m")?,
    })
}

/// Decode a [`Coordinate`] from an extension-typed point scalar (unwrapped to its coordinate
/// storage) or a bare coordinate `Struct` scalar. The per-row decode used by the distance fns.
pub(crate) fn coordinate_from_scalar(scalar: &Scalar) -> VortexResult<Coordinate> {
    match scalar.dtype().as_extension_opt() {
        Some(_) => coordinate_from_struct(&scalar.as_extension().to_storage_scalar()),
        None => coordinate_from_struct(scalar),
    }
}

/// Canonicalize a point column once and return its flat `x`/`y` `f64` columns. The bulk counterpart
/// to [`coordinate_from_scalar`]; distances use only `x`/`y`, so `z?`/`m?` are ignored.
pub(crate) fn xy_columns(
    points: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let storage = points
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone()
        .execute::<StructArray>(ctx)?;
    let xs = storage
        .unmasked_field_by_name("x")?
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let ys = storage
        .unmasked_field_by_name("y")?
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    Ok((xs, ys))
}
