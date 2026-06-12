// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Coordinate building blocks for geometry extension types: the `Struct<x, y[, z][, m]>` storage
//! of non-nullable `f64` fields, its [`Dimension`], and the decoded [`Coordinate`] value.
//!
//! The coordinate fields are:
//! - `x` — longitude or easting
//! - `y` — latitude or northing
//! - `z` (optional) — elevation
//! - `m` (optional) — measure: an arbitrary per-point value such as distance along a route or a
//!   timestamp

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
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
    pub(crate) fn from_field_names(names: &FieldNames) -> VortexResult<Dimension> {
        let mut strs = [""; 4];
        vortex_ensure!(
            names.len() <= strs.len(),
            "not a valid GeoArrow coordinate dimension: {names:?}"
        );
        for (slot, name) in strs.iter_mut().zip(names.iter()) {
            *slot = name.as_ref();
        }
        Ok(match &strs[..names.len()] {
            ["x", "y"] => Dimension::Xy,
            ["x", "y", "z"] => Dimension::Xyz,
            ["x", "y", "m"] => Dimension::Xym,
            ["x", "y", "z", "m"] => Dimension::Xyzm,
            _ => vortex_bail!("not a valid GeoArrow coordinate dimension: {names:?}"),
        })
    }
}

/// A decoded coordinate. `z`/`m` are `Some` iff the storage dimension includes them.
///
/// This is the native value produced when unpacking a [`Point`](crate::extension::Point) scalar;
/// the rest of the coordinate machinery is crate-internal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    /// The x (longitude/easting) ordinate.
    pub x: f64,
    /// The y (latitude/northing) ordinate.
    pub y: f64,
    /// The optional `z` (elevation) ordinate.
    pub z: Option<f64>,
    /// The optional `m` (measure) ordinate.
    pub m: Option<f64>,
}

impl Coordinate {
    /// A 2D coordinate (`z`/`m` unset).
    pub fn xy(x: f64, y: f64) -> Self {
        Coordinate {
            x,
            y,
            z: None,
            m: None,
        }
    }
}

impl Display for Coordinate {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        match (self.z, self.m) {
            (None, None) => write!(fmt, "POINT({} {})", self.x, self.y),
            (Some(z), None) => write!(fmt, "POINT Z ({} {} {})", self.x, self.y, z),
            (None, Some(m)) => write!(fmt, "POINT M ({} {} {})", self.x, self.y, m),
            (Some(z), Some(m)) => write!(fmt, "POINT ZM ({} {} {} {})", self.x, self.y, z, m),
        }
    }
}

/// Validate that `dtype` is a coordinate struct of non-nullable `f64` fields, returning its
/// [`Dimension`]. Any of the four GeoArrow dimensions validates.
pub(crate) fn coordinate_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("coordinate storage must be a Struct, was {dtype}");
    };
    for (name, field) in fields.names().iter().zip(fields.fields()) {
        vortex_ensure!(
            matches!(
                field,
                DType::Primitive(PType::F64, Nullability::NonNullable)
            ),
            "coordinate field {name} must be non-nullable f64, was {field}"
        );
    }
    Dimension::from_field_names(fields.names())
}

/// Decode a [`Coordinate`] from a coordinate `Struct<x, y[, z][, m]>` scalar (`z`/`m` read iff
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
    match scalar.as_extension_opt() {
        Some(ext_scalar) => coordinate_from_struct(&ext_scalar.to_storage_scalar()),
        None => coordinate_from_struct(scalar),
    }
}

/// Validated, executed `x`/`y` columns of a point array. The bulk counterpart to [`Coordinate`];
/// `z`/`m` are not executed.
pub(crate) struct ParsedCoordinates {
    /// The flat `f64` `x` column.
    pub(crate) xs: PrimitiveArray,
    /// The flat `f64` `y` column.
    pub(crate) ys: PrimitiveArray,
}

/// Validate a point column's coordinate storage (layout and non-nullability) and execute its
/// `x`/`y` columns.
pub(crate) fn parse_storage(
    points: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ParsedCoordinates> {
    let storage = points
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone()
        .execute::<StructArray>(ctx)?;
    coordinate_dimension(storage.dtype())?;
    vortex_ensure!(
        !storage.dtype().is_nullable(),
        "coordinate storage must be non-nullable to read unmasked ordinates, was {}",
        storage.dtype()
    );
    let xs = storage
        .unmasked_field_by_name("x")?
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let ys = storage
        .unmasked_field_by_name("y")?
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    Ok(ParsedCoordinates { xs, ys })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Coordinate;
    use super::parse_storage;
    use crate::extension::GeoMetadata;
    use crate::extension::Point;

    /// Display emits WKT, including `z`/`m` when present.
    #[test]
    fn display_is_wkt() {
        let coordinate = |z, m| Coordinate {
            x: 1.0,
            y: 2.0,
            z,
            m,
        };
        assert_eq!(coordinate(None, None).to_string(), "POINT(1 2)");
        assert_eq!(coordinate(Some(3.0), None).to_string(), "POINT Z (1 2 3)");
        assert_eq!(coordinate(None, Some(4.0)).to_string(), "POINT M (1 2 4)");
        assert_eq!(
            coordinate(Some(3.0), Some(4.0)).to_string(),
            "POINT ZM (1 2 3 4)"
        );
    }

    /// [`parse_storage`] reads the coordinate fields unmasked, so a nullable point column must
    /// be rejected at parse time rather than decoding null rows as garbage ordinates.
    #[test]
    fn parse_rejects_nullable_points() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let storage = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                PrimitiveArray::from_iter(vec![1.0f64]).into_array(),
                PrimitiveArray::from_iter(vec![2.0f64]).into_array(),
            ],
            1,
            Validity::AllValid,
        )?
        .into_array();
        let dtype = ExtDType::<Point>::try_new(GeoMetadata { crs: None }, storage.dtype().clone())?;
        let points = ExtensionArray::new(dtype.erased(), storage).into_array();

        assert!(parse_storage(&points, &mut ctx).is_err());
        Ok(())
    }
}
