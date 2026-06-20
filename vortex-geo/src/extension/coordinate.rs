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

use geoarrow::datatypes::Dimension as GeoArrowDimension;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
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

    /// The coordinate field names of this dimension, in GeoArrow order.
    pub(crate) fn field_names(self) -> &'static [&'static str] {
        match self {
            Dimension::Xy => &["x", "y"],
            Dimension::Xyz => &["x", "y", "z"],
            Dimension::Xym => &["x", "y", "m"],
            Dimension::Xyzm => &["x", "y", "z", "m"],
        }
    }
}

impl From<GeoArrowDimension> for Dimension {
    fn from(dim: GeoArrowDimension) -> Self {
        match dim {
            GeoArrowDimension::XY => Dimension::Xy,
            GeoArrowDimension::XYZ => Dimension::Xyz,
            GeoArrowDimension::XYM => Dimension::Xym,
            GeoArrowDimension::XYZM => Dimension::Xyzm,
        }
    }
}

impl From<Dimension> for GeoArrowDimension {
    fn from(dim: Dimension) -> Self {
        match dim {
            Dimension::Xy => GeoArrowDimension::XY,
            Dimension::Xyz => GeoArrowDimension::XYZ,
            Dimension::Xym => GeoArrowDimension::XYM,
            Dimension::Xyzm => GeoArrowDimension::XYZM,
        }
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

/// The canonical storage dtype for `dim`: a `Struct` of non-nullable `f64` coordinate fields,
/// with `nullability` at the struct (per-point) level. Inverse of [`coordinate_dimension`].
pub(crate) fn coordinate_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let names = dim.field_names();
    let fields = std::iter::repeat_n(
        DType::Primitive(PType::F64, Nullability::NonNullable),
        names.len(),
    )
    .collect::<Vec<_>>();
    DType::Struct(
        StructFields::new(FieldNames::from(names), fields),
        nullability,
    )
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::Coordinate;
    use super::Dimension;
    use super::coordinate_dimension;
    use super::coordinate_storage_dtype;

    /// Each dimension round-trips through its field names and canonical storage dtype.
    #[rstest]
    #[case::xy(Dimension::Xy, &["x", "y"])]
    #[case::xyz(Dimension::Xyz, &["x", "y", "z"])]
    #[case::xym(Dimension::Xym, &["x", "y", "m"])]
    #[case::xyzm(Dimension::Xyzm, &["x", "y", "z", "m"])]
    fn storage_dtype_roundtrips_dimension(
        #[case] dim: Dimension,
        #[case] names: &[&str],
    ) -> VortexResult<()> {
        assert_eq!(dim.field_names(), names);
        let dtype = coordinate_storage_dtype(dim, Nullability::NonNullable);
        assert_eq!(coordinate_dimension(&dtype)?, dim);
        Ok(())
    }

    /// Display emits WKT, including `z`/`m` when present.
    #[rstest]
    #[case::xy(None, None, "POINT(1 2)")]
    #[case::xyz(Some(3.0), None, "POINT Z (1 2 3)")]
    #[case::xym(None, Some(4.0), "POINT M (1 2 4)")]
    #[case::xyzm(Some(3.0), Some(4.0), "POINT ZM (1 2 3 4)")]
    fn display_is_wkt(#[case] z: Option<f64>, #[case] m: Option<f64>, #[case] expected: &str) {
        let coordinate = Coordinate {
            x: 1.0,
            y: 2.0,
            z,
            m,
        };
        assert_eq!(coordinate.to_string(), expected);
    }
}
