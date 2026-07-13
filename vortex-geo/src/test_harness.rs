// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared test helpers for the geospatial extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::extension::GeoMetadata;
use crate::extension::Point;
use crate::extension::Rect;
use crate::extension::coordinate::Coordinate;
use crate::extension::coordinate::coordinate_from_struct;

/// The WGS 84 (`EPSG:4326`) metadata tagged onto test geometry columns.
fn wgs84() -> GeoMetadata {
    GeoMetadata {
        crs: Some("EPSG:4326".to_string()),
    }
}

/// A `Point` column (CRS `EPSG:4326`) over the given x/y coordinates.
pub(crate) fn point_column(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
    let storage = StructArray::from_fields(&[
        ("x", PrimitiveArray::from_iter(xs).into_array()),
        ("y", PrimitiveArray::from_iter(ys).into_array()),
    ])?
    .into_array();
    let dtype = ExtDType::<Point>::try_new(wgs84(), storage.dtype().clone())?;
    Ok(ExtensionArray::new(dtype.erased(), storage).into_array())
}

/// A 2D `Rect` (`geoarrow.box`) column (CRS `EPSG:4326`) over `(xmin, ymin, xmax, ymax)` boxes.
pub(crate) fn rect_column(boxes: Vec<(f64, f64, f64, f64)>) -> VortexResult<ArrayRef> {
    let field = |select: fn(&(f64, f64, f64, f64)) -> f64| {
        PrimitiveArray::from_iter(boxes.iter().map(select)).into_array()
    };
    let storage = StructArray::from_fields(&[
        ("xmin", field(|b| b.0)),
        ("ymin", field(|b| b.1)),
        ("xmax", field(|b| b.2)),
        ("ymax", field(|b| b.3)),
    ])?
    .into_array();
    let dtype = ExtDType::<Rect>::try_new(wgs84(), storage.dtype().clone())?;
    Ok(ExtensionArray::new(dtype.erased(), storage).into_array())
}

/// Decode a [`Coordinate`] from an extension-typed point scalar (unwrapped to its coordinate
/// storage) or a bare coordinate `Struct` scalar — used to read back a single point in assertions.
pub(crate) fn coordinate_from_scalar(scalar: &Scalar) -> VortexResult<Coordinate> {
    match scalar.as_extension_opt() {
        Some(ext_scalar) => coordinate_from_struct(&ext_scalar.to_storage_scalar()),
        None => coordinate_from_struct(scalar),
    }
}
