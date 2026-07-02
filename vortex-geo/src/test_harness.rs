// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared test helpers for the geospatial extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::extension::GeoMetadata;
use crate::extension::MultiPolygon;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::coordinate::Coordinate;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::coordinate_from_struct;
use crate::extension::multipolygon_storage_dtype;
use crate::extension::polygon_storage_dtype;

/// The WGS 84 (`EPSG:4326`) metadata tagged onto test geometry columns.
fn wgs84() -> GeoMetadata {
    GeoMetadata {
        crs: Some("EPSG:4326".to_string()),
    }
}

/// A coordinate `Struct<x, y>` over the parallel x/y buffers.
fn xy_struct(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("x", PrimitiveArray::from_iter(xs).into_array()),
        ("y", PrimitiveArray::from_iter(ys).into_array()),
    ])?
    .into_array())
}

/// A `Point` column (CRS `EPSG:4326`) over the given x/y coordinates.
pub(crate) fn point_column(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
    let storage = xy_struct(xs, ys)?;
    let dtype = ExtDType::<Point>::try_new(wgs84(), storage.dtype().clone())?;
    Ok(ExtensionArray::new(dtype.erased(), storage).into_array())
}

/// A list offset as `i32`.
fn offset(n: usize) -> VortexResult<i32> {
    i32::try_from(n).map_err(|_| vortex_err!("geometry offset overflow"))
}

/// Wrap `values` in a non-nullable `List` with the given offsets.
fn list(values: ArrayRef, offsets: Vec<i32>) -> VortexResult<ArrayRef> {
    Ok(ListArray::try_new(
        values,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )?
    .into_array())
}

/// A `Polygon` column (CRS `EPSG:4326`). Each polygon is a list of rings; each ring a list of
/// `(x, y)` vertices. Stored as `List<List<Struct<x, y>>>`.
pub(crate) fn polygon_column(polygons: Vec<Vec<Vec<(f64, f64)>>>) -> VortexResult<ArrayRef> {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let mut ring_offsets = vec![0i32];
    let mut polygon_offsets = vec![0i32];
    for rings in &polygons {
        for ring in rings {
            for &(x, y) in ring {
                xs.push(x);
                ys.push(y);
            }
            ring_offsets.push(offset(xs.len())?);
        }
        polygon_offsets.push(offset(ring_offsets.len() - 1)?);
    }

    let storage = list(list(xy_struct(xs, ys)?, ring_offsets)?, polygon_offsets)?;
    let dtype = ExtDType::<Polygon>::try_new(
        wgs84(),
        polygon_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )?;
    Ok(ExtensionArray::try_new(dtype.erased(), storage)?.into_array())
}

/// One multipolygon: polygons → rings → `(x, y)` vertices.
pub(crate) type MultiPolygonRings = Vec<Vec<Vec<(f64, f64)>>>;

/// A `MultiPolygon` column (CRS `EPSG:4326`), stored as `List<List<List<Struct<x, y>>>>`.
pub(crate) fn multipolygon_column(multipolygons: Vec<MultiPolygonRings>) -> VortexResult<ArrayRef> {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let mut ring_offsets = vec![0i32];
    let mut polygon_offsets = vec![0i32];
    let mut multipolygon_offsets = vec![0i32];
    for polygons in &multipolygons {
        for rings in polygons {
            for ring in rings {
                for &(x, y) in ring {
                    xs.push(x);
                    ys.push(y);
                }
                ring_offsets.push(offset(xs.len())?);
            }
            polygon_offsets.push(offset(ring_offsets.len() - 1)?);
        }
        multipolygon_offsets.push(offset(polygon_offsets.len() - 1)?);
    }

    let rings = list(xy_struct(xs, ys)?, ring_offsets)?;
    let storage = list(list(rings, polygon_offsets)?, multipolygon_offsets)?;
    let dtype = ExtDType::<MultiPolygon>::try_new(
        wgs84(),
        multipolygon_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )?;
    Ok(ExtensionArray::try_new(dtype.erased(), storage)?.into_array())
}

/// Decode a [`Coordinate`] from an extension-typed point scalar (unwrapped to its coordinate
/// storage) or a bare coordinate `Struct` scalar — used to read back a single point in assertions.
pub(crate) fn coordinate_from_scalar(scalar: &Scalar) -> VortexResult<Coordinate> {
    match scalar.as_extension_opt() {
        Some(ext_scalar) => coordinate_from_struct(&ext_scalar.to_storage_scalar()),
        None => coordinate_from_struct(scalar),
    }
}
