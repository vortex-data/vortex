// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared test helpers for the spatial extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::extension::LineString;
use crate::extension::MultiLineString;
use crate::extension::MultiPoint;
use crate::extension::MultiPolygon;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Coordinate;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::coordinate_from_struct;
use crate::extension::linestring_storage_dtype;
use crate::extension::multilinestring_storage_dtype;
use crate::extension::multipoint_storage_dtype;
use crate::extension::multipolygon_storage_dtype;
use crate::extension::polygon_storage_dtype;

/// A fresh session with the spatial types, functions, and pruning rules registered.
pub fn spatial_session() -> VortexSession {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
}

/// The WGS 84 (`EPSG:4326`) metadata tagged onto test geometry columns.
fn wgs84() -> SpatialMetadata {
    SpatialMetadata {
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

/// A list offset as `i32`.
fn offset(n: usize) -> VortexResult<i32> {
    i32::try_from(n).map_err(|_| vortex_err!("geometry offset overflow"))
}

/// Wrap `elements` — built from `rows` flattened one level — in a non-nullable `List` with one
/// entry per row.
fn nest<T>(rows: &[Vec<T>], elements: ArrayRef) -> VortexResult<ArrayRef> {
    let mut offsets = vec![0i32];
    let mut len = 0usize;
    for row in rows {
        len += row.len();
        offsets.push(offset(len)?);
    }
    Ok(ListArray::try_new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::NonNullable,
    )?
    .into_array())
}

/// `List<Struct<x, y>>` storage: one list of `(x, y)` vertices per row.
fn vertex_lists(rows: &[Vec<(f64, f64)>]) -> VortexResult<ArrayRef> {
    let (xs, ys) = rows.iter().flatten().copied().unzip();
    nest(rows, xy_struct(xs, ys)?)
}

/// `List<List<Struct<x, y>>>` storage: one list of vertex lists per row.
fn vertex_list_lists(rows: &[Vec<Vec<(f64, f64)>>]) -> VortexResult<ArrayRef> {
    let inner: Vec<Vec<(f64, f64)>> = rows.iter().flatten().cloned().collect();
    nest(rows, vertex_lists(&inner)?)
}

/// Wrap `storage` in the geometry extension `vtable` (CRS `EPSG:4326`) with the canonical
/// `storage_dtype` of that type.
fn spatial_column<V: ExtVTable<Metadata = SpatialMetadata> + Default>(
    storage: ArrayRef,
    storage_dtype: DType,
) -> VortexResult<ArrayRef> {
    let dtype = ExtDType::<V>::try_new(wgs84(), storage_dtype)?;
    Ok(ExtensionArray::try_new(dtype.erased(), storage)?.into_array())
}

/// A `Point` column over the given x/y coordinates, stored as `Struct<x, y>`.
pub fn point_column(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
    let storage = xy_struct(xs, ys)?;
    let storage_dtype = storage.dtype().clone();
    spatial_column::<Point>(storage, storage_dtype)
}

/// A nullable `Point` column: `None` rows are null. Null rows carry placeholder coordinates in
/// storage that the spatial kernels must never decode (they filter nulls before decoding).
pub fn nullable_point_column(points: Vec<Option<(f64, f64)>>) -> VortexResult<ArrayRef> {
    let len = points.len();
    let valid = points.iter().map(Option::is_some);
    let xs = points.iter().map(|p| p.map_or(0.0, |(x, _)| x));
    let ys = points.iter().map(|p| p.map_or(0.0, |(_, y)| y));
    let storage = StructArray::try_new(
        FieldNames::from(["x", "y"]),
        vec![
            PrimitiveArray::from_iter(xs).into_array(),
            PrimitiveArray::from_iter(ys).into_array(),
        ],
        len,
        Validity::from_iter(valid),
    )?
    .into_array();
    let storage_dtype = storage.dtype().clone();
    spatial_column::<Point>(storage, storage_dtype)
}

/// A `LineString` column: each line a list of `(x, y)` vertices, stored as `List<Struct<x, y>>`.
pub fn linestring_column(lines: Vec<Vec<(f64, f64)>>) -> VortexResult<ArrayRef> {
    spatial_column::<LineString>(
        vertex_lists(&lines)?,
        linestring_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// A `MultiPoint` column: each row a list of `(x, y)` points, stored as `List<Struct<x, y>>`.
pub fn multipoint_column(points: Vec<Vec<(f64, f64)>>) -> VortexResult<ArrayRef> {
    spatial_column::<MultiPoint>(
        vertex_lists(&points)?,
        multipoint_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// A `Polygon` column: each polygon a list of rings, each ring a list of `(x, y)` vertices,
/// stored as `List<List<Struct<x, y>>>`.
pub fn polygon_column(polygons: Vec<Vec<Vec<(f64, f64)>>>) -> VortexResult<ArrayRef> {
    spatial_column::<Polygon>(
        vertex_list_lists(&polygons)?,
        polygon_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// A `MultiLineString` column: each row a list of lines, stored as `List<List<Struct<x, y>>>`.
pub fn multilinestring_column(multilines: Vec<Vec<Vec<(f64, f64)>>>) -> VortexResult<ArrayRef> {
    spatial_column::<MultiLineString>(
        vertex_list_lists(&multilines)?,
        multilinestring_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// One multipolygon: polygons → rings → `(x, y)` vertices.
pub type MultiPolygonRings = Vec<Vec<Vec<(f64, f64)>>>;

/// A `MultiPolygon` column, stored as `List<List<List<Struct<x, y>>>>`.
pub fn multipolygon_column(multipolygons: Vec<MultiPolygonRings>) -> VortexResult<ArrayRef> {
    let polygons: Vec<Vec<Vec<(f64, f64)>>> = multipolygons.iter().flatten().cloned().collect();
    spatial_column::<MultiPolygon>(
        nest(&multipolygons, vertex_list_lists(&polygons)?)?,
        multipolygon_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// A nullable `MultiPolygon` column: `None` rows are null (an empty-list placeholder in storage).
pub fn nullable_multipolygon_column(
    multipolygons: Vec<Option<MultiPolygonRings>>,
) -> VortexResult<ArrayRef> {
    let rows: Vec<MultiPolygonRings> = multipolygons
        .iter()
        .map(|row| row.clone().unwrap_or_default())
        .collect();
    let polygons: Vec<Vec<Vec<(f64, f64)>>> = rows.iter().flatten().cloned().collect();
    let mut offsets = vec![0i32];
    let mut len = 0usize;
    for row in &rows {
        len += row.len();
        offsets.push(offset(len)?);
    }
    let storage = ListArray::try_new(
        vertex_list_lists(&polygons)?,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::from_iter(multipolygons.iter().map(Option::is_some)),
    )?
    .into_array();
    spatial_column::<MultiPolygon>(
        storage,
        multipolygon_storage_dtype(Dimension::Xy, Nullability::Nullable),
    )
}

/// A 2D `Rect` (`geoarrow.box`) column over `(xmin, ymin, xmax, ymax)` boxes, stored as
/// `Struct<xmin, ymin, xmax, ymax>`.
pub fn rect_column(boxes: Vec<(f64, f64, f64, f64)>) -> VortexResult<ArrayRef> {
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
    spatial_column::<Rect>(
        storage,
        box_storage_dtype(Dimension::Xy, Nullability::NonNullable),
    )
}

/// A nullable 2D `Rect` (`geoarrow.box`) column: `None` rows are null boxes (placeholder ordinates
/// in storage). Tagged with default metadata, matching the `envelope` scalar fn and `GeometryAabb`
/// stat outputs.
pub fn nullable_rect_column(boxes: Vec<Option<(f64, f64, f64, f64)>>) -> VortexResult<ArrayRef> {
    let len = boxes.len();
    let field = |select: fn(&(f64, f64, f64, f64)) -> f64| {
        PrimitiveArray::from_iter(boxes.iter().map(|b| b.as_ref().map_or(0.0, select))).into_array()
    };
    let storage = StructArray::try_new(
        FieldNames::from(["xmin", "ymin", "xmax", "ymax"]),
        vec![
            field(|b| b.0),
            field(|b| b.1),
            field(|b| b.2),
            field(|b| b.3),
        ],
        len,
        Validity::from_iter(boxes.iter().map(Option::is_some)),
    )?
    .into_array();
    let ext = ExtDType::<Rect>::try_new(
        SpatialMetadata::default(),
        box_storage_dtype(Dimension::Xy, Nullability::Nullable),
    )?;
    Ok(ExtensionArray::try_new(ext.erased(), storage)?.into_array())
}

/// Decode a [`Coordinate`] from an extension-typed point scalar (unwrapped to its coordinate
/// storage) or a bare coordinate `Struct` scalar — used to read back a single point in assertions.
pub fn coordinate_from_scalar(scalar: &Scalar) -> VortexResult<Coordinate> {
    match scalar.as_extension_opt() {
        Some(ext_scalar) => coordinate_from_struct(&ext_scalar.to_storage_scalar()),
        None => coordinate_from_struct(scalar),
    }
}
