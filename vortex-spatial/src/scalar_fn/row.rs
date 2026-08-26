// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-function adapters for `geo_types` computation over native geometry columns.

use geo::BoundingRect;
use geo_types::Geometry;
use geo_types::Polygon as GeoPolygon;
use geo_types::Rect;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::OutputSink;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::extension::build_polygon_storage;
use crate::extension::coordinate::Dimension;
use crate::extension::geometries;
use crate::extension::is_native_geometry;
use crate::extension::polygon_storage_dtype;

type ConstBoundingRects = (Option<Option<Rect<f64>>>, Option<Option<Rect<f64>>>);

/// Visit a binary geometry predicate, hoisting the bounding rectangle of a single constant
/// operand out of the row loop.
pub(crate) fn visit_binary_geo_predicate<V: RowVisitor>(
    visitor: V,
    compute: fn(&Geometry<f64>, &Geometry<f64>) -> bool,
    precheck: fn(&Rect<f64>, &Rect<f64>) -> Option<bool>,
) -> VortexResult<V::VisitResult> {
    visitor.visit_prepared::<(GeometryRow, GeometryRow), bool, ConstBoundingRects>(
        |(left, right)| {
            (
                left.map(BoundingRect::bounding_rect),
                right.map(BoundingRect::bounding_rect),
            )
        },
        move |constant_bounds, (left, right)| {
            let result = match constant_bounds {
                (Some(Some(left_bounds)), None) => right
                    .bounding_rect()
                    .and_then(|right_bounds| precheck(left_bounds, &right_bounds)),
                (None, Some(Some(right_bounds))) => left
                    .bounding_rect()
                    .and_then(|left_bounds| precheck(&left_bounds, right_bounds)),
                _ => None,
            };
            result.unwrap_or_else(|| compute(left, right))
        },
    )
}

/// Marker for native geometry input elements: accepts any native geometry column and presents each
/// row as a decoded `geo_types` geometry.
///
/// The two operands of a binary geo function need not share a geometry type, since distance,
/// containment and intersection across types are all meaningful, so this validates only that the
/// column is _some_ native geometry.
pub(crate) struct GeometryRow;

// SAFETY: The row-loop view is a geometry slice. Its `ViewLen` is the slice length, so every index
// below that length is valid for both checked and unchecked access.
unsafe impl InputElement for GeometryRow {
    type Column = Vec<Geometry<f64>>;
    type View<'a> = &'a [Geometry<f64>];
    type Elem<'a> = &'a Geometry<f64>;

    // A geometry row is decoded from its coordinate storage, which behind a null row holds arbitrary
    // coordinates that need not describe a well-formed geometry.
    const DENSE_SAFE: bool = false;
    // Decoding builds a geometry from stored coordinates, and a malformed one in a *valid* row is a
    // domain error rather than an infrastructural failure.
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            is_native_geometry(dtype),
            "spatial: operand {dtype} is not a native geometry type"
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        geometries(&array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> &Geometry<f64> {
        &column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> &'a Geometry<f64>
    where
        Self: 'a,
    {
        &view[index]
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> &'a Geometry<f64>
    where
        Self: 'a,
    {
        // SAFETY: The caller established that `index` is below `ViewLen::len` for this view, which
        // is the geometry slice length.
        unsafe { view.get_unchecked(index) }
    }
}

/// Row output for native 2-D polygons.
pub(crate) struct PolygonSink {
    polygons: Vec<GeoPolygon<f64>>,
}

fn empty_polygon() -> GeoPolygon<f64> {
    GeoPolygon::new(geo_types::LineString::new(vec![]), vec![])
}

// SAFETY: `with_capacity` creates one initialized polygon per output row, and `Rows` is the
// corresponding mutable slice. Every in-bounds index therefore names one distinct initialized
// polygon. The sink remains safe to finish or drop after any row prefix.
unsafe impl OutputSink for PolygonSink {
    type Params = ();
    type Rows<'a> = &'a mut [GeoPolygon<f64>];
    type Row<'a> = &'a mut GeoPolygon<f64>;
    type WriteToken = ();

    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        Some(|_| {})
    }

    fn storage_dtype((): &Self::Params) -> DType {
        polygon_storage_dtype(Dimension::Xy, Nullability::NonNullable)
    }

    fn with_capacity(rows: usize, (): &Self::Params) -> VortexResult<Self> {
        Ok(Self {
            polygons: vec![empty_polygon(); rows],
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.polygons.as_mut_slice()
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        unsafe { rows.get_unchecked_mut(index) }
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        build_polygon_storage(&self.polygons)
    }
}
