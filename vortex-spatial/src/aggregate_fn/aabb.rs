// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The 2D axis-aligned bounding-box (AABB) aggregate for native geometry columns.

use geo::Rect as SpatialRect;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::box_corners;
use crate::extension::coordinate::ordinates;
use crate::extension::flatten_coordinates;
use crate::extension::is_native_geometry;

/// Aggregates a native geometry column's 2D axis-aligned bounding box (AABB) as a native
/// `geoarrow.box`, the spatial analogue of min/max. Also the default zone statistic for such
/// columns (via `zone_stat_default`).
#[derive(Clone, Debug)]
pub struct GeometryAabb;

/// Running union of geometry AABBs, or `None` until the first row. A transient
/// `geo::Rect` value - the persisted stat is the native box (see `to_scalar`).
pub struct AabbPartial {
    rect: Option<SpatialRect<f64>>,
}

impl AabbPartial {
    /// Grow the accumulated box to also cover `other`.
    fn merge(&mut self, other: SpatialRect<f64>) {
        self.rect = Some(self.rect.map_or(other, |cur| {
            SpatialRect::new(
                (
                    cur.min().x.min(other.min().x),
                    cur.min().y.min(other.min().y),
                ),
                (
                    cur.max().x.max(other.max().x),
                    cur.max().y.max(other.max().y),
                ),
            )
        }));
    }
}

/// The stat's type: the native `geoarrow.box` (2D), nullable so an empty group is a null box.
fn aabb_dtype() -> DType {
    DType::Extension(
        ExtDType::<Rect>::try_new(SpatialMetadata::default(), aabb_storage_dtype())
            .vortex_expect("2D box storage is a valid Rect")
            .erased(),
    )
}

/// The `Rect` storage `Struct<xmin, ymin, xmax, ymax>` backing the zone statistic.
fn aabb_storage_dtype() -> DType {
    box_storage_dtype(Dimension::Xy, Nullability::Nullable)
}

/// The AABB of the raw `x`/`y` slices, or `None` when empty.
fn aabb_of(xs: &[f64], ys: &[f64]) -> Option<SpatialRect<f64>> {
    (!xs.is_empty()).then(|| {
        let [xmin, ymin, xmax, ymax] = box_corners(xs, ys);
        SpatialRect::new((xmin, ymin), (xmax, ymax))
    })
}

/// Read an AABB stat scalar (a nullable native `geoarrow.box`) into a [`SpatialRect`], or `None` when
/// the scalar is null (an empty group).
fn rect_from_storage(scalar: &Scalar) -> VortexResult<Option<SpatialRect<f64>>> {
    if scalar.is_null() {
        return Ok(None);
    }
    let storage = scalar.as_extension().to_storage_scalar();
    let fields = storage.as_struct();
    let read = |name: &str| -> VortexResult<f64> {
        f64::try_from(
            &fields
                .field(name)
                .ok_or_else(|| vortex_err!("AABB missing {name}"))?,
        )
    };
    Ok(Some(SpatialRect::new(
        (read("xmin")?, read("ymin")?),
        (read("xmax")?, read("ymax")?),
    )))
}

/// Serialize a [`SpatialRect`] as a native `geoarrow.box` stat scalar (inverse of [`rect_from_storage`]).
fn rect_to_storage(rect: SpatialRect<f64>) -> Scalar {
    let storage = Scalar::struct_(
        aabb_storage_dtype(),
        vec![
            Scalar::primitive(rect.min().x, Nullability::NonNullable),
            Scalar::primitive(rect.min().y, Nullability::NonNullable),
            Scalar::primitive(rect.max().x, Nullability::NonNullable),
            Scalar::primitive(rect.max().y, Nullability::NonNullable),
        ],
    );
    Scalar::extension::<Rect>(SpatialMetadata::default(), storage)
}

impl AggregateFnVTable for GeometryAabb {
    type Options = EmptyOptions;
    type Partial = AabbPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.st.aabb");
        *ID
    }

    // Serializable so the zoned writer can persist this as a per-chunk stat. No options to encode.
    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_native_geometry(input_dtype).then(aabb_dtype)
    }

    fn zone_stat_default(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        is_native_geometry(input_dtype).then(|| self.bind(EmptyOptions))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(AabbPartial { rect: None })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if let Some(rect) = rect_from_storage(&other)? {
            partial.merge(rect);
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match partial.rect {
            Some(rect) => rect_to_storage(rect),
            None => Scalar::null(aabb_dtype()),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.rect = None;
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        // An AABB can always grow, so it is never saturated.
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        // Drop null rows before reading coordinates: a null geometry's storage holds placeholder
        // coordinates (e.g. `(0, 0)`) that would otherwise widen the zone box and drag it toward
        // the origin. `filter` collapses an all-true mask back to the input, so a null-free batch
        // passes through unchanged.
        //
        // TODO(perf): on nullable data this `filter` compacts the whole column even for a single
        // null before we min/max it. A validity-aware min/max straight over the raw x/y buffers
        // would skip that copy. Left as-is for now: this is a write-time zone stat, and the common
        // non-nullable case already costs nothing (the all-true mask makes `filter` a no-op).
        let valid = array.validity()?.execute_mask(array.len(), ctx)?;
        let array = array.filter(valid)?;
        // Null rows are gone, so every coordinate below belongs to a present geometry — the
        // `unmasked_field_by_name` reads are therefore safe. Min/max the raw x/y buffers directly:
        // cheap, and avoids `to_geometry`'s panic on empty points (which decoding would hit).
        let coords = flatten_coordinates(&array, ctx)?;
        let xs = ordinates(&coords, "x", ctx)?;
        let ys = ordinates(&coords, "y", ctx)?;
        if let Some(rect) = aabb_of(&xs, &ys) {
            partial.merge(rect);
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        // The stored partial is already the AABB struct, so finalizing is the identity.
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
    use geo::Rect as SpatialRect;
    use vortex_array::ArrayRef;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use super::AabbPartial;
    use super::GeometryAabb;
    use super::aabb_dtype;
    use super::rect_from_storage;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::spatial_session;

    /// One column of every native geometry type over the same `(x, y)` vertex set.
    fn every_native_column(vertices: &[(f64, f64)]) -> VortexResult<Vec<ArrayRef>> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = vertices.iter().copied().unzip();
        let flat = vertices.to_vec();
        Ok(vec![
            point_column(xs, ys)?,
            linestring_column(vec![flat.clone()])?,
            multipoint_column(vec![flat.clone()])?,
            polygon_column(vec![vec![flat.clone()]])?,
            multilinestring_column(vec![vec![flat.clone()]])?,
            multipolygon_column(vec![vec![vec![flat]]])?,
        ])
    }

    /// The aggregate must be serializable so the zoned writer can persist its zone-stat descriptor.
    #[test]
    fn serializes_for_zone_storage() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let metadata = GeometryAabb
            .serialize(&EmptyOptions)?
            .expect("GeometryAabb must be serializable to be stored as a zone statistic");
        GeometryAabb.deserialize(&metadata, &session)?;
        Ok(())
    }

    /// The AABB result's corners as `(xmin, ymin, xmax, ymax)`.
    fn aabb(result: &Scalar) -> VortexResult<(f64, f64, f64, f64)> {
        let rect = rect_from_storage(result)?.expect("non-null AABB");
        Ok((rect.min().x, rect.min().y, rect.max().x, rect.max().y))
    }

    /// The AABB of a Point column is the min/max of its coordinates, accumulated across batches.
    #[test]
    fn point_aabb_across_batches() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, dtype)?;

        acc.accumulate(&point_column(vec![1.0, 3.0], vec![2.0, 4.0])?, &mut ctx)?;
        acc.accumulate(&point_column(vec![-1.0], vec![5.0])?, &mut ctx)?;

        assert_eq!(aabb(&acc.finish()?)?, (-1.0, 2.0, 3.0, 5.0));
        Ok(())
    }

    /// Null rows contribute no extent: their placeholder coordinates must not widen the box toward
    /// the origin (matching min/max, which skip nulls).
    #[test]
    fn null_rows_do_not_widen_aabb() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // The valid points sit far from the origin; the null row's storage placeholder is (0, 0).
        let column = nullable_point_column(vec![Some((5.0, 6.0)), None, Some((7.0, 8.0))])?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, column.dtype().clone())?;
        acc.accumulate(&column, &mut ctx)?;

        // The box covers only the two valid points, never (0, 0).
        assert_eq!(aabb(&acc.finish()?)?, (5.0, 6.0, 7.0, 8.0));
        Ok(())
    }

    /// The AABB of a Polygon column unions every ring vertex - exercising the `List<List<Struct>>`
    /// unwrap, not just the bare Point struct.
    #[test]
    fn polygon_aabb_union_all_vertices() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Two rectangles: (0,0)-(2,3) and (5,5)-(7,8). The chunk AABB is their union: (0,0)-(7,8).
        let polygons = polygon_column(vec![
            vec![vec![(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)]],
            vec![vec![(5.0, 5.0), (7.0, 5.0), (7.0, 8.0), (5.0, 8.0)]],
        ])?;
        let dtype = polygons.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, dtype)?;
        acc.accumulate(&polygons, &mut ctx)?;

        assert_eq!(aabb(&acc.finish()?)?, (0.0, 0.0, 7.0, 8.0));
        Ok(())
    }

    /// Every native geometry type over the same vertex set yields the same AABB - the zone stat
    /// covers the whole type family.
    #[test]
    fn aabb_covers_every_native_geometry_type() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        for column in every_native_column(&[(1.0, 2.0), (-1.0, 5.0), (3.0, 4.0)])? {
            let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, column.dtype().clone())?;
            acc.accumulate(&column, &mut ctx)?;
            assert_eq!(
                aabb(&acc.finish()?)?,
                (-1.0, 2.0, 3.0, 5.0),
                "AABB mismatch for {}",
                column.dtype()
            );
        }
        Ok(())
    }

    /// The AABB of a MultiPolygon column unions every vertex of every polygon's rings - exercising
    /// the triple-`List` unwrap.
    #[test]
    fn multipolygon_aabb_union_all_vertices() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Multipolygon 0: squares (0,0)-(1,1) and (4,4)-(5,5); multipolygon 1: square (-3,7)-(-2,9).
        let multipolygons = multipolygon_column(vec![
            vec![
                vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
                vec![vec![(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)]],
            ],
            vec![vec![vec![
                (-3.0, 7.0),
                (-2.0, 7.0),
                (-2.0, 9.0),
                (-3.0, 9.0),
            ]]],
        ])?;
        let dtype = multipolygons.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, dtype)?;
        acc.accumulate(&multipolygons, &mut ctx)?;

        assert_eq!(aabb(&acc.finish()?)?, (-3.0, 0.0, 5.0, 9.0));
        Ok(())
    }

    /// `combine_partials` unions partial boxes - the path the zoned writer takes when a zone's
    /// array is chunked.
    #[test]
    fn combine_partials_unions_boxes() -> VortexResult<()> {
        let bbox = |xmin, ymin, xmax, ymax| AabbPartial {
            rect: Some(SpatialRect::new((xmin, ymin), (xmax, ymax))),
        };
        let mut partial = AabbPartial { rect: None };
        GeometryAabb.combine_partials(
            &mut partial,
            GeometryAabb.to_scalar(&bbox(0.0, 0.0, 1.0, 1.0))?,
        )?;
        GeometryAabb.combine_partials(
            &mut partial,
            GeometryAabb.to_scalar(&bbox(5.0, -2.0, 7.0, 3.0))?,
        )?;
        assert_eq!(
            aabb(&GeometryAabb.to_scalar(&partial)?)?,
            (0.0, -2.0, 7.0, 3.0)
        );
        Ok(())
    }

    /// A null partial (an empty group's AABB) is a no-op in `combine_partials`.
    #[test]
    fn combine_partials_ignores_null() -> VortexResult<()> {
        let mut partial = AabbPartial {
            rect: Some(SpatialRect::new((0.0, 0.0), (1.0, 1.0))),
        };
        GeometryAabb.combine_partials(&mut partial, Scalar::null(aabb_dtype()))?;
        assert_eq!(
            aabb(&GeometryAabb.to_scalar(&partial)?)?,
            (0.0, 0.0, 1.0, 1.0)
        );
        Ok(())
    }

    /// All-NaN coordinates: `f64::min`/`max` skip the NaNs and `geo::Rect` normalizes the result to
    /// a valid (whole-plane) box, so such a chunk is always kept. Sound - NaN-coordinate rows can
    /// never satisfy `distance <= r` anyway.
    #[test]
    fn all_nan_coordinates_kept() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let column = point_column(vec![f64::NAN, f64::NAN], vec![f64::NAN, f64::NAN])?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, column.dtype().clone())?;
        acc.accumulate(&column, &mut ctx)?;

        let (xmin, ymin, xmax, ymax) = aabb(&acc.finish()?)?;
        assert!(xmin <= xmax && ymin <= ymax);
        Ok(())
    }

    /// An empty group yields a null AABB.
    #[test]
    fn empty_group_is_null() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, dtype)?;
        assert!(acc.finish()?.is_null());
        Ok(())
    }

    /// After `initialize`, the registry yields a default zone statistic for every native geometry
    /// type (so the zoned writer stores it) but none for ordinary numeric columns.
    #[test]
    fn registered_as_geometry_zone_default() -> VortexResult<()> {
        let session = spatial_session();

        for column in every_native_column(&[(0.0, 0.0), (1.0, 1.0)])? {
            assert!(
                !session
                    .aggregate_fns()
                    .zone_stat_defaults(column.dtype())
                    .is_empty(),
                "a geometry zone-stat default should be discovered for {}",
                column.dtype()
            );
        }
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert!(
            session
                .aggregate_fns()
                .zone_stat_defaults(&i32_dtype)
                .is_empty(),
            "no geometry zone-stat default should apply to numeric columns"
        );
        Ok(())
    }
}
