// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An aggregate computing the 2D minimum bounding rectangle of a native geometry column as
//! `Struct<xmin, ymin, xmax, ymax>`. Stored as a zone statistic, it lets spatial filters prune
//! chunks whose bounding box cannot intersect the query region.

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::coordinates;
use crate::extension::is_native_geometry;

/// Aggregate computing the 2D minimum bounding rectangle of a native geometry column.
#[derive(Clone, Debug)]
pub struct GeometryBounds;

/// An axis-aligned bounding box.
#[derive(Clone, Copy)]
struct Bbox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
}

impl Bbox {
    /// The smallest box containing both `self` and `other`.
    fn union(self, other: Bbox) -> Bbox {
        Bbox {
            xmin: self.xmin.min(other.xmin),
            ymin: self.ymin.min(other.ymin),
            xmax: self.xmax.max(other.xmax),
            ymax: self.ymax.max(other.ymax),
        }
    }
}

/// Partial MBR accumulator: the union of every bounding box seen so far, or `None` when empty.
pub struct BoundsPartial {
    bbox: Option<Bbox>,
}

impl BoundsPartial {
    fn merge(&mut self, other: Bbox) {
        self.bbox = Some(match self.bbox {
            Some(cur) => cur.union(other),
            None => other,
        });
    }
}

/// `Struct<xmin, ymin, xmax, ymax>` of `f64`. Nullable so an empty group is a null MBR; the fields
/// are nullable too, so the pruning proof's `get_item` keeps a consistent nullable dtype.
fn bounds_dtype() -> DType {
    let coord = DType::Primitive(PType::F64, Nullability::Nullable);
    DType::Struct(
        StructFields::from_iter([
            ("xmin", coord.clone()),
            ("ymin", coord.clone()),
            ("xmax", coord.clone()),
            ("ymax", coord),
        ]),
        Nullability::Nullable,
    )
}

/// The bounding box of the coordinate slices, or `None` for an empty chunk.
fn bounds_of(xs: &[f64], ys: &[f64]) -> Option<Bbox> {
    if xs.is_empty() {
        return None;
    }
    let min_max = |vals: &[f64]| {
        vals.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    };
    let (xmin, xmax) = min_max(xs);
    let (ymin, ymax) = min_max(ys);
    Some(Bbox {
        xmin,
        ymin,
        xmax,
        ymax,
    })
}

impl AggregateFnVTable for GeometryBounds {
    type Options = EmptyOptions;
    type Partial = BoundsPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.geo.bounds");
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
        is_native_geometry(input_dtype).then(bounds_dtype)
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
        Ok(BoundsPartial { bbox: None })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let fields = other.as_struct();
        let read = |name: &str| -> VortexResult<f64> {
            f64::try_from(
                &fields
                    .field(name)
                    .ok_or_else(|| vortex_err!("bounds missing {name}"))?,
            )
        };
        partial.merge(Bbox {
            xmin: read("xmin")?,
            ymin: read("ymin")?,
            xmax: read("xmax")?,
            ymax: read("ymax")?,
        });
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match partial.bbox {
            Some(b) => Scalar::struct_(
                bounds_dtype(),
                vec![
                    Scalar::primitive(b.xmin, Nullability::Nullable),
                    Scalar::primitive(b.ymin, Nullability::Nullable),
                    Scalar::primitive(b.xmax, Nullability::Nullable),
                    Scalar::primitive(b.ymax, Nullability::Nullable),
                ],
            ),
            None => Scalar::null(bounds_dtype()),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.bbox = None;
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        // A bounding box can always grow, so it is never saturated.
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
        let coords = coordinates(&array, ctx)?;
        let xs = coords
            .unmasked_field_by_name("x")?
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let ys = coords
            .unmasked_field_by_name("y")?
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        if let Some(bbox) = bounds_of(xs.as_slice::<f64>(), ys.as_slice::<f64>()) {
            partial.merge(bbox);
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        // The stored partial is already the MBR struct, so finalizing is the identity.
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
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
    use vortex_error::vortex_err;

    use super::Bbox;
    use super::BoundsPartial;
    use super::GeometryBounds;
    use super::bounds_dtype;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    /// The aggregate must be serializable so the zoned writer can persist its zone-stat descriptor.
    #[test]
    fn serializes_for_zone_storage() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let metadata = GeometryBounds
            .serialize(&EmptyOptions)?
            .expect("GeometryBounds must be serializable to be stored as a zone statistic");
        GeometryBounds.deserialize(&metadata, &session)?;
        Ok(())
    }

    /// The MBR result's corners as `(xmin, ymin, xmax, ymax)`.
    fn mbr(result: &Scalar) -> VortexResult<(f64, f64, f64, f64)> {
        let fields = result.as_struct();
        let read = |name: &str| -> VortexResult<f64> {
            f64::try_from(
                &fields
                    .field(name)
                    .ok_or_else(|| vortex_err!("missing {name}"))?,
            )
        };
        Ok((read("xmin")?, read("ymin")?, read("xmax")?, read("ymax")?))
    }

    /// The MBR of a Point column is the min/max of its coordinates, accumulated across batches.
    #[test]
    fn point_bounds_across_batches() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;

        acc.accumulate(&point_column(vec![1.0, 3.0], vec![2.0, 4.0])?, &mut ctx)?;
        acc.accumulate(&point_column(vec![-1.0], vec![5.0])?, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (-1.0, 2.0, 3.0, 5.0));
        Ok(())
    }

    /// The MBR of a Polygon column unions every ring vertex — exercising the `List<List<Struct>>`
    /// unwrap, not just the bare Point struct.
    #[test]
    fn polygon_bounds_union_all_vertices() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // Two rectangles: (0,0)-(2,3) and (5,5)-(7,8). The chunk MBR is their union: (0,0)-(7,8).
        let polygons = polygon_column(vec![
            vec![vec![(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)]],
            vec![vec![(5.0, 5.0), (7.0, 5.0), (7.0, 8.0), (5.0, 8.0)]],
        ])?;
        let dtype = polygons.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        acc.accumulate(&polygons, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (0.0, 0.0, 7.0, 8.0));
        Ok(())
    }

    /// The MBR of a MultiPolygon column unions every vertex of every polygon's rings — exercising
    /// the triple-`List` unwrap.
    #[test]
    fn multipolygon_bounds_union_all_vertices() -> VortexResult<()> {
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
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        acc.accumulate(&multipolygons, &mut ctx)?;

        assert_eq!(mbr(&acc.finish()?)?, (-3.0, 0.0, 5.0, 9.0));
        Ok(())
    }

    /// `combine_partials` unions partial boxes — the path the zoned writer takes when a zone's
    /// array is chunked.
    #[test]
    fn combine_partials_unions_boxes() -> VortexResult<()> {
        let bbox = |xmin, ymin, xmax, ymax| BoundsPartial {
            bbox: Some(Bbox {
                xmin,
                ymin,
                xmax,
                ymax,
            }),
        };
        let mut partial = BoundsPartial { bbox: None };
        GeometryBounds.combine_partials(
            &mut partial,
            GeometryBounds.to_scalar(&bbox(0.0, 0.0, 1.0, 1.0))?,
        )?;
        GeometryBounds.combine_partials(
            &mut partial,
            GeometryBounds.to_scalar(&bbox(5.0, -2.0, 7.0, 3.0))?,
        )?;
        assert_eq!(
            mbr(&GeometryBounds.to_scalar(&partial)?)?,
            (0.0, -2.0, 7.0, 3.0)
        );
        Ok(())
    }

    /// A null partial (an empty group's MBR) is a no-op in `combine_partials`.
    #[test]
    fn combine_partials_ignores_null() -> VortexResult<()> {
        let mut partial = BoundsPartial {
            bbox: Some(Bbox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
            }),
        };
        GeometryBounds.combine_partials(&mut partial, Scalar::null(bounds_dtype()))?;
        assert_eq!(
            mbr(&GeometryBounds.to_scalar(&partial)?)?,
            (0.0, 0.0, 1.0, 1.0)
        );
        Ok(())
    }

    /// All-NaN coordinates fold to an inverted box (min > max). Sound to store: the pruning proof
    /// then skips the chunk, and NaN-coordinate rows can never satisfy `distance <= r` anyway.
    #[test]
    fn all_nan_coordinates_yield_inverted_box() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let column = point_column(vec![f64::NAN, f64::NAN], vec![f64::NAN, f64::NAN])?;
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, column.dtype().clone())?;
        acc.accumulate(&column, &mut ctx)?;

        let (xmin, ymin, xmax, ymax) = mbr(&acc.finish()?)?;
        assert!(xmin > xmax && ymin > ymax);
        Ok(())
    }

    /// An empty group yields a null MBR.
    #[test]
    fn empty_group_is_null() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let mut acc = Accumulator::try_new(GeometryBounds, EmptyOptions, dtype)?;
        assert!(acc.finish()?.is_null());
        Ok(())
    }

    /// After `initialize`, the registry yields a default zone statistic for geometry columns (so the
    /// zoned writer stores it) but none for ordinary numeric columns.
    #[test]
    fn registered_as_geometry_zone_default() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        assert!(
            !session
                .aggregate_fns()
                .zone_stat_defaults(&point_dtype)
                .is_empty(),
            "a geometry zone-stat default should be discovered for Point columns"
        );
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
