// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An aggregate computing the minimum bounding rectangle (2D) of a native
//! geometry column as `Struct<xmin, ymin, xmax, ymax>`. Stored as a zone statistic, it lets spatial
//! filters prune chunks whose bounding box cannot intersect the query region.

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

use crate::extension::coordinates;
use crate::extension::is_native_geometry;

/// Aggregate computing the minimum bounding rectangle of a native geometry column, as
/// `Struct<xmin, ymin, xmax, ymax>` of `f64`.
#[derive(Clone, Debug)]
pub struct GeometryBounds;

/// An axis-aligned bounding box: four `f64`s, all this aggregate needs to min/max coordinates.
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

/// `Struct<xmin, ymin, xmax, ymax>` of `f64`, nullable so an empty group yields a null MBR. The
/// coordinate fields are themselves nullable so that extracting one from the nullable struct (as the
/// pruning proof does) keeps a consistent nullable dtype.
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
        AggregateFnId::new("vortex.geo.bounds")
    }

    // Serializable so the zoned writer can persist this as a per-chunk stat. No options to encode.
    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8], _session: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_native_geometry(input_dtype).then(bounds_dtype)
    }

    fn zone_stat_default(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        // Geometry columns get a per-chunk bounding box for pruning.
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
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::GeometryBounds;
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

    /// The MBR of a Polygon column is the min/max over every ring vertex of every polygon —
    /// exercising the `List<List<Struct>>` unwrap, not just the bare Point struct.
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
        use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
        use vortex_array::dtype::DType;
        use vortex_array::dtype::Nullability;
        use vortex_array::dtype::PType;

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
