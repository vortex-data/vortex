// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The 2D axis-aligned bounding-box (AABB) aggregate for native geometry columns.

use geo::Rect as SpatialRect;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::box_corners;
use crate::extension::coordinate::ordinates;
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
        ExtDType::<Rect>::try_new(
            SpatialMetadata::default().with_dimension(Dimension::Xy),
            Nullability::Nullable,
        )
        .vortex_expect("Rect accepts metadata with a coordinate dimension")
        .erased(),
    )
}

/// The AABB of the raw `x`/`y` slices, or `None` when empty.
///
/// A box gives no bound for a row with a `NaN` coordinate, such as `POINT EMPTY`: its distance is
/// `NaN`, which some float orders put above every number and others below it. So a `NaN`
/// coordinate makes the box the whole plane, and pruning keeps the chunk whatever the order of the
/// predicate.
fn aabb_of(xs: &[f64], ys: &[f64]) -> Option<SpatialRect<f64>> {
    if xs.is_empty() {
        return None;
    }

    if xs.iter().chain(ys).any(|v| v.is_nan()) {
        return Some(SpatialRect::new(
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, f64::INFINITY),
        ));
    }

    let [xmin, ymin, xmax, ymax] = box_corners(xs, ys);
    Some(SpatialRect::new((xmin, ymin), (xmax, ymax)))
}

/// Union two optional bounding boxes.
fn union_rects(
    lhs: Option<SpatialRect<f64>>,
    rhs: Option<SpatialRect<f64>>,
) -> Option<SpatialRect<f64>> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(SpatialRect::new(
            (a.min().x.min(b.min().x), a.min().y.min(b.min().y)),
            (a.max().x.max(b.max().x), a.max().y.max(b.max().y)),
        )),
        (some, None) | (None, some) => some,
    }
}

/// Compute the 2D AABB over all valid geometries in a native geometry array, or `None` when empty
/// or all-null. Any `NaN` ordinate widens the result to the whole plane.
pub(crate) fn aabb_of_array(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<SpatialRect<f64>>> {
    vortex_ensure!(
        is_native_geometry(array.dtype()),
        "spatial: operand is not a native geometry extension type, was {}",
        array.dtype()
    );
    let mut active = array.validity()?.execute_mask(array.len(), ctx)?;
    if active.all_false() {
        return Ok(None);
    }

    let mut node = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();

    if array
        .dtype()
        .as_extension_opt()
        .is_some_and(|ext| ext.is::<Rect>())
    {
        let storage = node.filter(active)?.execute::<StructArray>(ctx)?;
        let xmins = ordinates(&storage, "xmin", ctx)?;
        let ymins = ordinates(&storage, "ymin", ctx)?;
        let xmaxs = ordinates(&storage, "xmax", ctx)?;
        let ymaxs = ordinates(&storage, "ymax", ctx)?;
        return Ok(union_rects(
            aabb_of(&xmins, &ymins),
            aabb_of(&xmaxs, &ymaxs),
        ));
    }

    // Propagate the union of referenced ranges through each `ListView` level instead of rebuilding
    // lists, so null rows and unreferenced gaps are skipped while shared/overlapping views are
    // scanned only once.
    while node.dtype().is_list() {
        let list_view = node.execute::<ListViewArray>(ctx)?;
        let valid = list_view.validity()?.execute_mask(list_view.len(), ctx)?;
        active = &active & &valid;
        if active.all_false() {
            return Ok(None);
        }
        let offsets = list_view
            .offsets()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        let sizes = list_view
            .sizes()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        let mut ranges = Vec::new();
        for idx in active.to_bit_buffer().set_indices() {
            let size = usize::try_from(sizes[idx])?;
            if size == 0 {
                continue;
            }
            let start = usize::try_from(offsets[idx])?;
            ranges.push((start, start + size));
        }
        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (start, end) in ranges {
            if let Some((_, last_end)) = merged.last_mut()
                && start <= *last_end
            {
                *last_end = (*last_end).max(end);
            } else {
                merged.push((start, end));
            }
        }
        let elements = list_view.elements().clone();
        active = Mask::from_slices(elements.len(), merged);
        if active.all_false() {
            return Ok(None);
        }
        node = elements;
    }

    let coords = node.filter(active)?.execute::<StructArray>(ctx)?;
    let xs = ordinates(&coords, "x", ctx)?;
    let ys = ordinates(&coords, "y", ctx)?;
    Ok(aabb_of(&xs, &ys))
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

/// Serialize a [`SpatialRect`] as a stat scalar of `dtype`, the native `geoarrow.box` that
/// [`aabb_dtype`] builds (inverse of [`rect_from_storage`]). The caller passes the dtype that it
/// holds, so each group reuses it instead of building it again.
fn rect_to_storage(rect: SpatialRect<f64>, dtype: &DType) -> Scalar {
    let ext_dtype = dtype.as_extension();
    let storage = Scalar::struct_(
        ext_dtype.storage_dtype().clone(),
        vec![
            Scalar::primitive(rect.min().x, Nullability::NonNullable),
            Scalar::primitive(rect.min().y, Nullability::NonNullable),
            Scalar::primitive(rect.max().x, Nullability::NonNullable),
            Scalar::primitive(rect.max().y, Nullability::NonNullable),
        ],
    );
    Scalar::extension_ref(ext_dtype.clone(), storage)
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
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(AabbPartial { rect: None })
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        // A null box is an empty group's AABB.
        Ok(AabbPartial {
            rect: rect_from_storage(scalar)?,
        })
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        if let Some(rect) = second.rect {
            first.merge(rect);
        }
        Ok(first)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(match partial.rect {
            Some(rect) => rect_to_storage(rect, args.partial_dtype),
            None => Scalar::null(args.partial_dtype.clone()),
        })
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        // An AABB can always grow, so it is never saturated.
        false
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => {
                if constant.is_empty() {
                    return Ok(());
                }
                ConstantArray::new(constant.scalar().clone(), 1).into_array()
            }
        };
        if let Some(rect) = aabb_of_array(&array, ctx)? {
            partial.merge(rect);
        }
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        // The stored partial is already the AABB struct, so finalizing is the identity.
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use geo::Rect as SpatialRect;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateDTypes;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::AabbPartial;
    use super::GeometryAabb;
    use super::rect_from_storage;
    use crate::extension::LineString;
    use crate::extension::Rect;
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::nullable_rect_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::rect_column;
    use crate::test_harness::spatial_session;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
        let session = VortexSession::empty();
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
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut ctx = SESSION.create_execution_ctx();

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

    /// `merge_partials` unions partial boxes - the path the zoned writer takes when a zone's
    /// array is chunked.
    #[test]
    fn merge_partials_unions_boxes() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let dtypes = AggregateDTypes::try_new(&GeometryAabb, &EmptyOptions, dtype)?;
        let bbox = |xmin, ymin, xmax, ymax| AabbPartial {
            rect: Some(SpatialRect::new((xmin, ymin), (xmax, ymax))),
        };
        let reduced = GeometryAabb.merge_partials(
            dtypes.args(&EmptyOptions),
            bbox(0.0, 0.0, 1.0, 1.0),
            bbox(5.0, -2.0, 7.0, 3.0),
        )?;
        assert_eq!(
            aabb(&GeometryAabb.to_scalar(dtypes.args(&EmptyOptions), &reduced)?)?,
            (0.0, -2.0, 7.0, 3.0)
        );
        Ok(())
    }

    /// An empty partial (an empty group's AABB) is a no-op in `merge_partials`.
    #[test]
    fn merge_partials_ignores_empty() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let dtypes = AggregateDTypes::try_new(&GeometryAabb, &EmptyOptions, dtype)?;
        let empty = GeometryAabb.empty_partial(dtypes.args(&EmptyOptions))?;
        let value = AabbPartial {
            rect: Some(SpatialRect::new((0.0, 0.0), (1.0, 1.0))),
        };
        let reduced = GeometryAabb.merge_partials(dtypes.args(&EmptyOptions), value, empty)?;
        assert_eq!(
            aabb(&GeometryAabb.to_scalar(dtypes.args(&EmptyOptions), &reduced)?)?,
            (0.0, 0.0, 1.0, 1.0)
        );
        Ok(())
    }

    /// A `NaN` coordinate makes the box the whole plane, also next to finite coordinates, so the
    /// chunk is always kept. Its row has a `NaN` distance, which can satisfy `distance >= r` or
    /// `distance <= r`, depending on the float order of the predicate and the sign of the `NaN`.
    #[rstest]
    #[case::all_nan(vec![f64::NAN, f64::NAN], vec![f64::NAN, f64::NAN])]
    #[case::one_nan_row(vec![0.0, f64::NAN], vec![0.0, f64::NAN])]
    #[case::one_nan_ordinate(vec![0.0, 1.0], vec![0.0, -f64::NAN])]
    fn nan_coordinates_widen_to_whole_plane(
        #[case] xs: Vec<f64>,
        #[case] ys: Vec<f64>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let column = point_column(xs, ys)?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, column.dtype().clone())?;
        acc.accumulate(&column, &mut ctx)?;

        let (xmin, ymin, xmax, ymax) = aabb(&acc.finish()?)?;
        assert_eq!(
            (xmin, ymin, xmax, ymax),
            (
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::INFINITY
            )
        );
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

    /// `Rect` columns (2D, 3D, nullable, and with `NaN` corners) accumulate their union AABB.
    #[test]
    fn rect_aabb_accumulates_2d_3d_nullable_and_nan() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let rects = rect_column(vec![(1.0, 2.0, 3.0, 4.0), (-2.0, 5.0, 0.0, 6.0)])?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, rects.dtype().clone())?;
        acc.accumulate(&rects, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (-2.0, 2.0, 3.0, 6.0));

        // Null rows carry (0, 0, 0, 0) placeholders that must not pull the box toward the origin.
        let nullable = nullable_rect_column(vec![
            Some((5.0, 6.0, 7.0, 8.0)),
            None,
            Some((9.0, 10.0, 11.0, 12.0)),
        ])?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, nullable.dtype().clone())?;
        acc.accumulate(&nullable, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (5.0, 6.0, 11.0, 12.0));

        // 3D `Rect` ignores z bounds and computes the 2D XY extent.
        let ordinate = |value: f64| PrimitiveArray::from_iter([value]).into_array();
        let storage = StructArray::from_fields(&[
            ("xmin", ordinate(1.0)),
            ("ymin", ordinate(2.0)),
            ("zmin", ordinate(-99.0)),
            ("xmax", ordinate(3.0)),
            ("ymax", ordinate(4.0)),
            ("zmax", ordinate(99.0)),
        ])?
        .into_array();
        let ext = ExtDType::<Rect>::try_new(
            SpatialMetadata::default().with_dimension(Dimension::Xyz),
            Nullability::NonNullable,
        )?;
        let xyz = ExtensionArray::try_new(ext.erased(), storage)?.into_array();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, xyz.dtype().clone())?;
        acc.accumulate(&xyz, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (1.0, 2.0, 3.0, 4.0));

        // A `NaN` corner widens the AABB to the whole plane.
        let nan_rect = rect_column(vec![(0.0, 0.0, f64::NAN, 1.0)])?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, nan_rect.dtype().clone())?;
        acc.accumulate(&nan_rect, &mut ctx)?;
        assert_eq!(
            aabb(&acc.finish()?)?,
            (
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::INFINITY
            )
        );
        Ok(())
    }

    /// Unreferenced coordinates in a `ListView` backing buffer (from slicing, filtered null rows,
    /// or non-contiguous `ListView` gaps) must not widen the AABB.
    #[test]
    fn listview_unreferenced_coordinates_are_ignored() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // 1. Sliced `LineString`: rows 0 and 2 hold outliers outside the slice window.
        let lines = linestring_column(vec![
            vec![(-1000.0, -1000.0), (1000.0, 1000.0)],
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(500.0, 500.0), (600.0, 600.0)],
        ])?
        .slice(1..2)?;
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, lines.dtype().clone())?;
        acc.accumulate(&lines, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (1.0, 2.0, 3.0, 4.0));

        // 2. Non-contiguous `ListView` with both a gap element (100, 100) and a null row pointing
        // to (-999, -999) in the backing elements buffer.
        let elements = StructArray::from_fields(&[
            (
                "x",
                PrimitiveArray::from_iter([-999.0f64, 1.0, 2.0, 100.0, 3.0]).into_array(),
            ),
            (
                "y",
                PrimitiveArray::from_iter([-999.0f64, 4.0, 5.0, 100.0, 6.0]).into_array(),
            ),
        ])?
        .into_array();
        let list_view = ListViewArray::try_new(
            elements,
            PrimitiveArray::from_iter([0u32, 1, 4]).into_array(),
            PrimitiveArray::from_iter([1u32, 2, 1]).into_array(),
            Validity::from_iter([false, true, true]),
            &mut ctx,
        )?
        .into_array();
        let ext = ExtDType::<LineString>::try_new(
            SpatialMetadata::default().with_dimension(Dimension::Xy),
            Nullability::Nullable,
        )?;
        let gapped = ExtensionArray::try_new(ext.clone().erased(), list_view)?.into_array();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, gapped.dtype().clone())?;
        acc.accumulate(&gapped, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (1.0, 4.0, 3.0, 6.0));

        // 3. Overlapping `ListView` views sharing the same vertex range plus a repeated constant
        // batch must compute the same box without expanding shared vertices.
        let shared = ListViewArray::try_new(
            StructArray::from_fields(&[
                (
                    "x",
                    PrimitiveArray::from_iter([-999.0f64, 2.0, 4.0, 999.0]).into_array(),
                ),
                (
                    "y",
                    PrimitiveArray::from_iter([-999.0f64, 3.0, 5.0, 999.0]).into_array(),
                ),
            ])?
            .into_array(),
            PrimitiveArray::from_iter([1u32, 1, 1, 0]).into_array(),
            PrimitiveArray::from_iter([2u32, 2, 1, 4]).into_array(),
            Validity::from_iter([true, true, true, false]),
            &mut ctx,
        )?
        .into_array();
        let overlapping = ExtensionArray::try_new(ext.erased(), shared)?.into_array();
        let mut acc =
            Accumulator::try_new(GeometryAabb, EmptyOptions, overlapping.dtype().clone())?;
        acc.accumulate(&overlapping, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (2.0, 3.0, 4.0, 5.0));

        let constant =
            ConstantArray::new(overlapping.execute_scalar(0, &mut ctx)?, 4_096).into_array();
        let mut acc = Accumulator::try_new(GeometryAabb, EmptyOptions, constant.dtype().clone())?;
        acc.accumulate(&constant, &mut ctx)?;
        assert_eq!(aabb(&acc.finish()?)?, (2.0, 3.0, 4.0, 5.0));
        Ok(())
    }

    /// After `initialize`, the registry yields a default zone statistic for every native geometry
    /// type (so the zoned writer stores it) but none for ordinary numeric columns.
    #[test]
    fn registered_as_geometry_zone_default() -> VortexResult<()> {
        let session = spatial_session();

        for column in every_native_column(&[(0.0, 0.0), (1.0, 1.0)])?
            .into_iter()
            .chain(std::iter::once(rect_column(vec![(0.0, 0.0, 1.0, 1.0)])?))
        {
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
