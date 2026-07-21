// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `envelope`: the per-row 2-D axis-aligned bounding box (AABB) of a native geometry column.
//!
//! A row-oriented consumer (e.g. bulk-loading an in-memory R-tree in a spatial-join operator)
//! reads the resulting box column back row by row.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::GeoMetadata;
use crate::extension::Rect;
use crate::extension::box_field_names;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::box_corners;
use crate::extension::coordinate::ordinates;
use crate::extension::flatten_row_offsets;
use crate::extension::validate_geometry_operands;

/// `envelope`: the axis-aligned bounding box of each geometry in a native geometry operand (a column
/// or a constant literal), as a native 2-D `geoarrow.box` ([`Rect`]) column.
///
/// 2-D only: only the `x`/`y` leaf ordinates are read, so any `z`/`m` are ignored and each box is
/// the XY extent — matching the [`GeometryAabb`](crate::aggregate_fn::GeometryAabb) aggregate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoEnvelope;

impl GeoEnvelope {
    /// A lazy `ScalarFnArray` computing the per-row bounding box of geometry operand `a`, which may
    /// be constant. The output length is taken from `a`.
    pub fn try_new_array(a: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoEnvelope, EmptyOptions).erased(),
            vec![a],
        )
    }
}

/// The output dtype: a nullable native 2-D box ([`Rect`], `geoarrow.box`) column. Nullable
/// because rows without a box — null or empty geometries — are null. Metadata is defaulted.
fn output_box_dtype() -> VortexResult<ExtDType<Rect>> {
    ExtDType::<Rect>::try_new(
        GeoMetadata::default(),
        box_storage_dtype(Dimension::Xy, Nullability::Nullable),
    )
}

/// Compute each row's 2-D bounding box: the smallest rectangle covering all of the row's
/// coordinates.
fn row_boxes(storage: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Vec<ArrayRef>, Validity)> {
    let len = storage.len();
    let valid = storage.validity()?.execute_mask(len, ctx)?;
    let (row_offsets, coords) = flatten_row_offsets(storage, ctx)?;

    // A row has a box iff it is valid and owns at least one coordinate (an empty geometry has
    // no box): the envelope-specific narrowing of the operand's mask, combined word-at-a-time.
    let non_empty = Mask::from(BitBuffer::collect_bool(len, |r| {
        row_offsets[r] < row_offsets[r + 1]
    }));
    let xs = ordinates(&coords, "x", ctx)?;
    let ys = ordinates(&coords, "y", ctx)?;

    // The output's four corner columns.
    let mut xmins = BufferMut::zeroed(len);
    let mut ymins = BufferMut::zeroed(len);
    let mut xmaxs = BufferMut::zeroed(len);
    let mut ymaxs = BufferMut::zeroed(len);

    for (r, (&start, &end)) in row_offsets.iter().zip(&row_offsets[1..]).enumerate() {
        let [xmin, ymin, xmax, ymax] = box_corners(&xs[start..end], &ys[start..end]);
        xmins[r] = xmin;
        ymins[r] = ymin;
        xmaxs[r] = xmax;
        ymaxs[r] = ymax;
    }

    Ok((
        vec![
            xmins.freeze().into_array(),
            ymins.freeze().into_array(),
            xmaxs.freeze().into_array(),
            ymaxs.freeze().into_array(),
        ],
        Validity::from_mask(&valid & &non_empty, Nullability::Nullable),
    ))
}

impl ScalarFnVTable for GeoEnvelope {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.geo.envelope");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("geometry"),
            _ => unreachable!("envelope has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_geometry_operands(dtypes)?;
        // Always nullable: an empty geometry has no box, so nulls can appear even over a
        // non-nullable operand.
        Ok(DType::Extension(output_box_dtype()?.erased()))
    }

    /// Compute each row's box directly over the native coordinate storage — no decode to
    /// `geo_types`, no Arrow round-trip. A null row, or a valid row that owns no coordinate (an
    /// empty geometry), yields a null box.
    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = args.get(0)?;
        let len = array.len();
        let ext = array
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("geo: envelope operand is not a geometry extension type"))?;
        let storage = array
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone();

        let (corners, output_validity) = if ext.is::<Rect>() {
            // A box is its own envelope: project the 2-D corner fields straight out of storage
            // (dropping any z/m bounds). A stored box cannot be empty, so the output validity
            // is exactly the operand's, kept lazy.
            let coords = storage.execute::<StructArray>(ctx)?;
            let corners = box_field_names(Dimension::Xy)
                .iter()
                .map(|name| coords.unmasked_field_by_name(name).cloned())
                .collect::<VortexResult<Vec<_>>>()?;
            (corners, array.validity()?.into_nullable())
        } else {
            row_boxes(storage, ctx)?
        };

        // Nullness lives at the box (struct) level: the corner fields stay non-nullable `f64`,
        // holding unspecified values under null rows.
        let storage = StructArray::try_new(
            FieldNames::from(box_field_names(Dimension::Xy)),
            corners,
            len,
            output_validity,
        )?
        .into_array();
        Ok(ExtensionArray::try_new(output_box_dtype()?.erased(), storage)?.into_array())
    }

    fn validity(&self, _: &Self::Options, _: &Expression) -> VortexResult<Option<Expression>> {
        // The output null mask is not derivable from the operand's validity alone: an empty
        // geometry yields a null box even where the operand is valid. Let the planner execute.
        Ok(None)
    }

    fn is_null_sensitive(&self, _: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;

    use super::GeoEnvelope;
    use crate::extension::GeoMetadata;
    use crate::extension::Rect;
    use crate::extension::box_storage_dtype;
    use crate::extension::coordinate::Dimension;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::nullable_rect_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::rect_column;

    /// Execute a `GeoEnvelope` over `array`, returning the lazy box column.
    fn boxes(array: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(GeoEnvelope::try_new_array(array)?.into_array())
    }

    /// A point's box is degenerate: both corners are the point itself.
    #[test]
    fn point_box_is_degenerate() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![1.0, 3.0], vec![2.0, 4.0])?;
        let expected =
            nullable_rect_column(vec![Some((1.0, 2.0, 1.0, 2.0)), Some((3.0, 4.0, 3.0, 4.0))])?;
        assert_arrays_eq!(boxes(points)?, expected, &mut ctx);
        Ok(())
    }

    /// A polygon's box is its extent: the min/max over every ring vertex, one box per row.
    #[test]
    fn polygon_box_is_extent() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let polygons = polygon_column(vec![
            vec![vec![(0.0, 0.0), (4.0, 0.0), (2.0, 5.0)]],
            vec![vec![(-3.0, 7.0), (-2.0, 7.0), (-2.0, 9.0), (-3.0, 9.0)]],
        ])?;
        let expected = nullable_rect_column(vec![
            Some((0.0, 0.0, 4.0, 5.0)),
            Some((-3.0, 7.0, -2.0, 9.0)),
        ])?;
        assert_arrays_eq!(boxes(polygons)?, expected, &mut ctx);
        Ok(())
    }

    /// A `Rect` row is its own bounding box.
    #[test]
    fn rect_box_is_itself() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let rects = rect_column(vec![(0.0, 0.0, 2.0, 3.0), (-1.0, -1.0, 1.0, 1.0)])?;
        let expected = nullable_rect_column(vec![
            Some((0.0, 0.0, 2.0, 3.0)),
            Some((-1.0, -1.0, 1.0, 1.0)),
        ])?;
        assert_arrays_eq!(boxes(rects)?, expected, &mut ctx);
        Ok(())
    }

    /// The `Rect` fast path projects the 2-D corners by name, so a 3-D box — whose `zmin`/`zmax`
    /// fields are interleaved between them in storage — yields its XY extent as a 2-D box.
    #[test]
    fn xyz_rect_drops_z_bounds() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let ordinate = |value: f64| PrimitiveArray::from_iter([value]).into_array();
        let storage = StructArray::from_fields(&[
            ("xmin", ordinate(0.0)),
            ("ymin", ordinate(1.0)),
            ("zmin", ordinate(-9.0)),
            ("xmax", ordinate(2.0)),
            ("ymax", ordinate(3.0)),
            ("zmax", ordinate(9.0)),
        ])?
        .into_array();
        let ext = ExtDType::<Rect>::try_new(
            GeoMetadata::default(),
            box_storage_dtype(Dimension::Xyz, Nullability::NonNullable),
        )?;
        let rects = ExtensionArray::try_new(ext.erased(), storage)?.into_array();

        let expected = nullable_rect_column(vec![Some((0.0, 1.0, 2.0, 3.0))])?;
        assert_arrays_eq!(boxes(rects)?, expected, &mut ctx);
        Ok(())
    }

    /// Every multi-vertex native type over the same vertex set yields that set's box, so the whole
    /// type family is covered (`Point` has its own degenerate-box test above).
    #[test]
    fn covers_every_native_geometry_type() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let vertices = vec![(1.0, 2.0), (-1.0, 5.0), (3.0, 4.0)];
        let columns = vec![
            linestring_column(vec![vertices.clone()])?,
            multipoint_column(vec![vertices.clone()])?,
            polygon_column(vec![vec![vertices.clone()]])?,
            multilinestring_column(vec![vec![vertices.clone()]])?,
            multipolygon_column(vec![vec![vec![vertices]]])?,
        ];
        let expected = nullable_rect_column(vec![Some((-1.0, 2.0, 3.0, 5.0))])?;
        for column in columns {
            assert_arrays_eq!(boxes(column)?, expected.clone(), &mut ctx);
        }
        Ok(())
    }

    /// Intermediate list levels with more than one part per row keep coordinates attributed to
    /// the right rows: row 0 owns two polygons (the first with two rings) whose extremes live in
    /// the second polygon, so composed bounds diverging from loop positions shows up here.
    #[test]
    fn uneven_nesting_keeps_rows_aligned() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let multipolygons = multipolygon_column(vec![
            vec![
                vec![vec![(0.0, 0.0), (1.0, 1.0)], vec![(0.2, 0.2), (0.8, 0.8)]],
                vec![vec![(5.0, -3.0), (6.0, 7.0)]],
            ],
            vec![vec![vec![(10.0, 10.0), (11.0, 12.0)]]],
        ])?;
        let expected = nullable_rect_column(vec![
            Some((0.0, -3.0, 6.0, 7.0)),
            Some((10.0, 10.0, 11.0, 12.0)),
        ])?;
        assert_arrays_eq!(boxes(multipolygons)?, expected, &mut ctx);
        Ok(())
    }

    /// An empty geometry (here a zero-part multipolygon) has no extent and yields a null box; other
    /// rows keep their boxes.
    #[test]
    fn empty_geometry_has_no_box() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let multipolygons = multipolygon_column(vec![
            vec![vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]],
            vec![],
        ])?;
        let expected = nullable_rect_column(vec![Some((0.0, 0.0, 1.0, 1.0)), None])?;
        assert_arrays_eq!(boxes(multipolygons)?, expected, &mut ctx);
        Ok(())
    }

    /// A geometry empty only at an inner level — here a polygon whose single ring has zero
    /// vertices, in the first row — has no box, exactly like one empty at the outer level.
    #[test]
    fn inner_empty_ring_yields_null_box() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let polygons = polygon_column(vec![
            vec![vec![]],
            vec![vec![(1.0, 2.0), (3.0, 4.0), (1.0, 4.0)]],
        ])?;
        let expected = nullable_rect_column(vec![None, Some((1.0, 2.0, 3.0, 4.0))])?;
        assert_arrays_eq!(boxes(polygons)?, expected, &mut ctx);
        Ok(())
    }

    /// A sliced operand keeps per-row boxes aligned: coordinates outside the slice window (still
    /// present in the sliced list's element buffer) must not leak into any row's box.
    #[test]
    fn sliced_operand_ignores_out_of_slice_coordinates() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let multipoints = multipoint_column(vec![
            vec![(-100.0, -100.0), (100.0, 100.0)],
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(5.0, 6.0)],
        ])?;
        let expected =
            nullable_rect_column(vec![Some((1.0, 2.0, 3.0, 4.0)), Some((5.0, 6.0, 5.0, 6.0))])?;
        assert_arrays_eq!(boxes(multipoints.slice(1..3)?)?, expected, &mut ctx);
        Ok(())
    }

    /// A null geometry row yields a null box, just like an empty geometry; valid rows keep theirs.
    #[test]
    fn null_row_yields_null_box() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = nullable_point_column(vec![Some((1.0, 2.0)), None, Some((3.0, 4.0))])?;
        let expected = nullable_rect_column(vec![
            Some((1.0, 2.0, 1.0, 2.0)),
            None,
            Some((3.0, 4.0, 3.0, 4.0)),
        ])?;
        assert_arrays_eq!(boxes(points)?, expected, &mut ctx);
        Ok(())
    }

    /// Valid, empty, and null rows stay positionally aligned: the valid row keeps its box, and the
    /// empty and null rows are both null.
    #[test]
    fn mixed_valid_empty_null_rows_align() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let multipolygons = nullable_multipolygon_column(vec![
            Some(vec![vec![vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ]]]),
            Some(vec![]),
            None,
        ])?;
        let expected = nullable_rect_column(vec![Some((0.0, 0.0, 1.0, 1.0)), None, None])?;
        assert_arrays_eq!(boxes(multipolygons)?, expected, &mut ctx);
        Ok(())
    }

    /// A constant-null operand yields an all-null box column.
    #[test]
    fn constant_null_is_all_null() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let null_const = ConstantArray::new(Scalar::null(point_dtype), 2).into_array();
        let expected = nullable_rect_column(vec![None, None])?;
        assert_arrays_eq!(boxes(null_const)?, expected, &mut ctx);
        Ok(())
    }

    /// Output is always nullable, even over a non-nullable operand, since an empty geometry has no
    /// box.
    #[test]
    fn output_is_always_nullable() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        assert!(!dtype.is_nullable());
        let out = GeoEnvelope.return_dtype(&EmptyOptions, &[dtype])?;
        assert!(out.is_nullable());
        Ok(())
    }

    /// A non-geometry operand dtype is rejected up front, before execution.
    #[test]
    fn non_geometry_operand_is_rejected() -> VortexResult<()> {
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert!(GeoEnvelope.return_dtype(&EmptyOptions, &[numeric]).is_err());
        Ok(())
    }
}
