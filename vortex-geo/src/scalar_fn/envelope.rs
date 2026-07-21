// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `envelope`: the per-row 2-D axis-aligned bounding box (AABB) of a native geometry column.
//!
//! A row-oriented consumer (e.g. bulk-loading an in-memory R-tree in a spatial-join operator)
//! reads the resulting box column back row by row.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
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
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::GeoMetadata;
use crate::extension::Rect;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
use crate::extension::coordinate::f64_field;
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

/// The output extension type: a nullable native 2-D `geoarrow.box` ([`Rect`]). Nullable because a
/// null row has no box. Metadata is defaulted.
fn output_ext_dtype() -> VortexResult<ExtDType<Rect>> {
    ExtDType::<Rect>::try_new(
        GeoMetadata::default(),
        box_storage_dtype(Dimension::Xy, Nullability::Nullable),
    )
}

/// The per-row 2-D bounding box of `array` as a nullable `Rect` (`geoarrow.box`) column, computed
/// directly over the native coordinate storage — no decode to `geo_types`, no Arrow round-trip.
///
/// Walks the nested `List` storage down to the leaf coordinate `Struct`, carrying which top-level
/// row owns each coordinate, then min/maxes x/y per row over the raw `f64` buffers. The ring/part
/// nesting is irrelevant to a bounding box, only which coordinates belong to a row matters. A null
/// row, or a valid row that owns no coordinate (an empty geometry), yields a null box.
fn envelopes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let len = array.len();
    let valid = array.validity()?.execute_mask(len, ctx)?;
    let ext = array
        .dtype()
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("geo: envelope operand is not a geometry extension type"))?;
    let storage = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();

    // Per-row min/max accumulators; `seen[r]` marks a row that owns at least one coordinate.
    let mut lo = vec![(f64::INFINITY, f64::INFINITY); len];
    let mut hi = vec![(f64::NEG_INFINITY, f64::NEG_INFINITY); len];
    let mut seen = vec![false; len];

    if ext.is::<Rect>() {
        // The bounding box of a box is itself: read its min/max fields straight from storage.
        let coords = storage.execute::<StructArray>(ctx)?;
        let xmin = f64_field(&coords, "xmin", ctx)?;
        let ymin = f64_field(&coords, "ymin", ctx)?;
        let xmax = f64_field(&coords, "xmax", ctx)?;
        let ymax = f64_field(&coords, "ymax", ctx)?;
        let (xmin, ymin) = (xmin.as_slice::<f64>(), ymin.as_slice::<f64>());
        let (xmax, ymax) = (xmax.as_slice::<f64>(), ymax.as_slice::<f64>());
        for r in 0..len {
            lo[r] = (xmin[r], ymin[r]);
            hi[r] = (xmax[r], ymax[r]);
            seen[r] = true;
        }
    } else {
        // Walk any `List` nesting down to the leaf coordinate `Struct`, mapping each element to the
        // top-level row that owns it. `ListView` offsets/sizes are arbitrary integer types and need
        // not be contiguous, so map every element explicitly rather than composing offset arrays.
        let u64_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let mut owner: Vec<usize> = (0..len).collect();
        let mut node = storage;
        while matches!(node.dtype(), DType::List(..)) {
            let list = node.execute::<Canonical>(ctx)?.into_listview();
            let offsets = list
                .offsets()
                .clone()
                .cast(u64_dtype.clone())?
                .execute::<PrimitiveArray>(ctx)?;
            let sizes = list
                .sizes()
                .clone()
                .cast(u64_dtype.clone())?
                .execute::<PrimitiveArray>(ctx)?;
            let (offsets, sizes) = (offsets.as_slice::<u64>(), sizes.as_slice::<u64>());
            let elements = list.elements().clone();
            let mut child = vec![0usize; elements.len()];
            for (i, &row) in owner.iter().enumerate() {
                let start = usize::try_from(offsets[i])
                    .map_err(|_| vortex_err!("geo: list offset exceeds usize"))?;
                let size = usize::try_from(sizes[i])
                    .map_err(|_| vortex_err!("geo: list size exceeds usize"))?;
                child[start..start + size].fill(row);
            }
            owner = child;
            node = elements;
        }
        let coords = node.execute::<StructArray>(ctx)?;
        let xs = f64_field(&coords, "x", ctx)?;
        let ys = f64_field(&coords, "y", ctx)?;
        for ((&r, &x), &y) in owner
            .iter()
            .zip(xs.as_slice::<f64>())
            .zip(ys.as_slice::<f64>())
        {
            lo[r] = (lo[r].0.min(x), lo[r].1.min(y));
            hi[r] = (hi[r].0.max(x), hi[r].1.max(y));
            seen[r] = true;
        }
    }

    // Build the nullable `Rect` column straight from the accumulators: a row is present iff it owns
    // a coordinate and its operand row is valid; absent rows are null (physical placeholder `0.0`).
    let present: Vec<bool> = (0..len).map(|r| seen[r] && valid.value(r)).collect();
    let ordinate = |src: &[(f64, f64)], axis: fn((f64, f64)) -> f64| {
        PrimitiveArray::from_iter((0..len).map(|r| if present[r] { axis(src[r]) } else { 0.0 }))
            .into_array()
    };
    let storage = StructArray::try_new(
        FieldNames::from(["xmin", "ymin", "xmax", "ymax"]),
        vec![
            ordinate(&lo, |c| c.0),
            ordinate(&lo, |c| c.1),
            ordinate(&hi, |c| c.0),
            ordinate(&hi, |c| c.1),
        ],
        len,
        Validity::from_iter(present.iter().copied()),
    )?
    .into_array();
    Ok(ExtensionArray::try_new(output_ext_dtype()?.erased(), storage)?.into_array())
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
        Ok(DType::Extension(output_ext_dtype()?.erased()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = args.get(0)?;
        envelopes(&array, ctx)
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
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;

    use super::GeoEnvelope;
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
