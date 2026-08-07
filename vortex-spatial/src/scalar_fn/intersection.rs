// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Intersection`: pairwise planar intersection of native polygons.

use geo::BooleanOps;
use geo_types::Geometry;
use geo_types::MultiPolygon as GeoMultiPolygon;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::MultiPolygon;
use crate::extension::Polygon;
use crate::extension::SpatialMetadata;
use crate::extension::coordinate::Dimension;
use crate::extension::multipolygon_storage_dtype;
use crate::scalar_fn::execute::execute_binary_geo_types;

/// Resolve CRS metadata shared by two polygon operands.
fn intersection_metadata(
    left: &SpatialMetadata,
    right: &SpatialMetadata,
) -> VortexResult<SpatialMetadata> {
    match (&left.crs, &right.crs) {
        (Some(left_crs), Some(right_crs)) => {
            vortex_ensure!(
                left_crs == right_crs,
                "spatial: intersection operands have different coordinate reference systems: \
                 {left_crs} and {right_crs}"
            );
            Ok(left.clone())
        }
        (Some(_), None) => Ok(left.clone()),
        (None, Some(_)) => Ok(right.clone()),
        (None, None) => Ok(SpatialMetadata::default()),
    }
}

/// Metadata carried by a validated native polygonal dtype.
fn polygonal_metadata(dtype: &DType) -> &SpatialMetadata {
    let extension = dtype.as_extension();
    if extension.is::<Polygon>() {
        extension.metadata::<Polygon>()
    } else if extension.is::<MultiPolygon>() {
        extension.metadata::<MultiPolygon>()
    } else {
        unreachable!("intersection operand was validated as polygonal")
    }
}

/// Resolve the native polygonal intersection overloads, which always return a MultiPolygon.
fn intersection_dtype(dtypes: &[DType]) -> VortexResult<ExtDType<MultiPolygon>> {
    vortex_ensure!(
        dtypes.len() == 2,
        "spatial: intersection requires exactly two polygonal operands, got {}",
        dtypes.len()
    );
    for dtype in dtypes {
        vortex_ensure!(
            dtype.as_extension_opt().is_some_and(|extension| {
                extension.is::<Polygon>() || extension.is::<MultiPolygon>()
            }),
            "spatial: intersection operand {dtype} is not a native Polygon or MultiPolygon"
        );
    }

    let metadata = intersection_metadata(
        polygonal_metadata(&dtypes[0]),
        polygonal_metadata(&dtypes[1]),
    )?;
    let nullability = Nullability::from(dtypes.iter().any(DType::is_nullable));
    ExtDType::try_new(
        metadata,
        multipolygon_storage_dtype(Dimension::Xy, nullability),
    )
}

/// Dispatch decoded geometry enums to `geo`'s concrete polygonal `BooleanOps` implementations.
fn polygonal_intersection(left: &Geometry<f64>, right: &Geometry<f64>) -> GeoMultiPolygon<f64> {
    match (left, right) {
        (Geometry::Polygon(left), Geometry::Polygon(right)) => left.intersection(right),
        (Geometry::Polygon(left), Geometry::MultiPolygon(right)) => left.intersection(right),
        (Geometry::MultiPolygon(left), Geometry::Polygon(right)) => left.intersection(right),
        (Geometry::MultiPolygon(left), Geometry::MultiPolygon(right)) => left.intersection(right),
        _ => unreachable!("intersection operands were validated as polygonal"),
    }
}

/// Compute the pairwise two-dimensional intersection of native `Polygon` or `MultiPolygon`
/// operands as a native `MultiPolygon`. Disjoint and boundary-only intersections produce an
/// empty `MultiPolygon`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialIntersection;

impl SpatialIntersection {
    /// A lazy `ScalarFnArray` intersecting two native polygonal operands by row.
    pub fn try_new_array(left: ArrayRef, right: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialIntersection, EmptyOptions).erased(),
            vec![left, right],
        )
    }
}

impl ScalarFnVTable for SpatialIntersection {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.intersection");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("left"),
            1 => ChildName::from("right"),
            _ => unreachable!("intersection has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Extension(intersection_dtype(dtypes)?.erased()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let left = args.get(0)?;
        let right = args.get(1)?;
        let output_dtype = intersection_dtype(&[left.dtype().clone(), right.dtype().clone()])?;
        execute_binary_geo_types(
            &left,
            &right,
            DType::Extension(output_dtype.erased()),
            polygonal_intersection,
            None,
            ctx,
        )
    }

    fn validity(
        &self,
        _: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use geo_types::Geometry;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Columnar;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::SpatialIntersection;
    use crate::extension::MultiPolygon;
    use crate::extension::geometries;
    use crate::scalar_fn::area::SpatialArea;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    fn square(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<(f64, f64)> {
        vec![
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
    }

    fn polygon_constant(
        ring: Vec<(f64, f64)>,
        len: usize,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let scalar = polygon_column(vec![vec![ring]])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    fn polygonal_column(ring: Vec<(f64, f64)>, multi: bool) -> VortexResult<ArrayRef> {
        if multi {
            multipolygon_column(vec![vec![vec![ring]]])
        } else {
            polygon_column(vec![vec![ring]])
        }
    }

    #[test]
    fn q9_area_pipeline_handles_overlap_disjoint_and_touching() -> VortexResult<()> {
        let left = polygon_column(vec![
            vec![square(0.0, 0.0, 2.0, 2.0)],
            vec![square(0.0, 0.0, 1.0, 1.0)],
            vec![square(0.0, 0.0, 1.0, 1.0)],
        ])?;
        let right = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(2.0, 2.0, 3.0, 3.0)],
            vec![square(1.0, 0.0, 2.0, 1.0)],
        ])?;
        let intersections = SpatialIntersection::try_new_array(left, right)?.into_array();
        assert!(intersections.dtype().as_extension().is::<MultiPolygon>());

        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let decoded = geometries(&intersections, &mut ctx)?;
        let polygon_counts = decoded
            .iter()
            .map(|geometry| match geometry {
                Geometry::MultiPolygon(multipolygon) => Ok(multipolygon.0.len()),
                other => Err(vortex_err!(
                    "intersection decoded as {other:?}, expected MultiPolygon"
                )),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        assert_eq!(polygon_counts, [1, 0, 0]);

        let areas = SpatialArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([1.0_f64, 0.0, 0.0]).into_array();
        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::polygon_polygon(false, false)]
    #[case::polygon_multipolygon(false, true)]
    #[case::multipolygon_polygon(true, false)]
    #[case::multipolygon_multipolygon(true, true)]
    fn supports_all_polygonal_combinations(
        #[case] left_multi: bool,
        #[case] right_multi: bool,
    ) -> VortexResult<()> {
        let left = polygonal_column(square(0.0, 0.0, 2.0, 2.0), left_multi)?;
        let right = polygonal_column(square(1.0, 1.0, 3.0, 3.0), right_multi)?;
        let intersections = SpatialIntersection::try_new_array(left, right)?.into_array();
        let areas = SpatialArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([1.0_f64]).into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn preserves_holes() -> VortexResult<()> {
        let left = polygon_column(vec![vec![
            square(0.0, 0.0, 4.0, 4.0),
            square(1.0, 1.0, 3.0, 3.0),
        ]])?;
        let right = polygon_column(vec![vec![square(2.0, 0.0, 5.0, 4.0)]])?;
        let intersections = SpatialIntersection::try_new_array(left, right)?.into_array();
        let areas = SpatialArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([6.0_f64]).into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn propagates_nulls() -> VortexResult<()> {
        let left = MaskedArray::try_new(
            polygon_column(vec![
                vec![square(0.0, 0.0, 2.0, 2.0)],
                vec![square(0.0, 0.0, 2.0, 2.0)],
            ])?,
            Validity::from_iter([true, false]),
        )?
        .into_array();
        let right = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(1.0, 1.0, 3.0, 3.0)],
        ])?;
        let intersections = SpatialIntersection::try_new_array(left, right)?.into_array();
        let areas = SpatialArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::new(vec![1.0_f64, 0.0], Validity::from_iter([true, false]))
            .into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::constant_left(true)]
    #[case::constant_right(false)]
    fn pairs_constants_with_columns(#[case] constant_left: bool) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let constant = polygon_constant(square(0.0, 0.0, 2.0, 2.0), 2, &mut ctx)?;
        let column = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(3.0, 3.0, 4.0, 4.0)],
        ])?;
        let (left, right) = if constant_left {
            (constant, column)
        } else {
            (column, constant)
        };

        let intersections = SpatialIntersection::try_new_array(left, right)?.into_array();
        let areas = SpatialArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([1.0_f64, 0.0]).into_array();
        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn two_constants_remain_constant() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let left = polygon_constant(square(0.0, 0.0, 2.0, 2.0), 3, &mut ctx)?;
        let right = polygon_constant(square(1.0, 1.0, 3.0, 3.0), 3, &mut ctx)?;

        let result = SpatialIntersection::try_new_array(left, right)?.into_array();
        let Columnar::Constant(constant) = result.execute::<Columnar>(&mut ctx)? else {
            return Err(vortex_err!(
                "intersection of two constants should remain constant"
            ));
        };
        assert_eq!(constant.len(), 3);
        Ok(())
    }

    #[rstest]
    #[case::none(0)]
    #[case::one(1)]
    #[case::three(3)]
    fn rejects_wrong_arity(#[case] arity: usize) -> VortexResult<()> {
        let dtype = polygon_column(vec![vec![]])?.dtype().clone();
        assert!(
            SpatialIntersection
                .return_dtype(&EmptyOptions, &vec![dtype; arity])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_polygonal_input() -> VortexResult<()> {
        let polygon = polygon_column(vec![vec![]])?;
        let point = point_column(vec![0.0], vec![0.0])?;
        assert!(SpatialIntersection::try_new_array(polygon, point).is_err());
        Ok(())
    }
}
