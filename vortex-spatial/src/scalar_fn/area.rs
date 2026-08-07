// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Area`: unsigned planar area of native geometries.

use geo::Area;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
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

use crate::extension::is_native_geometry;
use crate::scalar_fn::execute::execute_unary_geo_types;

/// Validate the native geometry operand accepted by `ST_Area`.
fn validate_area_operand(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: area requires exactly one geometry operand, got {}",
        dtypes.len()
    );
    vortex_ensure!(
        is_native_geometry(&dtypes[0]),
        "spatial: area operand {} is not a native geometry",
        dtypes[0]
    );
    Ok(())
}

/// Unsigned planar `ST_Area` of native geometries.
///
/// Points and line strings have zero area, polygons and multipolygons use their two-dimensional
/// coordinates, and rectangles use width times height. Higher coordinate dimensions are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialArea;

impl SpatialArea {
    /// A lazy `ScalarFnArray` computing the per-row area of a native geometry operand.
    pub fn try_new_array(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialArea, EmptyOptions).erased(),
            vec![array],
        )
    }
}

impl ScalarFnVTable for SpatialArea {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.area");
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
            _ => unreachable!("area has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_area_operand(dtypes)?;
        Ok(DType::Primitive(PType::F64, dtypes[0].nullability()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = args.get(0)?;
        execute_unary_geo_types(
            &array,
            DType::Primitive(PType::F64, array.dtype().nullability()),
            Area::unsigned_area,
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
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::SpatialArea;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::rect_column;

    #[rstest]
    #[case::point(point_column(vec![1.0], vec![2.0]), &[0.0])]
    #[case::line_string(
        linestring_column(vec![vec![(0.0, 0.0), (3.0, 4.0)]]),
        &[0.0]
    )]
    #[case::multi_point(
        multipoint_column(vec![vec![(0.0, 0.0), (1.0, 1.0)]]),
        &[0.0]
    )]
    #[case::multi_line_string(
        multilinestring_column(vec![vec![
            vec![(0.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 3.0)],
        ]]),
        &[0.0]
    )]
    #[case::polygon(
        polygon_column(vec![
            vec![
                vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.0, 0.0)],
                vec![(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)],
            ],
            vec![],
        ]),
        &[11.0, 0.0]
    )]
    #[case::multi_polygon(
        multipolygon_column(vec![vec![
            vec![vec![
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.0, 0.0),
            ]],
            vec![vec![
                (3.0, 0.0),
                (6.0, 0.0),
                (6.0, 3.0),
                (3.0, 3.0),
                (3.0, 0.0),
            ]],
        ]]),
        &[13.0]
    )]
    #[case::rect(rect_column(vec![(0.0, 0.0, 5.0, 3.0)]), &[15.0])]
    fn measures_native_geometries(
        #[case] geometry: VortexResult<ArrayRef>,
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let areas = SpatialArea::try_new_array(geometry?)?.into_array();
        let expected = PrimitiveArray::from_iter(expected.iter().copied()).into_array();
        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn propagates_nulls() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let multipolygons = nullable_multipolygon_column(vec![
            Some(vec![vec![vec![
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.0, 0.0),
            ]]]),
            None,
        ])?;
        let areas = SpatialArea::try_new_array(multipolygons)?.into_array();
        let expected =
            PrimitiveArray::new(vec![4.0f64, 0.0], Validity::from_iter([true, false])).into_array();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::none(0)]
    #[case::two(2)]
    fn rejects_wrong_arity(#[case] arity: usize) -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        assert!(
            SpatialArea
                .return_dtype(&EmptyOptions, &vec![dtype; arity])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_geometry_dtype() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(
            SpatialArea
                .return_dtype(&EmptyOptions, &[primitive])
                .is_err()
        );
        Ok(())
    }
}
