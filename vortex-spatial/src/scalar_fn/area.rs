// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Area`: unsigned planar area of native geometries.

use geo::Area;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::scalar_fn::row::GeometryRow;

/// Unsigned planar `ST_Area` of native geometries.
///
/// Points and line strings have zero area, polygons and multipolygons use their two-dimensional
/// coordinates, and rectangles use width times height. Higher coordinate dimensions are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialArea;

impl SpatialArea {
    /// A lazy `ScalarFnArray` computing the per-row area of a native geometry operand.
    pub fn try_new(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialArea, EmptyOptions).erased(),
            vec![array],
        )
    }
}

impl RowFn for SpatialArea {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["geometry"];
    const INFALLIBLE: bool = true;

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

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(GeometryRow,), f64>(|(geometry,)| geometry.unsigned_area())
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
        let areas = SpatialArea::try_new(geometry?)?.into_array();
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
        let areas = SpatialArea::try_new(multipolygons)?.into_array();
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
