// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_MakeLine`: construct a native line string between two native points.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
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
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::LineString;
use crate::extension::Point;
use crate::extension::SpatialMetadata;
use crate::extension::coordinate::coordinate_dimension;
use crate::extension::flatten_coordinates;
use crate::extension::linestring_array_from_point_pairs;
use crate::extension::linestring_storage_dtype;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_binary;

/// Validate the two point operands accepted by `ST_MakeLine`.
fn validate_make_line_operands(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 2,
        "spatial: make_line requires exactly two point operands, got {}",
        dtypes.len()
    );
    for dtype in dtypes {
        vortex_ensure!(
            dtype
                .as_extension_opt()
                .is_some_and(|extension| extension.is::<Point>()),
            "spatial: make_line operand {dtype} is not a native point"
        );
    }
    Ok(())
}

/// Resolve DuckDB's `ST_MakeLine` CRS propagation for two geometry operands.
fn make_line_metadata(
    left: &SpatialMetadata,
    right: &SpatialMetadata,
) -> VortexResult<SpatialMetadata> {
    match (&left.crs, &right.crs) {
        (Some(left_crs), Some(right_crs)) => {
            vortex_ensure!(
                left_crs == right_crs,
                "spatial: make_line operands have different coordinate reference systems: \
                 {left_crs} and {right_crs}"
            );
            Ok(left.clone())
        }
        (Some(_), None) => Ok(left.clone()),
        (None, Some(_)) => Ok(right.clone()),
        (None, None) => Ok(SpatialMetadata::default()),
    }
}

/// The native `LineString` dtype emitted by `ST_MakeLine`.
fn make_line_dtype(dtypes: &[DType]) -> VortexResult<ExtDType<LineString>> {
    validate_make_line_operands(dtypes)?;
    let left = dtypes[0].as_extension();
    let right = dtypes[1].as_extension();
    let dimension = coordinate_dimension(left.storage_dtype())?
        .promote(coordinate_dimension(right.storage_dtype())?);
    let metadata = make_line_metadata(left.metadata::<Point>(), right.metadata::<Point>())?;
    let nullability = Nullability::from(dtypes.iter().any(DType::is_nullable));
    ExtDType::try_new(metadata, linestring_storage_dtype(dimension, nullability))
}

/// Expose a point operand as coordinate fields without expanding a constant into value buffers.
fn point_coordinates(
    operand: Operand,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    match operand {
        Operand::Column(points) => flatten_coordinates(&points, ctx),
        Operand::Constant(point) => {
            let storage = point.as_extension().to_storage_scalar();
            let fields = storage.as_struct();
            let names = fields.names().clone();
            let arrays = names
                .iter()
                .map(|name| {
                    fields
                        .field(name)
                        .map(|value| ConstantArray::new(value, len).into_array())
                        .ok_or_else(|| vortex_err!("spatial: point coordinate missing {name}"))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            StructArray::try_new(names, arrays, len, Validity::NonNullable)
        }
    }
}

/// Build a native line-string column from dispatched point operands.
fn build_make_lines(
    operands: [Operand; 2],
    len: usize,
    valid: Mask,
    output_dtype: &ExtDType<LineString>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let [start, end] = operands;
    let starts = point_coordinates(start, len, ctx)?;
    let ends = point_coordinates(end, len, ctx)?;
    linestring_array_from_point_pairs(
        output_dtype,
        &starts,
        &ends,
        Validity::from_mask(valid, output_dtype.storage_dtype().nullability()),
    )
}

/// Execute `ST_MakeLine` after shared constant/column and null dispatch.
fn execute_make_line(
    execution: Execution<2>,
    output_dtype: &ExtDType<LineString>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(start), Operand::Constant(end)] => {
            let one = build_make_lines(
                [Operand::Constant(start), Operand::Constant(end)],
                1,
                Mask::new_true(1),
                output_dtype,
                ctx,
            )?;
            Ok(ConstantArray::new(one.execute_scalar(0, ctx)?, execution.len).into_array())
        }
        operands => build_make_lines(operands, execution.len, execution.valid, output_dtype, ctx),
    }
}

/// Construct `LineString`s from paired native point operands. The output's vertices preserve all
/// coordinate ordinates (`x`, `y`, and any `z`/`m`) and appear in operand order. When the points
/// have different dimensions, it promotes to their union and fills absent `z`/`m` ordinates with
/// zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialMakeLine;

impl SpatialMakeLine {
    /// A lazy `ScalarFnArray` constructing one two-vertex line string per pair of point operands.
    pub fn try_new(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialMakeLine, EmptyOptions).erased(),
            vec![a, b],
        )
    }
}

impl ScalarFnVTable for SpatialMakeLine {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.make_line");
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
            0 => ChildName::from("start"),
            1 => ChildName::from("end"),
            _ => unreachable!("make_line has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Extension(make_line_dtype(dtypes)?.erased()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let a = args.get(0)?;
        let b = args.get(1)?;
        let output_dtype = make_line_dtype(&[a.dtype().clone(), b.dtype().clone()])?;
        dispatch_binary(
            &a,
            &b,
            DType::Extension(output_dtype.clone().erased()),
            |execution, ctx| execute_make_line(execution, &output_dtype, ctx),
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
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::LineString as GeoLineString;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Columnar;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::SpatialMakeLine;
    use crate::extension::LineString;
    use crate::extension::Point;
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_dimension;
    use crate::extension::coordinate::ordinates;
    use crate::extension::flatten_coordinates;
    use crate::extension::geometries;
    use crate::test_harness::point_column;

    fn dimensional_point(
        dimension: Dimension,
        coordinate: [f64; 4],
        crs: Option<&str>,
    ) -> VortexResult<ArrayRef> {
        let mut fields = vec![
            ("x", PrimitiveArray::from_iter([coordinate[0]]).into_array()),
            ("y", PrimitiveArray::from_iter([coordinate[1]]).into_array()),
        ];
        if matches!(dimension, Dimension::Xyz | Dimension::Xyzm) {
            fields.push(("z", PrimitiveArray::from_iter([coordinate[2]]).into_array()));
        }
        if matches!(dimension, Dimension::Xym | Dimension::Xyzm) {
            fields.push(("m", PrimitiveArray::from_iter([coordinate[3]]).into_array()));
        }
        let storage = StructArray::from_fields(&fields)?.into_array();
        let dtype = ExtDType::<Point>::try_new(
            SpatialMetadata {
                crs: crs.map(str::to_owned),
            },
            storage.dtype().clone(),
        )?;
        Ok(ExtensionArray::try_new(dtype.erased(), storage)?.into_array())
    }

    fn point_constant(
        x: f64,
        y: f64,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let scalar = point_column(vec![x], vec![y])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    #[test]
    fn connects_paired_points_in_operand_order() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let starts = point_column(vec![0.0, 3.0], vec![0.0, 4.0])?;
        let ends = point_column(vec![3.0, 0.0], vec![4.0, 0.0])?;

        let lines = SpatialMakeLine::try_new(starts, ends)?.into_array();
        assert!(lines.dtype().as_extension().is::<LineString>());
        assert_eq!(
            geometries(&lines, &mut ctx)?,
            vec![
                Geometry::LineString(GeoLineString::new(vec![
                    Coord { x: 0.0, y: 0.0 },
                    Coord { x: 3.0, y: 4.0 },
                ])),
                Geometry::LineString(GeoLineString::new(vec![
                    Coord { x: 3.0, y: 4.0 },
                    Coord { x: 0.0, y: 0.0 },
                ])),
            ]
        );
        Ok(())
    }

    #[test]
    fn two_constants_are_built_once_and_remain_constant() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let starts = point_constant(0.0, 0.0, 3, &mut ctx)?;
        let ends = point_constant(3.0, 4.0, 3, &mut ctx)?;

        let result = SpatialMakeLine::try_new(starts, ends)?
            .into_array()
            .execute::<Columnar>(&mut ctx)?;
        let Columnar::Constant(lines) = result else {
            return Err(vortex_err!(
                "make_line of two constants should remain constant"
            ));
        };
        assert_eq!(lines.len(), 3);
        assert_eq!(
            geometries(&lines.into_array(), &mut ctx)?,
            vec![
                Geometry::LineString(GeoLineString::new(vec![
                    Coord { x: 0.0, y: 0.0 },
                    Coord { x: 3.0, y: 4.0 },
                ]));
                3
            ]
        );
        Ok(())
    }

    #[rstest]
    #[case::constant_start(true)]
    #[case::constant_end(false)]
    fn constant_and_column_are_paired_by_row(#[case] constant_start: bool) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let constant = point_constant(0.0, 0.0, 2, &mut ctx)?;
        let column = point_column(vec![3.0, 6.0], vec![4.0, 8.0])?;
        let (starts, ends) = if constant_start {
            (constant, column)
        } else {
            (column, constant)
        };

        let lines = SpatialMakeLine::try_new(starts, ends)?.into_array();
        let endpoints = [(3.0, 4.0), (6.0, 8.0)];
        let expected = endpoints
            .into_iter()
            .map(|(x, y)| {
                let constant = Coord { x: 0.0, y: 0.0 };
                let column = Coord { x, y };
                Geometry::LineString(GeoLineString::new(if constant_start {
                    vec![constant, column]
                } else {
                    vec![column, constant]
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(geometries(&lines, &mut ctx)?, expected);
        Ok(())
    }

    #[test]
    fn null_constant_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let starts = ConstantArray::new(Scalar::null(point_dtype), 2).into_array();
        let ends = point_column(vec![3.0, 6.0], vec![4.0, 8.0])?;

        let result = SpatialMakeLine::try_new(starts, ends)?
            .into_array()
            .execute::<Columnar>(&mut ctx)?;
        let Columnar::Constant(lines) = result else {
            return Err(vortex_err!(
                "make_line with a null constant should remain constant"
            ));
        };
        assert_eq!(lines.len(), 2);
        assert!(lines.scalar().is_null());
        assert!(lines.dtype().as_extension().is::<LineString>());
        Ok(())
    }

    #[rstest]
    #[case::xy_xyz(
        Dimension::Xy,
        [1.0, 2.0, 0.0, 0.0],
        Dimension::Xyz,
        [3.0, 4.0, 5.0, 0.0],
        Dimension::Xyz,
        Some([0.0, 5.0]),
        None
    )]
    #[case::xyz_xy(
        Dimension::Xyz,
        [1.0, 2.0, 5.0, 0.0],
        Dimension::Xy,
        [3.0, 4.0, 0.0, 0.0],
        Dimension::Xyz,
        Some([5.0, 0.0]),
        None
    )]
    #[case::xym_xyz(
        Dimension::Xym,
        [1.0, 2.0, 0.0, 6.0],
        Dimension::Xyz,
        [3.0, 4.0, 5.0, 0.0],
        Dimension::Xyzm,
        Some([0.0, 5.0]),
        Some([6.0, 0.0])
    )]
    #[case::xyz_xym(
        Dimension::Xyz,
        [1.0, 2.0, 5.0, 0.0],
        Dimension::Xym,
        [3.0, 4.0, 0.0, 6.0],
        Dimension::Xyzm,
        Some([5.0, 0.0]),
        Some([0.0, 6.0])
    )]
    fn promotes_mixed_point_dimensions(
        #[case] start_dimension: Dimension,
        #[case] start: [f64; 4],
        #[case] end_dimension: Dimension,
        #[case] end: [f64; 4],
        #[case] expected_dimension: Dimension,
        #[case] expected_z: Option<[f64; 2]>,
        #[case] expected_m: Option<[f64; 2]>,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let expected_x = [start[0], end[0]];
        let expected_y = [start[1], end[1]];
        let start = dimensional_point(start_dimension, start, None)?;
        let end = dimensional_point(end_dimension, end, None)?;

        let lines = SpatialMakeLine::try_new(start, end)?.into_array();
        let vertices = flatten_coordinates(&lines, &mut ctx)?;
        assert_eq!(coordinate_dimension(vertices.dtype())?, expected_dimension);
        for (name, expected) in [("x", expected_x), ("y", expected_y)] {
            assert_eq!(ordinates(&vertices, name, &mut ctx)?.as_slice(), expected);
        }
        for (name, expected) in [("z", expected_z), ("m", expected_m)] {
            if let Some(expected) = expected {
                assert_eq!(ordinates(&vertices, name, &mut ctx)?.as_slice(), expected);
            }
        }
        Ok(())
    }

    #[rstest]
    #[case::matching(Some("EPSG:4326"), Some("EPSG:4326"), Some("EPSG:4326"))]
    #[case::left_only(Some("EPSG:4326"), None, Some("EPSG:4326"))]
    #[case::right_only(None, Some("EPSG:3857"), Some("EPSG:3857"))]
    #[case::unreferenced(None, None, None)]
    fn propagates_compatible_crs(
        #[case] left_crs: Option<&str>,
        #[case] right_crs: Option<&str>,
        #[case] expected: Option<&str>,
    ) -> VortexResult<()> {
        let left = dimensional_point(Dimension::Xy, [0.0; 4], left_crs)?;
        let right = dimensional_point(Dimension::Xy, [1.0; 4], right_crs)?;

        let dtype = SpatialMakeLine.return_dtype(
            &EmptyOptions,
            &[left.dtype().clone(), right.dtype().clone()],
        )?;
        assert_eq!(
            dtype.as_extension().metadata::<LineString>().crs.as_deref(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rejects_mismatched_crs() -> VortexResult<()> {
        let left = dimensional_point(Dimension::Xy, [0.0; 4], Some("EPSG:4326"))?;
        let right = dimensional_point(Dimension::Xy, [1.0; 4], Some("EPSG:3857"))?;
        assert!(
            SpatialMakeLine
                .return_dtype(
                    &EmptyOptions,
                    &[left.dtype().clone(), right.dtype().clone()]
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_point_dtype() -> VortexResult<()> {
        let points = point_column(vec![0.0], vec![0.0])?;
        assert!(
            SpatialMakeLine
                .return_dtype(&EmptyOptions, std::slice::from_ref(points.dtype()))
                .is_err()
        );
        let non_point = DType::Bool(vortex_array::dtype::Nullability::NonNullable);
        assert!(
            SpatialMakeLine
                .return_dtype(&EmptyOptions, &[points.dtype().clone(), non_point])
                .is_err()
        );
        Ok(())
    }
}
