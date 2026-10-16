// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::list_project::Project;
use vortex_array::scalar_fn::fns::list_project::ProjectPaths;
use vortex_array::scalar_fn::fns::list_project::list_project;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

fn nested_list_view() -> VortexResult<(ArrayRef, Vec<Vec<FieldName>>)> {
    let nested_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let nested = StructArray::try_new(
        FieldNames::from(["x.y", "other"]),
        vec![
            buffer![20i32, 21, 22].into_array(),
            buffer![200i32, 201, 202].into_array(),
        ],
        3,
        nested_validity,
    )?;
    let element_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let elements = StructArray::try_new(
        FieldNames::from(["a.b", "nested", "unused"]),
        vec![
            buffer![10i32, 11, 12].into_array(),
            nested.into_array(),
            buffer![90i32, 91, 92].into_array(),
        ],
        3,
        element_validity,
    )?;
    let list_validity =
        Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array());
    let list = ListViewArray::try_new(
        elements.into_array(),
        buffer![2u64, 0, 1, 1].into_array(),
        buffer![1u64, 1, 1, 0].into_array(),
        list_validity,
    )?
    .into_array();
    let paths = vec![
        vec![FieldName::from("a.b")],
        vec![FieldName::from("nested"), FieldName::from("x.y")],
    ];
    Ok((list, paths))
}

#[test]
fn projection_preserves_list_and_struct_validity_and_view_offsets() -> VortexResult<()> {
    let (source, paths) = nested_list_view()?;
    let expr = list_project(&paths, Expression::Root, source.dtype())?;
    let projected = source.apply(&expr)?;

    let nested = StructArray::try_new(
        FieldNames::from(["x.y"]),
        vec![buffer![20i32, 21, 22].into_array()],
        3,
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    )?;
    let elements = StructArray::try_new(
        FieldNames::from(["a.b", "nested"]),
        vec![buffer![10i32, 11, 12].into_array(), nested.into_array()],
        3,
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    )?;
    let expected = ListViewArray::try_new(
        elements.into_array(),
        buffer![2u64, 0, 1, 1].into_array(),
        buffer![1u64, 1, 1, 0].into_array(),
        Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
    )?
    .into_array();

    let mut ctx = VortexSession::empty().create_execution_ctx();
    assert_arrays_eq!(projected, expected, &mut ctx);
    Ok(())
}

#[test]
fn empty_projection_keeps_list_rows_and_empty_lists() -> VortexResult<()> {
    let (source, _) = nested_list_view()?;
    let expr = list_project(&[], Expression::Root, source.dtype())?;
    let source_len = source.len();
    let result = source.apply(&expr)?;
    let element_dtype = result
        .dtype()
        .as_list_element_opt()
        .expect("list dtype")
        .as_struct_fields_opt()
        .expect("projected elements are structs");
    assert_eq!(element_dtype.nfields(), 0);
    assert_eq!(result.len(), source_len);
    Ok(())
}

#[test]
fn nested_lists_project_through_each_element_boundary() -> VortexResult<()> {
    let elements = StructArray::try_new(
        FieldNames::from(["x.y", "discard"]),
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![9i32, 8, 7].into_array(),
        ],
        3,
        Validity::NonNullable,
    )?;
    let inner = ListArray::try_new(
        elements.into_array(),
        buffer![0u32, 1, 3].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let outer = ListArray::try_new(inner, buffer![0u32, 2].into_array(), Validity::NonNullable)?
        .into_array();
    let paths = vec![vec![FieldName::from("x.y")]];
    let expr = list_project(&paths, Expression::Root, outer.dtype())?;
    let bound = expr.bind(outer.dtype())?;
    let source_paths = vortex_array::expr::analysis::referenced_field_paths(&bound)?
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        source_paths,
        vec![FieldPath::from(vec![
            Field::ElementType,
            Field::ElementType,
            Field::from("x.y"),
        ])]
    );
    let projected = outer.apply(&expr)?;

    let expected_elements = StructArray::try_new(
        FieldNames::from(["x.y"]),
        vec![buffer![1i32, 2, 3].into_array()],
        3,
        Validity::NonNullable,
    )?;
    let expected_inner = ListArray::try_new(
        expected_elements.into_array(),
        buffer![0u32, 1, 3].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let expected = ListArray::try_new(
        expected_inner,
        buffer![0u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = VortexSession::empty().create_execution_ctx();
    assert_arrays_eq!(projected, expected, &mut ctx);
    Ok(())
}

#[test]
fn dotted_paths_roundtrip_and_invalid_descendants_are_rejected() -> VortexResult<()> {
    let (source, paths) = nested_list_view()?;
    let project = Project;
    let options = ProjectPaths(paths.clone());
    let bytes = project.serialize(&options)?.expect("project metadata");
    let roundtrip = project.deserialize(&bytes, &VortexSession::empty())?;
    assert_eq!(roundtrip, options);

    assert!(
        project
            .deserialize(&[0xff, 0xff, 0xff, 0xff], &VortexSession::empty())
            .is_err()
    );
    let invalid = vec![
        vec![FieldName::from("a.b")],
        vec![FieldName::from("a.b"), FieldName::from("missing")],
    ];
    assert!(list_project(&invalid, Expression::Root, source.dtype()).is_err());
    Ok(())
}
