// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test how Arrow fixed-size lists import null elements.
//!
//! Both import paths use the element field's nullability. The session can also take that field
//! from a caller-supplied schema.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_array::Int32Array;
    use arrow_array::new_null_array;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_arrow::convert::from_arrow_dyn;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[derive(Clone, Copy)]
    enum ImportPath {
        Canonical,
        Session,
    }

    #[track_caller]
    fn assert_import(
        path: ImportPath,
        arrow: ArrowArrayRef,
        expected: FixedSizeListArray,
    ) -> VortexResult<()> {
        let imported = match path {
            ImportPath::Canonical => from_arrow_dyn(arrow.as_ref(), true)?,
            ImportPath::Session => {
                let field = Field::new("embedding", arrow.data_type().clone(), true);
                SESSION.arrow().from_arrow_array(arrow, &field)?
            }
        };

        assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_import_null_elements_in_null_lists(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let items = Float32Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0), // The first list is valid.
            None,
            None,
            None, // The middle list is null.
            Some(7.0),
            Some(8.0),
            Some(9.0), // The last list is valid.
        ]);
        let lists = NullBuffer::from(vec![true, false, true]);
        let arrow = ArrowFixedSizeListArray::try_new(item, 3, Arc::new(items), Some(lists))?;

        let expected = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0].into_array(),
            3,
            Validity::from_iter([true, false, true]),
            3,
        )?;
        assert_import(path, Arc::new(arrow), expected)
    }

    #[rstest]
    fn test_import_all_null_lists(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let item = Arc::new(Field::new("item", DataType::Int32, false));
        let arrow = new_null_array(&DataType::FixedSizeList(item, 2), 2);

        let expected =
            FixedSizeListArray::try_new(buffer![0i32; 4].into_array(), 2, Validity::AllInvalid, 2)?;
        assert_import(path, arrow, expected)
    }

    #[rstest]
    fn test_import_sliced_lists(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let item = Arc::new(Field::new("item", DataType::Int32, false));
        let items = Int32Array::from(vec![Some(1), Some(2), None, None, Some(5), Some(6)]);
        let lists = NullBuffer::from(vec![true, false, true]);
        let arrow = ArrowFixedSizeListArray::try_new(item, 2, Arc::new(items), Some(lists))?;

        let expected = FixedSizeListArray::try_new(
            buffer![0i32, 0, 5, 6].into_array(),
            2,
            Validity::from_iter([false, true]),
            2,
        )?;
        assert_import(path, Arc::new(arrow.slice(1, 2)), expected)
    }

    #[rstest]
    fn test_import_nested_all_null_lists(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let leaf = Arc::new(Field::new("item", DataType::Int32, false));
        let inner = Arc::new(Field::new("item", DataType::FixedSizeList(leaf, 2), false));
        let arrow = new_null_array(&DataType::FixedSizeList(inner, 2), 2);

        let expected_inner = FixedSizeListArray::try_new(
            buffer![0i32; 8].into_array(),
            2,
            Validity::NonNullable,
            4,
        )?;
        let expected =
            FixedSizeListArray::try_new(expected_inner.into_array(), 2, Validity::AllInvalid, 2)?;
        assert_import(path, arrow, expected)
    }

    #[rstest]
    fn test_import_null_element_in_valid_list(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let items = Arc::new(Float32Array::from(vec![
            Some(1.0),
            None, // The valid list contains a null.
            None,
            None, // The second list is null.
        ]));
        let lists = NullBuffer::from(vec![true, false]);

        // The valid list has a null child, which Arrow's checked constructor rejects.
        //
        // SAFETY: The four values match the Float32 field, each list has two elements, and the
        // bitmap has two bits.
        let arrow =
            unsafe { ArrowFixedSizeListArray::new_unchecked(item, 2, items, Some(lists), 2) };

        let expected = FixedSizeListArray::try_new(
            buffer![1.0f32, 0.0, 0.0, 0.0].into_array(),
            2,
            Validity::from_iter([true, false]),
            2,
        )?;
        assert_import(path, Arc::new(arrow), expected)
    }

    #[rstest]
    fn test_import_null_leaf_in_valid_nested_list(
        #[values(ImportPath::Canonical, ImportPath::Session)] path: ImportPath,
    ) -> VortexResult<()> {
        let leaf = Arc::new(Field::new("item", DataType::Int32, false));
        let inner_field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(Arc::clone(&leaf), 2),
            false,
        ));
        let leaves = Arc::new(Int32Array::from(vec![
            Some(1),
            None, // The first inner list is valid and contains a null.
            Some(3),
            Some(4), // The second inner list is valid.
            None,
            None, // The third inner list is null.
            None,
            None, // The fourth inner list is null.
        ]));
        let inner_validity = NullBuffer::from(vec![true, true, false, false]);

        // The first inner list has a null child, which Arrow's checked constructor rejects.
        //
        // SAFETY: The eight values match the Int32 field, each list has two elements, and the
        // bitmap has four bits.
        let inner = unsafe {
            ArrowFixedSizeListArray::new_unchecked(leaf, 2, leaves, Some(inner_validity), 4)
        };
        let outer_validity = NullBuffer::from(vec![true, false]);

        let arrow = ArrowFixedSizeListArray::try_new(
            inner_field,
            2,
            Arc::new(inner),
            Some(outer_validity),
        )?;

        let expected_inner = FixedSizeListArray::try_new(
            buffer![1i32, 0, 3, 4, 0, 0, 0, 0].into_array(),
            2,
            Validity::NonNullable,
            4,
        )?;
        let expected = FixedSizeListArray::try_new(
            expected_inner.into_array(),
            2,
            Validity::from_iter([true, false]),
            2,
        )?;
        assert_import(path, Arc::new(arrow), expected)
    }

    #[test]
    fn test_import_uses_declared_element_nullability() -> VortexResult<()> {
        let array_item = Arc::new(Field::new("item", DataType::Float32, true));
        let items = Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let arrow = ArrowFixedSizeListArray::try_new(array_item, 2, Arc::new(items), None)?;
        let caller_item = Arc::new(Field::new("item", DataType::Float32, false));
        let field = Field::new("embedding", DataType::FixedSizeList(caller_item, 2), true);

        let actual = SESSION.arrow().from_arrow_array(Arc::new(arrow), &field)?;
        let expected = FixedSizeListArray::try_new(
            buffer![1.0f32, 0.0, 3.0, 4.0].into_array(),
            2,
            Validity::AllValid,
            2,
        )?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
