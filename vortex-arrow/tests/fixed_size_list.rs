// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Import of Arrow fixed-size lists whose null lists hold null items.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
use arrow_array::Float32Array;
use arrow_array::Int32Array;
use arrow_array::new_null_array;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::NullBuffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use rstest::rstest;
use vortex_array::ArrayRef;
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

/// The two public entry points for importing an Arrow array.
#[derive(Clone, Copy)]
enum Importer {
    Dyn,
    Session,
}

impl Importer {
    fn import(self, array: ArrowArrayRef) -> VortexResult<ArrayRef> {
        match self {
            Importer::Dyn => from_arrow_dyn(array.as_ref(), true),
            Importer::Session => {
                let field = Field::new("embedding", array.data_type().clone(), true);
                SESSION.arrow().from_arrow_array(array, &field)
            }
        }
    }

    /// A part of the error for a null item in a valid list, in an array that also holds a null
    /// list. The session checks each item against the lists above it, so it names the null.
    fn visible_null_error(self) -> &'static str {
        match self {
            Importer::Dyn => "non-nullable Vortex array",
            Importer::Session => "must not be null unless an ancestor is null, got a null at index",
        }
    }
}

#[rstest]
fn null_items_in_null_lists(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let items = Float32Array::from(vec![
        Some(1.0),
        Some(2.0),
        Some(3.0),
        None,
        None,
        None,
        Some(7.0),
        Some(8.0),
        Some(9.0),
    ]);
    let lists = NullBuffer::new(BooleanBuffer::from(vec![true, false, true]));
    let arrow = ArrowFixedSizeListArray::try_new(item, 3, Arc::new(items), Some(lists))?;

    let actual = importer.import(Arc::new(arrow))?;

    assert_arrays_eq!(
        actual,
        FixedSizeListArray::new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0].into_array(),
            3,
            Validity::from_iter([true, false, true]),
            3
        ),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
fn all_null_lists(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let item = Arc::new(Field::new("item", DataType::Int32, false));
    let arrow = new_null_array(&DataType::FixedSizeList(item, 2), 2);

    let actual = importer.import(arrow)?;

    assert_arrays_eq!(
        actual,
        FixedSizeListArray::new(buffer![0i32; 4].into_array(), 2, Validity::AllInvalid, 2),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
fn sliced_lists(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let item = Arc::new(Field::new("item", DataType::Int32, false));
    let items = Int32Array::from(vec![Some(1), Some(2), None, None, Some(5), Some(6)]);
    let lists = NullBuffer::new(BooleanBuffer::from(vec![true, false, true]));
    let arrow = ArrowFixedSizeListArray::try_new(item, 2, Arc::new(items), Some(lists))?;

    let actual = importer.import(Arc::new(arrow.slice(1, 2)))?;

    assert_arrays_eq!(
        actual,
        FixedSizeListArray::new(
            buffer![0i32, 0, 5, 6].into_array(),
            2,
            Validity::from_iter([false, true]),
            2
        ),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
fn nested_all_null_lists(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let leaf = Arc::new(Field::new("item", DataType::Int32, false));
    let inner = Arc::new(Field::new("item", DataType::FixedSizeList(leaf, 2), false));
    let arrow = new_null_array(&DataType::FixedSizeList(inner, 2), 2);

    let actual = importer.import(arrow)?;

    assert_arrays_eq!(
        actual,
        FixedSizeListArray::new(
            FixedSizeListArray::new(buffer![0i32; 8].into_array(), 2, Validity::NonNullable, 4)
                .into_array(),
            2,
            Validity::AllInvalid,
            2
        ),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

/// Arrays imported over FFI are not validated, so a null item in a valid list can reach the
/// importer. It must stay an error rather than become a made-up value.
#[rstest]
fn null_item_in_valid_list_is_rejected(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let items = Float32Array::from(vec![Some(1.0), None, None, None]);
    let lists = NullBuffer::new(BooleanBuffer::from(vec![true, false]));
    // SAFETY: the buffers and child lengths are consistent, only the null masking rule is broken.
    let data = unsafe {
        ArrayDataBuilder::new(DataType::FixedSizeList(item, 2))
            .len(2)
            .nulls(Some(lists))
            .add_child_data(items.to_data())
            .build_unchecked()
    };

    let err = importer
        .import(Arc::new(ArrowFixedSizeListArray::from(data)))
        .unwrap_err();

    assert!(
        err.to_string().contains(importer.visible_null_error()),
        "unexpected error: {err}"
    );
    Ok(())
}

#[rstest]
fn null_leaf_in_valid_nested_list_is_rejected(
    #[values(Importer::Dyn, Importer::Session)] importer: Importer,
) -> VortexResult<()> {
    let leaf = Arc::new(Field::new("item", DataType::Int32, false));
    let inner = Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(Arc::clone(&leaf), 2),
        false,
    ));
    let leaves = Int32Array::from(vec![
        Some(1),
        None,
        Some(3),
        Some(4),
        None,
        None,
        None,
        None,
    ]);
    // SAFETY: the buffers and child lengths are consistent, only the null masking rule is broken.
    let inner_lists = unsafe {
        ArrayDataBuilder::new(DataType::FixedSizeList(leaf, 2))
            .len(4)
            .nulls(Some(NullBuffer::new(BooleanBuffer::from(vec![
                true, true, false, false,
            ]))))
            .add_child_data(leaves.to_data())
            .build_unchecked()
    };
    // SAFETY: as above, the null leaf at index 1 sits in a valid inner list of a valid outer list.
    let data = unsafe {
        ArrayDataBuilder::new(DataType::FixedSizeList(inner, 2))
            .len(2)
            .nulls(Some(NullBuffer::new(BooleanBuffer::from(vec![
                true, false,
            ]))))
            .add_child_data(inner_lists)
            .build_unchecked()
    };

    let err = importer
        .import(Arc::new(ArrowFixedSizeListArray::from(data)))
        .unwrap_err();

    assert!(
        err.to_string().contains(importer.visible_null_error()),
        "unexpected error: {err}"
    );
    Ok(())
}

#[rstest]
fn nullable_items_under_a_non_nullable_field_are_rejected(
    #[values(Importer::Session)] _importer: Importer,
) -> VortexResult<()> {
    let array_item = Arc::new(Field::new("item", DataType::Float32, true));
    let items = Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
    let arrow = ArrowFixedSizeListArray::try_new(array_item, 2, Arc::new(items), None)?;
    let caller_item = Arc::new(Field::new("item", DataType::Float32, false));
    let field = Field::new("embedding", DataType::FixedSizeList(caller_item, 2), true);

    let err = SESSION
        .arrow()
        .from_arrow_array(Arc::new(arrow), &field)
        .unwrap_err();

    assert!(
        err.to_string().contains("non-nullable Vortex array"),
        "unexpected error: {err}"
    );
    Ok(())
}
