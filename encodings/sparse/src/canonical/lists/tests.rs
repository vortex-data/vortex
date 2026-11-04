// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Sparse;

#[rstest]
fn empty_fill_reuses_elements(#[values(false, true)] null_fill: bool) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let elements = buffer![1i32, 2, 8, 9, 3].into_array();
    let lists = ListArray::try_new(
        elements.clone(),
        buffer![0u32, 2, 4, 5].into_array(),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    )?;

    let fill = if null_fill {
        Scalar::null(lists.dtype().clone())
    } else {
        Scalar::zero_value(lists.dtype())
    };
    let sparse = Sparse::try_new(
        buffer![1u32, 4, 7].into_array(),
        lists.into_array(),
        10,
        fill,
        &mut ctx,
    )?;

    let actual = sparse.into_array().execute_until::<List>(&mut ctx)?;
    let list = actual.as_::<List>();
    assert!(ArrayRef::ptr_eq(list.elements(), &elements));
    assert!(list.offsets().is::<RunEnd>());

    // The middle patch is null.
    let mut validity = vec![!null_fill; 10];
    validity[1] = true;
    validity[4] = false;
    validity[7] = true;
    let expected = ListArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0u32, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3].into_array(),
        Validity::from_iter(validity),
        &mut ctx,
    )?;

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn terminal_offset_wider_than_view_offsets() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let elements = Buffer::from(vec![1i32; 260]).into_array();
    let views = ListViewArray::try_new(
        elements.clone(),
        buffer![0u8, 130].into_array(),
        buffer![130u8, 130].into_array(),
        Validity::AllValid,
        &mut ctx,
    )?;

    // SAFETY: these two views cover contiguous, non-overlapping ranges 0..130 and 130..260.
    let views = unsafe { views.with_zero_copy_to_list(true) };

    let fill = Scalar::null(views.dtype().clone());
    let actual = Sparse::try_new(
        buffer![1u32, 3].into_array(),
        views.into_array(),
        5,
        fill,
        &mut ctx,
    )?
    .into_array()
    .execute::<ListViewArray>(&mut ctx)?;

    assert!(actual.offsets().is::<Sparse>());

    let expected = ListArray::try_new(
        elements,
        buffer![0u16, 0, 130, 130, 260, 260].into_array(),
        Validity::from_iter([false, true, false, true, false]),
        &mut ctx,
    )?;

    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn single_valid_patch_with_nulls() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let elements = buffer![7i32, 8, 9].into_array();
    let lists = ListArray::try_new(
        elements.clone(),
        buffer![0u32, 0, 3, 3].into_array(),
        Validity::from_iter([false, true, false]),
        &mut ctx,
    )?;
    let fill = Scalar::null(lists.dtype().clone());
    let sparse = Sparse::try_new(
        buffer![1u32, 3, 5].into_array(),
        lists.into_array(),
        7,
        fill,
        &mut ctx,
    )?;

    let actual = sparse.into_array().execute::<ListViewArray>(&mut ctx)?;
    // A single list needs no repeated gather, and the builder remains ready for Arrow export.
    assert!(actual.is_zero_copy_to_list());
    assert!(actual.offsets().is::<vortex_array::arrays::Primitive>());
    let expected = ListArray::try_new(
        elements,
        buffer![0u32, 0, 0, 0, 3, 3, 3, 3].into_array(),
        Validity::from_iter([false, false, false, true, false, false, false]),
        &mut ctx,
    )?;
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}
