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
fn empty_fill_reuses_elements(
    #[values(false, true)] null_fill: bool,
    #[values(false, true)] repeated_position: bool,
) -> VortexResult<()> {
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
        if repeated_position {
            buffer![1u32, 1, 7].into_array()
        } else {
            buffer![1u32, 4, 7].into_array()
        },
        lists.into_array(),
        10,
        fill,
        &mut ctx,
    )?;

    let actual = if repeated_position {
        sparse
            .into_array()
            .execute::<ListViewArray>(&mut ctx)?
            .into_array()
    } else {
        sparse.into_array().execute_until::<List>(&mut ctx)?
    };

    if let Some(list) = actual.as_opt::<List>() {
        assert!(ArrayRef::ptr_eq(list.elements(), &elements));
        assert!(list.offsets().is::<RunEnd>());
    } else {
        let view = actual.as_::<vortex_array::arrays::ListView>();
        assert!(ArrayRef::ptr_eq(view.elements(), &elements));
        assert!(view.offsets().is::<Sparse>());
        assert!(view.sizes().is::<Sparse>());
    }

    // The middle patch is null. At a repeated position it replaces the first, valid patch.
    let mut validity = vec![!null_fill; 10];
    validity[1] = !repeated_position;
    if !repeated_position {
        validity[4] = false;
    }
    validity[7] = true;
    let expected = ListArray::try_new(
        if repeated_position {
            buffer![3i32].into_array()
        } else {
            buffer![1i32, 2, 3].into_array()
        },
        if repeated_position {
            buffer![0u32, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1].into_array()
        } else {
            buffer![0u32, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3].into_array()
        },
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
