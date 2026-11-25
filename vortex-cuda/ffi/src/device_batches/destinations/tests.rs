// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex::dtype::StructFields;
use vortex::error::VortexError;
use vortex_cuda_macros::test as cuda_test;

use super::*;
use crate::device_batches::tests::batches_with_rows;
use crate::device_batches::tests::session;
use crate::device_batches::tests::take_error;

#[cuda_test]
fn test_shape_and_primitive_eligibility() -> VortexResult<()> {
    let session = session()?;
    let id = DType::Primitive(PType::I32, Nullability::NonNullable);
    let fields = StructFields::from_iter([
        ("id", id.clone()),
        ("maybe", DType::Primitive(PType::U8, Nullability::Nullable)),
        ("text", DType::Utf8(Nullability::NonNullable)),
    ]);
    for (dtype, expected) in [
        (
            DType::Struct(fields.clone(), Nullability::NonNullable),
            vec![Some(vx_ptype::PTYPE_I32), None, None],
        ),
        (DType::Struct(fields, Nullability::Nullable), vec![None; 3]),
        (id, vec![None]),
    ] {
        let mut handle = batches_with_rows(&session, dtype, vec![], 0)?;
        let (mut rows, mut columns) = (-1, usize::MAX);
        // SAFETY: The unpolled handle and disjoint output storage are live on this thread.
        let status = unsafe {
            vx_cuda_device_batches_shape(
                &mut handle,
                &raw mut rows,
                &raw mut columns,
                ptr::null_mut(),
            )
        };
        assert_eq!(status, VX_CUDA_OK);
        assert_eq!((rows, columns), (0, expected.len()));
        for (column, expected) in expected.into_iter().enumerate() {
            let mut eligible = expected.is_none();
            let mut ty = vx_ptype::PTYPE_F64;
            // SAFETY: The handle remains unpolled, and both outputs are writable and disjoint.
            let status = unsafe {
                vx_cuda_device_batches_primitive(
                    &mut handle,
                    column,
                    &raw mut eligible,
                    &raw mut ty,
                    ptr::null_mut(),
                )
            };
            assert_eq!(status, VX_CUDA_OK);
            assert_eq!(eligible, expected.is_some());
            assert_eq!(ty, expected.unwrap_or(vx_ptype::PTYPE_F64));
        }
    }
    Ok(())
}

#[cuda_test]
fn test_registration_rejects_invalid_plans_atomically() -> VortexResult<()> {
    let session = session()?;
    let int = DType::Primitive(PType::I32, Nullability::NonNullable);
    let dtype = DType::Struct(
        StructFields::from_iter([
            ("a", int.clone()),
            ("b", int),
            ("text", DType::Utf8(Nullability::NonNullable)),
        ]),
        Nullability::NonNullable,
    );
    let mut handle = batches_with_rows(&session, dtype, vec![], 1)?;
    let consumer = handle.ctx.stream().clone();
    let dst = consumer
        .alloc_zeros::<i32>(2)
        .map_err(VortexError::external)?;
    let valid = vx_cuda_column_destination {
        column: 0,
        data: dst.device_address() as *mut c_void,
        capacity_bytes: size_of::<i32>(),
    };
    let mut install = |plan: &[vx_cuda_column_destination]| {
        let mut error = ptr::null_mut();
        // SAFETY: Descriptors and error storage are live; the scan is never polled, and any
        // accepted storage is owned until after the handle is dropped.
        let status = unsafe {
            vx_cuda_device_batches_set_destinations(
                &mut handle,
                plan.as_ptr(),
                plan.len(),
                false,
                consumer.cu_stream().cast(),
                &raw mut error,
            )
        };
        (status, take_error(error))
    };
    assert_eq!(install(&[valid]), (VX_CUDA_OK, None));
    let col = |column| vx_cuda_column_destination { column, ..valid };
    let data = |data| vx_cuda_column_destination { data, ..valid };
    let short = vx_cuda_column_destination {
        capacity_bytes: 3,
        ..valid
    };
    for (plan, reason) in [
        (vec![col(3)], "out of range"),
        (vec![col(2)], "not eligible"),
        (vec![short], "needs 4"),
        (vec![data(ptr::null_mut())], "null destination"),
        (vec![data(valid.data.wrapping_byte_add(1))], "unaligned"),
        (vec![valid, valid], "duplicate"),
        (vec![col(1), valid], "overlapping"),
    ] {
        let (status, message) = install(&plan);
        assert_eq!(status, VX_CUDA_ERR);
        let message = message.expect("registration must return an owned error");
        assert!(message.contains(reason), "{message}");
    }
    let slots = &handle.destinations.as_ref().expect("original plan").slots;
    assert_eq!(slots[0].expect("original destination").data, valid.data);
    assert!(slots[1..].iter().all(Option::is_none));
    drop(handle);
    Ok(())
}
