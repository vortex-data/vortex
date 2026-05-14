// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_int;
use arrow_array::{Array as ArrowArrayTrait, StructArray};
use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::RecordBatch;
use crate::array::vx_array;

/// Exports a Vortex Array into pre-allocated Arrow FFI C Data structures.
///
/// # Safety
/// The caller must ensure that `out_array` and `out_schema` point to valid,
/// uninitialized or safely overwritable memory locations allocated by the host.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_array_export(
    array: *const vx_array,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> c_int {

    // Reference the incoming Vortex array
    let array = vx_array::as_ref(array);
    //let sarray = StructArray::from(array.as_ref());
    let record_batch = RecordBatch::try_from(array.as_ref());
    let struct_array = StructArray::from(record_batch.unwrap().clone());

    // Export schema + array
    let schema = FFI_ArrowSchema::try_from(struct_array.data_type()).unwrap();
    let array = FFI_ArrowArray::new(&struct_array.to_data());
    unsafe {
        std::ptr::write(out_array, array);
        std::ptr::write(out_schema, schema);
    }

    0
}