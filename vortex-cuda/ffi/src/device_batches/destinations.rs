// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pre-poll inspection and registration of caller-owned CUDA column storage.

use std::ffi::c_int;
use std::ffi::c_void;
use std::mem::size_of;
use std::slice;

use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_ffi::try_or;
use vortex_ffi::vx_error;
use vortex_ffi::vx_ptype;

use super::vx_cuda_device_batches;
use crate::VX_CUDA_ERR;
use crate::VX_CUDA_OK;

/// Caller-owned storage for one non-null primitive column of the projected scan.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_cuda_column_destination {
    /// Zero-based index in the projected schema, not the file schema.
    pub column: usize,
    /// Naturally aligned CUDA device address; may be null for zero capacity and zero rows.
    pub data: *mut c_void,
    /// Writable bytes starting at `data`.
    pub capacity_bytes: usize,
}

pub(super) struct Destinations {
    pub(super) slots: Vec<Option<vx_cuda_column_destination>>,
    pub(super) preserve_dictionaries: bool,
    pub(super) consumer_stream: *mut c_void,
    pub(super) rows_written: usize,
}

fn column_count(dtype: &DType) -> usize {
    match dtype {
        DType::Struct(fields, _) => fields.nfields(),
        _ => 1,
    }
}

fn primitive(dtype: &DType, column: usize) -> VortexResult<Option<PType>> {
    vortex_ensure!(column < column_count(dtype), "column {column} out of range");
    let DType::Struct(fields, Nullability::NonNullable) = dtype else {
        return Ok(None);
    };
    let field = fields
        .field_by_index(column)
        .ok_or_else(|| vortex_err!("missing column {column}"))?;
    Ok(match field {
        DType::Primitive(ptype, Nullability::NonNullable) => Some(ptype),
        _ => None,
    })
}

fn check_ptr<T>(ptr: *const T, name: &str) -> VortexResult<()> {
    vortex_ensure!(!ptr.is_null(), "null {name}");
    vortex_ensure!(ptr.is_aligned(), "unaligned {name}");
    Ok(())
}

/// Non-null, aligned `handle` must be live and exclusively borrowed for `'a` on its creating thread.
unsafe fn before_poll<'a>(
    handle: *mut vx_cuda_device_batches,
) -> VortexResult<&'a mut vx_cuda_device_batches> {
    check_ptr(handle, "vx_cuda_device_batches")?;
    // SAFETY: Pointer checks passed and the caller guarantees the handle's lifetime and exclusivity.
    let handle = unsafe { &mut *handle };
    vortex_ensure!(!handle.started, "device batch scan has already started");
    Ok(handle)
}

/// Return the exact scan row count and projected column count before the first poll.
/// A non-struct scan has one column, which cannot be registered as a destination.
/// Returns 0 on success or 1 on error (outputs unchanged).
/// As with `next`, a non-null `error_out` receives null on success or an owned error on failure.
///
/// # Safety
///
/// Use the live handle exclusively on its creating thread. Outputs must be disjoint from each
/// other and the handle. `rows` and `columns` must be writable when non-null and aligned.
/// Non-null `error_out` must be aligned and writable for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_shape(
    handle: *mut vx_cuda_device_batches,
    rows: *mut i64,
    columns: *mut usize,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        // SAFETY: The caller guarantees exclusive access on the creating thread.
        let handle = unsafe { before_poll(handle) }?;
        check_ptr(rows, "row count output")?;
        check_ptr(columns, "column count output")?;
        let row_count = i64::try_from(handle.row_count)?;
        let ncolumns = column_count(&handle.dtype);
        // SAFETY: Both outputs are checked and the caller provides disjoint writable storage.
        unsafe {
            rows.write(row_count);
            columns.write(ncolumns);
        }
        Ok(VX_CUDA_OK)
    })
}

/// Only non-nullable primitive fields of a top-level non-nullable struct are eligible, regardless
/// of encoding. Writes `ptype` only if eligible. Out-of-range indices are errors. Timing, status,
/// and error ownership are the same as for `vx_cuda_device_batches_shape`.
///
/// # Safety
///
/// The handle and output pointers have the same requirements as `vx_cuda_device_batches_shape`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_primitive(
    handle: *mut vx_cuda_device_batches,
    column: usize,
    eligible: *mut bool,
    ptype: *mut vx_ptype,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        // SAFETY: The caller guarantees exclusive access on the creating thread.
        let handle = unsafe { before_poll(handle) }?;
        check_ptr(eligible, "eligibility output")?;
        check_ptr(ptype, "primitive type output")?;
        let ty = primitive(&handle.dtype, column)?;
        // SAFETY: The checked outputs are disjoint and writable per the caller.
        unsafe {
            if let Some(ty) = ty {
                ptype.write(ty.into());
            }
            eligible.write(ty.is_some());
        }
        Ok(VX_CUDA_OK)
    })
}

/// Copy and install a destination plan before the first poll. Even an empty plan requests
/// cuDF-importable batches. When `preserve_dictionaries` is true, supported top-level dictionaries
/// are preserved; otherwise all dictionaries are decoded. Registered fields are
/// returned as batch-length Arrow Null fields; other fields remain in the Arrow output. Writes
/// append in poll order, after prior work on `consumer_stream`, and are covered by each batch's
/// sync event. A typed empty scan produces a live empty batch before EOS.
///
/// Returns 0 on success, 1 on error; errors leave the plan unchanged. Error ownership is as for
/// `vx_cuda_device_batches_shape`.
///
/// # Safety
///
/// Use a live handle exclusively on its creating thread. For nonzero `count`, `destinations` must
/// reference that many readable, initialized descriptors. Non-null `error_out` must be aligned,
/// writable, and disjoint from the inputs. The stream must belong to the scan's CUDA context
/// (null denotes the legacy default stream). Each destination must be exclusively writable device
/// storage on the scan device, disjoint from scan inputs and other destinations. Keep the stream
/// and buffers live through the scan. Order allocation, subsequent access, and reclamation on
/// the consumer stream (or establish equivalent dependencies); do not capture either stream.
/// If CUDA fails both to fence a submitted write and to synchronize its stream, `next` aborts:
/// returning control to the owner would allow it to free memory still in use.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_set_destinations(
    handle: *mut vx_cuda_device_batches,
    destinations: *const vx_cuda_column_destination,
    count: usize,
    preserve_dictionaries: bool,
    consumer_stream: *mut c_void,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        // SAFETY: The caller guarantees exclusive access on the creating thread.
        let handle = unsafe { before_poll(handle) }?;
        let ncolumns = column_count(&handle.dtype);
        vortex_ensure!(count <= ncolumns, "more destinations than columns");
        let descriptors = if count == 0 {
            &[]
        } else {
            check_ptr(destinations, "column destinations")?;
            vortex_ensure!(
                count <= isize::MAX as usize / size_of::<vx_cuda_column_destination>(),
                "destination descriptor array too large"
            );
            // SAFETY: The slice size and pointer were checked; the caller supplies readable descriptors.
            unsafe { slice::from_raw_parts(destinations, count) }
        };
        let mut slots = vec![None; ncolumns];
        let mut ranges = Vec::with_capacity(count);
        for dest in descriptors {
            let column = dest.column;
            let ty = primitive(&handle.dtype, column)?.ok_or_else(|| {
                vortex_err!("column {column} is not eligible for destination registration")
            })?;
            vortex_ensure!(
                slots[column].is_none(),
                "duplicate destination for column {column}"
            );
            let bytes = usize::try_from(handle.row_count)?
                .checked_mul(ty.byte_width())
                .ok_or_else(|| vortex_err!("column {column} byte count overflow"))?;
            vortex_ensure!(
                dest.capacity_bytes >= bytes,
                "column {column} needs {bytes} destination bytes"
            );
            if dest.capacity_bytes != 0 {
                vortex_ensure!(!dest.data.is_null(), "null destination for column {column}");
                let start = dest.data.addr();
                vortex_ensure!(
                    start.is_multiple_of(ty.byte_width()),
                    "unaligned destination for column {column}"
                );
                let end = start
                    .checked_add(dest.capacity_bytes)
                    .ok_or_else(|| vortex_err!("column {column} destination range overflows"))?;
                ranges.push((start, end));
            }
            slots[column] = Some(*dest);
        }
        ranges.sort_unstable();
        vortex_ensure!(
            ranges.windows(2).all(|w| w[0].1 <= w[1].0),
            "overlapping destination buffers"
        );
        handle.destinations = Some(Destinations {
            slots,
            preserve_dictionaries,
            consumer_stream,
            rows_written: 0,
        });
        Ok(VX_CUDA_OK)
    })
}

#[cfg(test)]
mod tests;
