// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building and reading Arrow C Data Interface structs in the guest's own linear memory.
//!
//! Decoded arrays cross the boundary as Arrow C structs. A kernel returns its result as a
//! [`Decoded`] and the SDK lays out the structs ([`write_primitive`]); child inputs arrive as
//! structs the SDK reads back ([`read_child`]). The layouts are plain bytes (see [`crate::abi`]),
//! so no Arrow library is needed.

use crate::abi::ARRAY_SIZE;
use crate::abi::ARROW_FLAG_NULLABLE;
use crate::abi::PType;
use crate::abi::SCHEMA_SIZE;
use crate::abi::array;
use crate::abi::schema;
use crate::error::GuestError;
use crate::error::GuestResult;
use crate::host::alloc_bytes;

/// A decoded primitive array a kernel returns. The values buffer must hold an entry at every
/// position; null positions may contain any bytes. `validity` is an LSB-first bitmap
/// (`ceil(len / 8)` bytes, 1 = valid); `None` means non-nullable.
pub struct Decoded {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values, `len * ptype.byte_width()` bytes.
    pub values: Vec<u8>,
    /// Optional validity bitmap.
    pub validity: Option<Vec<u8>>,
}

/// Write a primitive [`Decoded`] as Arrow C Data Interface structs in linear memory.
///
/// Returns a pointer to an 8-byte pair `[array_ptr: u32, schema_ptr: u32]` — the value a kernel's
/// `vx_decode` returns to the host.
pub fn write_primitive(decoded: &Decoded) -> i32 {
    let mut format = Vec::with_capacity(2);
    format.extend_from_slice(decoded.ptype.format_code().as_bytes());
    format.push(0);
    let format_ptr = alloc_bytes(&format);
    let values_ptr = alloc_bytes(&decoded.values);
    let validity_ptr = decoded
        .validity
        .as_ref()
        .map(|v| alloc_bytes(v))
        .unwrap_or(0);

    let mut buffers = [0u8; 8];
    buffers[0..4].copy_from_slice(&validity_ptr.to_le_bytes());
    buffers[4..8].copy_from_slice(&values_ptr.to_le_bytes());
    let buffers_ptr = alloc_bytes(&buffers);

    let mut schema_buf = [0u8; SCHEMA_SIZE];
    schema_buf[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
    let flags: i64 = if decoded.validity.is_some() {
        ARROW_FLAG_NULLABLE
    } else {
        0
    };
    schema_buf[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
    let schema_ptr = alloc_bytes(&schema_buf);

    let mut array_buf = [0u8; ARRAY_SIZE];
    array_buf[array::LENGTH..array::LENGTH + 8]
        .copy_from_slice(&(decoded.len as i64).to_le_bytes());
    let null_count: i64 = if decoded.validity.is_some() { -1 } else { 0 };
    array_buf[array::NULL_COUNT..array::NULL_COUNT + 8].copy_from_slice(&null_count.to_le_bytes());
    array_buf[array::N_BUFFERS..array::N_BUFFERS + 8].copy_from_slice(&2i64.to_le_bytes());
    array_buf[array::BUFFERS..array::BUFFERS + 4].copy_from_slice(&buffers_ptr.to_le_bytes());
    let array_ptr = alloc_bytes(&array_buf);

    let mut pair = [0u8; 8];
    pair[0..4].copy_from_slice(&array_ptr.to_le_bytes());
    pair[4..8].copy_from_slice(&schema_ptr.to_le_bytes());
    alloc_bytes(&pair) as i32
}

/// A read-only view of a child primitive array delivered by the host as Arrow C structs.
pub struct ChildView {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values (`len * ptype.byte_width()` bytes).
    pub values: &'static [u8],
    /// Validity bitmap, if the child is nullable.
    pub validity: Option<&'static [u8]>,
}

/// Parse the Arrow C Data Interface structs at `array_ptr`/`schema_ptr` in this module's linear
/// memory into a [`ChildView`].
///
/// # Safety
///
/// The host guarantees the structs and their buffers live in this module's memory for the duration
/// of the decode call, so the returned `'static` slices are valid until the call returns.
pub fn read_child(array_ptr: u32, schema_ptr: u32) -> GuestResult<ChildView> {
    unsafe {
        let format_ptr = load_u32(schema_ptr + schema::FORMAT as u32);
        let format = load_cstr(format_ptr)?;
        let ptype = PType::from_format(format)
            .ok_or(GuestError::new("child has unsupported Arrow format"))?;
        let len = load_i64(array_ptr + array::LENGTH as u32) as usize;
        let buffers_ptr = load_u32(array_ptr + array::BUFFERS as u32);
        let validity_ptr = load_u32(buffers_ptr);
        let values_ptr = load_u32(buffers_ptr + 4);

        let values = core::slice::from_raw_parts(values_ptr as *const u8, len * ptype.byte_width());
        let validity = if validity_ptr != 0 {
            Some(core::slice::from_raw_parts(
                validity_ptr as *const u8,
                len.div_ceil(8),
            ))
        } else {
            None
        };
        Ok(ChildView {
            ptype,
            len,
            values,
            validity,
        })
    }
}

unsafe fn load_u32(off: u32) -> u32 {
    let mut b = [0u8; 4];
    unsafe { core::ptr::copy_nonoverlapping(off as *const u8, b.as_mut_ptr(), 4) };
    u32::from_le_bytes(b)
}

unsafe fn load_i64(off: u32) -> i64 {
    let mut b = [0u8; 8];
    unsafe { core::ptr::copy_nonoverlapping(off as *const u8, b.as_mut_ptr(), 8) };
    i64::from_le_bytes(b)
}

unsafe fn load_cstr(ptr: u32) -> GuestResult<&'static str> {
    // Format codes are 1-3 ASCII bytes; scan a small bound for the NUL terminator.
    for n in 0..8u32 {
        let byte = unsafe { *((ptr + n) as *const u8) };
        if byte == 0 {
            let slice = unsafe { core::slice::from_raw_parts(ptr as *const u8, n as usize) };
            return core::str::from_utf8(slice)
                .map_err(|_| GuestError::new("non-utf8 Arrow format code"));
        }
    }
    Err(GuestError::new("unterminated Arrow format code"))
}
