// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Importing a Vortex array from Arrow C Data Interface structs that live in a WASM guest's linear
//! memory.
//!
//! Decoded arrays cross the host/guest boundary as the [Arrow C Data Interface]. The guest builds
//! the `ArrowSchema`/`ArrowArray` structs directly as plain bytes (no Arrow library needed); this
//! module reads that standard layout out of the guest's 32-bit address space, deep-copies the
//! buffers, reconstructs an Arrow array, and converts it to a Vortex array via
//! [`ArrayRef::from_arrow`].
//!
//! Because the boundary is wasm32, pointer fields are 4 bytes and there is no shared address space,
//! so we cannot hand Arrow a borrowed `FFI_ArrowArray`: we copy buffers out and build
//! [`arrow_data::ArrayData`] ourselves (Arrow's `from_ffi` is for same-address-space hand-off).
//!
//! Scope: primitive and boolean arrays, including a validity bitmap. Nested types (struct, list,
//! varbin/view) follow.
//!
//! [Arrow C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface.html

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::make_array;
use arrow_buffer::Buffer as ArrowBuffer;
use arrow_data::ArrayData;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Size of an `ArrowSchema` struct in the wasm32 C ABI.
const SCHEMA_SIZE: usize = 48;
/// Size of an `ArrowArray` struct in the wasm32 C ABI.
const ARRAY_SIZE: usize = 64;

/// `ArrowSchema` field offsets in the wasm32 C ABI (4-byte pointers, 8-byte/8-aligned `int64`).
mod schema {
    pub const FORMAT: usize = 0; // const char*
    pub const FLAGS: usize = 16; // int64 (after format/name/metadata ptrs + pad)
}

/// `ArrowArray` field offsets in the wasm32 C ABI.
mod array {
    pub const LENGTH: usize = 0; // int64
    pub const NULL_COUNT: usize = 8; // int64
    pub const OFFSET: usize = 16; // int64
    pub const N_BUFFERS: usize = 24; // int64
    pub const BUFFERS: usize = 40; // const void** (ptr)
}

/// Arrow flag: the field may contain nulls.
const ARROW_FLAG_NULLABLE: i64 = 2;

fn read_u32(mem: &[u8], off: u32) -> VortexResult<u32> {
    let off = off as usize;
    vortex_ensure!(off + 4 <= mem.len(), "arrow-ffi: u32 read out of bounds");
    Ok(u32::from_le_bytes(mem[off..off + 4].try_into().expect("4")))
}

fn read_i64(mem: &[u8], off: u32) -> VortexResult<i64> {
    let off = off as usize;
    vortex_ensure!(off + 8 <= mem.len(), "arrow-ffi: i64 read out of bounds");
    Ok(i64::from_le_bytes(mem[off..off + 8].try_into().expect("8")))
}

/// Read a NUL-terminated C string (the Arrow `format` field).
fn read_cstr(mem: &[u8], ptr: u32) -> VortexResult<&str> {
    let start = ptr as usize;
    vortex_ensure!(start <= mem.len(), "arrow-ffi: format ptr out of bounds");
    let end = mem[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|n| start + n)
        .ok_or_else(|| vortex_error::vortex_err!("arrow-ffi: unterminated format string"))?;
    std::str::from_utf8(&mem[start..end])
        .map_err(|_| vortex_error::vortex_err!("arrow-ffi: non-utf8 format string"))
}

fn copy_bytes(mem: &[u8], ptr: u32, len: usize) -> VortexResult<ArrowBuffer> {
    let start = ptr as usize;
    vortex_ensure!(
        start + len <= mem.len(),
        "arrow-ffi: buffer [{start}, {start}+{len}) out of bounds ({})",
        mem.len()
    );
    Ok(ArrowBuffer::from(&mem[start..start + len]))
}

/// The Arrow primitive layout for a format code: `(DataType, byte_width)`, or `None` for `Bool`
/// (a bitmap), or an error for unsupported formats.
fn primitive_layout(format: &str) -> VortexResult<(DataType, usize)> {
    Ok(match format {
        "c" => (DataType::Int8, 1),
        "C" => (DataType::UInt8, 1),
        "s" => (DataType::Int16, 2),
        "S" => (DataType::UInt16, 2),
        "i" => (DataType::Int32, 4),
        "I" => (DataType::UInt32, 4),
        "l" => (DataType::Int64, 8),
        "L" => (DataType::UInt64, 8),
        "e" => (DataType::Float16, 2),
        "f" => (DataType::Float32, 4),
        "g" => (DataType::Float64, 8),
        other => vortex_bail!("arrow-ffi: unsupported format code {other:?}"),
    })
}

/// Import a Vortex array from Arrow C Data Interface structs in `mem`.
///
/// `array_ptr` and `schema_ptr` are wasm32 offsets to the `ArrowArray` and `ArrowSchema` structs.
pub fn import(mem: &[u8], array_ptr: u32, schema_ptr: u32) -> VortexResult<ArrayRef> {
    let format = read_cstr(mem, read_u32(mem, schema_ptr + schema::FORMAT as u32)?)?;
    let flags = read_i64(mem, schema_ptr + schema::FLAGS as u32)?;
    let nullable = flags & ARROW_FLAG_NULLABLE != 0;

    let len = usize::try_from(read_i64(mem, array_ptr + array::LENGTH as u32)?)?;
    let offset = usize::try_from(read_i64(mem, array_ptr + array::OFFSET as u32)?)?;
    let n_buffers = read_i64(mem, array_ptr + array::N_BUFFERS as u32)?;
    let buffers_ptr = read_u32(mem, array_ptr + array::BUFFERS as u32)?;
    let _ = read_i64(mem, array_ptr + array::NULL_COUNT as u32)?;

    vortex_ensure!(
        n_buffers == 2,
        "arrow-ffi: primitive/bool expects 2 buffers (validity, values), got {n_buffers}"
    );
    let validity_ptr = read_u32(mem, buffers_ptr)?;
    let values_ptr = read_u32(mem, buffers_ptr + 4)?;

    let arrow = if format == "b" {
        // Boolean: values are a bitmap of (len + offset) bits.
        let nbytes = (len + offset).div_ceil(8);
        let values = copy_bytes(mem, values_ptr, nbytes)?;
        build_array(DataType::Boolean, len, offset, values, validity_ptr, mem)?
    } else {
        let (dtype, width) = primitive_layout(format)?;
        let values = copy_bytes(mem, values_ptr, (len + offset) * width)?;
        build_array(dtype, len, offset, values, validity_ptr, mem)?
    };

    ArrayRef::from_arrow(arrow.as_ref(), nullable)
}

/// Arrow C Data Interface format code for a Vortex primitive type.
fn format_code(ptype: PType) -> &'static str {
    match ptype {
        PType::I8 => "c",
        PType::U8 => "C",
        PType::I16 => "s",
        PType::U16 => "S",
        PType::I32 => "i",
        PType::U32 => "I",
        PType::I64 => "l",
        PType::U64 => "L",
        PType::F16 => "e",
        PType::F32 => "f",
        PType::F64 => "g",
    }
}

/// A writable view of a WASM guest's linear memory, used to lay out Arrow C structs for the guest.
///
/// `alloc` allocates `len` bytes in guest memory (via the guest's `vx_alloc`) and returns the
/// offset; `write` copies bytes to a previously allocated offset.
pub trait GuestMem {
    /// Allocate `len` bytes in guest memory, returning the offset.
    fn alloc(&mut self, len: u32) -> VortexResult<u32>;
    /// Write `bytes` at guest offset `off`.
    fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()>;
}

fn put(mem: &mut dyn GuestMem, bytes: &[u8]) -> VortexResult<u32> {
    let off = mem.alloc(u32::try_from(bytes.len().max(1))?)?;
    mem.write(off, bytes)?;
    Ok(off)
}

/// Export a canonical array as Arrow C Data Interface structs written into `mem`, returning the
/// `(array_ptr, schema_ptr)` offsets a guest can consume.
///
/// Scope mirrors [`import`]: primitive and boolean, including a validity bitmap.
pub fn export(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
    mem: &mut dyn GuestMem,
) -> VortexResult<(u32, u32)> {
    let (format, len, values, validity): (&str, usize, Vec<u8>, Validity) = match canonical {
        Canonical::Primitive(array) => (
            format_code(array.ptype()),
            array.len(),
            array.buffer_handle().to_host_sync().as_ref().to_vec(),
            array.validity()?,
        ),
        other => vortex_bail!(
            "arrow-ffi export supports primitive only, got {:?}",
            std::mem::discriminant(other)
        ),
    };

    let validity_ptr = match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        Validity::AllInvalid | Validity::Array(_) => {
            let bits = validity.execute_mask(len, ctx)?.to_bit_buffer();
            let bitmap = &bits.inner().as_slice()[..len.div_ceil(8)];
            put(mem, bitmap)?
        }
    };
    let nullable = !matches!(validity, Validity::NonNullable);

    let mut format_bytes = format.as_bytes().to_vec();
    format_bytes.push(0);
    let format_ptr = put(mem, &format_bytes)?;
    let values_ptr = put(mem, &values)?;

    let mut buffers = Vec::with_capacity(8);
    buffers.extend_from_slice(&validity_ptr.to_le_bytes());
    buffers.extend_from_slice(&values_ptr.to_le_bytes());
    let buffers_ptr = put(mem, &buffers)?;

    let mut schema = vec![0u8; SCHEMA_SIZE];
    schema[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
    let flags: i64 = if nullable { ARROW_FLAG_NULLABLE } else { 0 };
    schema[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
    let schema_ptr = put(mem, &schema)?;

    let mut array = vec![0u8; ARRAY_SIZE];
    array[self::array::LENGTH..self::array::LENGTH + 8]
        .copy_from_slice(&(len as i64).to_le_bytes());
    let null_count: i64 = if nullable { -1 } else { 0 };
    array[self::array::NULL_COUNT..self::array::NULL_COUNT + 8]
        .copy_from_slice(&null_count.to_le_bytes());
    array[self::array::N_BUFFERS..self::array::N_BUFFERS + 8].copy_from_slice(&2i64.to_le_bytes());
    array[self::array::BUFFERS..self::array::BUFFERS + 4]
        .copy_from_slice(&buffers_ptr.to_le_bytes());
    let array_ptr = put(mem, &array)?;

    Ok((array_ptr, schema_ptr))
}

fn build_array(
    dtype: DataType,
    len: usize,
    offset: usize,
    values: ArrowBuffer,
    validity_ptr: u32,
    mem: &[u8],
) -> VortexResult<ArrowArrayRef> {
    let null_bit_buffer = if validity_ptr != 0 {
        Some(copy_bytes(mem, validity_ptr, (len + offset).div_ceil(8))?)
    } else {
        None
    };
    let data = ArrayData::try_new(dtype, len, null_bit_buffer, offset, vec![values], vec![])
        .map_err(|e| vortex_error::vortex_err!("arrow-ffi: invalid array data: {e}"))?;
    Ok(make_array(data))
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    /// A `Vec`-backed [`GuestMem`] simulating guest linear memory for wasm-free tests.
    struct VecMem {
        mem: Vec<u8>,
    }

    impl VecMem {
        fn new() -> Self {
            // Reserve offset 0 so it reads as a null pointer.
            Self { mem: vec![0u8; 8] }
        }
    }

    impl GuestMem for VecMem {
        fn alloc(&mut self, len: u32) -> VortexResult<u32> {
            while !self.mem.len().is_multiple_of(8) {
                self.mem.push(0);
            }
            let off = self.mem.len() as u32;
            self.mem.resize(self.mem.len() + len as usize, 0);
            Ok(off)
        }

        fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()> {
            self.mem[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
    }

    #[test]
    fn export_then_import_round_trip_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let validity = Validity::from_iter([true, false, true, false, true]);
        let canonical =
            Canonical::Primitive(PrimitiveArray::new(buffer![1i64, 2, 3, 4, 5], validity));

        let mut mem = VecMem::new();
        let (array_ptr, schema_ptr) = export(&canonical, &mut ctx, &mut mem)?;
        let imported = import(&mem.mem, array_ptr, schema_ptr)?;

        assert_eq!(imported.len(), 5);
        let values = imported
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();
        assert_eq!(values.as_slice::<i64>(), &[1, 2, 3, 4, 5]);
        let bits = imported
            .validity()?
            .execute_mask(5, &mut ctx)?
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, false, true]);
        Ok(())
    }

    /// Lays out an Arrow C Data Interface image (wasm32) for a single primitive/bool array.
    struct ImageBuilder {
        mem: Vec<u8>,
    }

    impl ImageBuilder {
        fn new() -> Self {
            // Reserve a zero page so offset 0 reads as a null pointer.
            Self { mem: vec![0u8; 16] }
        }

        fn put(&mut self, bytes: &[u8]) -> u32 {
            // 8-align every region so struct int64 fields are aligned.
            while !self.mem.len().is_multiple_of(8) {
                self.mem.push(0);
            }
            let off = self.mem.len() as u32;
            self.mem.extend_from_slice(bytes);
            off
        }

        fn schema(&mut self, format: &str, nullable: bool) -> u32 {
            let mut fmt = format.as_bytes().to_vec();
            fmt.push(0);
            let format_ptr = self.put(&fmt);
            let mut s = vec![0u8; 48];
            s[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
            let flags: i64 = if nullable { ARROW_FLAG_NULLABLE } else { 0 };
            s[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
            self.put(&s)
        }

        fn array(&mut self, len: usize, values: &[u8], validity: Option<&[u8]>) -> u32 {
            let values_ptr = self.put(values);
            let validity_ptr = validity.map(|v| self.put(v)).unwrap_or(0);
            let mut buffers = Vec::new();
            buffers.extend_from_slice(&validity_ptr.to_le_bytes());
            buffers.extend_from_slice(&values_ptr.to_le_bytes());
            let buffers_ptr = self.put(&buffers);

            let null_count: i64 = if validity.is_some() { -1 } else { 0 };
            let mut a = vec![0u8; 64];
            a[array::LENGTH..array::LENGTH + 8].copy_from_slice(&(len as i64).to_le_bytes());
            a[array::NULL_COUNT..array::NULL_COUNT + 8].copy_from_slice(&null_count.to_le_bytes());
            a[array::OFFSET..array::OFFSET + 8].copy_from_slice(&0i64.to_le_bytes());
            a[array::N_BUFFERS..array::N_BUFFERS + 8].copy_from_slice(&2i64.to_le_bytes());
            a[array::BUFFERS..array::BUFFERS + 4].copy_from_slice(&buffers_ptr.to_le_bytes());
            self.put(&a)
        }
    }

    #[test]
    fn import_primitive_i32() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Vec<u8> = [10i32, 20, 30, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut b = ImageBuilder::new();
        let schema_ptr = b.schema("i", false);
        let array_ptr = b.array(4, &values, None);

        let imported = import(&b.mem, array_ptr, schema_ptr)?;
        assert_eq!(imported.len(), 4);
        let canonical = imported.execute::<vortex_array::Canonical>(&mut ctx)?;
        assert_eq!(
            canonical.into_primitive().as_slice::<i32>(),
            &[10, 20, 30, 40]
        );
        Ok(())
    }

    #[test]
    fn import_nullable_i32() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Vec<u8> = [1i32, 2, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // valid at 0,2,4 -> bits 1,0,1,0,1 -> 0b10101 = 0x15
        let validity = [0x15u8];
        let mut b = ImageBuilder::new();
        let schema_ptr = b.schema("i", true);
        let array_ptr = b.array(5, &values, Some(&validity));

        let imported = import(&b.mem, array_ptr, schema_ptr)?;
        assert_eq!(imported.len(), 5);
        let bits = imported
            .validity()?
            .execute_mask(5, &mut ctx)?
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, false, true]);
        Ok(())
    }
}
