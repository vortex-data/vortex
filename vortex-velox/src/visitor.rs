// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::mem::align_of;
use std::mem::size_of;
use std::mem::size_of_val;
use std::ptr;
use std::slice;
use std::sync::Arc;

use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Extension;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::ListView;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::MapArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::listview::ListViewArraySlotsExt;
use vortex::array::arrays::map::MapArrayExt;
use vortex::array::arrays::map::MapArraySlotsExt;
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::PType;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;
use vortex_array::ArrayView;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_ffi::try_or;
use vortex_ffi::vx_array;
use vortex_ffi::vx_array_ref;
use vortex_ffi::vx_error;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_ref;

use crate::array::ArrowMemoryReservation;
use crate::array::conservative_export_reservation;
use crate::array::parse_memory_callbacks;
use crate::array::vx_velox_arrow_memory_callbacks;

/// A fixed-width primitive value identifier in a semantic visitor block.
pub type vx_velox_primitive_type = u32;
/// Unsigned 8-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U8: vx_velox_primitive_type = 0;
/// Unsigned 16-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U16: vx_velox_primitive_type = 1;
/// Unsigned 32-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U32: vx_velox_primitive_type = 2;
/// Unsigned 64-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_U64: vx_velox_primitive_type = 3;
/// Signed 8-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I8: vx_velox_primitive_type = 4;
/// Signed 16-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I16: vx_velox_primitive_type = 5;
/// Signed 32-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I32: vx_velox_primitive_type = 6;
/// Signed 64-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I64: vx_velox_primitive_type = 7;
/// IEEE 754 binary16 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F16: vx_velox_primitive_type = 8;
/// IEEE 754 binary32 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F32: vx_velox_primitive_type = 9;
/// IEEE 754 binary64 primitive identifier.
pub const VX_VELOX_PRIMITIVE_F64: vx_velox_primitive_type = 10;
/// Signed 128-bit primitive identifier.
pub const VX_VELOX_PRIMITIVE_I128: vx_velox_primitive_type = 11;

fn primitive_type_id(value: PType) -> vx_velox_primitive_type {
    match value {
        PType::U8 => VX_VELOX_PRIMITIVE_U8,
        PType::U16 => VX_VELOX_PRIMITIVE_U16,
        PType::U32 => VX_VELOX_PRIMITIVE_U32,
        PType::U64 => VX_VELOX_PRIMITIVE_U64,
        PType::I8 => VX_VELOX_PRIMITIVE_I8,
        PType::I16 => VX_VELOX_PRIMITIVE_I16,
        PType::I32 => VX_VELOX_PRIMITIVE_I32,
        PType::I64 => VX_VELOX_PRIMITIVE_I64,
        PType::F16 => VX_VELOX_PRIMITIVE_F16,
        PType::F32 => VX_VELOX_PRIMITIVE_F32,
        PType::F64 => VX_VELOX_PRIMITIVE_F64,
    }
}

/// A fixed-width validity representation identifier for one visitor block.
pub type vx_velox_validity_kind = u32;
/// The type is not nullable.
pub const VX_VELOX_VALIDITY_NON_NULLABLE: vx_velox_validity_kind = 0;
/// Every value is valid.
pub const VX_VELOX_VALIDITY_ALL_VALID: vx_velox_validity_kind = 1;
/// Every value is null.
pub const VX_VELOX_VALIDITY_ALL_INVALID: vx_velox_validity_kind = 2;
/// A packed bitmap contains one valid bit per value.
pub const VX_VELOX_VALIDITY_BITMAP: vx_velox_validity_kind = 3;

/// A retained owner for buffers in a visitor block.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_buffer_owner {
    /// Set this field to `sizeof(vx_velox_buffer_owner)`.
    pub struct_size: usize,
    /// An opaque retained object.
    pub owner: *const c_void,
    /// Add one owner reference before the callback returns.
    pub retain: Option<unsafe extern "C" fn(owner: *const c_void)>,
    /// Release one retained owner reference.
    pub release: Option<unsafe extern "C" fn(owner: *const c_void)>,
    /// The exact sum of the value and validity allocation sizes retained by this owner.
    pub retained_bytes: usize,
}

/// A canonical primitive block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_primitive_view {
    /// Set this field to `sizeof(vx_velox_primitive_view)`.
    pub struct_size: usize,
    /// The physical type of each value.
    pub primitive_type: vx_velox_primitive_type,
    /// The logical decimal precision, or zero for a non-decimal block.
    pub decimal_precision: u32,
    /// The logical decimal scale, or zero for a non-decimal block.
    pub decimal_scale: i32,
    /// The number of logical values in the block.
    pub length: usize,
    /// The first value byte.
    pub values: *const u8,
    /// The number of value bytes.
    pub values_length: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains all pointers in this view.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty values buffer.
    pub values_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// Identifies the logical type of a variable-width binary block.
pub type vx_velox_varbin_kind = u32;
/// Identifies UTF-8 values.
pub const VX_VELOX_VARBIN_UTF8: vx_velox_varbin_kind = 0;
/// Identifies arbitrary binary values.
pub const VX_VELOX_VARBIN_BINARY: vx_velox_varbin_kind = 1;

/// Describes one retained payload buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_byte_buffer_view {
    /// The first payload byte.
    pub data: *const u8,
    /// The number of visible payload bytes.
    pub length: usize,
}

/// Defines the stable 16-byte variable-width binary view contract.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_binary_view {
    /// Stores the logical byte length.
    pub length: u32,
    /// Stores inline bytes, or prefix, buffer index, and offset for outlined values.
    pub data: [u8; 12],
}

/// A canonical variable-width binary block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_varbin_view {
    /// Set this field to `sizeof(vx_velox_varbin_view)`.
    pub struct_size: usize,
    /// Identifies UTF-8 or binary values.
    pub kind: vx_velox_varbin_kind,
    /// The number of logical values in the block.
    pub length: usize,
    /// The first 16-byte binary view.
    pub views: *const vx_velox_binary_view,
    /// The number of readable bytes in `views`.
    pub views_length: usize,
    /// The retained payload buffer descriptors.
    pub data_buffers: *const vx_velox_byte_buffer_view,
    /// The number of payload buffer descriptors.
    pub data_buffer_count: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of readable validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains all pointers in this view.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty view buffer.
    pub views_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A canonical packed Boolean block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_bool_view {
    /// Set this field to `sizeof(vx_velox_bool_view)`.
    pub struct_size: usize,
    /// The number of logical Boolean values.
    pub length: usize,
    /// The first packed value byte.
    pub values: *const u8,
    /// The number of readable value bytes.
    pub values_length: usize,
    /// The first logical value bit within `values`.
    pub values_bit_offset: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of readable validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains all pointers in this view.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty value buffer.
    pub values_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A dictionary block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_dictionary_view {
    /// Set this field to `sizeof(vx_velox_dictionary_view)`.
    pub struct_size: usize,
    /// The number of logical dictionary codes.
    pub length: usize,
    /// The canonical integer codes for this output window.
    pub codes: vx_velox_primitive_view,
    /// A borrowed prepared cursor for the dictionary values.
    pub values: *const vx_velox_export_cursor,
    /// The number of dictionary values.
    pub values_length: usize,
}

/// A constant block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_constant_view {
    /// Set this field to `sizeof(vx_velox_constant_view)`.
    pub struct_size: usize,
    /// The number of repeated logical values.
    pub length: usize,
    /// A borrowed prepared cursor with one canonical value.
    pub value: *const vx_velox_export_cursor,
}

/// A canonical struct block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_struct_view {
    /// Set this field to `sizeof(vx_velox_struct_view)`.
    pub struct_size: usize,
    /// The number of logical struct values in this window.
    pub length: usize,
    /// The first logical row in each field cursor.
    pub offset: usize,
    /// Borrowed prepared cursors in field order.
    pub fields: *const *const vx_velox_export_cursor,
    /// The number of field cursors.
    pub field_count: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of readable validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains the parent validity buffer.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A canonical list block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_list_view {
    /// Set this field to `sizeof(vx_velox_list_view)`.
    pub struct_size: usize,
    /// The number of logical lists in this window.
    pub length: usize,
    /// One non-negative element offset per list. Values remain absolute against `elements`.
    pub offsets: *const i32,
    /// One non-negative element count per list.
    pub sizes: *const i32,
    /// A borrowed prepared cursor for all referenced elements.
    pub elements: *const vx_velox_export_cursor,
    /// The number of values in the element cursor.
    pub elements_length: usize,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of readable validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains the complete offsets, sizes, and parent validity allocations.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty offsets buffer.
    pub offsets_alignment: usize,
    /// The guaranteed byte alignment of a non-empty sizes buffer.
    pub sizes_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A canonical map block delivered to Velox.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_map_view {
    /// Set this field to `sizeof(vx_velox_map_view)`.
    pub struct_size: usize,
    /// The number of logical maps in this window.
    pub length: usize,
    /// One non-negative entry offset per map. Values remain absolute against the child cursors.
    pub offsets: *const i32,
    /// One non-negative entry count per map.
    pub sizes: *const i32,
    /// A borrowed prepared cursor for all map keys.
    pub keys: *const vx_velox_export_cursor,
    /// A borrowed prepared cursor for all map values.
    pub values: *const vx_velox_export_cursor,
    /// The number of entries in each child cursor.
    pub entries_length: usize,
    /// True when each map asserts sorted keys.
    pub keys_sorted: bool,
    /// The validity representation.
    pub validity_kind: vx_velox_validity_kind,
    /// The first validity byte when `validity_kind` is `Bitmap`.
    pub validity: *const u8,
    /// The number of readable validity bytes.
    pub validity_length: usize,
    /// The first logical validity bit within `validity`.
    pub validity_bit_offset: usize,
    /// Retains the complete offsets, sizes, and parent validity allocations.
    pub buffers: vx_velox_buffer_owner,
    /// The guaranteed byte alignment of a non-empty offsets buffer.
    pub offsets_alignment: usize,
    /// The guaranteed byte alignment of a non-empty sizes buffer.
    pub sizes_alignment: usize,
    /// The guaranteed byte alignment of a non-empty validity buffer.
    pub validity_alignment: usize,
}

/// A single-shot subset request for the semantic visitor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_visit_request {
    /// Set this field to `sizeof(vx_velox_visit_request)`.
    pub struct_size: usize,
    /// Unique, increasing source positions. Null selects every row.
    pub rows: *const u64,
    /// The number of source positions.
    pub row_count: usize,
}

/// Host callbacks for Vortex array traversal.
///
/// One array visit calls the matching callback synchronously. Shared tables can receive concurrent
/// callbacks from simultaneous visits. `last_error` must return the calling thread's most recent
/// error. The string must remain valid until the next callback on that thread. Callbacks must catch
/// foreign exceptions and must not unwind across this ABI. The host owns the context.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_visitor {
    /// Set this field to `sizeof(vx_velox_visitor)`.
    pub struct_size: usize,
    /// Set this field to [`crate::VX_VELOX_ABI_VERSION`].
    pub abi_version: u32,
    /// An opaque callback context.
    pub context: *mut c_void,
    /// Consume one canonical primitive block. Zero means success.
    pub visit_primitive: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_primitive_view) -> i32,
    >,
    /// Return the last callback error as a null-terminated string.
    pub last_error: Option<unsafe extern "C" fn(context: *mut c_void) -> *const c_char>,
    /// Consume one canonical variable-width binary block. Zero means success.
    pub visit_varbin: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_varbin_view) -> i32,
    >,
    /// Consume one dictionary block. Zero means success.
    pub visit_dictionary: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_dictionary_view) -> i32,
    >,
    /// Consume one constant block. Zero means success.
    pub visit_constant: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_constant_view) -> i32,
    >,
    /// Consume one canonical packed Boolean block. Zero means success.
    pub visit_bool:
        Option<unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_bool_view) -> i32>,
    /// Consume one canonical struct block. Zero means success.
    pub visit_struct: Option<
        unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_struct_view) -> i32,
    >,
    /// Consume one canonical list block. Zero means success.
    pub visit_list:
        Option<unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_list_view) -> i32>,
    /// Consume one canonical map block. Zero means success.
    pub visit_map:
        Option<unsafe extern "C" fn(context: *mut c_void, view: *const vx_velox_map_view) -> i32>,
}

/// Retains one prepared Vortex array across several Velox output windows.
#[repr(C)]
pub struct vx_velox_export_cursor {
    export: CursorExport,
}

enum CursorExport {
    Primitive(PrimitiveExport),
    Bool(BoolExport),
    VarBin(VarBinExport),
    Dictionary(DictionaryExport),
    Constant(ConstantExport),
    Struct(StructExport),
    List(ListExport),
    Map(MapExport),
}

struct PackedBits(Box<[u64]>);

impl PackedBits {
    fn try_new(bits: vortex::buffer::BitBuffer) -> VortexResult<(Self, usize)> {
        let compact = bits
            .chunks()
            .iter_padded()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let allocation = size_of_val(compact.as_ref());
        Ok((Self(compact), allocation))
    }

    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr().cast()
    }

    fn len(&self) -> usize {
        size_of_val(self.0.as_ref())
    }
}

struct BoolOwner {
    values: PackedBits,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

impl BoolOwner {
    fn try_new(
        values: vortex::buffer::BitBuffer,
        validity: Option<vortex::buffer::BitBuffer>,
    ) -> VortexResult<Self> {
        let (values, values_allocation) = PackedBits::try_new(values)?;
        let (validity, validity_allocation) = match validity {
            Some(validity) => {
                let (validity, allocation) = PackedBits::try_new(validity)?;
                (Some(validity), allocation)
            }
            None => (None, 0),
        };
        let retained_bytes = values_allocation
            .checked_add(validity_allocation)
            .ok_or_else(|| vortex_err!("Boolean visitor retained byte count overflow"))?;
        Ok(Self {
            values,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

enum PrimitiveValues {
    Compact64(Box<[MaybeUninit<u64>]>),
    Compact128(Box<[MaybeUninit<u128>]>),
    Retained(ByteBuffer),
}

impl PrimitiveValues {
    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Compact64(values) => values.as_ptr().cast(),
            Self::Compact128(values) => values.as_ptr().cast(),
            Self::Retained(values) => values.as_ptr(),
        }
    }
}

struct PrimitiveOwner {
    values: PrimitiveValues,
    values_length: usize,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

enum RetainedBytes {
    Retained(ByteBuffer),
    Compact(Box<[u8]>),
}

impl RetainedBytes {
    fn try_new(handle: BufferHandle) -> VortexResult<(Self, usize)> {
        let buffer = handle.try_into_host_sync()?;
        let length = buffer.len();
        match buffer.try_into_mut() {
            Ok(buffer) => {
                let allocation_size = buffer.allocation_size();
                Ok((Self::Retained(buffer.freeze()), allocation_size))
            }
            Err(buffer) => {
                let compact = buffer.as_slice().to_vec().into_boxed_slice();
                Ok((Self::Compact(compact), length))
            }
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Retained(buffer) => buffer.as_ptr(),
            Self::Compact(buffer) => buffer.as_ptr(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Retained(buffer) => buffer.len(),
            Self::Compact(buffer) => buffer.len(),
        }
    }
}

enum RetainedViews {
    Retained(ByteBuffer),
    Compact(Box<[vx_velox_binary_view]>),
}

impl RetainedViews {
    fn try_new(handle: BufferHandle) -> VortexResult<(Self, usize)> {
        let buffer = handle.try_into_host_sync()?;
        if !buffer
            .len()
            .is_multiple_of(size_of::<vx_velox_binary_view>())
        {
            vortex_bail!(
                "Vortex variable-width view buffer has an invalid byte length: {}",
                buffer.len()
            );
        }
        match buffer.try_into_mut() {
            Ok(buffer) => {
                let allocation_size = buffer.allocation_size();
                Ok((Self::Retained(buffer.freeze()), allocation_size))
            }
            Err(buffer) => {
                let length = buffer.len() / size_of::<vx_velox_binary_view>();
                let mut compact = vec![
                    vx_velox_binary_view {
                        length: 0,
                        data: [0; 12],
                    };
                    length
                ]
                .into_boxed_slice();
                if !buffer.is_empty() {
                    // SAFETY: Both byte ranges have the checked identical size.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            buffer.as_ptr(),
                            compact.as_mut_ptr().cast::<u8>(),
                            buffer.len(),
                        )
                    };
                }
                let allocation = size_of_val(compact.as_ref());
                Ok((Self::Compact(compact), allocation))
            }
        }
    }

    fn as_ptr(&self) -> *const vx_velox_binary_view {
        match self {
            Self::Retained(buffer) => buffer.as_ptr().cast(),
            Self::Compact(buffer) => buffer.as_ptr(),
        }
    }
}

struct VarBinOwner {
    views: RetainedViews,
    _data: Box<[RetainedBytes]>,
    descriptors: Box<[vx_velox_byte_buffer_view]>,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    memory_reservation: Option<ArrowMemoryReservation>,
}

// SAFETY: The owner never mutates its buffers or pointer descriptors after construction.
// Every descriptor points into an immutable allocation that the same owner retains.
unsafe impl Send for VarBinOwner {}
// SAFETY: Shared access only reads immutable buffers and descriptors retained by this owner.
unsafe impl Sync for VarBinOwner {}

impl VarBinOwner {
    fn try_new(
        views: BufferHandle,
        mut buffers: Arc<[BufferHandle]>,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
    ) -> VortexResult<Self> {
        let (views, views_allocation) = RetainedViews::try_new(views)?;
        let handles = if let Some(handles) = Arc::get_mut(&mut buffers) {
            handles
                .iter_mut()
                .map(|handle| {
                    std::mem::replace(handle, BufferHandle::new_host(ByteBuffer::empty()))
                })
                .collect::<Vec<_>>()
        } else {
            buffers.iter().cloned().collect::<Vec<_>>()
        };
        let mut data_allocation = 0usize;
        let data = handles
            .into_iter()
            .map(|handle| {
                let (buffer, allocation) = RetainedBytes::try_new(handle)?;
                data_allocation = data_allocation
                    .checked_add(allocation)
                    .ok_or_else(|| vortex_err!("Vortex string payload allocation overflow"))?;
                Ok(buffer)
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();
        let descriptors = data
            .iter()
            .map(|buffer| vx_velox_byte_buffer_view {
                data: buffer.as_ptr(),
                length: buffer.len(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let descriptor_allocation = size_of_val(descriptors.as_ref());
        let (validity, validity_allocation) = retain_validity(validity, length)?;
        let retained_bytes = views_allocation
            .checked_add(data_allocation)
            .and_then(|bytes| bytes.checked_add(descriptor_allocation))
            .and_then(|bytes| bytes.checked_add(validity_allocation))
            .ok_or_else(|| vortex_err!("Vortex string retained byte count overflow"))?;
        Ok(Self {
            views,
            _data: data,
            descriptors,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

fn retain_validity(
    validity: Option<vortex::buffer::BitBuffer>,
    length: usize,
) -> VortexResult<(Option<PackedBits>, usize)> {
    let Some(validity) = validity else {
        return Ok((None, 0));
    };
    if validity.len() < length {
        vortex_bail!(
            "Vortex validity length is too small: {} for {length} values",
            validity.len()
        );
    }
    let validity = if validity.len() == length {
        validity
    } else {
        validity.slice(..length)
    };
    let (validity, allocation) = PackedBits::try_new(validity)?;
    Ok((Some(validity), allocation))
}

impl PrimitiveOwner {
    fn try_allocate(
        values_length: usize,
        values_alignment: usize,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
    ) -> VortexResult<Self> {
        let (values, values_allocation) = if values_alignment > align_of::<u64>() {
            if values_alignment > align_of::<u128>() {
                vortex_bail!(
                    "Primitive visitor does not support value alignment {values_alignment}"
                );
            }
            let values =
                vec![MaybeUninit::<u128>::uninit(); values_length.div_ceil(size_of::<u128>())]
                    .into_boxed_slice();
            let allocation = values
                .len()
                .checked_mul(size_of::<u128>())
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
            (PrimitiveValues::Compact128(values), allocation)
        } else {
            let values =
                vec![MaybeUninit::<u64>::uninit(); values_length.div_ceil(size_of::<u64>())]
                    .into_boxed_slice();
            let allocation = values
                .len()
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
            (PrimitiveValues::Compact64(values), allocation)
        };
        let (validity, validity_allocation) = retain_validity(validity, length)?;
        let retained_bytes = values_allocation
            .checked_add(validity_allocation)
            .ok_or_else(|| vortex_err!("Primitive visitor retained byte count overflow"))?;
        Ok(Self {
            values,
            values_length,
            validity,
            retained_bytes,
            memory_reservation: None,
        })
    }

    fn try_new(
        host_values: ByteBuffer,
        values_alignment: usize,
        validity: Option<vortex::buffer::BitBuffer>,
        length: usize,
        retain_values: bool,
    ) -> VortexResult<Self> {
        let values_length = host_values.len();
        let host_values = if retain_values {
            match host_values.try_into_mut() {
                Ok(values) => {
                    let values_allocation = values.allocation_size();
                    let (validity, validity_allocation) = retain_validity(validity, length)?;
                    let retained_bytes = values_allocation
                        .checked_add(validity_allocation)
                        .ok_or_else(|| {
                            vortex_err!("Primitive visitor retained byte count overflow")
                        })?;
                    return Ok(Self {
                        values: PrimitiveValues::Retained(values.freeze()),
                        values_length,
                        validity,
                        retained_bytes,
                        memory_reservation: None,
                    });
                }
                Err(values) => values,
            }
        } else {
            host_values
        };
        let mut owner = Self::try_allocate(values_length, values_alignment, validity, length)?;
        if !host_values.is_empty() {
            let (values_pointer, values_capacity) = match &mut owner.values {
                PrimitiveValues::Compact64(values) => (
                    values.as_mut_ptr().cast::<u8>(),
                    values.len() * size_of::<u64>(),
                ),
                PrimitiveValues::Compact128(values) => (
                    values.as_mut_ptr().cast::<u8>(),
                    values.len() * size_of::<u128>(),
                ),
                PrimitiveValues::Retained(_) => {
                    unreachable!("a newly allocated primitive owner must be compact")
                }
            };
            // SAFETY: The byte view spans the complete compact allocation.
            let values_bytes =
                unsafe { slice::from_raw_parts_mut(values_pointer, values_capacity) };
            values_bytes[..values_length].copy_from_slice(host_values.as_slice());
        }
        Ok(owner)
    }

    fn try_new_bitpacked_i64(
        array: ArrayView<'_, BitPacked>,
        validity: Option<vortex::buffer::BitBuffer>,
    ) -> VortexResult<Self> {
        let values_length = array
            .len()
            .checked_mul(size_of::<i64>())
            .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
        let mut owner =
            Self::try_allocate(values_length, align_of::<i64>(), validity, array.len())?;
        // SAFETY: The allocation uses `u64` alignment and contains at least `values_length` bytes.
        // The output slice covers exactly `array.len()` values and remains uniquely borrowed.
        let output = unsafe {
            slice::from_raw_parts_mut(
                match &mut owner.values {
                    PrimitiveValues::Compact64(values) => {
                        values.as_mut_ptr().cast::<MaybeUninit<i64>>()
                    }
                    PrimitiveValues::Compact128(_) | PrimitiveValues::Retained(_) => {
                        unreachable!("a newly allocated primitive owner must be compact")
                    }
                },
                array.len(),
            )
        };
        let mut scratch = [const { MaybeUninit::<i64>::uninit() }; FL_CHUNK_SIZE];
        array.unpacked_chunks(&mut scratch)?.decode_into(output);
        Ok(owner)
    }

    fn values(&self) -> *const u8 {
        if self.values_length == 0 {
            ptr::null()
        } else {
            self.values.as_ptr()
        }
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    fn set_memory_reservation(&mut self, reservation: ArrowMemoryReservation) {
        self.memory_reservation = Some(reservation);
    }
}

fn pointer_alignment(pointer: *const u8) -> usize {
    if pointer.is_null() {
        return 0;
    }
    1usize << pointer.addr().trailing_zeros()
}

fn primitive_width(primitive_type: vx_velox_primitive_type) -> VortexResult<usize> {
    Ok(match primitive_type {
        VX_VELOX_PRIMITIVE_U8 | VX_VELOX_PRIMITIVE_I8 => 1,
        VX_VELOX_PRIMITIVE_U16 | VX_VELOX_PRIMITIVE_I16 | VX_VELOX_PRIMITIVE_F16 => 2,
        VX_VELOX_PRIMITIVE_U32 | VX_VELOX_PRIMITIVE_I32 | VX_VELOX_PRIMITIVE_F32 => 4,
        VX_VELOX_PRIMITIVE_U64 | VX_VELOX_PRIMITIVE_I64 | VX_VELOX_PRIMITIVE_F64 => 8,
        VX_VELOX_PRIMITIVE_I128 => 16,
        _ => vortex_bail!("Unknown Vortex Velox primitive type: {primitive_type}"),
    })
}

fn cast_decimal_values<T, S>(values: Buffer<S>, validity: &Mask) -> VortexResult<ByteBuffer>
where
    T: NativeDecimalType,
    S: NativeDecimalType,
{
    let mut output = BufferMut::<T>::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        if !validity.value(index) {
            output.push(T::default());
            continue;
        }
        output.push(<T as vortex::dtype::BigCast>::from(value).ok_or_else(|| {
            vortex_err!(
                "Decimal value cannot be represented as {}",
                std::any::type_name::<T>()
            )
        })?);
    }
    Ok(output.freeze().into_byte_buffer())
}

fn normalized_decimal_values<T>(array: &DecimalArray, validity: &Mask) -> VortexResult<ByteBuffer>
where
    T: NativeDecimalType,
{
    if array.values_type() == T::DECIMAL_TYPE {
        return array.buffer_handle().clone().try_into_host_sync();
    }
    match array.values_type() {
        DecimalType::I8 => cast_decimal_values::<T, i8>(array.buffer::<i8>(), validity),
        DecimalType::I16 => cast_decimal_values::<T, i16>(array.buffer::<i16>(), validity),
        DecimalType::I32 => cast_decimal_values::<T, i32>(array.buffer::<i32>(), validity),
        DecimalType::I64 => cast_decimal_values::<T, i64>(array.buffer::<i64>(), validity),
        DecimalType::I128 => cast_decimal_values::<T, i128>(array.buffer::<i128>(), validity),
        DecimalType::I256 => cast_decimal_values::<T, vortex::dtype::i256>(
            array.buffer::<vortex::dtype::i256>(),
            validity,
        ),
    }
}

struct PrimitiveExport {
    primitive_type: vx_velox_primitive_type,
    decimal_precision: u32,
    decimal_scale: i32,
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<PrimitiveOwner>,
}

impl PrimitiveExport {
    fn try_new_decimal(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let retain_values = memory_callbacks.is_some();
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let decimal = array.execute::<DecimalArray>(&mut execution)?;
        let decimal_precision = u32::from(decimal.precision());
        let decimal_scale = i32::from(decimal.scale());
        let length = decimal.len();
        let mask = decimal
            .as_ref()
            .validity()?
            .execute_mask(length, &mut execution)?;
        let (primitive_type, host_values) = match decimal.precision() {
            1..=18 => (
                VX_VELOX_PRIMITIVE_I64,
                normalized_decimal_values::<i64>(&decimal, &mask)?,
            ),
            19..=38 => (
                VX_VELOX_PRIMITIVE_I128,
                normalized_decimal_values::<i128>(&decimal, &mask)?,
            ),
            precision => {
                vortex_bail!("Vortex Velox visitor does not support decimal precision {precision}")
            }
        };
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = PrimitiveOwner::try_new(
            host_values,
            primitive_width(primitive_type)?,
            validity,
            length,
            retain_values,
        )?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes())?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            primitive_type,
            decimal_precision,
            decimal_scale,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let retain_values = memory_callbacks.is_some();
        let direct_bitpacked = array.as_opt::<BitPacked>().filter(|bitpacked| {
            array.dtype().as_ptype() == PType::I64 && bitpacked.patches().is_none()
        });
        let values_length =
            array
                .len()
                .checked_mul(array.dtype().element_size().ok_or_else(|| {
                    vortex_err!("Primitive visitor received a variable-width array")
                })?)
                .ok_or_else(|| vortex_err!("Primitive visitor value byte count overflow"))?;
        let values_allocation = values_length
            .checked_add(size_of::<u64>() - 1)
            .ok_or_else(|| vortex_err!("Primitive visitor value allocation overflow"))?
            / size_of::<u64>()
            * size_of::<u64>();
        let validity_allocation = if array.dtype().is_nullable() {
            array
                .len()
                .div_ceil(u64::BITS as usize)
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Primitive visitor validity allocation overflow"))?
        } else {
            0
        };
        let peak_reservation =
            if direct_bitpacked.is_some() {
                values_allocation.checked_add(validity_allocation.checked_mul(2).ok_or_else(
                    || vortex_err!("Primitive visitor validity reservation overflow"),
                )?)
            } else {
                values_allocation
                    .checked_add(validity_allocation)
                    .and_then(|bytes| bytes.checked_mul(2))
            }
            .ok_or_else(|| vortex_err!("Primitive visitor memory reservation overflow"))?;
        let mut memory_reservation = match (memory_callbacks, peak_reservation) {
            (Some(callbacks), bytes) if bytes != 0 => {
                Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
            }
            _ => None,
        };

        let mut execution = session.create_execution_ctx();
        let (primitive_type, length, validity_kind, mut owner) = if let Some(bitpacked) =
            direct_bitpacked
        {
            let primitive_type = primitive_type_id(array.dtype().as_ptype());
            let length = array.len();
            let mask = bitpacked.validity()?.execute_mask(length, &mut execution)?;
            let (validity_kind, validity) = exported_validity(array.dtype().is_nullable(), mask);
            let owner = PrimitiveOwner::try_new_bitpacked_i64(bitpacked, validity)?;
            (primitive_type, length, validity_kind, owner)
        } else {
            let Canonical::Primitive(primitive) = array.execute::<Canonical>(&mut execution)?
            else {
                vortex_bail!("Primitive visitor received a non-primitive array");
            };
            let primitive_type = primitive_type_id(primitive.ptype());
            let length = primitive.len();
            let mask = primitive.validity()?.execute_mask(length, &mut execution)?;
            let (validity_kind, validity) =
                exported_validity(primitive.dtype().is_nullable(), mask);
            let host_values = primitive.into_data_parts().buffer.try_into_host_sync()?;
            let owner = PrimitiveOwner::try_new(
                host_values,
                primitive_width(primitive_type)?,
                validity,
                length,
                retain_values,
            )?;
            (primitive_type, length, validity_kind, owner)
        };
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes())?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            primitive_type,
            decimal_precision: 0,
            decimal_scale: 0,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn view(&self, offset: usize, length: usize) -> VortexResult<vx_velox_primitive_view> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let width = primitive_width(self.primitive_type)?;
        let byte_offset = offset
            .checked_mul(width)
            .ok_or_else(|| vortex_err!("Vortex Velox value offset overflow"))?;
        let values_length = length
            .checked_mul(width)
            .ok_or_else(|| vortex_err!("Vortex Velox value length overflow"))?;
        let values = if values_length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked export range lies within the retained primitive buffer.
            unsafe { self.owner.values().add(byte_offset) }
        };
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Primitive validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        Ok(vx_velox_primitive_view {
            struct_size: size_of::<vx_velox_primitive_view>(),
            primitive_type: self.primitive_type,
            decimal_precision: self.decimal_precision,
            decimal_scale: self.decimal_scale,
            length,
            values,
            values_length,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_primitive_owner),
                release: Some(release_primitive_owner),
                retained_bytes: self.owner.retained_bytes(),
            },
            values_alignment: pointer_alignment(values),
            validity_alignment: pointer_alignment(validity),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let view = self.view(offset, length)?;
        let callback = visitor
            .visit_primitive
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a primitive callback"))?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct BoolExport {
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<BoolOwner>,
}

impl BoolExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let Canonical::Bool(boolean) = array.execute::<Canonical>(&mut execution)? else {
            vortex_bail!("Boolean visitor received a non-Boolean array");
        };
        let length = boolean.len();
        let mask = boolean.validity()?.execute_mask(length, &mut execution)?;
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = BoolOwner::try_new(boolean.into_bit_buffer(), validity)?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes)?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (values, values_length, values_bit_offset) =
            packed_bits_window(&self.owner.values, offset, length)?;
        let (validity, validity_length, validity_bit_offset) = match &self.owner.validity {
            Some(validity) => packed_bits_window(validity, offset, length)?,
            None => (ptr::null(), 0, 0),
        };
        let view = vx_velox_bool_view {
            struct_size: size_of::<vx_velox_bool_view>(),
            length,
            values,
            values_length,
            values_bit_offset,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_bool_owner),
                release: Some(release_bool_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            values_alignment: pointer_alignment(values),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_bool
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a Boolean callback"))?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

fn packed_bits_window(
    bits: &PackedBits,
    offset: usize,
    length: usize,
) -> VortexResult<(*const u8, usize, usize)> {
    if length == 0 {
        return Ok((ptr::null(), 0, 0));
    }
    let word_bits = u64::BITS as usize;
    let byte_offset = offset / word_bits * size_of::<u64>();
    let bit_offset = offset % word_bits;
    let required_length = bit_offset
        .checked_add(length)
        .ok_or_else(|| vortex_err!("Packed Boolean window overflow"))?
        .div_ceil(u8::BITS as usize);
    let byte_length = bits
        .len()
        .checked_sub(byte_offset)
        .ok_or_else(|| vortex_err!("Packed Boolean window exceeds its owner"))?;
    if byte_length < required_length {
        vortex_bail!("Packed Boolean window exceeds its readable bytes");
    }
    // SAFETY: The caller validated the logical window against the owner length.
    let values = unsafe { bits.as_ptr().add(byte_offset) };
    Ok((values, byte_length, bit_offset))
}

struct VarBinExport {
    kind: vx_velox_varbin_kind,
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<VarBinOwner>,
}

impl VarBinExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let mut memory_reservation = match memory_callbacks {
            Some(callbacks) => Some(ArrowMemoryReservation::try_new(
                callbacks,
                conservative_export_reservation(&array, &mut execution)?,
            )?),
            None => None,
        };
        let is_nullable = array.dtype().is_nullable();
        let varbin = array.execute::<VarBinViewArray>(&mut execution)?;
        let length = varbin.len();
        let parts = varbin.into_data_parts();
        let kind = match parts.dtype {
            DType::Utf8(_) => VX_VELOX_VARBIN_UTF8,
            DType::Binary(_) => VX_VELOX_VARBIN_BINARY,
            dtype => vortex_bail!("Variable-width visitor received an invalid type: {dtype}"),
        };
        let mask = parts.validity.execute_mask(length, &mut execution)?;
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let mut owner = VarBinOwner::try_new(parts.views, parts.buffers, validity, length)?;
        if let Some(mut reservation) = memory_reservation.take() {
            reservation.reconcile(owner.retained_bytes)?;
            owner.set_memory_reservation(reservation);
        }
        Ok(Self {
            kind,
            length,
            validity_kind,
            owner: Arc::new(owner),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let view_byte_offset = offset
            .checked_mul(size_of::<vx_velox_binary_view>())
            .ok_or_else(|| vortex_err!("Vortex string view offset overflow"))?;
        let views_length = length
            .checked_mul(size_of::<vx_velox_binary_view>())
            .ok_or_else(|| vortex_err!("Vortex string view length overflow"))?;
        let views = if views_length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked export range lies within the retained view buffer.
            unsafe {
                self.owner
                    .views
                    .as_ptr()
                    .cast::<u8>()
                    .add(view_byte_offset)
                    .cast()
            }
        };
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("String validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let data_buffers = if self.owner.descriptors.is_empty() {
            ptr::null()
        } else {
            self.owner.descriptors.as_ptr()
        };
        let view = vx_velox_varbin_view {
            struct_size: size_of::<vx_velox_varbin_view>(),
            kind: self.kind,
            length,
            views,
            views_length,
            data_buffers,
            data_buffer_count: self.owner.descriptors.len(),
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_varbin_owner),
                release: Some(release_varbin_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            views_alignment: pointer_alignment(views.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor.visit_varbin.ok_or_else(|| {
            vortex_err!("Vortex Velox visitor requires a variable-width callback")
        })?;
        // SAFETY: The cursor retains every buffer in the view through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct DictionaryExport {
    codes: PrimitiveExport,
    values_length: usize,
    values: Box<vx_velox_export_cursor>,
}

impl DictionaryExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let dictionary = array.as_::<Dict>();
        let values = dictionary.values().clone();
        Ok(Self {
            codes: PrimitiveExport::try_new(dictionary.codes().clone(), session, memory_callbacks)?,
            values_length: values.len(),
            values: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new_canonical(values, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let codes = self.codes.view(offset, length)?;
        let view = vx_velox_dictionary_view {
            struct_size: size_of::<vx_velox_dictionary_view>(),
            length,
            codes,
            values: &raw const *self.values,
            values_length: self.values_length,
        };
        let callback = visitor
            .visit_dictionary
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a dictionary callback"))?;
        // SAFETY: The borrowed child cursor and every code buffer remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct ConstantExport {
    length: usize,
    value: Box<vx_velox_export_cursor>,
}

impl ConstantExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let length = array.len();
        let scalar = array.as_::<Constant>().scalar().clone();
        let value = ConstantArray::new(scalar, 1).into_array();
        Ok(Self {
            length,
            value: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new_canonical(value, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let view = vx_velox_constant_view {
            struct_size: size_of::<vx_velox_constant_view>(),
            length,
            value: &raw const *self.value,
        };
        let callback = visitor
            .visit_constant
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a constant callback"))?;
        // SAFETY: The borrowed child cursor remains live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct StructOwner {
    validity: Option<PackedBits>,
    retained_bytes: usize,
    _memory_reservation: Option<ArrowMemoryReservation>,
}

struct StructExport {
    length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<StructOwner>,
    fields: Box<[vx_velox_export_cursor]>,
    field_pointers: Box<[*const vx_velox_export_cursor]>,
}

impl StructExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let is_nullable = array.dtype().is_nullable();
        let mut execution = session.create_execution_ctx();
        let struct_array = array.execute::<StructArray>(&mut execution)?;
        let length = struct_array.len();
        let mask = struct_array
            .struct_validity()
            .execute_mask(length, &mut execution)?;
        let validity_reservation = if matches!(mask, Mask::Values(_)) {
            length
                .div_ceil(u64::BITS as usize)
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| vortex_err!("Struct validity reservation overflow"))?
        } else {
            0
        };
        let mut memory_reservation = match (memory_callbacks, validity_reservation) {
            (Some(callbacks), bytes) if bytes != 0 => {
                Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
            }
            _ => None,
        };
        let (validity_kind, validity) = exported_validity(is_nullable, mask);
        let (validity, retained_bytes) = retain_validity(validity, length)?;
        if let Some(reservation) = memory_reservation.as_mut() {
            reservation.reconcile(retained_bytes)?;
        }
        let owner = Arc::new(StructOwner {
            validity,
            retained_bytes,
            _memory_reservation: memory_reservation,
        });
        let fields = struct_array
            .iter_unmasked_fields()
            .map(|field| {
                Ok(vx_velox_export_cursor {
                    export: CursorExport::try_new(field.clone(), session, memory_callbacks)?,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();
        let field_pointers = fields
            .iter()
            .map(|field| field as *const vx_velox_export_cursor)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            length,
            validity_kind,
            owner,
            fields,
            field_pointers,
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Struct validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let view = vx_velox_struct_view {
            struct_size: size_of::<vx_velox_struct_view>(),
            length,
            offset,
            fields: if self.field_pointers.is_empty() {
                ptr::null()
            } else {
                self.field_pointers.as_ptr()
            },
            field_count: self.fields.len(),
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_struct_owner),
                release: Some(release_struct_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_struct
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a struct callback"))?;
        // SAFETY: The borrowed field cursors and parent validity remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct ListOwner {
    offsets: Box<[i32]>,
    sizes: Box<[i32]>,
    validity: Option<PackedBits>,
    retained_bytes: usize,
    _memory_reservation: Option<ArrowMemoryReservation>,
}

struct ListMetadata {
    length: usize,
    elements_length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
}

struct ListExport {
    length: usize,
    elements_length: usize,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
    elements: Box<vx_velox_export_cursor>,
}

fn list_metadata_value<T>(value: T, name: &str) -> VortexResult<i32>
where
    T: Copy + std::fmt::Display,
    i32: TryFrom<T>,
{
    i32::try_from(value)
        .map_err(|_| vortex_err!("Vortex list {name} exceeds the Velox vector limit: {value}"))
}

fn list_metadata_values(values: PrimitiveArray, name: &str) -> VortexResult<Box<[i32]>> {
    let values = values.reinterpret_cast(values.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(values.ptype(), |P| {
        values
            .as_slice::<P>()
            .iter()
            .map(|&value| list_metadata_value(value, name))
            .collect::<VortexResult<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    })
}

fn prepare_list_metadata(
    list: &ListViewArray,
    session: &vortex::session::VortexSession,
    memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
) -> VortexResult<ListMetadata> {
    let is_nullable = list.dtype().is_nullable();
    let mut execution = session.create_execution_ctx();
    let length = list.len();
    let elements_length = list.elements().len();
    if elements_length > i32::MAX as usize {
        vortex_bail!("Vortex list elements exceed the Velox vector limit: {elements_length}");
    }
    let mask = list
        .listview_validity()
        .execute_mask(length, &mut execution)?;
    let validity_reservation = if matches!(mask, Mask::Values(_)) {
        length
            .div_ceil(u64::BITS as usize)
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| vortex_err!("List validity reservation overflow"))?
    } else {
        0
    };
    let metadata_reservation = length
        .checked_mul(2 * size_of::<i32>())
        .ok_or_else(|| vortex_err!("List metadata reservation overflow"))?;
    let reservation = metadata_reservation
        .checked_add(validity_reservation)
        .ok_or_else(|| vortex_err!("List retained byte count overflow"))?;
    let mut memory_reservation = match (memory_callbacks, reservation) {
        (Some(callbacks), bytes) if bytes != 0 => {
            Some(ArrowMemoryReservation::try_new(callbacks, bytes)?)
        }
        _ => None,
    };
    let offsets = list_metadata_values(
        list.offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut execution)?,
        "offset",
    )?;
    let sizes = list_metadata_values(
        list.sizes()
            .clone()
            .execute::<PrimitiveArray>(&mut execution)?,
        "size",
    )?;
    let (validity_kind, validity) = exported_validity(is_nullable, mask);
    let (validity, validity_allocation) = retain_validity(validity, length)?;
    let retained_bytes = size_of_val(offsets.as_ref())
        .checked_add(size_of_val(sizes.as_ref()))
        .and_then(|bytes| bytes.checked_add(validity_allocation))
        .ok_or_else(|| vortex_err!("List retained byte count overflow"))?;
    if let Some(reservation) = memory_reservation.as_mut() {
        reservation.reconcile(retained_bytes)?;
    }
    Ok(ListMetadata {
        length,
        elements_length,
        validity_kind,
        owner: Arc::new(ListOwner {
            offsets,
            sizes,
            validity,
            retained_bytes,
            _memory_reservation: memory_reservation,
        }),
    })
}

impl ListExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let list = array.execute::<ListViewArray>(&mut execution)?;
        let elements = list.elements().clone();
        let metadata = prepare_list_metadata(&list, session, memory_callbacks)?;
        Ok(Self {
            length: metadata.length,
            elements_length: metadata.elements_length,
            validity_kind: metadata.validity_kind,
            owner: metadata.owner,
            elements: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(elements, session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("List validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let offsets = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.offsets.as_ptr().add(offset) }
        };
        let sizes = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.sizes.as_ptr().add(offset) }
        };
        let view = vx_velox_list_view {
            struct_size: size_of::<vx_velox_list_view>(),
            length,
            offsets,
            sizes,
            elements: &raw const *self.elements,
            elements_length: self.elements_length,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_list_owner),
                release: Some(release_list_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            offsets_alignment: pointer_alignment(offsets.cast()),
            sizes_alignment: pointer_alignment(sizes.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_list
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a list callback"))?;
        // SAFETY: The borrowed element cursor and parent buffers remain live through this call.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

struct MapExport {
    length: usize,
    entries_length: usize,
    keys_sorted: bool,
    validity_kind: vx_velox_validity_kind,
    owner: Arc<ListOwner>,
    keys: Box<vx_velox_export_cursor>,
    values: Box<vx_velox_export_cursor>,
}

impl MapExport {
    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        let mut execution = session.create_execution_ctx();
        let map = array.execute::<MapArray>(&mut execution)?;
        let keys_sorted = map.keys_sorted();
        let entries = map.entries().clone().downcast::<ListView>();
        let entry_values = entries.elements().clone();
        let entry_struct = entry_values.execute::<StructArray>(&mut execution)?;
        let fields = entry_struct.iter_unmasked_fields().collect::<Vec<_>>();
        if fields.len() != 2 {
            vortex_bail!(
                "Vortex map entries require two fields, got {}",
                fields.len()
            );
        }
        let metadata = prepare_list_metadata(&entries, session, memory_callbacks)?;
        Ok(Self {
            length: metadata.length,
            entries_length: metadata.elements_length,
            keys_sorted,
            validity_kind: metadata.validity_kind,
            owner: metadata.owner,
            keys: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(fields[0].clone(), session, memory_callbacks)?,
            }),
            values: Box::new(vx_velox_export_cursor {
                export: CursorExport::try_new(fields[1].clone(), session, memory_callbacks)?,
            }),
        })
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| vortex_err!("Vortex Velox export range overflow"))?;
        if end > self.length {
            vortex_bail!(
                "Vortex Velox export range is out of bounds: {offset}..{end}, array length {}",
                self.length
            );
        }
        let (validity, validity_length, validity_bit_offset) =
            if self.validity_kind == VX_VELOX_VALIDITY_BITMAP {
                packed_bits_window(
                    self.owner
                        .validity
                        .as_ref()
                        .ok_or_else(|| vortex_err!("Map validity bitmap is missing"))?,
                    offset,
                    length,
                )?
            } else {
                (ptr::null(), 0, 0)
            };
        let offsets = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.offsets.as_ptr().add(offset) }
        };
        let sizes = if length == 0 {
            ptr::null()
        } else {
            // SAFETY: The checked range lies within the metadata arrays.
            unsafe { self.owner.sizes.as_ptr().add(offset) }
        };
        let view = vx_velox_map_view {
            struct_size: size_of::<vx_velox_map_view>(),
            length,
            offsets,
            sizes,
            keys: &raw const *self.keys,
            values: &raw const *self.values,
            entries_length: self.entries_length,
            keys_sorted: self.keys_sorted,
            validity_kind: self.validity_kind,
            validity,
            validity_length,
            validity_bit_offset,
            buffers: vx_velox_buffer_owner {
                struct_size: size_of::<vx_velox_buffer_owner>(),
                owner: Arc::as_ptr(&self.owner).cast(),
                retain: Some(retain_list_owner),
                release: Some(release_list_owner),
                retained_bytes: self.owner.retained_bytes,
            },
            offsets_alignment: pointer_alignment(offsets.cast()),
            sizes_alignment: pointer_alignment(sizes.cast()),
            validity_alignment: pointer_alignment(validity),
        };
        let callback = visitor
            .visit_map
            .ok_or_else(|| vortex_err!("Vortex Velox visitor requires a map callback"))?;
        // SAFETY: The borrowed child cursors and parent buffers remain live through this callback.
        let status = unsafe { callback(visitor.context, &raw const view) };
        if status != 0 {
            vortex_bail!("{}", callback_error(visitor, status));
        }
        Ok(())
    }
}

impl CursorExport {
    fn date_storage(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
    ) -> VortexResult<Option<vortex::array::ArrayRef>> {
        let DType::Extension(ext_dtype) = array.dtype() else {
            return Ok(None);
        };
        let Some(time_unit) = ext_dtype.metadata_opt::<Date>() else {
            return Ok(None);
        };
        if *time_unit != TimeUnit::Days {
            vortex_bail!(
                "Vortex Velox visitor does not support date unit {time_unit}; Velox DATE uses days"
            );
        }

        if let Some(extension) = array.as_opt::<Extension>() {
            return Ok(Some(extension.storage_array().clone()));
        }
        let mut execution = session.create_execution_ctx();
        let extension = array.execute::<ExtensionArray>(&mut execution)?;
        Ok(Some(extension.storage_array().clone()))
    }

    fn try_new_canonical(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        if matches!(array.dtype(), DType::Map(..)) {
            Ok(Self::Map(MapExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::List(..)) {
            Ok(Self::List(ListExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Struct(..)) {
            Ok(Self::Struct(StructExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Decimal(..)) {
            Ok(Self::Primitive(PrimitiveExport::try_new_decimal(
                array,
                session,
                memory_callbacks,
            )?))
        } else if let Some(storage) = Self::date_storage(array.clone(), session)? {
            Ok(Self::Primitive(PrimitiveExport::try_new(
                storage,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Bool(_)) {
            Ok(Self::Bool(BoolExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_)) {
            Ok(Self::VarBin(VarBinExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else {
            Ok(Self::Primitive(PrimitiveExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        }
    }

    fn try_new(
        array: vortex::array::ArrayRef,
        session: &vortex::session::VortexSession,
        memory_callbacks: Option<vx_velox_arrow_memory_callbacks>,
    ) -> VortexResult<Self> {
        if array.is::<Dict>() {
            Ok(Self::Dictionary(DictionaryExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else if array.is::<Constant>() {
            Ok(Self::Constant(ConstantExport::try_new(
                array,
                session,
                memory_callbacks,
            )?))
        } else {
            Self::try_new_canonical(array, session, memory_callbacks)
        }
    }

    fn visit(&self, offset: usize, length: usize, visitor: &vx_velox_visitor) -> VortexResult<()> {
        match self {
            Self::Primitive(export) => export.visit(offset, length, visitor),
            Self::Bool(export) => export.visit(offset, length, visitor),
            Self::VarBin(export) => export.visit(offset, length, visitor),
            Self::Dictionary(export) => export.visit(offset, length, visitor),
            Self::Constant(export) => export.visit(offset, length, visitor),
            Self::Struct(export) => export.visit(offset, length, visitor),
            Self::List(export) => export.visit(offset, length, visitor),
            Self::Map(export) => export.visit(offset, length, visitor),
        }
    }
}

fn exported_validity(
    is_nullable: bool,
    mask: Mask,
) -> (vx_velox_validity_kind, Option<vortex::buffer::BitBuffer>) {
    if !is_nullable {
        return (VX_VELOX_VALIDITY_NON_NULLABLE, None);
    }
    match mask {
        Mask::AllTrue(_) => (VX_VELOX_VALIDITY_ALL_VALID, None),
        Mask::AllFalse(_) => (VX_VELOX_VALIDITY_ALL_INVALID, None),
        Mask::Values(values) => (VX_VELOX_VALIDITY_BITMAP, Some(values.bit_buffer().clone())),
    }
}

unsafe extern "C" fn retain_primitive_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<PrimitiveOwner>()) };
}

unsafe extern "C" fn release_primitive_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<PrimitiveOwner>()) });
}

unsafe extern "C" fn retain_bool_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<BoolOwner>()) };
}

unsafe extern "C" fn release_bool_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<BoolOwner>()) });
}

unsafe extern "C" fn retain_varbin_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<VarBinOwner>()) };
}

unsafe extern "C" fn release_varbin_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<VarBinOwner>()) });
}

unsafe extern "C" fn retain_struct_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<StructOwner>()) };
}

unsafe extern "C" fn release_struct_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<StructOwner>()) });
}

unsafe extern "C" fn retain_list_owner(owner: *const c_void) {
    // SAFETY: The visitor receives a pointer from `Arc::as_ptr` while one strong reference lives.
    unsafe { Arc::increment_strong_count(owner.cast::<ListOwner>()) };
}

unsafe extern "C" fn release_list_owner(owner: *const c_void) {
    // SAFETY: Each release matches a prior retain of this `Arc` pointer.
    drop(unsafe { Arc::from_raw(owner.cast::<ListOwner>()) });
}

fn validate_visitor(visitor: &vx_velox_visitor) -> VortexResult<()> {
    if visitor.struct_size < size_of::<vx_velox_visitor>() {
        vortex_bail!(
            "Vortex Velox visitor structure is too small: expected at least {}, got {}",
            size_of::<vx_velox_visitor>(),
            visitor.struct_size
        );
    }
    if visitor.abi_version != crate::VX_VELOX_ABI_VERSION {
        vortex_bail!(
            "Unsupported Vortex Velox ABI version: expected {}, got {}",
            crate::VX_VELOX_ABI_VERSION,
            visitor.abi_version
        );
    }
    Ok(())
}

fn callback_error(visitor: &vx_velox_visitor, status: i32) -> String {
    let Some(last_error) = visitor.last_error else {
        return format!("Velox visitor failed with status {status}");
    };
    // SAFETY: The callback contract returns null or a valid null-terminated string.
    let message = unsafe { last_error(visitor.context) };
    if message.is_null() {
        return format!("Velox visitor failed with status {status}");
    }
    // SAFETY: The callback keeps the string valid until the next callback.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn selected_array(
    array: &vortex::array::ArrayRef,
    request: &vx_velox_visit_request,
) -> VortexResult<vortex::array::ArrayRef> {
    if request.rows.is_null() {
        if request.row_count != 0 {
            vortex_bail!("A null visitor row pointer requires a zero row count");
        }
        return Ok(array.clone());
    }
    // SAFETY: The caller supplies `row_count` readable positions.
    let rows = unsafe { slice::from_raw_parts(request.rows, request.row_count) };
    let mut previous = None;
    for row in rows {
        let position = usize::try_from(*row)
            .map_err(|_| vortex_err!("Visitor row does not fit usize: {}", row))?;
        if position >= array.len() {
            vortex_bail!(
                "Visitor row is out of bounds: row {}, array length {}",
                row,
                array.len()
            );
        }
        if previous.is_some_and(|previous| previous >= *row) {
            vortex_bail!("Visitor rows must be unique and increasing");
        }
        previous = Some(*row);
    }
    let dense = rows.len() == array.len()
        && rows
            .iter()
            .enumerate()
            .all(|(position, row)| *row == position as u64);
    if dense {
        return Ok(array.clone());
    }
    array.take(PrimitiveArray::from_iter(rows.iter().copied()).into_array())
}

fn visit_array(
    array: vortex::array::ArrayRef,
    session: &vortex::session::VortexSession,
    visitor: &vx_velox_visitor,
) -> VortexResult<()> {
    let length = array.len();
    CursorExport::try_new_canonical(array, session, None)?.visit(0, length, visitor)
}

/// Create one export cursor for several Velox output windows.
///
/// # Safety
///
/// The session and array pointers must identify live handles.
/// The memory callbacks must identify a complete, thread-safe callback table.
/// `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_export_cursor_new(
    session: *const vx_session,
    array: *const vx_array,
    memory_callbacks: *const vx_velox_arrow_memory_callbacks,
    error_out: *mut *mut vx_error,
) -> *mut vx_velox_export_cursor {
    try_or(error_out, ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let memory_callbacks = unsafe { parse_memory_callbacks(memory_callbacks)? };
        Ok(Box::into_raw(Box::new(vx_velox_export_cursor {
            export: CursorExport::try_new(array.clone(), session, Some(memory_callbacks))?,
        })))
    })
}

/// Free one export cursor.
///
/// # Safety
///
/// The pointer must be null or come from [`vx_velox_export_cursor_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_export_cursor_free(cursor: *mut vx_velox_export_cursor) {
    if !cursor.is_null() {
        // SAFETY: The pointer came from `Box::into_raw` and is freed once.
        drop(unsafe { Box::from_raw(cursor) });
    }
}

/// Visit one contiguous range from a retained export cursor.
///
/// # Safety
///
/// The cursor and visitor pointers must remain live until this call returns.
/// Concurrent calls are valid. The caller must not free the cursor before all calls return.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_export_cursor_visit(
    cursor: *const vx_velox_export_cursor,
    offset: usize,
    length: usize,
    visitor: *const vx_velox_visitor,
    error_out: *mut *mut vx_error,
) -> i32 {
    try_or(error_out, 1, || {
        let cursor = unsafe {
            cursor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox export cursor must not be null"))?
        };
        let visitor = unsafe {
            visitor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visitor must not be null"))?
        };
        validate_visitor(visitor)?;
        cursor.export.visit(offset, length, visitor)?;
        Ok(0)
    })
}

/// Visit one Vortex array through host semantic callbacks.
///
/// The request selects source positions once. Callback block positions are compact and follow the
/// request order.
///
/// # Safety
///
/// Every pointer must be null or valid for the documented access. The array and session handles
/// must remain live until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_visit(
    session: *const vx_session,
    array: *const vx_array,
    request: *const vx_velox_visit_request,
    visitor: *const vx_velox_visitor,
    error_out: *mut *mut vx_error,
) -> i32 {
    try_or(error_out, 1, || {
        let session = unsafe { vx_session_ref(session)? };
        let array = unsafe { vx_array_ref(array)? };
        let request = unsafe {
            request
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visit request must not be null"))?
        };
        if request.struct_size < size_of::<vx_velox_visit_request>() {
            vortex_bail!(
                "Vortex Velox visit request is too small: expected at least {}, got {}",
                size_of::<vx_velox_visit_request>(),
                request.struct_size
            );
        }
        let visitor = unsafe {
            visitor
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox visitor must not be null"))?
        };
        validate_visitor(visitor)?;
        visit_array(selected_array(array, request)?, session, visitor)?;
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use std::mem::align_of;
    use std::ptr;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::ListViewArray;
    use vortex::array::arrays::MapArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::FieldNames;
    use vortex::dtype::MapDType;
    use vortex::dtype::Nullability;
    use vortex::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_error::vortex_ensure;
    use vortex_fastlanes::BitPackedData;
    use vortex_ffi::vx_array_new_with;
    use vortex_ffi::vx_session_free;
    use vortex_ffi::vx_session_new_with;

    use super::*;
    use crate::api::vx_velox_array_free;

    #[derive(Default)]
    struct TestMemory {
        retained_bytes: AtomicUsize,
    }

    unsafe extern "C" fn retain_test_memory(_context: *mut c_void) {}

    unsafe extern "C" fn release_test_memory(_context: *mut c_void) {}

    unsafe extern "C" fn reserve_test_memory(context: *mut c_void, bytes: usize) -> i32 {
        // SAFETY: The test context stays live through every callback.
        let memory = unsafe { &*context.cast::<TestMemory>() };
        memory.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
        0
    }

    unsafe extern "C" fn free_test_memory(context: *mut c_void, bytes: usize) {
        // SAFETY: The test context stays live through every callback.
        let memory = unsafe { &*context.cast::<TestMemory>() };
        memory.retained_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn test_memory_callbacks(memory: &mut TestMemory) -> vx_velox_arrow_memory_callbacks {
        vx_velox_arrow_memory_callbacks {
            struct_size: size_of::<vx_velox_arrow_memory_callbacks>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (memory as *mut TestMemory).cast(),
            retain_context: Some(retain_test_memory),
            release_context: Some(release_test_memory),
            report_allocation: Some(reserve_test_memory),
            report_free: Some(free_test_memory),
            last_error: None,
        }
    }

    #[rstest]
    #[case(PType::U8, VX_VELOX_PRIMITIVE_U8)]
    #[case(PType::U16, VX_VELOX_PRIMITIVE_U16)]
    #[case(PType::U32, VX_VELOX_PRIMITIVE_U32)]
    #[case(PType::U64, VX_VELOX_PRIMITIVE_U64)]
    #[case(PType::I8, VX_VELOX_PRIMITIVE_I8)]
    #[case(PType::I16, VX_VELOX_PRIMITIVE_I16)]
    #[case(PType::I32, VX_VELOX_PRIMITIVE_I32)]
    #[case(PType::I64, VX_VELOX_PRIMITIVE_I64)]
    #[case(PType::F16, VX_VELOX_PRIMITIVE_F16)]
    #[case(PType::F32, VX_VELOX_PRIMITIVE_F32)]
    #[case(PType::F64, VX_VELOX_PRIMITIVE_F64)]
    fn maps_primitive_types(#[case] input: PType, #[case] expected: vx_velox_primitive_type) {
        assert_eq!(primitive_type_id(input), expected);
    }

    #[test]
    fn date_days_use_i32_storage_and_millisecond_dates_are_rejected() -> VortexResult<()> {
        let session = vortex::session::VortexSession::empty();
        let days = TemporalArray::new_date(
            PrimitiveArray::from_option_iter([Some(-1_i32), None, Some(19_000)]).into_array(),
            TimeUnit::Days,
        )
        .into_array();
        let CursorExport::Primitive(export) =
            CursorExport::try_new_canonical(days, &session, None)?
        else {
            vortex_bail!("date visitor did not produce primitive storage");
        };
        assert_eq!(export.primitive_type, VX_VELOX_PRIMITIVE_I32);
        let view = export.view(0, 3)?;
        // SAFETY: The export owns three readable i32 values.
        let values = unsafe { slice::from_raw_parts(view.values.cast::<i32>(), 3) };
        assert_eq!(values, [-1, 0, 19_000]);
        assert_eq!(view.validity_kind, VX_VELOX_VALIDITY_BITMAP);

        let milliseconds = TemporalArray::new_date(
            PrimitiveArray::from_iter([86_400_000_i64]).into_array(),
            TimeUnit::Milliseconds,
        )
        .into_array();
        let error = match CursorExport::try_new_canonical(milliseconds, &session, None) {
            Ok(_) => vortex_bail!("millisecond date visitor unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Velox DATE uses days"));
        Ok(())
    }

    #[test]
    fn decimals_normalize_to_velox_storage_widths() -> VortexResult<()> {
        let session = vortex::session::VortexSession::empty();
        let short = DecimalArray::new(
            buffer![1_i8, -2, 3],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        )
        .into_array();
        let short = PrimitiveExport::try_new_decimal(short, &session, None)?;
        assert_eq!(short.primitive_type, VX_VELOX_PRIMITIVE_I64);
        let short_view = short.view(0, 3)?;
        assert_eq!(short_view.decimal_precision, 18);
        assert_eq!(short_view.decimal_scale, 2);
        // SAFETY: The export owns three readable i64 values.
        let short_values = unsafe { slice::from_raw_parts(short_view.values.cast::<i64>(), 3) };
        assert_eq!(short_values, [1, -2, 3]);

        let nullable_short = DecimalArray::new(
            buffer![1_i128, i128::MAX],
            DecimalDType::new(18, 2),
            Validity::from_iter([true, false]),
        )
        .into_array();
        let nullable_short = PrimitiveExport::try_new_decimal(nullable_short, &session, None)?;
        let nullable_short_view = nullable_short.view(0, 2)?;
        // SAFETY: The export owns two readable i64 values.
        let nullable_short_values =
            unsafe { slice::from_raw_parts(nullable_short_view.values.cast::<i64>(), 2) };
        assert_eq!(nullable_short_values, [1, 0]);
        assert_eq!(nullable_short_view.validity_kind, VX_VELOX_VALIDITY_BITMAP);

        let long = DecimalArray::new(
            buffer![1_i64, -2, 3],
            DecimalDType::new(30, 4),
            Validity::NonNullable,
        )
        .into_array();
        let long = PrimitiveExport::try_new_decimal(long, &session, None)?;
        assert_eq!(long.primitive_type, VX_VELOX_PRIMITIVE_I128);
        let long_view = long.view(0, 3)?;
        assert_eq!(long_view.decimal_precision, 30);
        assert_eq!(long_view.decimal_scale, 4);
        // SAFETY: The export owns three readable i128 values.
        let long_values = unsafe { slice::from_raw_parts(long_view.values.cast::<i128>(), 3) };
        assert_eq!(long_values, [1, -2, 3]);

        let unsupported = DecimalArray::new(
            buffer![1_i8],
            DecimalDType::new(39, 0),
            Validity::NonNullable,
        )
        .into_array();
        let error = match PrimitiveExport::try_new_decimal(unsupported, &session, None) {
            Ok(_) => vortex_bail!("precision 39 decimal visitor unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("decimal precision 39"));
        Ok(())
    }

    #[test]
    fn dictionary_export_preserves_code_width_and_nullable_children() -> VortexResult<()> {
        let session = vortex::session::VortexSession::empty();
        let code_cases: [(ArrayRef, vx_velox_primitive_type); 4] = [
            (buffer![0_u8, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U8),
            (buffer![0_u16, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U16),
            (buffer![0_u32, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U32),
            (buffer![0_u64, 1, 0].into_array(), VX_VELOX_PRIMITIVE_U64),
        ];
        for (codes, expected_type) in code_cases {
            let dictionary = DictArray::try_new(codes, buffer![10_i64, 20].into_array())?;
            let CursorExport::Dictionary(export) =
                CursorExport::try_new(dictionary.into_array(), &session, None)?
            else {
                vortex_bail!("dictionary export lost its outer encoding");
            };
            assert_eq!(export.codes.primitive_type, expected_type);
            assert_eq!(export.values_length, 2);
            assert!(matches!(export.values.export, CursorExport::Primitive(_)));
        }

        let codes = PrimitiveArray::from_option_iter([Some(0_u8), None, Some(1)]).into_array();
        let values = PrimitiveArray::from_option_iter([Some(10_i64), None]).into_array();
        let dictionary = DictArray::try_new(codes, values)?;
        let CursorExport::Dictionary(export) =
            CursorExport::try_new(dictionary.into_array(), &session, None)?
        else {
            vortex_bail!("nullable dictionary export lost its outer encoding");
        };
        assert_eq!(export.codes.validity_kind, VX_VELOX_VALIDITY_BITMAP);
        let CursorExport::Primitive(values) = &export.values.export else {
            vortex_bail!("nullable dictionary values lost their primitive representation");
        };
        assert_eq!(values.validity_kind, VX_VELOX_VALIDITY_BITMAP);
        Ok(())
    }

    #[test]
    fn constant_export_preserves_null_value() -> VortexResult<()> {
        let session = vortex::session::VortexSession::empty();
        let constant = ConstantArray::new(Scalar::null_native::<i64>(), 10).into_array();
        let CursorExport::Constant(export) = CursorExport::try_new(constant, &session, None)?
        else {
            vortex_bail!("constant export lost its outer encoding");
        };
        assert_eq!(export.length, 10);
        let CursorExport::Primitive(value) = &export.value.export else {
            vortex_bail!("null constant lost its primitive representation");
        };
        assert_eq!(value.length, 1);
        assert_eq!(value.validity_kind, VX_VELOX_VALIDITY_ALL_INVALID);
        Ok(())
    }

    #[test]
    fn struct_export_preserves_children_and_nonzero_window() -> VortexResult<()> {
        #[derive(Default)]
        struct StructCapture {
            length: usize,
            offset: usize,
            fields: *const *const vx_velox_export_cursor,
            field_count: usize,
            validity: *const u8,
            validity_bit_offset: usize,
            owner: Option<vx_velox_buffer_owner>,
        }

        unsafe extern "C" fn capture_struct(
            context: *mut c_void,
            view: *const vx_velox_struct_view,
        ) -> i32 {
            if context.is_null() || view.is_null() {
                return 1;
            }
            // SAFETY: The test passes pointers to live capture and view objects.
            let (capture, view) = unsafe { (&mut *context.cast::<StructCapture>(), &*view) };
            let Some(retain) = view.buffers.retain else {
                return 2;
            };
            // SAFETY: The visitor owner is live for the callback.
            unsafe { retain(view.buffers.owner) };
            capture.length = view.length;
            capture.offset = view.offset;
            capture.fields = view.fields;
            capture.field_count = view.field_count;
            capture.validity = view.validity;
            capture.validity_bit_offset = view.validity_bit_offset;
            capture.owner = Some(view.buffers);
            0
        }

        let session = vortex::session::VortexSession::empty();
        let length: usize = 130;
        let dictionary = DictArray::try_new(
            PrimitiveArray::from_iter((0..length).map(|index| [0_u8, 1][index % 2])).into_array(),
            buffer![10_i64, 20].into_array(),
        )?
        .into_array();
        let constant = ConstantArray::new(Scalar::from(7_i64), length).into_array();
        let parent_validity = Validity::from_iter((0..length).map(|index| index % 9 != 0));
        let struct_array = StructArray::new(
            FieldNames::from(["dictionary", "constant"]),
            [dictionary, constant],
            length,
            parent_validity,
        )
        .into_array();
        let CursorExport::Struct(export) = CursorExport::try_new(struct_array, &session, None)?
        else {
            vortex_bail!("struct export lost its outer encoding");
        };
        assert!(matches!(
            export.fields[0].export,
            CursorExport::Dictionary(_)
        ));
        assert!(matches!(export.fields[1].export, CursorExport::Constant(_)));

        let mut capture = StructCapture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: Some(capture_struct),
            visit_list: None,
            visit_map: None,
        };
        export.visit(65, 63, &visitor)?;
        assert_eq!(capture.length, 63);
        assert_eq!(capture.offset, 65);
        assert_eq!(capture.field_count, 2);
        assert_eq!(capture.validity_bit_offset, 1);
        // SAFETY: The export retains both field cursors until it is dropped below.
        assert_eq!(unsafe { *capture.fields }, &raw const export.fields[0]);
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("struct callback returned no validity owner"))?;
        drop(export);
        // SAFETY: The callback retained the parent owner before the cursor was dropped.
        assert!(
            unsafe {
                *capture
                    .validity
                    .add(capture.validity_bit_offset / u8::BITS as usize)
            } != 0
        );
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("struct owner returned no release callback"))?;
        // SAFETY: This release matches the callback retain above.
        unsafe { release(owner.owner) };
        Ok(())
    }

    #[test]
    fn list_export_preserves_elements_window_and_accounting() -> VortexResult<()> {
        #[derive(Default)]
        struct ListCapture {
            length: usize,
            offsets: *const i32,
            sizes: *const i32,
            elements_length: usize,
            validity: *const u8,
            validity_bit_offset: usize,
            owner: Option<vx_velox_buffer_owner>,
        }

        unsafe extern "C" fn capture_list(
            context: *mut c_void,
            view: *const vx_velox_list_view,
        ) -> i32 {
            if context.is_null() || view.is_null() {
                return 1;
            }
            // SAFETY: The test passes pointers to live capture and view objects.
            let (capture, view) = unsafe { (&mut *context.cast::<ListCapture>(), &*view) };
            let Some(retain) = view.buffers.retain else {
                return 2;
            };
            // SAFETY: The visitor owner is live for the callback.
            unsafe { retain(view.buffers.owner) };
            capture.length = view.length;
            capture.offsets = view.offsets;
            capture.sizes = view.sizes;
            capture.elements_length = view.elements_length;
            capture.validity = view.validity;
            capture.validity_bit_offset = view.validity_bit_offset;
            capture.owner = Some(view.buffers);
            0
        }

        let session = vortex::session::VortexSession::empty();
        let length = 130;
        let elements = DictArray::try_new(
            buffer![0_u8, 1, 0, 1, 0, 1].into_array(),
            PrimitiveArray::from_option_iter([Some(10_i64), None]).into_array(),
        )?
        .into_array();
        let offsets = PrimitiveArray::from_iter((0..length).map(|index| [0_u32, 2, 4][index % 3]));
        let sizes =
            PrimitiveArray::from_iter((0..length).map(|index| if index % 10 == 0 { 0 } else { 2 }));
        let validity = Validity::from_iter((0..length).map(|index| index % 9 != 0));
        let list = ListViewArray::new(elements, offsets.into_array(), sizes.into_array(), validity)
            .into_array();
        let mut memory = TestMemory::default();
        let CursorExport::List(export) =
            CursorExport::try_new(list, &session, Some(test_memory_callbacks(&mut memory)))?
        else {
            vortex_bail!("list export lost its outer encoding");
        };
        assert!(matches!(
            export.elements.export,
            CursorExport::Dictionary(_)
        ));
        let expected_parent_bytes =
            length * 2 * size_of::<i32>() + length.div_ceil(u64::BITS as usize) * size_of::<u64>();
        assert_eq!(export.owner.retained_bytes, expected_parent_bytes);

        let mut capture = ListCapture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: Some(capture_list),
            visit_map: None,
        };
        export.visit(65, 63, &visitor)?;
        assert_eq!(capture.length, 63);
        assert_eq!(capture.elements_length, 6);
        assert_eq!(capture.validity_bit_offset, 1);
        // SAFETY: The retained owner keeps both metadata arrays live.
        assert_eq!(unsafe { *capture.offsets }, 4);
        // SAFETY: The retained owner keeps both metadata arrays live.
        assert_eq!(unsafe { *capture.sizes }, 2);
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("list callback returned no owner"))?;
        drop(export);
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            expected_parent_bytes
        );
        // SAFETY: The callback retained the owner before the export was dropped.
        assert_eq!(unsafe { *capture.offsets.add(1) }, 0);
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("list owner returned no release callback"))?;
        // SAFETY: This release matches the callback retain above.
        unsafe { release(owner.owner) };
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn map_export_preserves_children_window_and_accounting() -> VortexResult<()> {
        #[derive(Default)]
        struct MapCapture {
            length: usize,
            offsets: *const i32,
            sizes: *const i32,
            keys: *const vx_velox_export_cursor,
            values: *const vx_velox_export_cursor,
            entries_length: usize,
            keys_sorted: bool,
            validity_bit_offset: usize,
            owner: Option<vx_velox_buffer_owner>,
        }

        unsafe extern "C" fn capture_map(
            context: *mut c_void,
            view: *const vx_velox_map_view,
        ) -> i32 {
            if context.is_null() || view.is_null() {
                return 1;
            }
            // SAFETY: The test passes pointers to live capture and view objects.
            let (capture, view) = unsafe { (&mut *context.cast::<MapCapture>(), &*view) };
            let Some(retain) = view.buffers.retain else {
                return 2;
            };
            // SAFETY: The visitor owner is live for the callback.
            unsafe { retain(view.buffers.owner) };
            capture.length = view.length;
            capture.offsets = view.offsets;
            capture.sizes = view.sizes;
            capture.keys = view.keys;
            capture.values = view.values;
            capture.entries_length = view.entries_length;
            capture.keys_sorted = view.keys_sorted;
            capture.validity_bit_offset = view.validity_bit_offset;
            capture.owner = Some(view.buffers);
            0
        }

        let session = vortex::session::VortexSession::empty();
        let keys = DictArray::try_new(
            buffer![0_u8, 1, 0, 1, 0, 1].into_array(),
            buffer![10_i64, 20].into_array(),
        )?
        .into_array();
        let values = ConstantArray::new(Scalar::from(7_i64), 6).into_array();
        let entries = StructArray::new(
            FieldNames::from(["key", "value"]),
            [keys, values],
            6,
            Validity::NonNullable,
        )
        .into_array();
        let entry_lists = ListViewArray::new(
            entries,
            buffer![0_u32, 2, 4].into_array(),
            buffer![2_u32, 2, 2].into_array(),
            Validity::from_iter([true, false, true]),
        );
        let map_dtype = MapDType::try_new(
            DType::Primitive(PType::I64, Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::NonNullable),
            true,
        )?;
        let map = MapArray::try_new(map_dtype, entry_lists)?.into_array();
        let mut memory = TestMemory::default();
        let CursorExport::Map(export) =
            CursorExport::try_new(map, &session, Some(test_memory_callbacks(&mut memory)))?
        else {
            vortex_bail!("map export lost its outer encoding");
        };
        assert!(matches!(export.keys.export, CursorExport::Dictionary(_)));
        assert!(matches!(export.values.export, CursorExport::Constant(_)));
        let expected_parent_bytes = 3 * 2 * size_of::<i32>() + size_of::<u64>();
        assert_eq!(export.owner.retained_bytes, expected_parent_bytes);

        let mut capture = MapCapture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: Some(capture_map),
        };
        export.visit(1, 2, &visitor)?;
        assert_eq!(capture.length, 2);
        assert_eq!(capture.entries_length, 6);
        assert!(capture.keys_sorted);
        assert_eq!(capture.validity_bit_offset, 1);
        assert_eq!(capture.keys, &raw const *export.keys);
        assert_eq!(capture.values, &raw const *export.values);
        // SAFETY: The retained owner keeps both metadata arrays live.
        assert_eq!(unsafe { *capture.offsets }, 2);
        // SAFETY: The retained owner keeps both metadata arrays live.
        assert_eq!(unsafe { *capture.sizes }, 2);
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("map callback returned no owner"))?;
        drop(export);
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            expected_parent_bytes
        );
        // SAFETY: The callback retained the owner before the export was dropped.
        assert_eq!(unsafe { *capture.offsets.add(1) }, 4);
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("map owner returned no release callback"))?;
        // SAFETY: This release matches the callback retain above.
        unsafe { release(owner.owner) };
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[derive(Default)]
    struct Capture {
        primitive_type: Option<vx_velox_primitive_type>,
        length: usize,
        values: *const u8,
        values_length: usize,
        values_alignment: usize,
        validity: *const u8,
        validity_length: usize,
        validity_bit_offset: usize,
        validity_alignment: usize,
        retained_bytes: usize,
        validity_kind: Option<vx_velox_validity_kind>,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_primitive(
        context: *mut c_void,
        view: *const vx_velox_primitive_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live `Capture` and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<Capture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.primitive_type = Some(view.primitive_type);
        capture.length = view.length;
        capture.values = view.values;
        capture.values_length = view.values_length;
        capture.values_alignment = view.values_alignment;
        capture.validity = view.validity;
        capture.validity_length = view.validity_length;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.validity_alignment = view.validity_alignment;
        capture.retained_bytes = view.buffers.retained_bytes;
        capture.validity_kind = Some(view.validity_kind);
        capture.owner = Some(view.buffers);
        0
    }

    fn release_capture(capture: &Capture) -> VortexResult<()> {
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        // SAFETY: This release matches the retain in `capture_primitive`.
        unsafe { release(owner.owner) };
        Ok(())
    }

    #[derive(Default)]
    struct VarBinCapture {
        struct_size: usize,
        kind: Option<vx_velox_varbin_kind>,
        length: usize,
        views: *const vx_velox_binary_view,
        views_length: usize,
        views_alignment: usize,
        data_buffers: *const vx_velox_byte_buffer_view,
        data_buffer_count: usize,
        validity: *const u8,
        validity_length: usize,
        validity_bit_offset: usize,
        validity_alignment: usize,
        validity_kind: Option<vx_velox_validity_kind>,
        retained_bytes: usize,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_varbin(
        context: *mut c_void,
        view: *const vx_velox_varbin_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live capture and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<VarBinCapture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.struct_size = view.struct_size;
        capture.kind = Some(view.kind);
        capture.length = view.length;
        capture.views = view.views;
        capture.views_length = view.views_length;
        capture.views_alignment = view.views_alignment;
        capture.data_buffers = view.data_buffers;
        capture.data_buffer_count = view.data_buffer_count;
        capture.validity = view.validity;
        capture.validity_length = view.validity_length;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.validity_alignment = view.validity_alignment;
        capture.validity_kind = Some(view.validity_kind);
        capture.retained_bytes = view.buffers.retained_bytes;
        capture.owner = Some(view.buffers);
        0
    }

    fn release_varbin_capture(capture: &VarBinCapture) -> VortexResult<()> {
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained string owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("string owner did not return a release callback"))?;
        // SAFETY: This release matches the retain in `capture_varbin`.
        unsafe { release(owner.owner) };
        Ok(())
    }

    #[derive(Default)]
    struct BoolCapture {
        length: usize,
        values: *const u8,
        values_bit_offset: usize,
        validity: *const u8,
        validity_bit_offset: usize,
        validity_kind: Option<vx_velox_validity_kind>,
        retained_bytes: usize,
        owner: Option<vx_velox_buffer_owner>,
    }

    unsafe extern "C" fn capture_bool(
        context: *mut c_void,
        view: *const vx_velox_bool_view,
    ) -> i32 {
        if context.is_null() || view.is_null() {
            return 1;
        }
        // SAFETY: The test passes pointers to live capture and view objects.
        let (capture, view) = unsafe { (&mut *context.cast::<BoolCapture>(), &*view) };
        let Some(retain) = view.buffers.retain else {
            return 2;
        };
        // SAFETY: The visitor owner is live for the callback.
        unsafe { retain(view.buffers.owner) };
        capture.length = view.length;
        capture.values = view.values;
        capture.values_bit_offset = view.values_bit_offset;
        capture.validity = view.validity;
        capture.validity_bit_offset = view.validity_bit_offset;
        capture.validity_kind = Some(view.validity_kind);
        capture.retained_bytes = view.buffers.retained_bytes;
        capture.owner = Some(view.buffers);
        0
    }

    fn release_bool_capture(capture: &BoolCapture) -> VortexResult<()> {
        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained Boolean owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("Boolean owner did not return a release callback"))?;
        // SAFETY: This release matches the retain in `capture_bool`.
        unsafe { release(owner.owner) };
        Ok(())
    }

    #[expect(
        clippy::host_endian_bytes,
        reason = "The Vortex binary-view fields use the host C ABI layout"
    )]
    unsafe fn captured_varbin_value(capture: &VarBinCapture, index: usize) -> Option<&[u8]> {
        if capture.validity_kind == Some(VX_VELOX_VALIDITY_BITMAP) {
            let bit_index = capture.validity_bit_offset + index;
            // SAFETY: The callback contract retains the bitmap for every captured row.
            let byte = unsafe { *capture.validity.add(bit_index / 8) };
            if byte & (1 << (bit_index % 8)) == 0 {
                return None;
            }
        }
        // SAFETY: The callback contract retains `length` readable views.
        let view = unsafe { &*capture.views.add(index) };
        let length = view.length as usize;
        const INLINE_LENGTH: usize = size_of::<vx_velox_binary_view>() - size_of::<u32>();
        if length <= INLINE_LENGTH {
            return Some(&view.data[..length]);
        }
        let buffer_index =
            u32::from_ne_bytes([view.data[4], view.data[5], view.data[6], view.data[7]]) as usize;
        let offset =
            u32::from_ne_bytes([view.data[8], view.data[9], view.data[10], view.data[11]]) as usize;
        // SAFETY: The callback contract retains all payload descriptors.
        let buffer = unsafe { &*capture.data_buffers.add(buffer_index) };
        // SAFETY: Canonical Vortex views contain validated payload ranges.
        Some(unsafe { slice::from_raw_parts(buffer.data.add(offset), length) })
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable), VX_VELOX_VARBIN_UTF8)]
    #[case(DType::Binary(Nullability::Nullable), VX_VELOX_VARBIN_BINARY)]
    fn varbin_cursor_retains_mixed_views_across_nonzero_window(
        #[case] dtype: DType,
        #[case] expected_kind: vx_velox_varbin_kind,
    ) -> VortexResult<()> {
        let utf8_expected: [Option<&[u8]>; 7] = [
            Some(b""),
            Some(b"a"),
            None,
            Some(b"abcdefghijkl"),
            Some(b"abcdefghijklm"),
            Some("vortex 🌀 outlined".as_bytes()),
            Some(b"tail"),
        ];
        let binary_expected: [Option<&[u8]>; 7] = [
            Some(b""),
            Some(b"\xff"),
            None,
            Some(b"abcdefghijkl"),
            Some(b"\x00abcdefghijklm"),
            Some(b"\xff\x00 binary outlined value"),
            Some(b"tail"),
        ];
        let expected = if matches!(dtype, DType::Utf8(_)) {
            utf8_expected
        } else {
            binary_expected
        };
        let session = vx_session_new_with(|session| session);
        let varbin = VarBinViewArray::from_iter(expected, dtype);
        let array = vx_array_new_with(varbin.into_array());
        let mut error = ptr::null_mut();
        let mut memory = TestMemory::default();
        let memory_callbacks = test_memory_callbacks(&mut memory);
        // SAFETY: The session and array handles remain live until cursor creation finishes.
        let cursor = unsafe {
            vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
        };
        vortex_ensure!(!cursor.is_null(), "string cursor creation failed");
        vortex_ensure!(error.is_null(), "string cursor returned an error");

        let mut capture = VarBinCapture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: Some(capture_varbin),
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        // SAFETY: The cursor and callback state remain live through the call.
        let status = unsafe {
            vx_velox_export_cursor_visit(cursor, 1, 5, &raw const visitor, &raw mut error)
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "string export window returned an error");
        assert_eq!(capture.struct_size, size_of::<vx_velox_varbin_view>());
        assert_eq!(capture.kind, Some(expected_kind));
        assert_eq!(capture.length, 5);
        assert_eq!(capture.views_length, 5 * size_of::<vx_velox_binary_view>());
        assert!(capture.views_alignment >= align_of::<vx_velox_binary_view>());
        assert_eq!(capture.views.addr() % align_of::<vx_velox_binary_view>(), 0);
        assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
        assert_eq!(capture.validity_bit_offset, 1);
        assert!(capture.validity_length >= 1);
        assert!(capture.validity_alignment >= align_of::<u64>());
        assert_eq!(capture.validity.addr() % align_of::<u64>(), 0);
        assert!(capture.data_buffer_count >= 1);
        assert!(!capture.data_buffers.is_null());
        assert_eq!(
            capture.retained_bytes,
            memory.retained_bytes.load(Ordering::Relaxed)
        );

        // SAFETY: Each owned handle is freed once. The callback retained the string owner.
        unsafe {
            vx_velox_export_cursor_free(cursor);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            capture.retained_bytes
        );
        for (index, expected) in expected[1..6].iter().enumerate() {
            // SAFETY: The retained owner keeps every captured pointer live.
            let actual = unsafe { captured_varbin_value(&capture, index) };
            assert_eq!(actual, *expected);
        }
        release_varbin_capture(&capture)?;
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn varbin_shared_buffers_compact_into_exact_owned_storage() -> VortexResult<()> {
        let length = 130_usize;
        let strings = VarBinViewArray::from_iter(
            (0..length).map(|index| {
                (index % 11 != 0).then(|| format!("outlined string value {index:03}"))
            }),
            DType::Utf8(Nullability::Nullable),
        );
        let parts = strings.into_data_parts();
        let views_length = parts.views.try_to_host_sync()?.len();
        let data_length = parts
            .buffers
            .iter()
            .map(|buffer| Ok(buffer.try_to_host_sync()?.len()))
            .sum::<VortexResult<usize>>()?;
        let descriptor_length = parts.buffers.len() * size_of::<vx_velox_byte_buffer_view>();
        let validity_length = length.div_ceil(u64::BITS as usize) * size_of::<u64>();
        let expected_retained = views_length + data_length + descriptor_length + validity_length;

        let retained_views = parts.views.clone();
        let retained_buffers = Arc::<[BufferHandle]>::clone(&parts.buffers);
        let mut execution = vortex::session::VortexSession::empty().create_execution_ctx();
        let mask = parts.validity.execute_mask(length, &mut execution)?;
        let (_, validity) = exported_validity(true, mask);
        let owner = VarBinOwner::try_new(parts.views, parts.buffers, validity, length)?;

        assert!(matches!(owner.views, RetainedViews::Compact(_)));
        assert!(
            owner
                ._data
                .iter()
                .all(|buffer| matches!(buffer, RetainedBytes::Compact(_)))
        );
        assert_eq!(owner.retained_bytes, expected_retained);
        drop(retained_views);
        drop(retained_buffers);
        Ok(())
    }

    #[test]
    fn retained_varbin_buffers_report_complete_unique_allocations() -> VortexResult<()> {
        let alignment = vortex::buffer::Alignment::new(256);
        let mut payload = BufferMut::<u8>::with_capacity_aligned(17, alignment);
        payload.extend(0..17);
        let expected_payload_allocation = payload.allocation_size();
        let (retained_payload, payload_allocation) =
            RetainedBytes::try_new(BufferHandle::new_host(payload.freeze()))?;
        assert!(matches!(retained_payload, RetainedBytes::Retained(_)));
        assert_eq!(payload_allocation, expected_payload_allocation);
        assert!(payload_allocation > 17);

        let mut views = BufferMut::<u8>::with_capacity_aligned(
            2 * size_of::<vx_velox_binary_view>(),
            alignment,
        );
        views.extend(std::iter::repeat_n(
            0,
            2 * size_of::<vx_velox_binary_view>(),
        ));
        let expected_views_allocation = views.allocation_size();
        let (retained_views, views_allocation) =
            RetainedViews::try_new(BufferHandle::new_host(views.freeze()))?;
        assert!(matches!(retained_views, RetainedViews::Retained(_)));
        assert_eq!(views_allocation, expected_views_allocation);
        assert!(views_allocation > 2 * size_of::<vx_velox_binary_view>());
        Ok(())
    }

    #[test]
    fn word_aligned_windows_rebase_validity_buffers() -> VortexResult<()> {
        let session = vortex::session::VortexSession::empty();
        let primitive = PrimitiveArray::from_option_iter(
            (0..130).map(|index| (index % 7 != 0).then_some(index as i64)),
        )
        .into_array();
        let primitive = PrimitiveExport::try_new(primitive, &session, None)?;
        let primitive_first = primitive.view(0, 64)?;
        let primitive_second = primitive.view(64, 64)?;
        assert_eq!(primitive_first.validity_bit_offset, 0);
        assert_eq!(primitive_second.validity_bit_offset, 0);
        // SAFETY: Both pointers lie in the retained validity allocation.
        assert_eq!(primitive_second.validity, unsafe {
            primitive_first.validity.add(size_of::<u64>())
        });

        let strings = VarBinViewArray::from_iter(
            (0..130).map(|index| (index % 11 != 0).then(|| format!("value-{index}"))),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let strings = VarBinExport::try_new(strings, &session, None)?;
        let mut first = VarBinCapture::default();
        let first_visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut first).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: Some(capture_varbin),
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        strings.visit(0, 64, &first_visitor)?;

        let mut second = VarBinCapture::default();
        let second_visitor = vx_velox_visitor {
            context: (&raw mut second).cast(),
            ..first_visitor
        };
        strings.visit(64, 64, &second_visitor)?;
        assert_eq!(first.validity_bit_offset, 0);
        assert_eq!(second.validity_bit_offset, 0);
        // SAFETY: Both pointers lie in the retained validity allocation.
        assert_eq!(second.validity, unsafe {
            first.validity.add(size_of::<u64>())
        });
        release_varbin_capture(&first)?;
        release_varbin_capture(&second)?;
        Ok(())
    }

    #[test]
    fn bool_cursor_retains_nonzero_window_and_exact_accounting() -> VortexResult<()> {
        let expected = (0..130)
            .map(|index| (index % 11 != 0).then_some(index % 3 == 0))
            .collect::<Vec<_>>();
        let session = vx_session_new_with(|session| session);
        let boolean = BoolArray::from_iter(expected.iter().copied());
        let array = vx_array_new_with(boolean.into_array());
        let mut error = ptr::null_mut();
        let mut memory = TestMemory::default();
        let memory_callbacks = test_memory_callbacks(&mut memory);
        // SAFETY: The session and array handles remain live until cursor creation finishes.
        let cursor = unsafe {
            vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
        };
        vortex_ensure!(!cursor.is_null(), "Boolean cursor creation failed");
        vortex_ensure!(error.is_null(), "Boolean cursor returned an error");

        let mut capture = BoolCapture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: None,
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: Some(capture_bool),
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        // SAFETY: The cursor and callback state remain live through the call.
        let status = unsafe {
            vx_velox_export_cursor_visit(cursor, 65, 63, &raw const visitor, &raw mut error)
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "Boolean export window returned an error");
        assert_eq!(capture.length, 63);
        assert_eq!(capture.values_bit_offset, 1);
        assert_eq!(capture.validity_bit_offset, 1);
        assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
        assert_eq!(capture.retained_bytes, 6 * size_of::<u64>());
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 48);

        // SAFETY: Each owned handle is freed once. The callback retained the Boolean owner.
        unsafe {
            vx_velox_export_cursor_free(cursor);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            capture.retained_bytes
        );
        for (relative_index, expected) in expected[65..128].iter().enumerate() {
            let value_bit = capture.values_bit_offset + relative_index;
            let validity_bit = capture.validity_bit_offset + relative_index;
            // SAFETY: The retained buffers cover every captured value and validity bit.
            let (actual, is_valid) = unsafe {
                (
                    *capture.values.add(value_bit / 8) & (1 << (value_bit % 8)) != 0,
                    *capture.validity.add(validity_bit / 8) & (1 << (validity_bit % 8)) != 0,
                )
            };
            assert_eq!(is_valid, expected.is_some());
            if let Some(expected) = expected {
                assert_eq!(actual, *expected);
            }
        }
        release_bool_capture(&capture)?;
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn export_cursor_reuses_one_prepared_array_across_windows() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let array = vx_array_new_with(
            PrimitiveArray::from_option_iter([Some(10_i64), None, Some(30), Some(40), Some(50)])
                .into_array(),
        );
        let mut error = ptr::null_mut();
        let mut memory = TestMemory::default();
        let memory_callbacks = test_memory_callbacks(&mut memory);
        // SAFETY: The session and array handles remain live until cursor creation finishes.
        let cursor = unsafe {
            vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
        };
        vortex_ensure!(!cursor.is_null(), "export cursor creation failed");
        vortex_ensure!(error.is_null(), "export cursor returned an error");
        assert!(memory.retained_bytes.load(Ordering::Relaxed) >= 48);

        let mut first = Capture::default();
        let first_visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut first).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        // SAFETY: The cursor and callback state remain live through the call.
        let status = unsafe {
            vx_velox_export_cursor_visit(cursor, 1, 2, &raw const first_visitor, &raw mut error)
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "first export window returned an error");
        assert_eq!(first.length, 2);
        assert_eq!(first.validity_bit_offset, 1);
        // SAFETY: The callback retained two readable i64 values.
        let first_values = unsafe { slice::from_raw_parts(first.values.cast::<i64>(), 2) };
        assert_eq!(first_values, [0, 30]);
        assert_eq!(
            first.retained_bytes,
            memory.retained_bytes.load(Ordering::Relaxed)
        );
        let owner = first
            .owner
            .ok_or_else(|| vortex_err!("first export window returned no owner"))?
            .owner;
        release_capture(&first)?;

        let mut second = Capture::default();
        let second_visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut second).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        // SAFETY: The cursor and callback state remain live through the call.
        let status = unsafe {
            vx_velox_export_cursor_visit(cursor, 3, 2, &raw const second_visitor, &raw mut error)
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "second export window returned an error");
        assert_eq!(second.length, 2);
        assert_eq!(second.validity_bit_offset, 3);
        assert_eq!(
            second
                .owner
                .ok_or_else(|| vortex_err!("second export window returned no owner"))?
                .owner,
            owner
        );

        // SAFETY: Each owned handle is freed exactly once. The second callback retained the owner.
        unsafe {
            vx_velox_export_cursor_free(cursor);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        // SAFETY: The retained cursor owner keeps these two i64 values live.
        let second_values = unsafe { slice::from_raw_parts(second.values.cast::<i64>(), 2) };
        assert_eq!(second_values, [40, 50]);
        release_capture(&second)?;
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn export_cursor_decodes_sliced_bitpacked_into_exact_owner() -> VortexResult<()> {
        let session = vx_session_new_with(|session| {
            vortex_fastlanes::initialize(&session);
            session
        });
        let session_ref = unsafe { vx_session_ref(session)? };
        let values = (0..2_050).map(|index| (index % 7 != 0).then_some(i64::from(index % 100)));
        let primitive = PrimitiveArray::from_option_iter(values).into_array();
        let mut execution = session_ref.create_execution_ctx();
        let bitpacked = BitPackedData::encode(&primitive, 7, &mut execution)?;
        vortex_ensure!(
            bitpacked.patches().is_none(),
            "test bit-packed array unexpectedly contains patches"
        );
        let slice_begin = 113;
        let slice_end = 1_941;
        let sliced = bitpacked.into_array().slice(slice_begin..slice_end)?;
        let array = vx_array_new_with(sliced);
        let mut error = ptr::null_mut();
        let mut memory = TestMemory::default();
        let memory_callbacks = test_memory_callbacks(&mut memory);
        // SAFETY: The session and array handles remain live until cursor creation finishes.
        let cursor = unsafe {
            vx_velox_export_cursor_new(session, array, &raw const memory_callbacks, &raw mut error)
        };
        vortex_ensure!(!cursor.is_null(), "export cursor creation failed");
        vortex_ensure!(error.is_null(), "export cursor returned an error");
        let sliced_length = slice_end - slice_begin;
        let expected_retained = sliced_length * size_of::<i64>()
            + sliced_length.div_ceil(u64::BITS as usize) * size_of::<u64>();
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            expected_retained
        );

        let window_offset = 997;
        let window_length = 6;
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        // SAFETY: The cursor and callback state remain live through the call.
        let status = unsafe {
            vx_velox_export_cursor_visit(
                cursor,
                window_offset,
                window_length,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "export window returned an error");
        assert_eq!(capture.primitive_type, Some(VX_VELOX_PRIMITIVE_I64));
        assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
        assert_eq!(
            capture.validity_bit_offset,
            window_offset % u64::BITS as usize
        );
        assert_eq!(capture.retained_bytes, expected_retained);
        // SAFETY: The callback retained `window_length` readable i64 values.
        let actual = unsafe { slice::from_raw_parts(capture.values.cast::<i64>(), window_length) };
        for (relative_index, value) in actual.iter().enumerate() {
            let sliced_index = window_offset + relative_index;
            let source_index = slice_begin + sliced_index;
            // SAFETY: The retained bitmap covers every row in the sliced array.
            let validity_index = capture.validity_bit_offset + relative_index;
            let validity_byte = unsafe { *capture.validity.add(validity_index / 8) };
            let is_valid = validity_byte & (1 << (validity_index % 8)) != 0;
            assert_eq!(is_valid, source_index % 7 != 0);
            if is_valid {
                assert_eq!(*value, i64::try_from(source_index % 100)?);
            }
        }

        // SAFETY: Each owned handle is freed exactly once. The callback retained the owner.
        unsafe {
            vx_velox_export_cursor_free(cursor);
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            expected_retained
        );
        release_capture(&capture)?;
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn patched_bitpacked_uses_retained_canonical_fallback() -> VortexResult<()> {
        let session = vx_session_new_with(|session| {
            vortex_fastlanes::initialize(&session);
            session
        });
        let session_ref = unsafe { vx_session_ref(session)? };
        let expected = [1_u64, 2, 3, u64::MAX];
        let primitive = PrimitiveArray::from_iter(expected).into_array();
        let mut execution = session_ref.create_execution_ctx();
        let bitpacked = BitPackedData::encode(&primitive, 2, &mut execution)?;
        vortex_ensure!(
            bitpacked.patches().is_some(),
            "test bit-packed array unexpectedly omitted patches"
        );
        let mut memory = TestMemory::default();
        let export = PrimitiveExport::try_new(
            bitpacked.into_array(),
            session_ref,
            Some(test_memory_callbacks(&mut memory)),
        )?;
        assert!(matches!(export.owner.values, PrimitiveValues::Retained(_)));
        assert_eq!(
            memory.retained_bytes.load(Ordering::Relaxed),
            export.owner.retained_bytes()
        );
        // SAFETY: The export owner contains `expected.len()` initialized u64 values.
        let actual =
            unsafe { slice::from_raw_parts(export.owner.values().cast::<u64>(), expected.len()) };
        assert_eq!(actual, expected);
        drop(export);
        assert_eq!(memory.retained_bytes.load(Ordering::Relaxed), 0);
        unsafe { vx_session_free(session) };
        Ok(())
    }

    #[test]
    fn visits_sparse_nullable_values_with_retained_buffers() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let array = vx_array_new_with(
            PrimitiveArray::from_option_iter([Some(10_i64), None, Some(30), Some(40)]).into_array(),
        );
        let rows = [1_u64, 3];
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: rows.as_ptr(),
            row_count: rows.len(),
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        let mut error = ptr::null_mut();
        // SAFETY: Every handle and callback object stays live for this call.
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_eq!(capture.primitive_type, Some(VX_VELOX_PRIMITIVE_I64));
        assert_eq!(capture.length, 2);
        assert_eq!(capture.values_length, 2 * size_of::<i64>());
        assert!(capture.values_alignment.is_power_of_two());
        assert_eq!(capture.values.addr() % capture.values_alignment, 0);
        assert_eq!(capture.validity_kind, Some(VX_VELOX_VALIDITY_BITMAP));
        assert_eq!(capture.validity_length, size_of::<u64>());
        assert_eq!(capture.validity_bit_offset, 0);
        assert!(capture.validity_alignment.is_power_of_two());
        assert_eq!(capture.validity.addr() % capture.validity_alignment, 0);
        assert_eq!(
            capture.retained_bytes,
            capture.values_length + size_of::<u64>()
        );
        // SAFETY: The callback retained the owner before storing these pointers.
        let values = unsafe { slice::from_raw_parts(capture.values.cast::<i64>(), 2) };
        assert_eq!(values, [0, 40]);
        // SAFETY: The retained validity pointer has one readable word.
        let validity = unsafe { *capture.validity };
        assert_eq!(validity & 0b11, 0b10);

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        // SAFETY: This release matches the retain in `capture_primitive`.
        unsafe { release(owner.owner) };
        // SAFETY: Each owned handle is freed exactly once.
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn copies_sliced_values_into_exact_owned_storage() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let source = PrimitiveArray::from_iter(0_i32..16);
        let source_values = source.buffer_handle().try_to_host_sync()?;
        // SAFETY: The source contains sixteen i32 values. The fifth value is in bounds.
        let source_slice = unsafe { source_values.as_ptr().add(5 * size_of::<i32>()) };
        drop(source_values);
        let array = vx_array_new_with(source.into_array().slice(5..8)?);
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: ptr::null(),
            row_count: 0,
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_eq!(capture.values_length, 3 * size_of::<i32>());
        assert_eq!(capture.retained_bytes, 2 * size_of::<u64>());
        assert_ne!(capture.values, source_slice);
        // SAFETY: Each owned handle is freed exactly once. The callback retained the value owner.
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        // SAFETY: The retained compact buffer contains three i32 values.
        let values = unsafe { slice::from_raw_parts(capture.values.cast::<i32>(), 3) };
        assert_eq!(values, [5, 6, 7]);
        assert!(capture.values_alignment.is_power_of_two());
        assert_eq!(capture.values.addr() % capture.values_alignment, 0);
        assert_eq!(capture.validity_alignment, 0);

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        unsafe { release(owner.owner) };
        Ok(())
    }

    #[test]
    fn copies_validity_into_word_padded_storage() -> VortexResult<()> {
        let session = vx_session_new_with(|session| session);
        let session_ref = unsafe { vx_session_ref(session)? };
        let primitive = PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]);
        let mut execution = session_ref.create_execution_ctx();
        let Mask::Values(mask) = primitive
            .validity()?
            .execute_mask(primitive.len(), &mut execution)?
        else {
            vortex_bail!("Expected bitmap validity");
        };
        let expected_validity = mask.bit_buffer().inner().as_ptr();
        let array = vx_array_new_with(primitive.into_array());
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: ptr::null(),
            row_count: 0,
        };
        let mut capture = Capture::default();
        let visitor = vx_velox_visitor {
            struct_size: size_of::<vx_velox_visitor>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            context: (&raw mut capture).cast(),
            visit_primitive: Some(capture_primitive),
            last_error: None,
            visit_varbin: None,
            visit_dictionary: None,
            visit_constant: None,
            visit_bool: None,
            visit_struct: None,
            visit_list: None,
            visit_map: None,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            vx_velox_array_visit(
                session,
                array,
                &raw const request,
                &raw const visitor,
                &raw mut error,
            )
        };
        assert_eq!(status, 0);
        vortex_ensure!(error.is_null(), "visitor returned an error");
        assert_ne!(capture.validity, expected_validity);
        assert_eq!(capture.validity_bit_offset, 0);
        assert_eq!(capture.validity_length, size_of::<u64>());
        assert!(capture.validity_alignment >= align_of::<u64>());
        assert_eq!(
            capture.retained_bytes,
            capture.values_length.div_ceil(size_of::<u64>()) * size_of::<u64>() + size_of::<u64>()
        );

        let owner = capture
            .owner
            .ok_or_else(|| vortex_err!("visitor did not return a retained owner"))?;
        let release = owner
            .release
            .ok_or_else(|| vortex_err!("visitor owner did not return a release callback"))?;
        unsafe { release(owner.owner) };
        unsafe {
            vx_velox_array_free(array);
            vx_session_free(session);
        }
        Ok(())
    }

    #[test]
    fn rejects_unsorted_rows() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();
        let rows = [2_u64, 1];
        let request = vx_velox_visit_request {
            struct_size: size_of::<vx_velox_visit_request>(),
            rows: rows.as_ptr(),
            row_count: rows.len(),
        };
        match selected_array(&array, &request) {
            Ok(_) => vortex_bail!("unsorted rows unexpectedly succeeded"),
            Err(error) => assert!(error.to_string().contains("unique and increasing")),
        }
        Ok(())
    }
}
