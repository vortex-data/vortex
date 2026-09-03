// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::ffi::c_void;

mod export;

pub use export::vx_velox_export_cursor;

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
