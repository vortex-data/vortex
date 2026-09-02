// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::ptr;
use std::slice;

use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr::root;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_error::vortex_bail;
use vortex_ffi::try_or;
use vortex_ffi::vx_array;
use vortex_ffi::vx_data_source;
use vortex_ffi::vx_data_source_scan_with;
use vortex_ffi::vx_dtype;
use vortex_ffi::vx_dtype_new_with;
use vortex_ffi::vx_error;
use vortex_ffi::vx_expression;
use vortex_ffi::vx_expression_new_with;
use vortex_ffi::vx_expression_ref;
use vortex_ffi::vx_partition;
use vortex_ffi::vx_scalar;
use vortex_ffi::vx_scan;
use vortex_ffi::vx_session;
use vortex_ffi::vx_view;

// The base FFI wrappers are opaque C handles despite their private Rust payloads.
#[allow(improper_ctypes)]
mod ffi {
    use super::*;

    unsafe extern "C-unwind" {
        pub fn vx_error_message(error: *const vx_error) -> vx_view;
        pub fn vx_error_free(error: *const vx_error);
        pub fn vx_session_new() -> *mut vx_session;
        pub fn vx_session_free(session: *const vx_session);
        pub fn vx_dtype_free(dtype: *const vx_dtype);
        pub fn vx_scalar_new_bool(value: bool, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_i8(value: i8, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_i16(value: i16, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_i32(value: i32, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_i64(value: i64, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_f32(value: f32, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_f64(value: f64, nullable: bool) -> *mut vx_scalar;
        pub fn vx_scalar_new_utf8(
            value: vx_view,
            nullable: bool,
            error_out: *mut *mut vx_error,
        ) -> *mut vx_scalar;
        pub fn vx_scalar_new_binary(
            data: *const u8,
            length: usize,
            nullable: bool,
            error_out: *mut *mut vx_error,
        ) -> *mut vx_scalar;
        pub fn vx_scalar_new_list(
            element_dtype: *const vx_dtype,
            elements: *const *const vx_scalar,
            length: usize,
            nullable: bool,
            error_out: *mut *mut vx_error,
        ) -> *mut vx_scalar;
        pub fn vx_scalar_free(scalar: *const vx_scalar);
        pub fn vx_expression_literal(
            scalar: *const vx_scalar,
            error_out: *mut *mut vx_error,
        ) -> *mut vx_expression;
        pub fn vx_expression_free(expression: *const vx_expression);
        pub fn vx_data_source_free(data_source: *const vx_data_source);
        pub fn vx_scan_free(scan: *const vx_scan);
        pub fn vx_scan_next_partition(
            scan: *mut vx_scan,
            error_out: *mut *mut vx_error,
        ) -> *mut vx_partition;
        pub fn vx_partition_free(partition: *const vx_partition);
        pub fn vx_partition_next(
            partition: *mut vx_partition,
            error_out: *mut *mut vx_error,
        ) -> *const vx_array;
        pub fn vx_array_free(array: *const vx_array);
        pub fn vx_array_len(array: *const vx_array) -> usize;
        pub fn vx_array_slice(
            array: *const vx_array,
            begin: usize,
            end: usize,
            error_out: *mut *mut vx_error,
        ) -> *const vx_array;
    }

    unsafe extern "C" {
        pub fn vx_expression_root() -> *mut vx_expression;
        pub fn vx_expression_get_item(
            name: vx_view,
            child: *const vx_expression,
        ) -> *mut vx_expression;
        pub fn vx_expression_and(
            expressions: *const *const vx_expression,
            length: usize,
        ) -> *mut vx_expression;
        pub fn vx_expression_or(
            expressions: *const *const vx_expression,
            length: usize,
        ) -> *mut vx_expression;
        pub fn vx_expression_not(child: *const vx_expression) -> *mut vx_expression;
        pub fn vx_expression_is_null(child: *const vx_expression) -> *mut vx_expression;
        pub fn vx_expression_list_contains(
            list: *const vx_expression,
            value: *const vx_expression,
        ) -> *mut vx_expression;
    }
}

/// A fixed-width primitive type identifier for Velox scalar construction.
pub type vx_velox_ptype = u32;
/// Unsigned 8-bit integer type identifier.
pub const VX_VELOX_PTYPE_U8: vx_velox_ptype = 0;
/// Unsigned 16-bit integer type identifier.
pub const VX_VELOX_PTYPE_U16: vx_velox_ptype = 1;
/// Unsigned 32-bit integer type identifier.
pub const VX_VELOX_PTYPE_U32: vx_velox_ptype = 2;
/// Unsigned 64-bit integer type identifier.
pub const VX_VELOX_PTYPE_U64: vx_velox_ptype = 3;
/// Signed 8-bit integer type identifier.
pub const VX_VELOX_PTYPE_I8: vx_velox_ptype = 4;
/// Signed 16-bit integer type identifier.
pub const VX_VELOX_PTYPE_I16: vx_velox_ptype = 5;
/// Signed 32-bit integer type identifier.
pub const VX_VELOX_PTYPE_I32: vx_velox_ptype = 6;
/// Signed 64-bit integer type identifier.
pub const VX_VELOX_PTYPE_I64: vx_velox_ptype = 7;
/// 16-bit floating-point type identifier.
pub const VX_VELOX_PTYPE_F16: vx_velox_ptype = 8;
/// 32-bit floating-point type identifier.
pub const VX_VELOX_PTYPE_F32: vx_velox_ptype = 9;
/// 64-bit floating-point type identifier.
pub const VX_VELOX_PTYPE_F64: vx_velox_ptype = 10;

/// A fixed-width binary expression operator identifier.
pub type vx_velox_binary_operator = u32;
/// Equality operator identifier.
pub const VX_VELOX_OPERATOR_EQ: vx_velox_binary_operator = 0;
/// Inequality operator identifier.
pub const VX_VELOX_OPERATOR_NOT_EQ: vx_velox_binary_operator = 1;
/// Greater-than operator identifier.
pub const VX_VELOX_OPERATOR_GT: vx_velox_binary_operator = 2;
/// Greater-than-or-equal operator identifier.
pub const VX_VELOX_OPERATOR_GTE: vx_velox_binary_operator = 3;
/// Less-than operator identifier.
pub const VX_VELOX_OPERATOR_LT: vx_velox_binary_operator = 4;
/// Less-than-or-equal operator identifier.
pub const VX_VELOX_OPERATOR_LTE: vx_velox_binary_operator = 5;
/// Kleene logical AND operator identifier.
pub const VX_VELOX_OPERATOR_KLEENE_AND: vx_velox_binary_operator = 6;
/// Kleene logical OR operator identifier.
pub const VX_VELOX_OPERATOR_KLEENE_OR: vx_velox_binary_operator = 7;

/// A fixed-width row-selection mode identifier.
pub type vx_velox_scan_selection_include = u32;
/// Include every row.
pub const VX_VELOX_SELECTION_ALL: vx_velox_scan_selection_include = 0;
/// Include the supplied row indexes.
pub const VX_VELOX_SELECTION_INCLUDE: vx_velox_scan_selection_include = 1;
/// Exclude the supplied row indexes.
pub const VX_VELOX_SELECTION_EXCLUDE: vx_velox_scan_selection_include = 2;

/// A stable row selection for one scan request.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_velox_scan_selection {
    /// The selected row indexes.
    pub indices: *const u64,
    /// The number of selected row indexes.
    pub length: usize,
    /// The selection mode.
    pub include: vx_velox_scan_selection_include,
}

/// Stable options for one Vortex scan.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_scan_options {
    /// Set this field to `sizeof(vx_velox_scan_options)`.
    pub struct_size: usize,
    /// Set this field to [`crate::VX_VELOX_ABI_VERSION`].
    pub abi_version: u32,
    /// The projected expression, or null for every field.
    pub projection: *const vx_expression,
    /// The exact filter expression, or null for no filter.
    pub filter: *const vx_expression,
    /// The first row in the scan range.
    pub row_range_begin: u64,
    /// One past the final row in the scan range.
    pub row_range_end: u64,
    /// An optional row-index selection.
    pub selection: vx_velox_scan_selection,
    /// The maximum output row count, or zero for no limit.
    pub limit: u64,
    /// Return rows in storage order.
    pub ordered: bool,
}

impl Default for vx_velox_scan_selection {
    fn default() -> Self {
        Self {
            indices: ptr::null(),
            length: 0,
            include: VX_VELOX_SELECTION_ALL,
        }
    }
}

/// Return the message stored in an adapter error.
///
/// # Safety
///
/// `error` must point to a live error handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_error_message(error: *const vx_error) -> vx_view {
    unsafe { ffi::vx_error_message(error) }
}

/// Free an adapter error.
///
/// # Safety
///
/// `error` must be null or an owned error handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_error_free(error: *const vx_error) {
    unsafe { ffi::vx_error_free(error) };
}

/// Create a default Vortex session for Velox.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_velox_session_new() -> *mut vx_session {
    unsafe { ffi::vx_session_new() }
}

/// Free a Vortex session.
///
/// # Safety
///
/// `session` must be null or an owned session handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_session_free(session: *const vx_session) {
    unsafe { ffi::vx_session_free(session) };
}

fn primitive_type(ptype: vx_velox_ptype) -> vortex_error::VortexResult<PType> {
    match ptype {
        VX_VELOX_PTYPE_U8 => Ok(PType::U8),
        VX_VELOX_PTYPE_U16 => Ok(PType::U16),
        VX_VELOX_PTYPE_U32 => Ok(PType::U32),
        VX_VELOX_PTYPE_U64 => Ok(PType::U64),
        VX_VELOX_PTYPE_I8 => Ok(PType::I8),
        VX_VELOX_PTYPE_I16 => Ok(PType::I16),
        VX_VELOX_PTYPE_I32 => Ok(PType::I32),
        VX_VELOX_PTYPE_I64 => Ok(PType::I64),
        VX_VELOX_PTYPE_F16 => Ok(PType::F16),
        VX_VELOX_PTYPE_F32 => Ok(PType::F32),
        VX_VELOX_PTYPE_F64 => Ok(PType::F64),
        _ => vortex_bail!("Unknown Vortex Velox primitive type identifier: {ptype}"),
    }
}

/// Create a primitive dtype for a list literal.
///
/// # Safety
///
/// `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_dtype_new_primitive(
    ptype: vx_velox_ptype,
    nullable: bool,
    error_out: *mut *mut vx_error,
) -> *const vx_dtype {
    try_or(error_out, ptr::null(), || {
        Ok(vx_dtype_new_with(DType::Primitive(
            primitive_type(ptype)?,
            Nullability::from(nullable),
        )))
    })
}

/// Free a dtype.
///
/// # Safety
///
/// `dtype` must be null or an owned dtype handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_dtype_free(dtype: *const vx_dtype) {
    unsafe { ffi::vx_dtype_free(dtype) };
}

/// Create a Boolean scalar.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_velox_scalar_new_bool(value: bool, nullable: bool) -> *mut vx_scalar {
    unsafe { ffi::vx_scalar_new_bool(value, nullable) }
}

macro_rules! scalar_primitive_wrapper {
    ($name:ident, $source:ident, $type:ty, $description:literal) => {
        #[doc = $description]
        #[unsafe(no_mangle)]
        pub extern "C-unwind" fn $name(value: $type, nullable: bool) -> *mut vx_scalar {
            unsafe { ffi::$source(value, nullable) }
        }
    };
}

scalar_primitive_wrapper!(
    vx_velox_scalar_new_i8,
    vx_scalar_new_i8,
    i8,
    "Create an i8 scalar."
);
scalar_primitive_wrapper!(
    vx_velox_scalar_new_i16,
    vx_scalar_new_i16,
    i16,
    "Create an i16 scalar."
);
scalar_primitive_wrapper!(
    vx_velox_scalar_new_i32,
    vx_scalar_new_i32,
    i32,
    "Create an i32 scalar."
);
scalar_primitive_wrapper!(
    vx_velox_scalar_new_i64,
    vx_scalar_new_i64,
    i64,
    "Create an i64 scalar."
);
scalar_primitive_wrapper!(
    vx_velox_scalar_new_f32,
    vx_scalar_new_f32,
    f32,
    "Create an f32 scalar."
);
scalar_primitive_wrapper!(
    vx_velox_scalar_new_f64,
    vx_scalar_new_f64,
    f64,
    "Create an f64 scalar."
);

/// Create a UTF-8 scalar.
///
/// # Safety
///
/// `value` and `error_out` must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scalar_new_utf8(
    value: vx_view,
    nullable: bool,
    error_out: *mut *mut vx_error,
) -> *mut vx_scalar {
    unsafe { ffi::vx_scalar_new_utf8(value, nullable, error_out) }
}

/// Create a binary scalar.
///
/// # Safety
///
/// `data` must identify `length` bytes. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scalar_new_binary(
    data: *const u8,
    length: usize,
    nullable: bool,
    error_out: *mut *mut vx_error,
) -> *mut vx_scalar {
    unsafe { ffi::vx_scalar_new_binary(data, length, nullable, error_out) }
}

/// Create a list scalar.
///
/// # Safety
///
/// Every pointer must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scalar_new_list(
    element_dtype: *const vx_dtype,
    elements: *const *const vx_scalar,
    length: usize,
    nullable: bool,
    error_out: *mut *mut vx_error,
) -> *mut vx_scalar {
    unsafe { ffi::vx_scalar_new_list(element_dtype, elements, length, nullable, error_out) }
}

/// Free a scalar.
///
/// # Safety
///
/// `scalar` must be null or an owned scalar handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scalar_free(scalar: *const vx_scalar) {
    unsafe { ffi::vx_scalar_free(scalar) };
}

/// Create a literal expression.
///
/// # Safety
///
/// `scalar` must point to a live scalar. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_expression_literal(
    scalar: *const vx_scalar,
    error_out: *mut *mut vx_error,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_literal(scalar, error_out) }
}

/// Create a root expression.
#[unsafe(no_mangle)]
pub extern "C" fn vx_velox_expression_root() -> *mut vx_expression {
    unsafe { ffi::vx_expression_root() }
}

/// Create a field expression.
///
/// # Safety
///
/// `child` must point to a live expression. `name` must identify valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_get_item(
    name: vx_view,
    child: *const vx_expression,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_get_item(name, child) }
}

fn binary_operator(operator: vx_velox_binary_operator) -> vortex_error::VortexResult<Operator> {
    match operator {
        VX_VELOX_OPERATOR_EQ => Ok(Operator::Eq),
        VX_VELOX_OPERATOR_NOT_EQ => Ok(Operator::NotEq),
        VX_VELOX_OPERATOR_GT => Ok(Operator::Gt),
        VX_VELOX_OPERATOR_GTE => Ok(Operator::Gte),
        VX_VELOX_OPERATOR_LT => Ok(Operator::Lt),
        VX_VELOX_OPERATOR_LTE => Ok(Operator::Lte),
        VX_VELOX_OPERATOR_KLEENE_AND => Ok(Operator::And),
        VX_VELOX_OPERATOR_KLEENE_OR => Ok(Operator::Or),
        _ => vortex_bail!("Unknown Vortex Velox binary operator identifier: {operator}"),
    }
}

/// Create a binary expression.
///
/// # Safety
///
/// Both operands must point to live expressions. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_expression_binary(
    operator: vx_velox_binary_operator,
    left: *const vx_expression,
    right: *const vx_expression,
    error_out: *mut *mut vx_error,
) -> *mut vx_expression {
    try_or(error_out, ptr::null_mut(), || {
        let operator = binary_operator(operator)?;
        let left = unsafe { vx_expression_ref(left)? }.clone();
        let right = unsafe { vx_expression_ref(right)? }.clone();
        Ok(vx_expression_new_with(
            Binary.new_expr(operator, [left, right]),
        ))
    })
}

/// Create a conjunction from expressions.
///
/// # Safety
///
/// `expressions` must identify `length` live expression pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_and(
    expressions: *const *const vx_expression,
    length: usize,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_and(expressions, length) }
}

/// Create a disjunction from expressions.
///
/// # Safety
///
/// `expressions` must identify `length` live expression pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_or(
    expressions: *const *const vx_expression,
    length: usize,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_or(expressions, length) }
}

/// Create a logical negation.
///
/// # Safety
///
/// `child` must point to a live expression.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_not(
    child: *const vx_expression,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_not(child) }
}

/// Create a null test.
///
/// # Safety
///
/// `child` must point to a live expression.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_is_null(
    child: *const vx_expression,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_is_null(child) }
}

/// Create a list membership test.
///
/// # Safety
///
/// Both operands must point to live expressions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_expression_list_contains(
    list: *const vx_expression,
    value: *const vx_expression,
) -> *mut vx_expression {
    unsafe { ffi::vx_expression_list_contains(list, value) }
}

/// Free an expression.
///
/// # Safety
///
/// `expression` must be null or an owned expression handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_expression_free(expression: *const vx_expression) {
    unsafe { ffi::vx_expression_free(expression) };
}

/// Free a data source.
///
/// # Safety
///
/// `data_source` must be null or an owned data-source handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_data_source_free(data_source: *const vx_data_source) {
    unsafe { ffi::vx_data_source_free(data_source) };
}

unsafe fn scan_options(options: &vx_velox_scan_options) -> vortex_error::VortexResult<ScanRequest> {
    if options.struct_size < size_of::<vx_velox_scan_options>() {
        vortex_bail!(
            "Vortex Velox scan options are too small: expected at least {}, got {}",
            size_of::<vx_velox_scan_options>(),
            options.struct_size
        );
    }
    if options.abi_version != crate::VX_VELOX_ABI_VERSION {
        vortex_bail!(
            "Unsupported Vortex Velox ABI version: expected {}, got {}",
            crate::VX_VELOX_ABI_VERSION,
            options.abi_version
        );
    }
    let projection = if options.projection.is_null() {
        root()
    } else {
        unsafe { vx_expression_ref(options.projection)? }.clone()
    };
    let filter = if options.filter.is_null() {
        None
    } else {
        Some(unsafe { vx_expression_ref(options.filter)? }.clone())
    };
    let indices = if options.selection.length == 0 {
        &[]
    } else {
        if options.selection.indices.is_null() {
            vortex_bail!("Vortex Velox scan selection indices must not be null");
        }
        unsafe { slice::from_raw_parts(options.selection.indices, options.selection.length) }
    };
    let selection = match options.selection.include {
        VX_VELOX_SELECTION_ALL => {
            if !indices.is_empty() {
                vortex_bail!("An all-rows selection must not contain row indices");
            }
            Selection::All
        }
        VX_VELOX_SELECTION_INCLUDE => {
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::copy_from(indices))?)
        }
        VX_VELOX_SELECTION_EXCLUDE => {
            Selection::ExcludeByIndex(StrictSortedBuffer::try_new(Buffer::copy_from(indices))?)
        }
        include => vortex_bail!("Unknown Vortex Velox scan selection identifier: {include}"),
    };
    if options.row_range_end != 0 && options.row_range_begin > options.row_range_end {
        vortex_bail!(
            "Vortex Velox row range is invalid: {}..{}",
            options.row_range_begin,
            options.row_range_end
        );
    }
    Ok(ScanRequest {
        projection,
        filter,
        row_range: (options.row_range_begin != 0 || options.row_range_end != 0)
            .then_some(options.row_range_begin..options.row_range_end),
        selection,
        limit: (options.limit != 0).then_some(options.limit),
        ordered: options.ordered,
        partition_selection: Selection::All,
        partition_range: None,
    })
}

/// Start a scan through the stable adapter options.
///
/// # Safety
///
/// Every pointer must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_data_source_scan(
    data_source: *const vx_data_source,
    options: *const vx_velox_scan_options,
    error_out: *mut *mut vx_error,
) -> *mut vx_scan {
    try_or(error_out, ptr::null_mut(), || {
        let request = if options.is_null() {
            ScanRequest::default()
        } else {
            unsafe { scan_options(&*options)? }
        };
        unsafe { vx_data_source_scan_with(data_source, request) }
    })
}

/// Free a scan.
///
/// # Safety
///
/// `scan` must be null or an owned scan handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scan_free(scan: *const vx_scan) {
    unsafe { ffi::vx_scan_free(scan) };
}

/// Return the next partition from a scan.
///
/// # Safety
///
/// `scan` must point to a live scan. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_scan_next_partition(
    scan: *mut vx_scan,
    error_out: *mut *mut vx_error,
) -> *mut vx_partition {
    unsafe { ffi::vx_scan_next_partition(scan, error_out) }
}

/// Free a partition.
///
/// # Safety
///
/// `partition` must be null or an owned partition handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_partition_free(partition: *const vx_partition) {
    unsafe { ffi::vx_partition_free(partition) };
}

/// Return the next array from a partition.
///
/// # Safety
///
/// `partition` must point to a live partition. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_partition_next(
    partition: *mut vx_partition,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    unsafe { ffi::vx_partition_next(partition, error_out) }
}

/// Free an array.
///
/// # Safety
///
/// `array` must be null or an owned array handle.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_free(array: *const vx_array) {
    unsafe { ffi::vx_array_free(array) };
}

/// Return an array length.
///
/// # Safety
///
/// `array` must point to a live array.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_len(array: *const vx_array) -> usize {
    unsafe { ffi::vx_array_len(array) }
}

/// Slice an array.
///
/// # Safety
///
/// `array` must point to a live array. `error_out` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_array_slice(
    array: *const vx_array,
    begin: usize,
    end: usize,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    unsafe { ffi::vx_array_slice(array, begin, end, error_out) }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn translates_scan_options() -> VortexResult<()> {
        let options = vx_velox_scan_options {
            struct_size: size_of::<vx_velox_scan_options>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            projection: ptr::null(),
            filter: ptr::null(),
            row_range_begin: 10,
            row_range_end: 20,
            selection: vx_velox_scan_selection::default(),
            limit: 5,
            ordered: true,
        };
        let translated = unsafe { scan_options(&options)? };
        assert_eq!(translated.row_range, Some(10..20));
        assert_eq!(translated.limit, Some(5));
        assert!(translated.ordered);
        Ok(())
    }

    #[test]
    fn builds_list_membership_expression() {
        let mut error = ptr::null_mut();
        let dtype =
            unsafe { vx_velox_dtype_new_primitive(VX_VELOX_PTYPE_I64, false, &raw mut error) };
        assert!(error.is_null());
        let values = [
            vx_velox_scalar_new_i64(10, false),
            vx_velox_scalar_new_i64(20, false),
        ];
        let list = unsafe {
            vx_velox_scalar_new_list(
                dtype,
                values.as_ptr().cast(),
                values.len(),
                false,
                &raw mut error,
            )
        };
        assert!(error.is_null());
        let list_literal = unsafe { vx_velox_expression_literal(list, &raw mut error) };
        assert!(error.is_null());
        let root = vx_velox_expression_root();
        let name = vx_view {
            ptr: c"value".as_ptr(),
            len: 5,
        };
        let value = unsafe { vx_velox_expression_get_item(name, root) };
        let membership = unsafe { vx_velox_expression_list_contains(list_literal, value) };
        assert!(!membership.is_null());

        unsafe {
            vx_velox_expression_free(membership);
            vx_velox_expression_free(value);
            vx_velox_expression_free(root);
            vx_velox_expression_free(list_literal);
            vx_velox_scalar_free(list);
            for value in values {
                vx_velox_scalar_free(value);
            }
            vx_velox_dtype_free(dtype);
        }
    }

    #[test]
    fn rejects_wrong_scan_abi_before_source_access() {
        let options = vx_velox_scan_options {
            struct_size: size_of::<vx_velox_scan_options>(),
            abi_version: crate::VX_VELOX_ABI_VERSION + 1,
            projection: ptr::null(),
            filter: ptr::null(),
            row_range_begin: 0,
            row_range_end: 0,
            selection: vx_velox_scan_selection::default(),
            limit: 0,
            ordered: false,
        };
        let mut error = ptr::null_mut();
        let scan =
            unsafe { vx_velox_data_source_scan(ptr::null(), &raw const options, &raw mut error) };
        assert!(scan.is_null());
        assert!(!error.is_null());
        unsafe { vx_velox_error_free(error) };
    }

    #[test]
    fn rejects_unknown_fixed_width_identifiers() {
        let mut error = ptr::null_mut();
        let dtype = unsafe { vx_velox_dtype_new_primitive(u32::MAX, false, &raw mut error) };
        assert!(dtype.is_null());
        assert!(!error.is_null());
        unsafe { vx_velox_error_free(error) };

        error = ptr::null_mut();
        let expression = unsafe {
            vx_velox_expression_binary(u32::MAX, ptr::null(), ptr::null(), &raw mut error)
        };
        assert!(expression.is_null());
        assert!(!error.is_null());
        unsafe { vx_velox_error_free(error) };

        let options = vx_velox_scan_options {
            struct_size: size_of::<vx_velox_scan_options>(),
            abi_version: crate::VX_VELOX_ABI_VERSION,
            projection: ptr::null(),
            filter: ptr::null(),
            row_range_begin: 0,
            row_range_end: 0,
            selection: vx_velox_scan_selection {
                include: u32::MAX,
                ..Default::default()
            },
            limit: 0,
            ordered: false,
        };
        assert!(unsafe { scan_options(&options) }.is_err());
    }
}
