// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Private C-handle support for the Velox adapter.

use std::any::Any;
use std::ffi::c_char;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scan::DataSource;
use vortex::scan::DataSourceScanRef;
use vortex::scan::PartitionRef;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

pub(crate) fn ffi_runtime() -> &'static CurrentThreadRuntime {
    &RUNTIME
}

macro_rules! ffi_handle {
    ($name:ident, $inner:ty, $free:ident) => {
        #[repr(transparent)]
        pub struct $name($inner);

        #[allow(dead_code)]
        impl $name {
            pub(crate) fn new(value: $inner) -> *mut Self {
                Box::into_raw(Box::new(Self(value)))
            }

            pub(crate) unsafe fn as_ref<'a>(pointer: *const Self) -> &'a $inner {
                // SAFETY: Callers validate pointer ownership and lifetime at the C boundary.
                &unsafe { &*pointer }.0
            }

            pub(crate) unsafe fn as_mut<'a>(pointer: *mut Self) -> &'a mut $inner {
                // SAFETY: Callers validate unique pointer ownership at the C boundary.
                &mut unsafe { &mut *pointer }.0
            }
        }

        pub(crate) unsafe fn $free(pointer: *const $name) {
            if !pointer.is_null() {
                // SAFETY: The caller transfers one handle created by `new`.
                drop(unsafe { Box::from_raw(pointer.cast_mut()) });
            }
        }
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct vx_velox_view {
    pub ptr: *const c_char,
    pub len: usize,
}

impl vx_velox_view {
    pub(crate) fn from_str(value: &str) -> Self {
        Self {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    pub(crate) unsafe fn as_bytes<'a>(&self) -> VortexResult<&'a [u8]> {
        if self.ptr.is_null() {
            vortex_ensure!(self.len == 0, "null view pointer with non-zero length");
            return Ok(&[]);
        }
        // SAFETY: The caller provides `len` readable bytes.
        Ok(unsafe { slice::from_raw_parts(self.ptr.cast(), self.len) })
    }

    pub(crate) unsafe fn as_str<'a>(&self) -> VortexResult<&'a str> {
        std::str::from_utf8(unsafe { self.as_bytes() }?)
            .map_err(|error| vortex_err!("invalid UTF-8: {error}"))
    }
}

pub(crate) struct AdapterError {
    message: Arc<str>,
}

ffi_handle!(vx_velox_error, AdapterError, vx_error_free);
ffi_handle!(vx_velox_session, VortexSession, vx_session_free);
ffi_handle!(vx_velox_dtype, DType, vx_dtype_free);
ffi_handle!(vx_velox_scalar, Scalar, vx_scalar_free);
ffi_handle!(vx_velox_expression, Expression, vx_expression_free);
ffi_handle!(
    vx_velox_data_source,
    MultiLayoutDataSource,
    vx_data_source_free
);
ffi_handle!(vx_velox_array, ArrayRef, vx_array_free);

pub(crate) enum ScanState {
    Pending(DataSourceScanRef),
    Started(PartitionStream),
    Finished,
}

ffi_handle!(vx_velox_scan, ScanState, vx_scan_free);

pub(crate) enum PartitionState {
    Pending(PartitionRef),
    Started(SendableArrayStream),
    Finished,
}

ffi_handle!(vx_velox_partition, PartitionState, vx_partition_free);

fn clear_error(error_out: *mut *mut vx_velox_error) {
    if !error_out.is_null() {
        // SAFETY: The caller provides writable storage for one pointer.
        unsafe { error_out.write(ptr::null_mut()) };
    }
}

fn write_error(error_out: *mut *mut vx_velox_error, message: impl Into<Arc<str>>) {
    if !error_out.is_null() {
        // SAFETY: The caller provides writable storage for one pointer.
        unsafe {
            error_out.write(vx_velox_error::new(AdapterError {
                message: message.into(),
            }))
        };
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("panic in Vortex Velox adapter: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("panic in Vortex Velox adapter: {message}")
    } else {
        "panic in Vortex Velox adapter".to_string()
    }
}

pub(crate) fn try_or<T>(
    error_out: *mut *mut vx_velox_error,
    error_value: T,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(value)) => {
            clear_error(error_out);
            value
        }
        Ok(Err(error)) => {
            write_error(error_out, error.to_string());
            error_value
        }
        Err(payload) => {
            write_error(error_out, panic_message(payload.as_ref()));
            error_value
        }
    }
}

pub(crate) unsafe fn vx_error_message(error: *const vx_velox_error) -> vx_velox_view {
    vx_velox_view::from_str(&unsafe { vx_velox_error::as_ref(error) }.message)
}

pub(crate) fn vx_session_new() -> *mut vx_velox_session {
    vx_velox_session::new(VortexSession::default().with_handle(RUNTIME.handle()))
}

pub(crate) unsafe fn vx_session_clone(session: *const vx_velox_session) -> *mut vx_velox_session {
    vx_velox_session::new(unsafe { vx_velox_session::as_ref(session) }.clone())
}

pub(crate) unsafe fn vx_session_ref<'a>(
    session: *const vx_velox_session,
) -> VortexResult<&'a VortexSession> {
    vortex_ensure!(!session.is_null(), "Vortex Velox session must not be null");
    Ok(unsafe { vx_velox_session::as_ref(session) })
}

#[cfg(test)]
pub(crate) fn vx_session_new_with(
    configure: impl FnOnce(VortexSession) -> VortexSession,
) -> *mut vx_velox_session {
    vx_velox_session::new(configure(
        VortexSession::default().with_handle(RUNTIME.handle()),
    ))
}

pub(crate) fn vx_dtype_new_with(dtype: DType) -> *const vx_velox_dtype {
    vx_velox_dtype::new(dtype)
}

pub(crate) unsafe fn vx_scalar_new_bool(value: bool, nullable: bool) -> *mut vx_velox_scalar {
    vx_velox_scalar::new(Scalar::bool(value, Nullability::from(nullable)))
}

macro_rules! scalar_primitive {
    ($name:ident, $type:ty) => {
        pub(crate) unsafe fn $name(value: $type, nullable: bool) -> *mut vx_velox_scalar {
            vx_velox_scalar::new(Scalar::primitive(value, Nullability::from(nullable)))
        }
    };
}

scalar_primitive!(vx_scalar_new_i8, i8);
scalar_primitive!(vx_scalar_new_i16, i16);
scalar_primitive!(vx_scalar_new_i32, i32);
scalar_primitive!(vx_scalar_new_i64, i64);
scalar_primitive!(vx_scalar_new_f32, f32);
scalar_primitive!(vx_scalar_new_f64, f64);

pub(crate) unsafe fn vx_scalar_new_utf8(
    value: vx_velox_view,
    nullable: bool,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_scalar {
    try_or(error_out, ptr::null_mut(), || {
        Ok(vx_velox_scalar::new(Scalar::utf8(
            unsafe { value.as_str() }?.to_owned(),
            Nullability::from(nullable),
        )))
    })
}

pub(crate) unsafe fn vx_scalar_new_binary(
    data: *const u8,
    length: usize,
    nullable: bool,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_scalar {
    try_or(error_out, ptr::null_mut(), || {
        if length != 0 {
            vortex_ensure!(!data.is_null(), "binary data pointer must not be null");
        }
        let bytes = if length == 0 {
            &[]
        } else {
            // SAFETY: The caller provides `length` readable bytes.
            unsafe { slice::from_raw_parts(data, length) }
        };
        Ok(vx_velox_scalar::new(Scalar::binary(
            bytes.to_vec(),
            Nullability::from(nullable),
        )))
    })
}

pub(crate) unsafe fn vx_scalar_new_list(
    element_dtype: *const vx_velox_dtype,
    elements: *const *const vx_velox_scalar,
    length: usize,
    nullable: bool,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_scalar {
    try_or(error_out, ptr::null_mut(), || {
        vortex_ensure!(
            !element_dtype.is_null(),
            "list element dtype must not be null"
        );
        if length != 0 {
            vortex_ensure!(!elements.is_null(), "list elements must not be null");
        }
        let values = if length == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(elements, length) }
                .iter()
                .enumerate()
                .map(|(index, scalar)| {
                    vortex_ensure!(!scalar.is_null(), "list scalar {index} must not be null");
                    Ok(unsafe { vx_velox_scalar::as_ref(*scalar) }
                        .clone()
                        .into_value())
                })
                .collect::<VortexResult<Vec<_>>>()?
        };
        Ok(vx_velox_scalar::new(Scalar::try_new(
            DType::List(
                Arc::new(unsafe { vx_velox_dtype::as_ref(element_dtype) }.clone()),
                Nullability::from(nullable),
            ),
            Some(ScalarValue::Tuple(values)),
        )?))
    })
}

pub(crate) unsafe fn vx_scalar_new_extension(
    dtype: *const vx_velox_dtype,
    storage: *const vx_velox_scalar,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_scalar {
    try_or(error_out, ptr::null_mut(), || {
        vortex_ensure!(!dtype.is_null(), "extension dtype must not be null");
        vortex_ensure!(!storage.is_null(), "extension storage must not be null");
        let dtype = unsafe { vx_velox_dtype::as_ref(dtype) };
        let storage = unsafe { vx_velox_scalar::as_ref(storage) };
        let DType::Extension(extension) = dtype else {
            vortex_bail!("dtype is not an extension type: {dtype}");
        };
        vortex_ensure!(
            storage
                .dtype()
                .eq_ignore_nullability(extension.storage_dtype()),
            "storage dtype {} does not match extension storage dtype {}",
            storage.dtype(),
            extension.storage_dtype()
        );
        Ok(vx_velox_scalar::new(Scalar::try_new(
            dtype.clone(),
            storage.value().cloned(),
        )?))
    })
}

pub(crate) unsafe fn vx_expression_literal(
    scalar: *const vx_velox_scalar,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_expression {
    try_or(error_out, ptr::null_mut(), || {
        vortex_ensure!(!scalar.is_null(), "literal scalar must not be null");
        Ok(vx_velox_expression::new(lit(unsafe {
            vx_velox_scalar::as_ref(scalar)
        }
        .clone())))
    })
}

pub(crate) fn vx_expression_new_with(expression: Expression) -> *mut vx_velox_expression {
    vx_velox_expression::new(expression)
}

pub(crate) unsafe fn vx_expression_ref<'a>(
    expression: *const vx_velox_expression,
) -> VortexResult<&'a Expression> {
    vortex_ensure!(
        !expression.is_null(),
        "Vortex Velox expression must not be null"
    );
    Ok(unsafe { vx_velox_expression::as_ref(expression) })
}

pub(crate) fn vx_expression_root() -> *mut vx_velox_expression {
    vx_velox_expression::new(root())
}

pub(crate) unsafe fn vx_expression_get_item(
    name: vx_velox_view,
    child: *const vx_velox_expression,
) -> *mut vx_velox_expression {
    let Ok(name) = (unsafe { name.as_str() }) else {
        return ptr::null_mut();
    };
    vx_velox_expression::new(get_item(
        FieldName::from(name),
        unsafe { vx_velox_expression::as_ref(child) }.clone(),
    ))
}

unsafe fn expression_slice<'a>(
    expressions: *const *const vx_velox_expression,
    length: usize,
) -> &'a [*const vx_velox_expression] {
    if length == 0 {
        &[]
    } else {
        // SAFETY: The caller provides `length` expression pointers.
        unsafe { slice::from_raw_parts(expressions, length) }
    }
}

pub(crate) unsafe fn vx_expression_and(
    expressions: *const *const vx_velox_expression,
    length: usize,
) -> *mut vx_velox_expression {
    and_collect(
        unsafe { expression_slice(expressions, length) }
            .iter()
            .map(|expression| unsafe { vx_velox_expression::as_ref(*expression) }.clone()),
    )
    .map_or(ptr::null_mut(), vx_velox_expression::new)
}

pub(crate) unsafe fn vx_expression_or(
    expressions: *const *const vx_velox_expression,
    length: usize,
) -> *mut vx_velox_expression {
    or_collect(
        unsafe { expression_slice(expressions, length) }
            .iter()
            .map(|expression| unsafe { vx_velox_expression::as_ref(*expression) }.clone()),
    )
    .map_or(ptr::null_mut(), vx_velox_expression::new)
}

pub(crate) unsafe fn vx_expression_not(
    child: *const vx_velox_expression,
) -> *mut vx_velox_expression {
    vx_velox_expression::new(not(unsafe { vx_velox_expression::as_ref(child) }.clone()))
}

pub(crate) unsafe fn vx_expression_is_null(
    child: *const vx_velox_expression,
) -> *mut vx_velox_expression {
    vx_velox_expression::new(is_null(
        unsafe { vx_velox_expression::as_ref(child) }.clone(),
    ))
}

pub(crate) unsafe fn vx_expression_list_contains(
    list: *const vx_velox_expression,
    value: *const vx_velox_expression,
) -> *mut vx_velox_expression {
    vx_velox_expression::new(list_contains(
        unsafe { vx_velox_expression::as_ref(list) }.clone(),
        unsafe { vx_velox_expression::as_ref(value) }.clone(),
    ))
}

pub(crate) fn vx_data_source_new_with(
    data_source: MultiLayoutDataSource,
) -> *const vx_velox_data_source {
    vx_velox_data_source::new(data_source)
}

pub(crate) unsafe fn vx_data_source_scan_with(
    data_source: *const vx_velox_data_source,
    request: ScanRequest,
) -> VortexResult<*mut vx_velox_scan> {
    vortex_ensure!(
        !data_source.is_null(),
        "Vortex Velox data source must not be null"
    );
    RUNTIME.block_on(async {
        let scan = unsafe { vx_velox_data_source::as_ref(data_source) }
            .scan(request)
            .await?;
        Ok(vx_velox_scan::new(ScanState::Pending(scan)))
    })
}

pub(crate) unsafe fn vx_scan_next_partition(
    scan: *mut vx_velox_scan,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_partition {
    try_or(error_out, ptr::null_mut(), || {
        vortex_ensure!(!scan.is_null(), "Vortex Velox scan must not be null");
        let scan = unsafe { vx_velox_scan::as_mut(scan) };
        let state = std::mem::replace(scan, ScanState::Finished);
        let mut stream = match state {
            ScanState::Pending(scan) => scan.partitions(),
            ScanState::Started(stream) => stream,
            ScanState::Finished => return Ok(ptr::null_mut()),
        };
        match RUNTIME.block_on(stream.next()) {
            Some(partition) => {
                *scan = ScanState::Started(stream);
                Ok(vx_velox_partition::new(PartitionState::Pending(partition?)))
            }
            None => Ok(ptr::null_mut()),
        }
    })
}

pub(crate) unsafe fn vx_partition_next(
    partition: *mut vx_velox_partition,
    error_out: *mut *mut vx_velox_error,
) -> *const vx_velox_array {
    try_or(error_out, ptr::null(), || {
        vortex_ensure!(
            !partition.is_null(),
            "Vortex Velox partition must not be null"
        );
        let partition = unsafe { vx_velox_partition::as_mut(partition) };
        let state = std::mem::replace(partition, PartitionState::Finished);
        let mut stream = match state {
            PartitionState::Pending(partition) => partition.execute()?,
            PartitionState::Started(stream) => stream,
            PartitionState::Finished => return Ok(ptr::null()),
        };
        match RUNTIME.block_on(stream.next()) {
            Some(array) => {
                *partition = PartitionState::Started(stream);
                Ok(vx_velox_array::new(array?))
            }
            None => Ok(ptr::null()),
        }
    })
}

pub(crate) fn vx_array_new_with(array: ArrayRef) -> *const vx_velox_array {
    vx_velox_array::new(array)
}

pub(crate) unsafe fn vx_array_ref<'a>(array: *const vx_velox_array) -> VortexResult<&'a ArrayRef> {
    vortex_ensure!(!array.is_null(), "Vortex Velox array must not be null");
    Ok(unsafe { vx_velox_array::as_ref(array) })
}

pub(crate) unsafe fn vx_array_len(array: *const vx_velox_array) -> usize {
    unsafe { vx_velox_array::as_ref(array) }.len()
}

pub(crate) unsafe fn vx_array_slice(
    array: *const vx_velox_array,
    begin: usize,
    end: usize,
    error_out: *mut *mut vx_velox_error,
) -> *const vx_velox_array {
    try_or(error_out, ptr::null(), || {
        let array = unsafe { vx_array_ref(array) }?;
        vortex_ensure!(begin <= end, "array slice begin exceeds end");
        vortex_ensure!(end <= array.len(), "array slice end exceeds array length");
        Ok(vx_velox_array::new(array.slice(begin..end)?))
    })
}
