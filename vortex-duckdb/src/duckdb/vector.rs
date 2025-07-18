// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, CString};
use std::ptr;

use bitvec::macros::internal::funty::Fundamental;
use bitvec::slice::BitSlice;
use vortex::error::{VortexResult, VortexUnwrap, vortex_bail, vortex_err};

use crate::cpp::duckdb_vx_error;
use crate::duckdb::data::Data;
use crate::duckdb::{LogicalType, SelectionVector, Value};
use crate::{cpp, wrapper};

pub const DUCKDB_STANDARD_VECTOR_SIZE: usize = 2048;

wrapper!(Vector, cpp::duckdb_vector, cpp::duckdb_destroy_vector);

/// Safety: It is safe to mark `Vector` as `Send` as the memory it points is `Send`.
///
/// Exceptions from a raw pointer not being `Send` would be pointing to
/// thread-local storage or other types that are not `Send`, e.g. `RefCell`.
///
/// ```no_test
/// pub struct Vector {
///     ptr: *mut _duckdb_vector,
///     owned: bool,
/// }
/// ```
unsafe impl Send for Vector {}

impl Vector {
    /// Create a new vector with the given type and capacity.
    pub fn with_capacity(logical_type: LogicalType, len: usize) -> Self {
        unsafe { Self::own(cpp::duckdb_create_vector(logical_type.as_ptr(), len as _)) }
    }

    /// Converts the vector to a constant value.
    pub fn reference_value(&mut self, value: &Value) {
        unsafe {
            cpp::duckdb_vector_reference_value(self.as_ptr(), value.as_ptr());
        }
    }

    /// Populates this vector by reference to another.
    pub fn reference(&mut self, other: &Vector) {
        unsafe { cpp::duckdb_vector_reference_vector(self.as_ptr(), other.as_ptr()) }
    }

    /// Slice the vector to a new dictionary vector, using the current vector's values and
    /// the provided selection vector.
    pub fn slice_to_dictionary(&mut self, sel_vec: SelectionVector, sel_vec_length: usize) {
        unsafe {
            cpp::duckdb_vx_vector_slice_to_dictionary(
                self.as_ptr(),
                sel_vec.as_ptr(),
                sel_vec_length as _,
            )
        }
    }

    // Used to by duckdb to know the dictionary value length (since each vector doesn't know its own
    // length only its capacity).
    pub fn set_dictionary_len(&mut self, len: u32) {
        unsafe { cpp::duckdb_vx_set_dictionary_vector_length(self.as_ptr(), len) }
    }

    // A pipeline-scoped id to assert dictionary vector value uniqueness
    pub fn set_dictionary_id(&mut self, dict_id: String) {
        let dict_id = CString::new(dict_id)
            .map_err(|e| vortex_err!("cstr creation error {e}"))
            .vortex_unwrap();
        unsafe {
            cpp::duckdb_vx_set_dictionary_vector_id(
                self.ptr,
                dict_id.as_ptr(),
                dict_id.as_bytes().len().as_u32(),
            )
        }
    }

    pub fn to_sequence(&mut self, start: i64, stop: i64, capacity: u64) {
        unsafe { cpp::duckdb_vx_sequence_vector(self.ptr, start, stop, capacity) }
    }

    /// Converts a vector into a flat uncompressed vector vortex call this `canonicalize`.
    pub fn flatten(&self, length: u64) {
        unsafe { cpp::duckdb_vector_flatten(self.as_ptr(), length) }
    }

    // NOTE(ngates): vector doesn't hold its own length. Which makes writing a safe
    //  Rust API annoying...
    pub unsafe fn as_slice_mut<T>(&mut self, length: usize) -> &mut [T] {
        let ptr = unsafe { cpp::duckdb_vector_get_data(self.as_ptr()) };
        unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), length) }
    }

    pub fn as_slice_with_len<T>(&self, length: usize) -> &[T] {
        let ptr = unsafe { cpp::duckdb_vector_get_data(self.as_ptr()) };
        unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), length) }
    }

    // TODO(joe): remove this once move away from arrow conversion
    pub fn slow_row_is_null(&self, row: u64) -> bool {
        // this is the formula, given a validity vector to extract validity bit as row_idx.
        // use idx_t entry_idx = row_idx / 64; idx_t idx_in_entry = row_idx % 64; bool is_valid = validity_mask[entry_idx] & (1 « idx_in_entry);
        // as the row is valid function is slower
        let valid = unsafe {
            let validity = cpp::duckdb_vector_get_validity(self.ptr);

            // validity can return a NULL pointer if the entire vector is valid
            if validity.is_null() {
                return false;
            }

            cpp::duckdb_validity_row_is_valid(validity, row)
        };

        !valid
    }

    pub fn add_string_buffer<T>(&self, buffer: T) {
        let data = Data::from(Box::new(buffer));
        unsafe { cpp::duckdb_vx_string_vector_add_buffer(self.as_ptr(), data.into_ptr()) }
    }

    /// Assigns the element at the specified index with a string value.
    /// FIXME(ngates): remove this.
    pub fn assign_string_element(&self, idx: usize, value: &CStr) {
        unsafe { cpp::duckdb_vector_assign_string_element(self.as_ptr(), idx as _, value.as_ptr()) }
    }

    pub fn logical_type(&self) -> LogicalType {
        unsafe { LogicalType::own(cpp::duckdb_vector_get_column_type(self.as_ptr())) }
    }

    #[allow(clippy::expect_used)]
    pub fn ensure_validity_slice(&mut self) -> &mut BitSlice<u64> {
        unsafe { cpp::duckdb_vector_ensure_validity_writable(self.as_ptr()) };
        self.validity_slice_mut()
            .expect("we just ensured the validity slice is allocated")
    }

    /// Returns the validity slice of the vector, if it exists.
    pub fn validity_slice_mut(&mut self) -> Option<&mut BitSlice<u64>> {
        let ptr = unsafe { cpp::duckdb_vector_get_validity(self.as_ptr()) };
        unsafe { ptr.as_mut() }.map(|ptr| {
            let len = DUCKDB_STANDARD_VECTOR_SIZE / 64;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            BitSlice::from_slice_mut(slice)
        })
    }

    pub fn try_to_string(&self, len: u64) -> VortexResult<String> {
        let mut err: duckdb_vx_error = ptr::null_mut();
        let debug =
            unsafe { cpp::duckdb_vector_to_string(self.as_ptr(), len.as_u64(), &raw mut err) };
        if !err.is_null() {
            vortex_bail!("{}", unsafe {
                CStr::from_ptr(cpp::duckdb_vx_error_value(err)).to_string_lossy()
            })
        }
        let string = unsafe { CStr::from_ptr(debug).to_string_lossy() }.to_string();
        unsafe { cpp::duckdb_free(debug.cast_mut().cast()) };
        Ok(string)
    }
}
