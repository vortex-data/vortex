// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, CString, c_void};
use std::ptr;

use arrow_buffer;
use arrow_buffer::Buffer;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::slice::BitSlice;
use bitvec::view::BitView;
use vortex::error::{VortexResult, VortexUnwrap, vortex_bail, vortex_err};

use crate::cpp::duckdb_vx_error;
use crate::duckdb::vector_buffer::VectorBuffer;
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
    /// Create a new vector with the given type.
    pub fn new(logical_type: LogicalType) -> Self {
        unsafe { Self::own(cpp::duckdb_create_vector(logical_type.as_ptr(), 0)) }
    }

    /// Create a new vector with the given type and capacity.
    pub fn with_capacity(logical_type: LogicalType, len: usize) -> Self {
        unsafe { Self::own(cpp::duckdb_create_vector(logical_type.as_ptr(), len as _)) }
    }

    /// Converts the vector is a constant vector with every element `value`.
    pub fn reference_value(&mut self, value: &Value) {
        unsafe {
            cpp::duckdb_vector_reference_value(self.as_ptr(), value.as_ptr());
        }
    }

    /// Reference the data from another vector.
    ///
    /// After calling this, both vectors share ownership of the same underlying data.
    pub fn reference(&mut self, other: &Vector) {
        unsafe { cpp::duckdb_vector_reference_vector(self.as_ptr(), other.as_ptr()) }
    }

    /// Creates a dictionary vector for a given values vector and selection vector.
    ///
    /// A dictionary holds a strong reference to all memory it uses.
    ///
    /// `dictionary` differs from `slice_to_dictionary` in that it initializes hash caching.
    /// See: <https://github.com/duckdb/duckdb/blob/0dcf633f603a629981d089202f93b9080cb1a3e9/src/common/types/vector.cpp#L293>
    pub fn dictionary(
        &self,
        dict: &Vector,
        dictionary_size: usize,
        sel_vec: &SelectionVector,
        count: usize,
    ) {
        unsafe {
            cpp::duckdb_vx_vector_dictionary(
                self.as_ptr(),
                dict.as_ptr(),
                dictionary_size as _,
                sel_vec.as_ptr(),
                count as _,
            )
        }
    }

    // Used to by duckdb to know the dictionary value length (since each vector doesn't know its own
    // length only its capacity).
    pub fn set_dictionary_len(&mut self, len: u32) {
        unsafe { cpp::duckdb_vx_set_dictionary_vector_length(self.as_ptr(), len) }
    }

    // A operator-scoped id to assert dictionary vector value uniqueness
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

    pub fn row_is_null(&self, row: u64) -> bool {
        let validity = unsafe { cpp::duckdb_vector_get_validity(self.ptr) };

        // validity can return a NULL pointer if the entire vector is valid
        if validity.is_null() {
            return false;
        }

        // Direct bit manipulation for better performance
        let entry_idx = row / 64;
        let idx_in_entry = row % 64;
        let validity_u64_ptr = validity as *const u64;
        let validity_entry = unsafe { *validity_u64_ptr.add(entry_idx as usize) };
        let is_valid = (validity_entry & (1u64 << idx_in_entry)) != 0;

        !is_valid
    }

    pub unsafe fn set_vector_buffer(&self, buffer: &VectorBuffer) {
        unsafe { cpp::duckdb_vx_vector_set_vector_data_buffer(self.as_ptr(), buffer.as_ptr()) }
    }

    pub fn add_string_vector_buffer(&self, buffer: &VectorBuffer) {
        unsafe {
            cpp::duckdb_vx_string_vector_add_vector_data_buffer(self.as_ptr(), buffer.as_ptr())
        }
    }

    /// Sets the data pointer for the vector. This is the start of the values array in the vector.
    pub unsafe fn set_data_ptr<T>(&self, ptr: *mut T) {
        unsafe { cpp::duckdb_vx_vector_set_data_ptr(self.as_ptr(), ptr as *mut c_void) }
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
    /// Ensure the validity slice is writable.
    ///
    /// # SAFETY
    ///
    /// The provided capacity *must* be the actual capacity of this vector.
    pub unsafe fn ensure_validity_bitslice(&mut self, capacity: usize) -> &mut BitSlice<u64> {
        unsafe { self.ensure_validity_slice(capacity) }.view_bits_mut()
    }

    #[allow(clippy::expect_used)]
    /// Ensure the validity slice is writable.
    ///
    /// # SAFETY
    ///
    /// The provided capacity *must* be the actual capacity of this vector.
    pub unsafe fn ensure_validity_slice(&mut self, capacity: usize) -> &mut [u64] {
        unsafe {
            cpp::duckdb_vector_ensure_validity_writable(self.as_ptr());
            self.validity_slice_mut(capacity)
                .expect("we just ensured the validity slice is allocated")
        }
    }

    /// Returns the validity slice of the vector, if it exists.
    ///
    /// # SAFETY
    ///
    /// The provided capacity *must* be the actual capacity of this vector.
    pub unsafe fn validity_slice_mut(&mut self, capacity: usize) -> Option<&mut [u64]> {
        let ptr = unsafe { cpp::duckdb_vector_get_validity(self.as_ptr()) };
        unsafe { ptr.as_mut() }.map(|ptr| {
            let len = capacity.div_ceil(64);
            unsafe { std::slice::from_raw_parts_mut(ptr, len) }
        })
    }

    /// Returns the validity slice of the vector, if it exists.
    ///
    /// # SAFETY
    ///
    /// The provided capacity *must* be the actual capacity of this vector.
    pub unsafe fn validity_bitslice_mut(&mut self, capacity: usize) -> Option<&mut BitSlice<u64>> {
        unsafe { self.validity_slice_mut(capacity) }.map(|slice| slice.view_bits_mut())
    }

    pub fn validity_ref(&self, len: usize) -> ValidityRef<'_> {
        let validity_ptr = unsafe { cpp::duckdb_vector_get_validity(self.as_ptr()) };

        if validity_ptr.is_null() {
            // All values are valid - no null buffer needed
            return ValidityRef {
                validity: None,
                len,
            };
        }

        let num_bytes = len.div_ceil(64);

        ValidityRef {
            validity: Some(unsafe { std::slice::from_raw_parts(validity_ptr, num_bytes) }),
            len,
        }
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

    pub fn list_vector_reserve(&self, required_capacity: u64) -> VortexResult<()> {
        let state = unsafe { cpp::duckdb_list_vector_reserve(self.as_ptr(), required_capacity) };
        match state {
            cpp::duckdb_state::DuckDBSuccess => Ok(()),
            cpp::duckdb_state::DuckDBError => vortex_bail!("vector was nullptr!"),
        }
    }

    pub fn list_vector_get_child(&self) -> Self {
        unsafe { Self::borrow(cpp::duckdb_list_vector_get_child(self.as_ptr())) }
    }

    pub fn array_vector_get_child(&self) -> Self {
        // SAFETY: duckdb_array_vector_get_child dereferences the vector pointer which must be
        // valid and point to an ARRAY type vector. The returned child vector is borrowed and
        // remains valid as long as the parent vector is valid.
        unsafe { Self::borrow(cpp::duckdb_array_vector_get_child(self.as_ptr())) }
    }

    pub fn list_vector_set_size(&self, size: u64) -> VortexResult<()> {
        let state = unsafe { cpp::duckdb_list_vector_set_size(self.as_ptr(), size) };
        match state {
            cpp::duckdb_state::DuckDBSuccess => Ok(()),
            cpp::duckdb_state::DuckDBError => vortex_bail!("vector was nullptr!"),
        }
    }
}

pub struct ValidityRef<'a> {
    validity: Option<&'a [u64]>,
    len: usize,
}

impl ValidityRef<'_> {
    pub fn is_valid(&self, row: usize) -> bool {
        let Some(validity) = self.validity else {
            return true;
        };
        // Direct bit manipulation for better performance
        let entry_idx = row / 64;
        let idx_in_entry = row % 64;
        let validity_entry = validity[entry_idx];
        (validity_entry & (1u64 << idx_in_entry)) != 0
    }

    /// Creates a NullBuffer directly from the DuckDB validity mask for optimal performance.
    ///
    /// Returns None if all values are valid (no null buffer needed).
    pub fn to_null_buffer(&self) -> Option<arrow_buffer::NullBuffer> {
        let Some(validity) = self.validity else {
            // All values are valid - no null buffer needed
            return None;
        };

        // Create copy of the buffer from the DuckDB validity mask.
        let buffer = Buffer::from_iter(validity.iter().cloned());

        let boolean_buffer = arrow_buffer::BooleanBuffer::new(buffer, 0, self.len);
        Some(arrow_buffer::NullBuffer::new(boolean_buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpp::DUCKDB_TYPE;

    #[test]
    fn test_create_null_buffer_all_valid() {
        // Test case where all values are valid - should return None
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let vector = Vector::with_capacity(logical_type, len);

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        assert!(null_buffer.is_none(), "Expected None for all-valid vector");
    }

    #[test]
    fn test_create_null_buffer_with_nulls() {
        // Test case with some null values
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Set some positions as null
        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(1, false); // null at position 1
        validity_slice.set(3, false); // null at position 3
        validity_slice.set(7, false); // null at position 7

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        assert!(
            null_buffer.is_some(),
            "Expected Some(NullBuffer) for vector with nulls"
        );

        let null_buffer = null_buffer.unwrap();
        assert_eq!(null_buffer.len(), len);

        // Check that the right positions are null
        assert!(null_buffer.is_valid(0));
        assert!(null_buffer.is_null(1));
        assert!(null_buffer.is_valid(2));
        assert!(null_buffer.is_null(3));
        assert!(null_buffer.is_valid(4));
        assert!(null_buffer.is_valid(5));
        assert!(null_buffer.is_valid(6));
        assert!(null_buffer.is_null(7));
        assert!(null_buffer.is_valid(8));
        assert!(null_buffer.is_valid(9));
    }

    #[test]
    fn test_create_null_buffer_single_element() {
        // Test with a single element that is null
        let len = 1;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(0, false); // null at position 0

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        assert!(null_buffer.is_some());

        let null_buffer = null_buffer.unwrap();
        assert_eq!(null_buffer.len(), 1);
        assert!(null_buffer.is_null(0));
    }

    #[test]
    fn test_create_null_buffer_single_element_valid() {
        // Test with a single valid element
        let len = 1;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Ensure validity slice exists but don't set any nulls
        // SAFETY: Vector was created with this length.
        let _validity_slice = unsafe { vector.ensure_validity_bitslice(len) };

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        assert!(null_buffer.is_some());

        let null_buffer = null_buffer.unwrap();
        assert_eq!(null_buffer.len(), 1);
        assert!(null_buffer.is_valid(0));
    }

    #[test]
    fn test_create_null_buffer_empty() {
        // Test with zero length
        let len = 0;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let vector = Vector::with_capacity(logical_type, len);

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        // Even with zero length, if validity mask doesn't exist, should return None
        assert!(null_buffer.is_none());
    }

    #[test]
    fn test_create_null_buffer_all_nulls() {
        // Test case where all values are null
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        // Set all positions as null
        for i in 0..len {
            validity_slice.set(i, false);
        }

        let validity = vector.validity_ref(len);
        let null_buffer = validity.to_null_buffer();
        assert!(null_buffer.is_some());

        let null_buffer = null_buffer.unwrap();
        assert_eq!(null_buffer.len(), len);

        // Check that all positions are null
        for i in 0..len {
            assert!(null_buffer.is_null(i), "Element {i} should be null");
        }
    }

    #[test]
    fn test_row_is_null_all_valid() {
        // Test case where all values are valid (no validity mask)
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let vector = Vector::with_capacity(logical_type, len);

        let validity = vector.validity_ref(len);

        // When there's no validity mask, all rows should be valid (not null)
        for i in 0..len {
            assert!(validity.is_valid(i), "Row {i} should not be null");
        }
    }

    #[test]
    fn test_row_is_null_with_nulls() {
        // Test case with some null values
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Set some positions as null
        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(1, false); // null at position 1
        validity_slice.set(3, false); // null at position 3
        validity_slice.set(7, false); // null at position 7

        let validity = vector.validity_ref(len);

        // Check each position
        assert!(validity.is_valid(0), "Row 0 should not be null");
        assert!(!validity.is_valid(1), "Row 1 should be null");
        assert!(validity.is_valid(2), "Row 2 should not be null");
        assert!(!validity.is_valid(3), "Row 3 should be null");
        assert!(validity.is_valid(4), "Row 4 should not be null");
        assert!(validity.is_valid(5), "Row 5 should not be null");
        assert!(validity.is_valid(6), "Row 6 should not be null");
        assert!(!validity.is_valid(7), "Row 7 should be null");
        assert!(validity.is_valid(8), "Row 8 should not be null");
        assert!(validity.is_valid(9), "Row 9 should not be null");
    }

    #[test]
    fn test_row_is_null_all_nulls() {
        // Test case where all values are null
        let len = 10;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        // Set all positions as null
        for i in 0..len {
            validity_slice.set(i, false);
        }

        let validity = vector.validity_ref(len);

        // Check that all positions are null
        for i in 0..len {
            assert!(!validity.is_valid(i), "Row {i} should be null");
        }
    }

    #[test]
    fn test_row_is_null_single_element() {
        // Test with a single element that is null
        let len = 1;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(0, false); // null at position 0

        let validity = vector.validity_ref(len);

        assert!(!validity.is_valid(0), "Single element should be null");
    }

    #[test]
    fn test_row_is_null_single_element_valid() {
        // Test with a single valid element
        let len = 1;
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Ensure validity slice exists but element is valid
        // SAFETY: Vector was created with this length.
        let _validity_slice = unsafe { vector.ensure_validity_bitslice(len) };

        let validity = vector.validity_ref(len);

        assert!(validity.is_valid(0), "Single element should be valid");
    }

    #[test]
    fn test_row_is_null_byte_boundaries() {
        // Test bit manipulation across 64-bit boundaries
        let len = 128; // More than 64 bits
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };

        // Set specific positions as null to test bit manipulation
        validity_slice.set(0, false); // First bit of first u64
        validity_slice.set(63, false); // Last bit of first u64
        validity_slice.set(64, false); // First bit of second u64
        validity_slice.set(127, false); // Last bit of second u64 (if it exists)

        let validity = vector.validity_ref(len);

        // Test the null positions
        assert!(!validity.is_valid(0), "Row 0 should be null");
        assert!(!validity.is_valid(63), "Row 63 should be null");
        assert!(!validity.is_valid(64), "Row 64 should be null");
        if len > 127 {
            assert!(!validity.is_valid(127), "Row 127 should be null");
        }

        // Test some valid positions
        assert!(validity.is_valid(1), "Row 1 should be valid");
        assert!(validity.is_valid(32), "Row 32 should be valid");
        assert!(validity.is_valid(62), "Row 62 should be valid");
        assert!(validity.is_valid(65), "Row 65 should be valid");
    }

    #[test]
    fn test_dictionary() {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);

        let mut dict = Vector::with_capacity(logical_type.clone(), 2);
        let dict_slice = unsafe { dict.as_slice_mut::<i32>(2) };
        dict_slice[0] = 100;
        dict_slice[1] = 200;

        let vector = Vector::with_capacity(logical_type, 3);

        let mut sel_vec = SelectionVector::with_capacity(3);
        let sel_slice = unsafe { sel_vec.as_slice_mut(3) };
        sel_slice[0] = 0;
        sel_slice[1] = 1;
        sel_slice[2] = 0;

        vector.dictionary(&dict, 2, &sel_vec, 3);
        vector.flatten(3);

        assert_eq!(vector.as_slice_with_len::<i32>(3), &[100, 200, 100]);
    }
}
