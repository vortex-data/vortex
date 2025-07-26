// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_mask::Mask;

/// A vector is the atomic unit of data in Vortex.
///
/// TODO(ngates): is it self-contained? i.e. does it store a DType, and stats, etc.? Surely not...
///   Is it generically typed? Would be nice if it were, over the element type for example. But
///   it's much better to have a consistent shape that can be downcast into impls.
///
/// One problem here is we can't really wrap up a mutable elements array from some external source.
/// We could solve this using unsafe and a lifetime object (e.g. Arc<dyn Any>), as well as storing
/// the raw pointer. Managing who has write access is quite fiddly though. For now, we just use
/// regular vortex buffers and copy into the external source when needed.
///
/// The next problem is custom vectors, e.g. what if we have VarBinView with FSST encoding? I
/// guess the thing here is that some view vectors want to defer access to their children or
/// data buffers somehow. But eventually, they do want to access the underlying data. We also need
/// to decide whether such view vectors use pointers or offsets to access their data.
pub struct Vector {
    // The buffer containing the fixed-width elements of the vector.
    elements: Option<ByteBuffer>,
    // The validity mask for the vector, indicating which elements are valid.
    validity: Mask,
    // A selection over the elements and validity of the vector.
    selection: Selection,
    // Additional buffers of data used by the vector, such as string data.
    data: Vec<ByteBuffer>,
    // Additional vectors used by the vector, such as dictionary values.
    children: Vec<Vector>,
}

pub enum Selection {
    // Select all elements in the vector, up to the given length.
    All { len: usize },
    // The element in the vector to be considered the constant value.
    Constant { element: usize, len: usize },
    // A selection that is a boolean mask, with true indicating the element is selected.
    Filter(Mask),
    // A selection that is a list of indices, with each index indicating an element to select.
    // Enabling duplicate values, reordering, etc.
    Indices(Vec<usize>),
}

/// Primitive access to a vector's elements.
pub struct PrimitiveVector<'a, T: NativePType> {
    vector: &'a mut Vector,
}

impl<'a, T: NativePType> AsMut<[T]> for PrimitiveVector<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        let elements = self
            .vector
            .elements
            .as_mut()
            .vortex_expect("Vector has no elements");
        let len = elements.len() / size_of::<T>();

        let ptr = elements.as_mut_ptr() as *mut T;
        assert!(ptr.is_aligned(), "Pointer is not aligned to T's alignment");
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}
