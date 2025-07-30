// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{NativePType, PType};
use vortex_error::VortexExpect;
use vortex_mask::Mask;

/// A vector is the atomic unit of data in Vortex.
///
/// TODO(ngates): is it self-contained? i.e. does it store a DType, and stats, etc.? Surely not...
///   Is it generically typed? Would be nice if it were, over the element type for example. But
///   it's much better to have a consistent shape that can be downcast into impls?
///
/// One problem here is we can't really wrap up a mutable elements array from some external source.
/// We could solve this using unsafe and a lifetime object (e.g. Arc<dyn Any>), as well as storing
/// the raw pointer. Managing who has write access is quite fiddly though. For now, we just use
/// regular vortex buffers and copy into the external source when needed.
///
/// ## Why is this a single type-erased struct?
///
/// If we used generics at this level, we would taint all functions that use this type with a
/// generic type. To remove the generic, we'd need to introduce a trait, at which point we're
/// forced into both dynamic dispatch, and boxed return types. It's not a terrible idea?
/// The other thing we want to support is implementing common operations over only vectors, for
/// example filter and take can be implemented by shuffling the elements list only. But we can also
/// do this with traits.
///
/// Maybe one benefit is that e.g. exporting a constant vector just writes data into the actual
/// vector, rather than constructing heap-allocated scalars. Particularly useful for nested data.
/// In other words, the pre-allocated buffers can be re-used even when switching between vector
/// types. But this can also be done with into_parts style operations.
///
/// ## How do we handle custom encodings, e.g. FSST, RoaringBitMap, ZStd, etc.?
///
/// I could imagine a VarBinView vector (i.e. it has 16-byte views in its elements buffer), but
/// is able to delay decompression of the data buffers. These could be stored as child arrays and
/// decompressed on-demand, since this is now an opaque operation (and then call export on the
/// child data arrays using a slices mask? We'd be masking binary data... that sounds slow))
///
/// What about dictionary arrays? Are they even important at this level? Well, they are for export
/// to DuckDB, since we can return a DictionaryVector. But maybe that logic is held within the
/// export_to_duckdb compute function, and therefore it runs an export of the values array first,
/// before exporting the codes arrays and directly returning the result to DuckDB. This would only
/// work for top-level dictionaries, but nested dictionaries are probably gross anyway!
///
/// ## Can we re-use parts of the pipeline to avoid common-subexpression elimination?
///
/// This gets tricky... Suppose we start with a Vortex expression. We can then pass that naively
/// through pipeline construction. This now represents a physical execution plan. At this point,
/// we could in theory optimize the pipeline by removing common sub-expressions, such as
/// canonicalizing the same field multiple times to pass into two comparison operators.
///
/// We'd then need some way to buffer the intermediate results as both expressions are driven.
/// Maybe this works?
pub struct Vector {
    vtype: VType,
    // The buffer containing the fixed-width elements of the vector.
    elements: ByteBuffer,
    // The validity mask for the vector, indicating which elements are valid.
    validity: Mask,
    // A selection over the elements and validity of the vector.
    // FIXME(ngates): using a selection mask means rank-based operations are expensive, vs
    //  selection indices which are always constant time.
    selection: Selection,
    // Additional buffers of data used by the vector, such as string data.
    data: Vec<ByteBuffer>,
    // Additional vectors used by the vector, such as dictionary values. Maybe these should be
    // arrays actually?
    children: Vec<Vector>,
}

/// Matches the variant types of our logical DTypes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VType {
    Primitive(PType),
    VarBin,
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

impl Vector {
    pub fn as_primitive_mut<T: NativePType>(&mut self) -> PrimitiveVectorMut<'_, u8> {
        assert_eq!(
            self.vtype,
            VType::Primitive(T::PTYPE),
            "Vector is not primitive"
        );
        PrimitiveVectorMut::<T>(self)
    }
}

/// Primitive access to a vector's elements.
pub struct PrimitiveVector<'a, T: NativePType>(&'a Vector);
pub struct PrimitiveVectorMut<'a, T: NativePType>(&'a mut Vector);

impl<'a, T: NativePType> Deref for PrimitiveVectorMut<'a, T> {
    type Target = PrimitiveVector<'a, T>;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute::<&PrimitiveVectorMut<'a, T>, &PrimitiveVector<'a, T>>(&self) }
    }
}

impl<'a, T: NativePType> AsMut<[T]> for PrimitiveVectorMut<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        let elements = self
            .0
            .elements
            .as_mut()
            .vortex_expect("Vector has no elements");
        let len = elements.len() / size_of::<T>();

        let ptr = elements.as_mut_ptr() as *mut T;
        assert!(ptr.is_aligned(), "Pointer is not aligned to T's alignment");
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}
