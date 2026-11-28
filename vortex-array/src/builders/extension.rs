// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::ChildBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::ExtScalar;
use crate::scalar::Scalar;

/// The builder for building a [`ExtensionArray`].
pub struct ExtensionBuilder {
    dtype: DType,
    storage: ChildBuilder,
}

impl ExtensionBuilder {
    /// Creates a new `ExtensionBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(ext_dtype: ExtDTypeRef) -> Self {
        Self::new_in(ext_dtype, BufferAllocatorRef::static_ref())
    }

    /// Creates a new `ExtensionBuilder` with the default capacity using `allocator`.
    pub fn new_in(ext_dtype: ExtDTypeRef, allocator: &BufferAllocatorRef) -> Self {
        Self::with_capacity_in(ext_dtype, DEFAULT_BUILDER_CAPACITY, allocator)
    }

    /// Creates a new `ExtensionBuilder` with the given `capacity`.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(ext_dtype: ExtDTypeRef, capacity: usize) -> Self {
        Self::with_capacity_in(ext_dtype, capacity, BufferAllocatorRef::static_ref())
    }

    /// Creates a new `ExtensionBuilder` with `capacity` using `allocator`.
    pub fn with_capacity_in(
        ext_dtype: ExtDTypeRef,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self {
            storage: ChildBuilder::with_capacity(ext_dtype.storage_dtype(), capacity, allocator),
            dtype: DType::Extension(ext_dtype),
        }
    }

    /// Appends an extension `value` to the builder.
    pub fn append_value(&mut self, value: ExtScalar) -> VortexResult<()> {
        self.storage.append_scalar(&value.to_storage_scalar())
    }

    /// Appends the values of a canonical [`ExtensionArray`] to the builder by appending its
    /// storage array to the underlying storage builder.
    pub(crate) fn append_extension_array(
        &mut self,
        array: ArrayView<'_, Extension>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.storage.append_array(array.storage_array(), ctx)
    }

    /// Finishes the builder directly into a [`ExtensionArray`].
    pub fn finish_into_extension(&mut self) -> ExtensionArray {
        let storage = self.storage.finish();
        // SAFETY: the storage child builder was created with this extension dtype's storage dtype
        // and rejects an appended array of any other dtype.
        unsafe { ExtensionArray::new_unchecked(self.ext_dtype(), storage) }
    }

    /// The [`ExtDType`] of this builder.
    ///
    /// [`ExtDType`]: crate::dtype::extension::ExtDType
    fn ext_dtype(&self) -> ExtDTypeRef {
        if let DType::Extension(ext_dtype) = &self.dtype {
            ext_dtype.clone()
        } else {
            unreachable!()
        }
    }
}

impl ArrayBuilder for ExtensionBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.storage.append_zeros(n)
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.storage.append_nulls(n)
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "ExtensionBuilder received a scalar with the wrong dtype"
        );

        self.append_value(scalar.as_extension())
    }

    fn reserve_chunks(&mut self, additional: usize) {
        self.storage.reserve_chunks(additional)
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.storage.reserve_exact(capacity)
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_extension().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Extension(self.finish_into_extension())
    }
}
