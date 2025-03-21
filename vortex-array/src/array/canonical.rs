use std::sync::{Arc, OnceLock};

use vortex_error::VortexResult;

use super::{Array, ArrayRef, ArrayVariants, ArrayVisitor};
use crate::Canonical;
use crate::builders::ArrayBuilder;

/// Implementation trait for canonicalization functions.
///
/// These functions should not be called directly, rather their equivalents on the base
/// [`crate::Array`] trait should be used.
pub trait ArrayCanonicalImpl {
    /// Returns the canonical representation of the array.
    ///
    /// ## Post-conditions
    /// - The length is equal to that of the input array.
    /// - The [`vortex_dtype::DType`] is equal to that of the input array.
    fn _to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// ## Post-conditions
    /// - The length of the builder is incremented by the length of the input array.
    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let canonical = self._to_canonical()?;
        builder.extend_from_array(canonical.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct CachedCanonicalArray {
    inner: ArrayRef,
    canonical: Arc<OnceLock<Canonical>>,
}

impl CachedCanonicalArray {
    pub fn from(array: ArrayRef) -> ArrayRef {
        if array.is_canonical() {
            array
        } else {
            CachedCanonicalArray {
                inner: array.clone(),
                canonical: Arc::new(OnceLock::new()),
            }
            .into_array()
        }
    }
}

impl ArrayVisitor for CachedCanonicalArray {
    fn children(&self) -> Vec<ArrayRef> {
        self.inner.children()
    }

    fn nchildren(&self) -> usize {
        self.inner.nchildren()
    }

    fn children_names(&self) -> Vec<String> {
        self.inner.children_names()
    }

    fn buffers(&self) -> Vec<vortex_buffer::ByteBuffer> {
        self.inner.buffers()
    }

    fn nbuffers(&self) -> usize {
        self.inner.nbuffers()
    }

    fn metadata(&self) -> Option<Vec<u8>> {
        self.inner.metadata()
    }

    fn metadata_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.metadata_fmt(f)
    }
}

impl ArrayVariants for CachedCanonicalArray {
    fn as_null_typed(&self) -> Option<&dyn crate::variants::NullArrayTrait> {
        self.inner.as_null_typed()
    }

    fn as_bool_typed(&self) -> Option<&dyn crate::variants::BoolArrayTrait> {
        self.inner.as_bool_typed()
    }

    fn as_primitive_typed(&self) -> Option<&dyn crate::variants::PrimitiveArrayTrait> {
        self.inner.as_primitive_typed()
    }

    fn as_utf8_typed(&self) -> Option<&dyn crate::variants::Utf8ArrayTrait> {
        self.inner.as_utf8_typed()
    }

    fn as_binary_typed(&self) -> Option<&dyn crate::variants::BinaryArrayTrait> {
        self.inner.as_binary_typed()
    }

    fn as_struct_typed(&self) -> Option<&dyn crate::variants::StructArrayTrait> {
        self.inner.as_struct_typed()
    }

    fn as_list_typed(&self) -> Option<&dyn crate::variants::ListArrayTrait> {
        self.inner.as_list_typed()
    }

    fn as_extension_typed(&self) -> Option<&dyn crate::variants::ExtensionArrayTrait> {
        self.inner.as_extension_typed()
    }
}

impl Array for CachedCanonicalArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self.inner.clone().as_any_arc()
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(self.clone())
    }

    fn into_array(self) -> ArrayRef
    where
        Self: Sized,
    {
        Arc::new(self)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn dtype(&self) -> &vortex_dtype::DType {
        self.inner.dtype()
    }

    fn encoding(&self) -> crate::EncodingId {
        self.inner.encoding()
    }

    fn vtable(&self) -> crate::vtable::VTableRef {
        self.inner.vtable()
    }

    fn find_kernel(
        &self,
        compute_fn: &dyn crate::compute::ComputeFn,
    ) -> Option<crate::compute::KernelRef> {
        self.inner.find_kernel(compute_fn)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.inner.is_valid(index)
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.inner.is_invalid(index)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        self.inner.all_valid()
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        self.inner.all_invalid()
    }

    fn valid_count(&self) -> VortexResult<usize> {
        self.inner.valid_count()
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        self.inner.invalid_count()
    }

    fn validity_mask(&self) -> VortexResult<vortex_mask::Mask> {
        self.inner.validity_mask()
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        self.inner.append_to_builder(builder)
    }

    fn statistics(&self) -> crate::stats::StatsSetRef<'_> {
        self.inner.statistics()
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        // unused cache
        let _ = self
            .canonical
            .get_or_try_init(|| self.inner.to_canonical())
            .cloned();
        self.inner.to_canonical()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn to_canonical_impl(&self) -> VortexResult<Canonical> {
        self.inner.to_canonical_impl()
    }
}
