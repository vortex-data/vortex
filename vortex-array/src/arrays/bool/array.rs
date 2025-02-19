use std::any::Any;
use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use arrow_buffer::BooleanBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};
use vortex_flatbuffers::dtype::Null;
use vortex_mask::Mask;

use crate::array::{Array, ArrayCanonicalImpl, ArrayRef, ArrayValidityImpl, ArrayVariantsImpl};
use crate::stats::StatsSet;
use crate::validity::Validity;
use crate::variants::BoolArrayTrait;
use crate::Canonical;

#[derive(Clone, Debug)]
pub struct BoolArray {
    dtype: DType,
    buffer: BooleanBuffer,
    validity: Validity,
    stats: StatsSet,
}

impl BoolArray {
    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Nullability`].
    pub fn new(buffer: BooleanBuffer, nullability: Nullability) -> Self {
        Self::new_unchecked(buffer, nullability.into())
    }

    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Validity`].
    ///
    /// Returns an error if the buffer and validity length mismatch.
    pub fn try_new(buffer: BooleanBuffer, validity: Validity) -> VortexResult<Self> {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() != len {
                vortex_bail!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len(),
                    len
                );
            }
        }
        Ok(Self::new_unchecked(buffer, validity))
    }

    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Validity`], without checking
    /// any invariants.
    pub fn new_unchecked(buffer: BooleanBuffer, validity: Validity) -> Self {
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats: StatsSet::default(),
        }
    }

    /// Returns the underlying [`BooleanBuffer`] of the array.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        &self.buffer
    }
}

impl Array for BoolArray {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    #[inline]
    fn to_array(&self) -> ArrayRef {
        Arc::new(self.clone())
    }

    #[inline]
    fn into_array(self) -> ArrayRef
    where
        Self: Sized,
    {
        Arc::new(self)
    }

    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl ArrayCanonicalImpl for BoolArray {
    #[inline]
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }

    #[inline]
    fn _to_builder(&self, _builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        todo!()
    }
}

impl ArrayValidityImpl for BoolArray {
    #[inline]
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    #[inline]
    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    #[inline]
    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    #[inline]
    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len())
    }
}

impl ArrayVariantsImpl for BoolArray {
    fn _as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for BoolArray {}
