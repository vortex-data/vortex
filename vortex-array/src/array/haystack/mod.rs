// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lookups of needles among the elements of an array.
//!
//! An [`ArrayHaystack`] holds the non-null elements of a set as a lookup array: an array whose
//! encoding is built to find needles, and which canonicalizes to the elements it holds. Vortex
//! builds a sorted, a bitmap, a hashed or a row haystack from the elements. An encoding outside
//! Vortex becomes a lookup array by implementing [`HaystackVTable`].

mod bytes;
mod integer;
mod rows;
#[cfg(test)]
mod tests;

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// An encoding whose arrays can be the lookup array of an [`ArrayHaystack`].
///
/// Every row of a lookup array is an element of the set, so a lookup array holds no null rows. It
/// may hold an element more than once.
pub trait HaystackVTable: VTable {
    /// One bit per needle, set when the needle is an element of `array`, and the validity of the
    /// needles.
    ///
    /// The needles have the dtype of `array`, ignoring nullability. The bit of a null needle has
    /// no effect.
    fn contains(
        array: ArrayView<'_, Self>,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)>;
}

/// The [`HaystackVTable::contains`] of the encoding of a lookup array, with the array erased.
type ContainsFn =
    fn(&ArrayRef, &ArrayRef, &mut ExecutionCtx) -> VortexResult<(BitBuffer, Validity)>;

/// Looks up needles among the elements of a lookup array, for many batches of needles.
///
/// The haystack counterpart of [`RepeatedArrayProbe`](crate::RepeatedArrayProbe): where a probe
/// reads the rows of an array by index, a haystack answers whether each row of a batch of
/// needles is an element of the array. A haystack pays off only over repeated lookups, so it
/// has no one-off form. It is built eagerly. A holder that must build it lazily, or share it
/// between threads, keeps it in a shared cell.
#[derive(Clone)]
pub struct ArrayHaystack {
    array: ArrayRef,
    contains: ContainsFn,
}

impl ArrayHaystack {
    /// A haystack over a lookup array of your own encoding.
    pub fn new<V: HaystackVTable>(array: Array<V>) -> Self {
        Self {
            array: array.into_array(),
            contains: contains_of::<V>,
        }
    }

    /// Builds the haystack of the elements of `array`.
    ///
    /// A null element never equals a needle, so the haystack leaves null elements out. Integers,
    /// and floats by their bit patterns, go into a bitmap over their span when the span is dense,
    /// and into a sorted array otherwise, as do decimals by their unscaled values. UTF-8 and
    /// binary values go into hash tables, and nested values into sorted row indices with the same
    /// comparator as equality.
    ///
    /// # Errors
    ///
    /// Fails when the elements cannot be executed.
    pub fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let valid = array.validity()?.execute_mask(array.len(), ctx)?;
        let members = if valid.all_true() {
            array.clone()
        } else {
            array.filter(valid)?
        };

        match members.dtype() {
            DType::Primitive(..) | DType::Decimal(..) => integer::try_new(members, ctx),
            DType::Utf8(_) | DType::Binary(_) => Ok(Self::new(bytes::try_new(members, ctx)?)),
            _ => Ok(Self::new(rows::try_new(members, ctx)?)),
        }
    }

    /// The lookup array, which canonicalizes to the elements of the haystack.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// One bit per needle, set when the needle is an element, and the validity of the needles.
    ///
    /// The needles have the dtype of the lookup array, ignoring nullability.
    ///
    /// # Errors
    ///
    /// Fails when the needles cannot be executed.
    pub fn contains(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        (self.contains)(&self.array, needles, ctx)
    }
}

fn contains_of<V: HaystackVTable>(
    array: &ArrayRef,
    needles: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(BitBuffer, Validity)> {
    V::contains(array.as_::<V>(), needles, ctx)
}

/// The operations and validity that the lookup arrays of Vortex share.
///
/// Every row is valid. A row is read from the canonical form, since a lookup array exists for
/// lookups, not for reads.
#[derive(Clone, Debug)]
pub(crate) struct HaystackOps;

impl<V: VTable> ValidityVTable<V> for HaystackOps {
    fn validity(array: ArrayView<'_, V>) -> VortexResult<Validity> {
        Ok(Validity::from(array.dtype().nullability()))
    }
}

impl<V: VTable> OperationsVTable<V> for HaystackOps {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array
            .array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array()
            .execute_scalar(index, ctx)
    }
}

/// The child in `slot` of a lookup array of Vortex, which validation found present.
fn slot(slots: &[Option<ArrayRef>], slot: usize) -> &ArrayRef {
    slots[slot]
        .as_ref()
        .unwrap_or_else(|| vortex_error::vortex_panic!("Haystack slot {slot} must be present"))
}

/// The buffer and serialization methods that every lookup array of Vortex shares: it keeps its
/// data in children and in memory, and it is never serialized.
macro_rules! haystack_vtable_common {
    ($name:literal) => {
        fn nbuffers(_array: $crate::array::ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(
            _array: $crate::array::ArrayView<'_, Self>,
            _idx: usize,
        ) -> $crate::buffer::BufferHandle {
            ::vortex_error::vortex_panic!("A haystack has no buffers")
        }

        fn buffer_name(_array: $crate::array::ArrayView<'_, Self>, _idx: usize) -> Option<String> {
            None
        }

        fn with_buffers(
            &self,
            array: $crate::array::ArrayView<'_, Self>,
            buffers: &[$crate::buffer::BufferHandle],
        ) -> ::vortex_error::VortexResult<$crate::array::ArrayParts<Self>> {
            $crate::array::with_empty_buffers(self, array, buffers)
        }

        fn serialize(
            _array: $crate::array::ArrayView<'_, Self>,
            _session: &::vortex_session::VortexSession,
        ) -> ::vortex_error::VortexResult<Option<Vec<u8>>> {
            ::vortex_error::vortex_bail!("{} is never serialized", $name)
        }

        fn deserialize(
            &self,
            _dtype: &$crate::dtype::DType,
            _len: usize,
            _metadata: &[u8],
            _buffers: &[$crate::buffer::BufferHandle],
            _children: &dyn $crate::serde::ArrayChildren,
            _session: &::vortex_session::VortexSession,
        ) -> ::vortex_error::VortexResult<$crate::array::ArrayParts<Self>> {
            ::vortex_error::vortex_bail!("{} is never serialized", $name)
        }
    };
}

use haystack_vtable_common;
