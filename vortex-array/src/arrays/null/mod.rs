// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::Precision;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::BaseArrayVTable;
use crate::vtable::CanonicalVTable;
use crate::vtable::NotSupported;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

mod compute;

vtable!(Null);

impl VTable for NullVTable {
    type Array = NullArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.null")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        NullVTable.as_vtable()
    }

    fn metadata(_array: &NullArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        _dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<NullArray> {
        Ok(NullArray::new(len))
    }

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "NullArray has no children, got {}",
            children.len()
        );
        Ok(())
    }
}

/// A array where all values are null.
///
/// This mirrors the Apache Arrow Null array encoding and provides an efficient representation
/// for arrays containing only null values. No actual data is stored, only the length.
///
/// All operations on null arrays return null values or indicate invalid data.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::NullArray;
/// use vortex_array::IntoArray;
///
/// // Create a null array with 5 elements
/// let array = NullArray::new(5);
///
/// // Slice the array - still contains nulls
/// let sliced = array.slice(1..3);
/// assert_eq!(sliced.len(), 2);
///
/// // All elements are null
/// let scalar = array.scalar_at(0);
/// assert!(scalar.is_null());
/// ```
#[derive(Clone, Debug)]
pub struct NullArray {
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct NullVTable;

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            stats_set: Default::default(),
        }
    }
}

impl BaseArrayVTable<NullVTable> for NullVTable {
    fn len(array: &NullArray) -> usize {
        array.len
    }

    fn dtype(_array: &NullArray) -> &DType {
        &DType::Null
    }

    fn stats(array: &NullArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &NullArray, state: &mut H, _precision: Precision) {
        array.len.hash(state);
    }

    fn array_eq(array: &NullArray, other: &NullArray, _precision: Precision) -> bool {
        array.len == other.len
    }
}

impl VisitorVTable<NullVTable> for NullVTable {
    fn visit_buffers(_array: &NullArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(_array: &NullArray, _visitor: &mut dyn ArrayChildVisitor) {}
}

impl CanonicalVTable<NullVTable> for NullVTable {
    fn canonicalize(array: &NullArray) -> Canonical {
        Canonical::Null(array.clone())
    }
}

impl OperationsVTable<NullVTable> for NullVTable {
    fn slice(_array: &NullArray, range: Range<usize>) -> ArrayRef {
        NullArray::new(range.len()).into_array()
    }

    fn scalar_at(_array: &NullArray, _index: usize) -> Scalar {
        Scalar::null(DType::Null)
    }
}

impl ValidityVTable<NullVTable> for NullVTable {
    fn is_valid(_array: &NullArray, _index: usize) -> bool {
        false
    }

    fn all_valid(array: &NullArray) -> bool {
        array.is_empty()
    }

    fn all_invalid(array: &NullArray) -> bool {
        !array.is_empty()
    }

    fn validity(_array: &NullArray) -> VortexResult<Validity> {
        Ok(Validity::AllInvalid)
    }

    fn validity_mask(array: &NullArray) -> Mask {
        Mask::AllFalse(array.len)
    }
}
