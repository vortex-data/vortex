// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::take::array::TakeArray;
use crate::arrays::take::execute::execute_take;
use crate::arrays::take::execute::execute_take_fast_paths;
use crate::arrays::take::rules::PARENT_RULES;
use crate::arrays::take::rules::RULES;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::BaseArrayVTable;
use crate::vtable::NotSupported;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

vtable!(Take);

#[derive(Debug)]
pub struct TakeVTable;

impl TakeVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.take");
}

impl VTable for TakeVTable {
    type Array = TakeArray;
    type Metadata = TakeMetadata;
    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(TakeMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Take array is not serializable")
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("Take array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &TakeMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        // The indices child determines the length - use u64 as the index type
        let indices_dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        let indices = children.get(1, &indices_dtype, len)?;
        let child = children.get(0, dtype, 0)?; // child length is unknown from metadata
        Ok(TakeArray {
            child,
            indices,
            stats: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "TakeArray expects exactly 2 children, got {}",
            children.len()
        );
        let mut iter = children.into_iter();
        array.child = iter
            .next()
            .vortex_expect("children length already validated");
        array.indices = iter
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn canonicalize(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(canonical) = execute_take_fast_paths(array, ctx)? {
            return Ok(canonical);
        }

        // Execute both child and indices to canonical form
        let child_canonical = array.child.clone().execute::<Canonical>(ctx)?;
        let indices_canonical = array.indices.clone().execute::<Canonical>(ctx)?;

        // Execute the take operation
        let canonical = execute_take(child_canonical, indices_canonical.into_array())?;

        // Verify the resulting length and type
        let result_len = array.indices.len();
        vortex_ensure!(
            canonical.as_ref().len() == result_len,
            "Take result length mismatch: expected {}, got {}",
            result_len,
            canonical.as_ref().len()
        );

        Ok(canonical)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }
}

impl BaseArrayVTable<TakeVTable> for TakeVTable {
    fn len(array: &TakeArray) -> usize {
        array.indices.len()
    }

    fn dtype(array: &TakeArray) -> &DType {
        // The dtype is the child's dtype with potentially nullable from indices
        // For now, return child's dtype - nullability adjustment happens during execution
        array.child.dtype()
    }

    fn stats(array: &TakeArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &TakeArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.indices.array_hash(state, precision);
    }

    fn array_eq(array: &TakeArray, other: &TakeArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
            && array.indices.array_eq(&other.indices, precision)
    }
}

impl OperationsVTable<TakeVTable> for TakeVTable {
    fn scalar_at(array: &TakeArray, index: usize) -> VortexResult<Scalar> {
        // Get the index value at position `index` from the indices array
        let idx_scalar = array.indices.scalar_at(index)?;
        if idx_scalar.is_null() {
            return Ok(Scalar::null(array.child.dtype().as_nullable()));
        }
        let idx: usize = idx_scalar.as_ref().try_into()?;
        array.child.scalar_at(idx)
    }
}

impl ValidityVTable<TakeVTable> for TakeVTable {
    fn validity(array: &TakeArray) -> VortexResult<Validity> {
        // The validity of a take array depends on both:
        // 1. The validity of the indices (null indices produce null values)
        // 2. The validity of the child at the taken positions
        // We return the child's validity taken at the indices positions
        array.child.validity()?.take(&array.indices)
    }
}

impl VisitorVTable<TakeVTable> for TakeVTable {
    fn visit_buffers(_array: &TakeArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &TakeArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
        visitor.visit_child("indices", &array.indices);
    }
}

pub struct TakeMetadata;

impl Debug for TakeMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TakeMetadata")
    }
}
