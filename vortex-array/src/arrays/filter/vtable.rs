// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_compute::filter::Filter;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::Precision;
use crate::arrays::filter::array::FilterArray;
use crate::arrays::filter::execute::execute_filter;
use crate::arrays::filter::execute::execute_filter_fast_paths;
use crate::arrays::filter::rules::PARENT_RULES;
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

vtable!(Filter);

#[derive(Debug)]
pub struct FilterVTable;

impl FilterVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.filter");
}

impl VTable for FilterVTable {
    type Array = FilterArray;
    type Metadata = FilterMetadata;
    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(FilterMetadata(array.mask.clone()))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // TODO(joe): make this configurable
        vortex_bail!("Filter array is not serializable")
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("Filter array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &FilterMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        assert_eq!(len, metadata.0.true_count());
        let child = children.get(0, dtype, metadata.0.len())?;
        Ok(FilterArray {
            child,
            mask: metadata.0.clone(),
            stats: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "FilterArray expects exactly 1 child, got {}",
            children.len()
        );
        array.child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(canonical) = execute_filter_fast_paths(array, ctx)? {
            return Ok(canonical);
        }
        let Mask::Values(mask_values) = &array.mask else {
            unreachable!("`execute_filter_fast_paths` handles AllTrue and AllFalse")
        };

        // We rely on the optimization pass that runs prior to this execution for filter pushdown,
        // so now we can just execute the filter without worrying.
        let canonical = execute_filter(array.child.clone().execute(ctx)?, mask_values);

        // Verify the resulting length and type.
        let result_len = array.mask.true_count();
        vortex_ensure!(
            canonical.as_ref().len() == result_len,
            "Filter result length mismatch: expected {}, got {}",
            result_len,
            canonical.as_ref().len()
        );
        vortex_ensure!(
            canonical.as_ref().dtype() == array.dtype(),
            "Filter result dtype mismatch: expected {:?}, got {:?}",
            array.dtype(),
            canonical.as_ref().dtype()
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
}

impl BaseArrayVTable<FilterVTable> for FilterVTable {
    fn len(array: &FilterArray) -> usize {
        array.mask.true_count()
    }

    fn dtype(array: &FilterArray) -> &DType {
        array.child.dtype()
    }

    fn stats(array: &FilterArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &FilterArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.mask.array_hash(state, precision);
    }

    fn array_eq(array: &FilterArray, other: &FilterArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision) && array.mask.array_eq(&other.mask, precision)
    }
}

impl OperationsVTable<FilterVTable> for FilterVTable {
    fn scalar_at(array: &FilterArray, index: usize) -> VortexResult<Scalar> {
        let rank_idx = array.mask.rank(index);
        array.child.scalar_at(rank_idx)
    }
}

impl ValidityVTable<FilterVTable> for FilterVTable {
    fn validity(array: &FilterArray) -> VortexResult<Validity> {
        array.child.validity()?.filter(&array.mask)
    }

    fn validity_mask(array: &FilterArray) -> VortexResult<Mask> {
        Ok(Filter::filter(&array.child.validity_mask()?, &array.mask))
    }
}

impl VisitorVTable<FilterVTable> for FilterVTable {
    fn visit_buffers(_array: &FilterArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FilterArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
    }
}

pub struct FilterMetadata(pub(super) Mask);

impl Debug for FilterMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} / {} => {}",
            self.0.true_count(),
            self.0.len(),
            self.0.density()
        )
    }
}
