// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_traits::{ConstOne, PrimInt};
use vortex_array::Array;
use vortex_array::operator::slice::SliceOperator;
use vortex_array::operator::{
    LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::vec::Selection;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{
    BindContext, Element, Kernel, KernelContext, N, PipelinedOperator, RowSelection,
};
use vortex_array::vtable::PipelineVTable;
use vortex_dtype::{DType, NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_err};

use crate::{SequenceArray, SequenceVTable};

impl PipelineVTable<SequenceVTable> for SequenceVTable {
    fn to_operator(array: &SequenceArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for SequenceArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.base().hash(state);
        self.multiplier().hash(state);
        self.dtype().hash(state);
        self.bounds().hash(state);
    }
}

impl OperatorEq for SequenceArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.base() == other.base()
            && self.multiplier() == other.multiplier()
            && self.dtype() == other.dtype()
            && self.bounds() == other.bounds()
    }
}

impl Operator for SequenceArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        Array::dtype(self.as_ref())
    }

    fn bounds(&self) -> LengthBounds {
        Array::len(self.as_ref()).into()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn reduce_parent(
        &self,
        parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        // Push down slice
        if let Some(slice) = parent.as_any().downcast_ref::<SliceOperator>() {
            let range = slice.range();
            return Ok(Some(Arc::new(SequenceArray::unchecked_new(
                self.index_value(range.start),
                self.multiplier(),
                self.ptype(),
                self.dtype().nullability(),
                range.len(),
            ))));
        }

        Ok(None)
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

impl PipelinedOperator for SequenceArray {
    fn row_selection(&self) -> RowSelection {
        RowSelection::Domain(self.as_ref().len())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        Ok(match_each_integer_ptype!(self.ptype(), |T| {
            if self.multiplier().as_primitive::<T>() == <T as ConstOne>::ONE {
                Box::new(SequenceKernel::<T> {
                    base: self.base().as_primitive::<T>(),
                    len: Array::len(self.as_ref()),
                })
            } else {
                Box::new(MultiplierSequenceKernel::<T> {
                    base: self.base().as_primitive::<T>(),
                    multiplier: self.multiplier().as_primitive::<T>(),
                    len: Array::len(self.as_ref()),
                })
            }
        }))
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

struct SequenceKernel<T> {
    base: T,
    len: usize,
}

impl<T: Element + NativePType + PrimInt> Kernel for SequenceKernel<T> {
    fn step(
        &self,
        _ctx: &KernelContext,
        step_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        // TODO(ngates): benchmark and optimize this
        let values = out.as_array_mut::<T>();
        let offset = step_idx * N;

        // Check if we're in the final chunk to avoid overflow
        if (offset + N) > self.len {
            selection.try_iter_ones(|i| {
                values[i] = self.base
                    + T::from_usize(offset + i)
                        .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?;
                Ok(())
            })?;
        } else {
            for i in 0..N {
                values[i] = self.base
                    + T::from_usize(offset + i)
                        .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?;
            }
        }

        match selection.true_count() {
            0 | N => out.set_selection(Selection::Prefix),
            _ => out.set_selection(Selection::Mask),
        }

        Ok(())
    }
}

struct MultiplierSequenceKernel<T> {
    base: T,
    multiplier: T,
    len: usize,
}

impl<T: Element + NativePType + PrimInt> Kernel for MultiplierSequenceKernel<T> {
    fn step(
        &self,
        _ctx: &KernelContext,
        chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        // TODO(ngates): benchmark and optimize this. We should use addition not multiplication
        let values = out.as_array_mut::<T>();
        let offset = chunk_idx * N;

        if (offset + N) > self.len {
            selection.try_iter_ones(|i| {
                values[i] = self.base
                    + self
                        .multiplier
                        .checked_mul(
                            &T::from_usize(offset + i)
                                .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?,
                        )
                        .ok_or_else(|| vortex_err!("Overflow computing sequence value"))?;
                Ok(())
            })?;
        } else {
            for i in 0..N {
                values[i] = self.base
                    + self
                        .multiplier
                        .checked_mul(
                            &T::from_usize(offset + i)
                                .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?,
                        )
                        .ok_or_else(|| vortex_err!("Overflow computing sequence value"))?;
            }
        }

        match selection.true_count() {
            0 | N => out.set_selection(Selection::Prefix),
            _ => out.set_selection(Selection::Mask),
        }

        Ok(())
    }
}
