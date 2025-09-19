// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BindContext, Operator, OperatorId, OperatorRef, PipelinedOperator, VectorId,
};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext};
use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::slice;
use std::sync::Arc;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::Mask;

#[derive(Debug)]
pub struct FilterOperator {
    child: OperatorRef,
    mask: Mask,
}

impl PartialEq for FilterOperator {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.mask.eq(&other.mask)
    }
}
impl Eq for FilterOperator {}

impl Hash for FilterOperator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        // Hash the discriminant first
        std::mem::discriminant(&self.mask).hash(state);
        match &self.mask {
            Mask::AllTrue(len) => len.hash(state),
            Mask::AllFalse(len) => len.hash(state),
            Mask::Values(values) => {
                Arc::as_ptr(values).hash(state);
            }
        }
    }
}

impl FilterOperator {
    pub fn new(child: OperatorRef, mask: Mask) -> FilterOperator {
        assert_eq!(
            child.len(),
            mask.len(),
            "Mask length must match child length"
        );
        FilterOperator { child, mask }
    }
}

impl Operator for FilterOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.filter")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn len(&self) -> usize {
        self.mask.true_count()
    }

    fn children(&self) -> &[OperatorRef] {
        slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(FilterOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            mask: self.mask.clone(),
        }))
    }

    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
        // If none of the children are position-preserving, we cannot push down the filter.
        if !self
            .child
            .children()
            .iter()
            .enumerate()
            .any(|(i, child)| child.is_position_preserving(i).unwrap_or_default())
        {
            return Ok(None);
        }

        // We push down the filter operator to any child that is aligned to the parent.
        let children = (0..self.child.nchildren())
            .map(|i| {
                let child = self.child.children()[i].clone();

                if child.is_position_preserving(i).unwrap_or_default() {
                    // Push-down the filter to this child.
                    Arc::new(FilterOperator::new(child, self.mask.clone()))
                } else {
                    child
                }
            })
            .collect();

        Ok(Some(self.child.clone().with_children(children)?))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        // TODO(ngates): do we decide if we're super sparse that we should be batch executed?
        //  Seems like a weird place to make that decision though...
        Some(self)
    }
}

impl PipelinedOperator for FilterOperator {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let child = ctx.children()[0];
        match self.dtype() {
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    Ok(Box::new(PrimitiveFilterKernel::<T> {
                        child,
                        mask: self.mask.clone(),
                        _marker: std::marker::PhantomData,
                    }) as Box<dyn Kernel>)
                })
            }
            _ => vortex_bail!("FilterOperator only supports primitive dtypes"),
        }
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![0]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

struct PrimitiveFilterKernel<T> {
    child: VectorId,
    mask: Mask,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Element + NativePType> Kernel for PrimitiveFilterKernel<T> {
    fn step(&mut self, ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        todo!()
    }
}
