// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::operator::{LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef};
use crate::pipeline::bits::BitView;
use crate::pipeline::vec::Selection;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{
    BindContext, Element, Kernel, KernelContext, N, PipelinedOperator, RowSelection,
};
use crate::vtable::PipelineVTable;

impl PipelineVTable<ConstantVTable> for ConstantVTable {
    fn to_operator(array: &ConstantArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for ConstantArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.scalar.hash(state);
        self.len.hash(state);
    }
}

impl OperatorEq for ConstantArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.scalar == other.scalar && self.len == other.len
    }
}

impl Operator for ConstantArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn bounds(&self) -> LengthBounds {
        self.len.into()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }
}

impl PipelinedOperator for ConstantArray {
    fn row_selection(&self) -> RowSelection {
        RowSelection::Domain(self.len)
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert!(matches!(
            self.dtype(),
            DType::Bool(_) | DType::Primitive(..)
        ));
        match self.scalar.dtype() {
            DType::Bool(_) => Ok(Box::new(BoolConstantKernel {
                value: self
                    .scalar
                    .as_bool()
                    .value()
                    .vortex_expect("scalar value not bool"),
            })),
            DType::Primitive(..) => Ok(match_each_native_ptype!(
                self.scalar.as_primitive().ptype(),
                |T| {
                    Box::new(ConstantKernel::<T> {
                        value: self
                            .scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("scalar value not of type T"),
                    })
                }
            )),
            _ => todo!(
                "Unsupported scalar type for constant: {:?}",
                self.scalar.dtype()
            ),
        }
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

/// Kernel that produces constant primitive values.
pub struct ConstantKernel<T: NativePType> {
    value: T,
}

/// Kernel that produces constant boolean values.
pub struct BoolConstantKernel {
    value: bool,
}

impl<T: Element + NativePType> Kernel for ConstantKernel<T> {
    fn step(
        &self,
        _ctx: &KernelContext,
        _chunk_idx: usize,
        _selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        // TODO(ngates): benchmark whether to populate the true indices, or the entire vector.
        out.as_array_mut::<T>()[..N].fill(self.value);
        out.set_selection(Selection::Prefix);
        Ok(())
    }
}

impl Kernel for BoolConstantKernel {
    fn step(
        &self,
        _ctx: &KernelContext,
        _chunk_idx: usize,
        _selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        // TODO(ngates): benchmark whether to populate the true indices, or the entire vector.
        out.as_array_mut::<bool>()[..N].fill(self.value);
        out.set_selection(Selection::Prefix);
        Ok(())
    }
}
