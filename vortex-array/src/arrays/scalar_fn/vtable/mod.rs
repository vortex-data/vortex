// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod operations;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::scalar_fn::array::ScalarFnArrayExt;
use crate::arrays::scalar_fn::array::ScalarFnData;
use crate::arrays::scalar_fn::rules::PARENT_RULES;
use crate::arrays::scalar_fn::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::matcher::Matcher;
use crate::scalar_fn;
use crate::scalar_fn::ArrayReduceNode;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`ScalarFn`]-encoded Vortex array.
pub type ScalarFnArray = Array<ScalarFn>;

#[derive(Clone, Debug)]
pub struct ScalarFn {
    pub(super) id: ScalarFnId,
}

impl ArrayHash for ScalarFnData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.scalar_fn().hash(state);
    }
}

impl ArrayEq for ScalarFnData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.scalar_fn() == other.scalar_fn()
    }
}

impl VTable for ScalarFn {
    type TypedArrayData = ScalarFnData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        self.id
    }

    fn validate(
        &self,
        data: &ScalarFnData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let scalar_fn = data.scalar_fn();
        vortex_ensure!(
            scalar_fn.id() == self.id,
            "ScalarFnArray data scalar_fn does not match vtable"
        );

        let missing_children = slots.iter().filter(|slot| slot.is_none()).count();
        vortex_ensure!(
            missing_children == 0,
            "ScalarFnArray requires every child slot to be present, got {missing_children} missing"
        );

        let arity = scalar_fn.signature().arity();
        vortex_ensure!(
            arity.matches(slots.len()),
            "ScalarFnArray requires {arity} children, got {}",
            slots.len()
        );
        vortex_ensure!(
            slots.iter().flatten().all(|c| c.len() == len),
            "All child arrays must have the same length as the scalar function array"
        );

        let child_dtypes = slots
            .iter()
            .flatten()
            .map(|c| c.dtype().clone())
            .collect_vec();
        vortex_ensure!(
            scalar_fn.return_dtype(&child_dtypes)? == *dtype,
            "ScalarFnArray dtype does not match scalar function return dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ScalarFnArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Not supported
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("Deserialization of ScalarFnVTable metadata is not supported");
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        array
            .scalar_fn()
            .signature()
            .child_name(idx)
            .as_ref()
            .to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let args = VecExecutionArgs::new(array.children(), array.len());
        array
            .scalar_fn()
            .execute(&args, ctx)
            .map(ExecutionResult::done)
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// A matcher that matches any scalar function expression.
#[derive(Debug)]
pub struct AnyScalarFn;
impl Matcher for AnyScalarFn {
    type Match<'a> = ArrayView<'a, ScalarFn>;

    fn matches(array: &ArrayRef) -> bool {
        array.is::<ScalarFn>()
    }

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        array.as_opt::<ScalarFn>()
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug, Default)]
pub struct ExactScalarFn<F: scalar_fn::ScalarFnVTable>(PhantomData<F>);

impl<F: scalar_fn::ScalarFnVTable> Matcher for ExactScalarFn<F> {
    type Match<'a> = ScalarFnArrayView<'a, F>;

    fn matches(array: &ArrayRef) -> bool {
        if let Some(scalar_fn_array) = array.as_opt::<ScalarFn>() {
            scalar_fn_array.data().scalar_fn().is::<F>()
        } else {
            false
        }
    }

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        let scalar_fn_array = array.as_opt::<ScalarFn>()?;
        let scalar_fn_data = scalar_fn_array.data();
        let scalar_fn = scalar_fn_data.scalar_fn().downcast_ref::<F>()?;
        Some(ScalarFnArrayView {
            array,
            vtable: scalar_fn.vtable(),
            options: scalar_fn.options(),
        })
    }
}

pub struct ScalarFnArrayView<'a, F: scalar_fn::ScalarFnVTable> {
    array: &'a ArrayRef,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<F: scalar_fn::ScalarFnVTable> Deref for ScalarFnArrayView<'_, F> {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}

impl ValidityVTable<ScalarFn> for ScalarFn {
    fn validity(view: ArrayView<'_, ScalarFn>) -> VortexResult<Validity> {
        let node = ArrayReduceNode::new(view.as_ref());
        Ok(Validity::Array(match node.validity()? {
            ReduceNodeValidity::Reduced(reduced) => reduced.into_array(),
            // We get validity only after evaluating this node. To avoid
            // infinite recursion, IsNotNull(x) -> x.validity() symbolically
            // rewrites only in the Reduced() case.
            ReduceNodeValidity::Irreducible => IsNotNull::new(view.as_ref().clone()).into_array(),
        }))
    }
}
