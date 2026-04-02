// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod operations;
mod validity;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::arrays::scalar_fn::rules::PARENT_RULES;
use crate::arrays::scalar_fn::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::expr::Expression;
use crate::matcher::Matcher;
use crate::scalar_fn;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::VecExecutionArgs;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

vtable!(ScalarFn, ScalarFnVTable);

#[derive(Clone, Debug)]
pub struct ScalarFnVTable {
    pub(super) scalar_fn: ScalarFnRef,
}

impl VTable for ScalarFnVTable {
    type Array = ScalarFnArray;
    type Metadata = ScalarFnMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(array: &Self::Array) -> &Self {
        &array.vtable
    }

    fn id(&self) -> ArrayId {
        self.scalar_fn.id()
    }

    fn len(array: &ScalarFnArray) -> usize {
        array.len
    }

    fn dtype(array: &ScalarFnArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ScalarFnArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ScalarFnArray, state: &mut H, precision: Precision) {
        array.len.hash(state);
        array.dtype.hash(state);
        array.scalar_fn().hash(state);
        for child in array.iter_children() {
            child.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ScalarFnArray, other: &ScalarFnArray, precision: Precision) -> bool {
        if array.len != other.len {
            return false;
        }
        if array.dtype != other.dtype {
            return false;
        }
        if array.scalar_fn() != other.scalar_fn() {
            return false;
        }
        for (child, other_child) in array.iter_children().zip(other.iter_children()) {
            if !child.array_eq(other_child, precision) {
                return false;
            }
        }
        true
    }

    fn nbuffers(_array: &ScalarFnArray) -> usize {
        0
    }

    fn buffer(_array: &ScalarFnArray, idx: usize) -> BufferHandle {
        vortex_panic!("ScalarFnArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ScalarFnArray, _idx: usize) -> Option<String> {
        None
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        let child_dtypes = array.iter_children().map(|c| c.dtype().clone()).collect();
        Ok(ScalarFnMetadata {
            scalar_fn: array.scalar_fn().clone(),
            child_dtypes,
        })
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Not supported
        Ok(None)
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("Deserialization of ScalarFnVTable metadata is not supported");
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &ScalarFnMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        let children: Vec<_> = metadata
            .child_dtypes
            .iter()
            .enumerate()
            .map(|(idx, child_dtype)| children.get(idx, child_dtype, len))
            .try_collect()?;

        #[cfg(debug_assertions)]
        {
            let child_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
            vortex_error::vortex_ensure!(
                &metadata.scalar_fn.return_dtype(&child_dtypes)? == dtype,
                "Return dtype mismatch when building ScalarFnArray"
            );
        }

        Ok(ScalarFnArray {
            vtable: ScalarFnVTable {
                scalar_fn: metadata.scalar_fn.clone(),
            },
            dtype: dtype.clone(),
            len,
            slots: children.into_iter().map(Some).collect(),
            stats: Default::default(),
        })
    }

    fn slots(array: &ScalarFnArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(array: &ScalarFnArray, idx: usize) -> String {
        array
            .scalar_fn()
            .signature()
            .child_name(idx)
            .as_ref()
            .to_string()
    }

    fn with_slots(array: &mut ScalarFnArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        ctx.log(format_args!("scalar_fn({}): executing", array.scalar_fn()));
        let args = VecExecutionArgs::new(array.children(), array.len);
        array
            .scalar_fn()
            .execute(&args, ctx)
            .map(ExecutionResult::done)
    }

    fn reduce(array: &Array<Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Array factory functions for scalar functions.
pub trait ScalarFnArrayExt: scalar_fn::ScalarFnVTable {
    fn try_new_array(
        &self,
        len: usize,
        options: Self::Options,
        children: impl Into<Vec<ArrayRef>>,
    ) -> VortexResult<ArrayRef> {
        let scalar_fn = scalar_fn::ScalarFn::new(self.clone(), options).erased();

        let children = children.into();
        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "All child arrays must have the same length as the scalar function array"
        );

        let child_dtypes = children.iter().map(|c| c.dtype().clone()).collect_vec();
        let dtype = scalar_fn.return_dtype(&child_dtypes)?;

        Ok(ScalarFnArray {
            vtable: ScalarFnVTable { scalar_fn },
            dtype,
            len,
            slots: children.into_iter().map(Some).collect(),
            stats: Default::default(),
        }
        .into_array())
    }
}
impl<V: scalar_fn::ScalarFnVTable> ScalarFnArrayExt for V {}

/// A matcher that matches any scalar function expression.
#[derive(Debug)]
pub struct AnyScalarFn;
impl Matcher for AnyScalarFn {
    type Match<'a> = &'a ScalarFnArray;

    fn try_match(array: &dyn DynArray) -> Option<Self::Match<'_>> {
        array.as_opt::<ScalarFnVTable>()
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug, Default)]
pub struct ExactScalarFn<F: scalar_fn::ScalarFnVTable>(PhantomData<F>);

impl<F: scalar_fn::ScalarFnVTable> Matcher for ExactScalarFn<F> {
    type Match<'a> = ScalarFnArrayView<'a, F>;

    fn matches(array: &dyn DynArray) -> bool {
        if let Some(scalar_fn_array) = array.as_opt::<ScalarFnVTable>() {
            scalar_fn_array.scalar_fn().is::<F>()
        } else {
            false
        }
    }

    fn try_match(array: &dyn DynArray) -> Option<Self::Match<'_>> {
        let scalar_fn_array = array.as_opt::<ScalarFnVTable>()?;
        let scalar_fn = scalar_fn_array.scalar_fn().downcast_ref::<F>()?;
        Some(ScalarFnArrayView {
            array,
            vtable: scalar_fn.vtable(),
            options: scalar_fn.options(),
        })
    }
}

pub struct ScalarFnArrayView<'a, F: scalar_fn::ScalarFnVTable> {
    array: &'a dyn DynArray,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<F: scalar_fn::ScalarFnVTable> Deref for ScalarFnArrayView<'_, F> {
    type Target = dyn DynArray;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}

// Used only in this method to allow constrained using of Expression evaluate.
#[derive(Clone)]
struct ArrayExpr;

#[derive(Clone, Debug)]
struct FakeEq<T>(T);

impl<T> PartialEq<Self> for FakeEq<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T> Eq for FakeEq<T> {}

impl<T> Hash for FakeEq<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl Display for FakeEq<ArrayRef> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.encoding_id())
    }
}

impl scalar_fn::ScalarFnVTable for ArrayExpr {
    type Options = FakeEq<ArrayRef>;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.array")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        todo!()
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", options.0.encoding_id())
    }

    fn return_dtype(&self, options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(options.0.dtype().clone())
    }

    fn execute(
        &self,
        options: &Self::Options,
        _args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        crate::Executable::execute(options.0.clone(), ctx)
    }

    fn validity(
        &self,
        options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let validity_array = options.0.validity()?.to_array(options.0.len());
        Ok(Some(ArrayExpr.new_expr(FakeEq(validity_array), [])))
    }
}
