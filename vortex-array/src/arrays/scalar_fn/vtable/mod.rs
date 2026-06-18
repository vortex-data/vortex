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

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::ParentRef;
use crate::array::ParentView;
use crate::array::VTable;
use crate::arrays::scalar_fn::array::ScalarFnArrayExt;
use crate::arrays::scalar_fn::array::ScalarFnData;
use crate::arrays::scalar_fn::rules::PARENT_RULES;
use crate::arrays::scalar_fn::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::expr::Expression;
use crate::matcher::AsParent;
use crate::matcher::Matcher;
use crate::scalar_fn;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::VecExecutionArgs;
use crate::serde::ArrayChildren;

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
        vortex_ensure!(
            data.scalar_fn.id() == self.id,
            "ScalarFnArray data scalar_fn does not match vtable"
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
            data.scalar_fn.return_dtype(&child_dtypes)? == *dtype,
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
        ctx.log(format_args!("scalar_fn({}): executing", array.scalar_fn()));
        let args = VecExecutionArgs::new(array.children(), array.len());
        array
            .scalar_fn()
            .execute(&args, ctx)
            .map(ExecutionResult::done)
    }

    fn reduce(array: ParentView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Array factory functions for scalar functions.
pub trait ScalarFnFactoryExt: scalar_fn::ScalarFnVTable {
    /// Build the [`ArrayParts<ScalarFn>`] for this scalar function applied to `children`.
    ///
    /// Stops short of allocating the backing `ArrayRef`, so callers can drive the parts
    /// through [`ArrayParts::optimize`] and only pay the wrapper allocation when no
    /// reduction fires.
    #[inline]
    fn try_new_array_parts(
        &self,
        len: usize,
        options: Self::Options,
        children: impl Into<Vec<ArrayRef>>,
    ) -> VortexResult<ArrayParts<ScalarFn>> {
        let scalar_fn = scalar_fn::TypedScalarFnInstance::new(self.clone(), options).erased();
        Array::<ScalarFn>::try_new_parts(scalar_fn, children.into(), len)
    }

    /// Build a materialized scalar-function array for this scalar function applied to
    /// `children`. Equivalent to [`try_new_array_parts`](Self::try_new_array_parts) followed
    /// by [`ArrayParts::into_array`].
    fn try_new_array(
        &self,
        len: usize,
        options: Self::Options,
        children: impl Into<Vec<ArrayRef>>,
    ) -> VortexResult<ArrayRef> {
        Ok(self
            .try_new_array_parts(len, options, children)?
            .into_array())
    }
}
impl<V: scalar_fn::ScalarFnVTable> ScalarFnFactoryExt for V {}

/// A matcher that matches any scalar function expression.
#[derive(Debug)]
pub struct AnyScalarFn;
impl Matcher for AnyScalarFn {
    type Match<'a> = ParentView<'a, ScalarFn>;

    fn try_match<'a, P: AsParent>(parent: &'a P) -> Option<Self::Match<'a>> {
        parent.as_opt::<ScalarFn>()
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug, Default)]
pub struct ExactScalarFn<F: scalar_fn::ScalarFnVTable>(PhantomData<F>);

impl<F: scalar_fn::ScalarFnVTable> ExactScalarFn<F> {
    #[inline]
    fn from_view(view: ParentView<'_, ScalarFn>) -> Option<ScalarFnArrayView<'_, F>> {
        let scalar_fn = view.data().scalar_fn().downcast_ref::<F>()?;
        Some(ScalarFnArrayView {
            view,
            vtable: scalar_fn.vtable(),
            options: scalar_fn.options(),
        })
    }
}

impl<F: scalar_fn::ScalarFnVTable> Matcher for ExactScalarFn<F> {
    type Match<'a> = ScalarFnArrayView<'a, F>;

    /// Skip the `ParentView` + `ScalarFnArrayView` construction that the default
    /// `try_match(...).is_some()` would do. Two cheap downcasts suffice: encoding
    /// id, then scalar function id.
    fn matches<P: AsParent>(parent: &P) -> bool {
        parent
            .typed_data::<ScalarFn>()
            .is_some_and(|data| data.scalar_fn().is::<F>())
    }

    fn try_match<'a, P: AsParent>(parent: &'a P) -> Option<Self::Match<'a>> {
        Self::from_view(parent.as_opt::<ScalarFn>()?)
    }
}

/// A typed view over a [`ScalarFn`] array exposing the concrete `F`-typed `vtable`
/// and `options`.
///
/// Wraps a [`ParentView<'_, ScalarFn>`], so the view works for heap arrays and
/// stack-allocated construction parts alike. It does not expose implicit `ArrayRef`
/// access; callers must explicitly materialize the underlying parent view if they
/// need an owned array.
pub struct ScalarFnArrayView<'a, F: scalar_fn::ScalarFnVTable> {
    view: ParentView<'a, ScalarFn>,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<'a, F: scalar_fn::ScalarFnVTable> ScalarFnArrayView<'a, F> {
    /// Returns the underlying [`ScalarFn`]-typed parent view.
    #[inline]
    pub fn view(&self) -> ParentView<'a, ScalarFn> {
        self.view
    }

    /// Returns the child array at the given slot.
    ///
    /// Reads from `slots()` directly without forcing stack-backed parents to
    /// materialize.
    pub fn child_at(&self, idx: usize) -> &ArrayRef {
        self.view.slots()[idx]
            .as_ref()
            .vortex_expect("ScalarFnArray child slot")
    }

    /// Alias for [`Self::child_at`].
    #[inline]
    pub fn get_child(&self, idx: usize) -> &ArrayRef {
        self.child_at(idx)
    }

    /// Returns the number of child slots.
    #[inline]
    pub fn child_count(&self) -> usize {
        self.view.slots().len()
    }

    /// Iterates over the array's children.
    pub fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        (0..self.child_count()).map(|idx| self.child_at(idx))
    }

    /// Collects the children into a `Vec` of cloned `ArrayRef`s.
    pub fn children(&self) -> Vec<ArrayRef> {
        self.iter_children().cloned().collect()
    }
}

impl<F: scalar_fn::ScalarFnVTable> Copy for ScalarFnArrayView<'_, F> {}

impl<F: scalar_fn::ScalarFnVTable> Clone for ScalarFnArrayView<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F: scalar_fn::ScalarFnVTable> Deref for ScalarFnArrayView<'a, F> {
    type Target = ParentView<'a, ScalarFn>;

    #[inline]
    fn deref(&self) -> &ParentView<'a, ScalarFn> {
        &self.view
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
        static ID: CachedId = CachedId::new("vortex.array");
        *ID
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
