// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod operations;
mod validity;
mod visitor;

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::Range;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::arrays::scalar_fn::rules::PARENT_RULES;
use crate::arrays::scalar_fn::rules::RULES;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::expr;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::ScalarFn;
use crate::expr::VTableExt;
use crate::matcher::Matcher;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

vtable!(ScalarFn);

#[derive(Clone, Debug)]
pub struct ScalarFnVTable;

impl VTable for ScalarFnVTable {
    type Array = ScalarFnArray;
    type Metadata = ScalarFnMetadata;
    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(array: &Self::Array) -> ArrayId {
        array.scalar_fn.id()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        let child_dtypes = array.children().iter().map(|c| c.dtype().clone()).collect();
        Ok(ScalarFnMetadata {
            scalar_fn: array.scalar_fn.clone(),
            child_dtypes,
        })
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Not supported
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
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
            // This requires a new Arc, but we plan to remove this later anyway.
            scalar_fn: metadata.scalar_fn.clone(),
            dtype: dtype.clone(),
            len,
            children,
            stats: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == array.children.len(),
            "ScalarFnArray expects {} children, got {}",
            array.children.len(),
            children.len()
        );
        array.children = children;
        Ok(())
    }

    fn canonicalize(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        let args = ExecutionArgs {
            inputs: array.children.clone(),
            row_count: array.len,
            ctx,
        };

        let result = array
            .scalar_fn
            .execute(args)?
            .into_array()
            .execute::<Canonical>(ctx)?;

        debug_assert_eq!(
            result.dtype(),
            &array.dtype,
            "Scalar function {} returned dtype {:?} but expected {:?}",
            array.scalar_fn,
            result.dtype(),
            array.dtype
        );

        debug_assert_eq!(
            result.len(),
            array.len(),
            "Scalar function {} returned len {:?} but expected {:?}",
            array.scalar_fn,
            result.len(),
            array.len()
        );

        Ok(result)
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let children: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray {
                scalar_fn: array.scalar_fn.clone(),
                dtype: array.dtype.clone(),
                len: range.len(),
                children,
                stats: Default::default(),
            }
            .into_array(),
        ))
    }
}

/// Array factory functions for scalar functions.
pub trait ScalarFnArrayExt: expr::VTable {
    fn try_new_array(
        &'static self,
        len: usize,
        options: Self::Options,
        children: impl Into<Vec<ArrayRef>>,
    ) -> VortexResult<ArrayRef> {
        let scalar_fn = ScalarFn::new_static(self, options);

        let children = children.into();
        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "All child arrays must have the same length as the scalar function array"
        );

        let child_dtypes = children.iter().map(|c| c.dtype().clone()).collect_vec();
        let dtype = scalar_fn.return_dtype(&child_dtypes)?;

        Ok(ScalarFnArray {
            scalar_fn,
            dtype,
            len,
            children,
            stats: Default::default(),
        }
        .into_array())
    }
}
impl<V: expr::VTable> ScalarFnArrayExt for V {}

/// A matcher that matches any scalar function expression.
#[derive(Debug)]
pub struct AnyScalarFn;
impl Matcher for AnyScalarFn {
    type Match<'a> = &'a ScalarFnArray;

    fn try_match(array: &dyn Array) -> Option<Self::Match<'_>> {
        array.as_opt::<ScalarFnVTable>()
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug, Default)]
pub struct ExactScalarFn<F: expr::VTable>(PhantomData<F>);

impl<F: expr::VTable> Matcher for ExactScalarFn<F> {
    type Match<'a> = ScalarFnArrayView<'a, F>;

    fn matches(array: &dyn Array) -> bool {
        if let Some(scalar_fn_array) = array.as_opt::<ScalarFnVTable>() {
            scalar_fn_array.scalar_fn().is::<F>()
        } else {
            false
        }
    }

    fn try_match(array: &dyn Array) -> Option<Self::Match<'_>> {
        let scalar_fn_array = array.as_opt::<ScalarFnVTable>()?;
        let scalar_fn_vtable = scalar_fn_array
            .scalar_fn
            .vtable()
            .as_any()
            .downcast_ref::<F>()
            .vortex_expect("ScalarFn VTable type mismatch in ExactScalarFn matcher");
        let scalar_fn_options = scalar_fn_array
            .scalar_fn
            .options()
            .as_any()
            .downcast_ref::<F::Options>()
            .vortex_expect("ScalarFn options type mismatch in ExactScalarFn matcher");
        Some(ScalarFnArrayView {
            array,
            vtable: scalar_fn_vtable,
            options: scalar_fn_options,
        })
    }
}

pub struct ScalarFnArrayView<'a, F: expr::VTable> {
    array: &'a dyn Array,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<F: expr::VTable> Deref for ScalarFnArrayView<'_, F> {
    type Target = dyn Array;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}

// Used only in this method to allow constrained using of Expression evaluate.
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

impl expr::VTable for ArrayExpr {
    type Options = FakeEq<ArrayRef>;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.array")
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
        args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        crate::Executable::execute(options.0.clone(), args.ctx)
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
