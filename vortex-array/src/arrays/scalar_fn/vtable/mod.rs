// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

use std::marker::PhantomData;
use std::ops::Deref;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_vector::Datum;
use vortex_vector::Vector;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ConstantVTable;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::arrays::scalar_fn::rules::PARENT_RULES;
use crate::arrays::scalar_fn::rules::RULES;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::expr;
use crate::expr::ExecutionArgs;
use crate::expr::ExprVTable;
use crate::expr::ScalarFn;
use crate::matchers::MatchKey;
use crate::matchers::Matcher;
use crate::serde::ArrayChildren;
use crate::vectors::VectorIntoArray;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

vtable!(ScalarFn);

#[derive(Clone, Debug)]
pub struct ScalarFnVTable {
    vtable: ExprVTable,
}

impl ScalarFnVTable {
    pub fn new(vtable: ExprVTable) -> Self {
        Self { vtable }
    }
}

impl VTable for ScalarFnVTable {
    type Array = ScalarFnArray;
    type Metadata = ScalarFnMetadata;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        self.vtable.id()
    }

    fn encoding(array: &Self::Array) -> ArrayVTable {
        array.vtable.clone()
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
        &self,
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
            vtable: self.to_vtable(),
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        // NOTE: we don't use iterators here to make the profiles easier to read!
        let mut datums = Vec::with_capacity(array.children.len());
        let mut input_dtypes = Vec::with_capacity(array.children.len());
        for child in array.children.iter() {
            match child.as_opt::<ConstantVTable>() {
                None => datums.push(Datum::Vector(child.clone().execute::<Vector>(ctx)?)),
                Some(constant) => datums.push(Datum::Scalar(constant.scalar().to_vector_scalar())),
            }
            input_dtypes.push(child.dtype().clone());
        }

        let args = ExecutionArgs {
            datums,
            dtypes: input_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        };

        // TODO(joe): should this go via Vector or canonical?
        Ok(array
            .scalar_fn
            .execute(args)?
            .unwrap_into_vector(array.len)
            .into_array(array.dtype()))
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

        let array_vtable: ArrayVTable = ScalarFnVTable {
            vtable: scalar_fn.vtable().clone(),
        }
        .into_vtable();

        Ok(ScalarFnArray {
            vtable: array_vtable,
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
    type View<'a> = &'a ScalarFnArray;

    fn key(&self) -> MatchKey {
        MatchKey::Any
    }

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        array.as_opt::<ScalarFnVTable>()
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug)]
pub struct ExactScalarFn<F: expr::VTable> {
    id: ArrayId,
    _phantom: PhantomData<F>,
}

impl<F: expr::VTable> From<&'static F> for ExactScalarFn<F> {
    fn from(value: &'static F) -> Self {
        Self {
            id: value.id(),
            _phantom: PhantomData,
        }
    }
}

impl<F: expr::VTable> Matcher for ExactScalarFn<F> {
    type View<'a> = ScalarFnArrayView<'a, F>;

    fn key(&self) -> MatchKey {
        MatchKey::Array(self.id.clone())
    }

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        if array.encoding_id() != self.id {
            return None;
        }

        let scalar_fn_array = array
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("Array encoding ID matched but downcast to ScalarFnVTable failed");
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
    array: &'a ArrayRef,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<F: expr::VTable> Deref for ScalarFnArrayView<'_, F> {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}
