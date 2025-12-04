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
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorMutOps;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::execution::ExecutionCtx;
use crate::expr;
use crate::expr::ExecutionArgs;
use crate::expr::ExprVTable;
use crate::expr::ScalarFn;
use crate::optimizer::rules::MatchKey;
use crate::optimizer::rules::Matcher;
use crate::serde::ArrayChildren;
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
            bound: array.bound.clone(),
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
                &metadata.bound.return_dtype(&child_dtypes)? == dtype,
                "Return dtype mismatch when building ScalarFnArray"
            );
        }

        Ok(ScalarFnArray {
            // This requires a new Arc, but we plan to remove this later anyway.
            vtable: self.to_vtable(),
            bound: metadata.bound.clone(),
            dtype: dtype.clone(),
            len,
            children,
            stats: Default::default(),
        })
    }

    fn batch_execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        let input_dtypes: Vec<_> = array.children().iter().map(|c| c.dtype().clone()).collect();
        let input_datums = array
            .children()
            .iter()
            .map(|child| child.batch_execute(ctx).map(Datum::Vector))
            .try_collect()?;
        let ctx = ExecutionArgs {
            datums: input_datums,
            dtypes: input_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        };

        let datum = array.bound.execute(ctx)?;
        let vector = match datum {
            Datum::Scalar(s) => s.repeat(array.len).freeze(),
            Datum::Vector(v) => v,
        };
        Ok(vector)
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
        let bound = ScalarFn::new_static(self, options);

        let children = children.into();
        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "All child arrays must have the same length as the scalar function array"
        );

        let child_dtypes = children.iter().map(|c| c.dtype().clone()).collect_vec();
        let dtype = bound.return_dtype(&child_dtypes)?;

        let array_vtable: ArrayVTable = ScalarFnVTable {
            vtable: bound.vtable().clone(),
        }
        .into_vtable();

        Ok(ScalarFnArray {
            vtable: array_vtable,
            bound,
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
        let scalar_fn_array = array.as_opt::<ScalarFnVTable>()?;
        let scalar_fn_vtable = scalar_fn_array
            .bound
            .vtable()
            .as_any()
            .downcast_ref::<F>()?;
        let scalar_fn_options = scalar_fn_array
            .bound
            .options()
            .as_any()
            .downcast_ref::<F::Options>()?;
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
