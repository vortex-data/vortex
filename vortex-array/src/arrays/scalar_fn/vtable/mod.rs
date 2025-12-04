// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::LazyLock;

use itertools::Itertools;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_vector::Vector;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::metadata::ScalarFnMetadata;
use crate::execution::ExecutionCtx;
use crate::expr::functions;
use crate::expr::functions::scalar::ScalarFn;
use crate::optimizer::rules::MatchKey;
use crate::optimizer::rules::Matcher;
use crate::serde::ArrayChildren;
use crate::session::ArraySession;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

// TODO(ngates): canonicalize doesn't currently take a session, therefore we cannot dispatch
//  to registered scalar function kernels. We therefore hold our own non-pluggable session here
//  that contains all the built-in kernels while we migrate over to "execute" instead of canonicalize.
static SCALAR_FN_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

vtable!(ScalarFn);

#[derive(Clone, Debug)]
pub struct ScalarFnVTable {
    vtable: functions::ScalarFnVTable,
}

impl ScalarFnVTable {
    pub fn new(vtable: functions::ScalarFnVTable) -> Self {
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

    fn batch_execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        let input_dtypes: Vec<_> = array.children().iter().map(|c| c.dtype().clone()).collect();
        let input_datums = array
            .children()
            .iter()
            .map(|child| child.batch_execute(ctx))
            .try_collect()?;
        let ctx = functions::ExecutionArgs::new(
            array.len(),
            array.dtype.clone(),
            input_dtypes,
            input_datums,
        );
        Ok(array
            .scalar_fn
            .execute(&ctx)?
            .into_vector()
            .vortex_expect("Vector inputs should return vector outputs"))
    }
}

/// Array factory functions for scalar functions.
pub trait ScalarFnArrayExt: functions::VTable {
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
impl<V: functions::VTable> ScalarFnArrayExt for V {}

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
pub struct ExactScalarFn<F: functions::VTable> {
    id: ArrayId,
    _phantom: PhantomData<F>,
}

impl<F: functions::VTable> From<&'static F> for ExactScalarFn<F> {
    fn from(value: &'static F) -> Self {
        Self {
            id: value.id(),
            _phantom: PhantomData,
        }
    }
}

impl<F: functions::VTable> Matcher for ExactScalarFn<F> {
    type View<'a> = ScalarFnArrayView<'a, F>;

    fn key(&self) -> MatchKey {
        MatchKey::Array(self.id.clone())
    }

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        let scalar_fn_array = array.as_opt::<ScalarFnVTable>()?;
        let scalar_fn_vtable = scalar_fn_array
            .scalar_fn
            .vtable()
            .as_any()
            .downcast_ref::<F>()?;
        let scalar_fn_options = scalar_fn_array
            .scalar_fn
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

pub struct ScalarFnArrayView<'a, F: functions::VTable> {
    array: &'a ArrayRef,
    pub vtable: &'a F,
    pub options: &'a F::Options,
}

impl<F: functions::VTable> Deref for ScalarFnArrayView<'_, F> {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}
