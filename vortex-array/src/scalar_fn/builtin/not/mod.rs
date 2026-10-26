// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Expression that logically inverts boolean values.
#[derive(Clone)]
pub struct Not;

impl Not {
    /// Creates a lazy boolean inversion of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not boolean data.
    pub fn try_new(input: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Not.bind(EmptyOptions), [input])
    }
}

impl ScalarFnVTable for Not {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.not");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Not expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        if !matches!(child_dtype, DType::Bool(_)) {
            vortex_bail!(
                "Not expression expects a boolean child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype.clone())
    }

    fn execute(
        &self,
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;

        // For constant boolean
        if let Some(scalar) = child.as_constant() {
            let value = match scalar.as_bool().value() {
                Some(b) => Scalar::bool(!b, child.dtype().nullability()),
                None => Scalar::null(child.dtype().clone()),
            };
            return Ok(ConstantArray::new(value, args.row_count()).into_array());
        }

        // For boolean array
        if let Some(bool) = child.as_opt::<Bool>() {
            return Ok(BoolArray::new(!bool.to_bit_buffer(), bool.validity()?).into_array());
        }

        // Otherwise, execute and try again
        child.execute::<ArrayRef>(ctx)?.not()
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}
