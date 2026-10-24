// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::display::ExprDisplay;

/// Expression that represents a literal scalar value.
#[derive(Clone)]
pub struct Literal;

impl ScalarFnVTable for Literal {
    type Options = Scalar;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.literal");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LiteralOpts {
                value: Some(instance.into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let ops = pb::LiteralOpts::decode(_metadata)?;
        Scalar::from_proto(
            ops.value
                .as_ref()
                .ok_or_else(|| vortex_err!("Literal metadata missing value"))?,
            session,
        )
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _instance: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(
        &self,
        scalar: &Scalar,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", scalar)
    }

    fn return_dtype(&self, options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(options.dtype().clone())
    }

    fn execute(
        &self,
        scalar: &Scalar,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(scalar.clone(), args.row_count()).into_array())
    }

    fn validity<T: ReduceNode>(
        &self,
        scalar: &Scalar,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        Ok(ReduceNodeValidity::Reduced(
            node.new_constant(scalar.is_valid().into()),
        ))
    }

    fn is_strict(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _instance: &Self::Options) -> bool {
        true
    }
}
