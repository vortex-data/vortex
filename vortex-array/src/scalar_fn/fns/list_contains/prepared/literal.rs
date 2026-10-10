// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::PreparedSetArray;
use super::PreparedSetData;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::DType;
use crate::expr::display::ExprDisplay;
use crate::proto::scalar as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// A literal list prepared as a set for membership probes.
///
/// [`ListContains`] puts this in place of a literal list when an expression is optimized, so that
/// every batch the expression is applied to probes one shared [`PreparedSetData`]: the list is
/// materialized, sorted and hashed once per expression rather than once per batch. Applied to an
/// array, it gives a [`PreparedSetArray`] of the batch's length in constant time.
///
/// It serializes as the list, and deserializes into a set prepared again.
///
/// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
#[derive(Clone, Debug)]
pub struct PreparedSetLiteral;

impl ScalarFnVTable for PreparedSetLiteral {
    type Options = PreparedSetData;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.prepared_set_literal");
        *ID
    }

    fn serialize(&self, set: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(pb::Scalar::from(set.list()).encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let list = Scalar::from_proto(&pb::Scalar::decode(metadata)?, session)?;
        PreparedSetData::try_new(list)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(
        &self,
        set: &Self::Options,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", set.list())
    }

    fn return_dtype(&self, set: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(set.list().dtype().clone())
    }

    fn execute(
        &self,
        set: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(PreparedSetArray::new(set.clone(), args.row_count()).into_array())
    }

    fn validity<T: ReduceNode>(
        &self,
        _set: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        // The list is never null.
        Ok(ReduceNodeValidity::Reduced(node.new_constant(true.into())))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}
