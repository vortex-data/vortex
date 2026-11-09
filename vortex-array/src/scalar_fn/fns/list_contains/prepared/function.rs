// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::PreparedSetData;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
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
use crate::scalar_fn::ScalarFnVTableExt;

/// A constant, non-null list whose elements are prepared as a set for membership lookups.
///
/// [`ListContains`] puts this in place of a literal list when an expression is optimized, so that
/// every batch the expression is applied to looks up its needles in one shared
/// [`PreparedSetData`]: the list is materialized, sorted and hashed once per expression rather
/// than once per batch. In an array tree, the constant list of a [`ListContains`] node becomes
/// this function with no children.
///
/// Every row of such an array holds the list. A slice or a filter of it keeps the options, so it
/// looks up in the same set. A [`ListContainsElementReduce`] rule of the needle encoding can push
/// the function into the needle's parts with slices of the array, so that the chunks of a chunked
/// needle all look up in one set, and a [`ListContainsElementKernel`] of the needle encoding gets
/// the set through [`PreparedSet::set_of`].
///
/// It serializes as the list, and deserializes into a set prepared again.
///
/// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
/// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
/// [`ListContainsElementReduce`]: crate::scalar_fn::fns::list_contains::ListContainsElementReduce
#[derive(Clone, Debug)]
pub struct PreparedSet;

impl PreparedSet {
    /// Prepares the non-null list scalar `list` as a set repeated `len` times.
    ///
    /// [`ListContains`] prepares a constant list itself. A kernel crate can use this to test its
    /// [`ListContainsElementKernel`] against a prepared set.
    ///
    /// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
    /// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
    ///
    /// # Errors
    ///
    /// Fails when `list` is not a list, or is null.
    pub fn try_new_array(list: Scalar, len: usize) -> VortexResult<ArrayRef> {
        Ok(Self::new_array(PreparedSetData::try_new(list)?, len))
    }

    /// An array of `len` rows that share the prepared set `set`.
    pub(crate) fn new_array(set: PreparedSetData, len: usize) -> ArrayRef {
        ScalarFnArray::try_new_with_len(Self.bind(set), [], len)
            .vortex_expect("A prepared set has no children, and its dtype is the list dtype")
            .into_array()
    }

    /// The prepared set that every row of `array` holds, when `array` is a [`PreparedSet`].
    pub fn set_of(array: &ArrayRef) -> Option<&PreparedSetData> {
        array
            .as_opt::<ScalarFn>()?
            .data()
            .scalar_fn()
            .as_opt::<Self>()
    }
}

impl ScalarFnVTable for PreparedSet {
    type Options = PreparedSetData;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.prepared_set");
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
        // The rows hold the list only. The haystack is of no use to the canonical form.
        Ok(ConstantArray::new(set.list().clone(), args.row_count()).into_array())
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
