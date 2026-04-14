// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFn;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// Options payload for a foreign scalar function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ForeignScalarFnOptions {
    metadata: Vec<u8>,
    arity: usize,
}

impl ForeignScalarFnOptions {
    pub fn new(metadata: Vec<u8>, arity: usize) -> Self {
        Self { metadata, arity }
    }
}

impl Display for ForeignScalarFnOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "foreign(arity={}, metadata={}B)",
            self.arity,
            self.metadata.len()
        )
    }
}

/// Scalar function placeholder used when deserializing an unknown scalar function ID.
#[derive(Clone, Debug)]
pub struct ForeignScalarFnVTable {
    id: ScalarFnId,
}

impl ForeignScalarFnVTable {
    pub fn new(id: ScalarFnId) -> Self {
        Self { id }
    }

    pub fn make_scalar_fn(id: ScalarFnId, metadata: Vec<u8>, arity: usize) -> ScalarFnRef {
        ScalarFn::new(Self::new(id), ForeignScalarFnOptions::new(metadata, arity)).erased()
    }
}

impl ScalarFnVTable for ForeignScalarFnVTable {
    type Options = ForeignScalarFnOptions;

    fn id(&self) -> ScalarFnId {
        self.id
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.metadata.clone()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(ForeignScalarFnOptions::new(metadata.to_vec(), 0))
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.arity)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::new_arc(format!("arg{child_idx}").into())
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id)?;
        for i in 0..expr.children().len() {
            if i > 0 {
                write!(f, ", ")?;
            }
            expr.child(i).fmt_sql(f)?;
        }
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Variant(Nullability::Nullable))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("Cannot execute unknown scalar function '{}'", self.id);
    }
}
