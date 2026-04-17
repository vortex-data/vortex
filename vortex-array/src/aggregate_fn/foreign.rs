// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Options payload for a foreign aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ForeignAggregateFnOptions {
    metadata: Vec<u8>,
}

impl ForeignAggregateFnOptions {
    pub fn new(metadata: Vec<u8>) -> Self {
        Self { metadata }
    }
}

impl Display for ForeignAggregateFnOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "foreign(metadata={}B)", self.metadata.len())
    }
}

/// Aggregate-function placeholder used when deserializing an unknown aggregate function ID.
#[derive(Clone, Debug)]
pub struct ForeignAggregateFnVTable {
    id: AggregateFnId,
}

impl ForeignAggregateFnVTable {
    pub fn new(id: AggregateFnId) -> Self {
        Self { id }
    }
}

impl AggregateFnVTable for ForeignAggregateFnVTable {
    type Options = ForeignAggregateFnOptions;
    type Partial = ();

    fn id(&self) -> AggregateFnId {
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
        Ok(ForeignAggregateFnOptions::new(metadata.to_vec()))
    }

    fn return_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        None
    }

    fn partial_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        None
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }

    fn combine_partials(&self, _partial: &mut Self::Partial, _other: Scalar) -> VortexResult<()> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }

    fn to_scalar(&self, _partial: &Self::Partial) -> VortexResult<Scalar> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }

    fn reset(&self, _partial: &mut Self::Partial) {}

    fn is_saturated(&self, _state: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        _state: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }

    fn finalize(&self, _states: ArrayRef) -> VortexResult<ArrayRef> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }

    fn finalize_scalar(&self, _partial: &Self::Partial) -> VortexResult<Scalar> {
        vortex_bail!("Cannot execute unknown aggregate function '{}'", self.id)
    }
}

pub fn new_foreign_aggregate_fn(id: AggregateFnId, metadata: Vec<u8>) -> AggregateFnRef {
    AggregateFn::new(
        ForeignAggregateFnVTable::new(id),
        ForeignAggregateFnOptions::new(metadata),
    )
    .erased()
}
