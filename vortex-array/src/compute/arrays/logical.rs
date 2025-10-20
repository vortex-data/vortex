// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, VTable};
use crate::{vtable, EncodingId, EncodingRef};
use arrow_array::{Array, ArrayRef};
use vortex_dtype::DType;

#[derive(Clone, Debug)]
pub enum LogicalOperator {
    And,
    AndKleene,
    AndNot,
    Or,
    OrKleene,
}

#[derive(Clone, Debug)]
pub struct LogicalArray {
    lhs: ArrayRef,
    rhs: ArrayRef,
    operator: LogicalOperator,
    dtype: DType,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct LogicalEncoding;

vtable!(Logical);

impl VTable for LogicalVTable {
    type Array = LogicalArray;
    type Encoding = LogicalEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
    type VisitorVTable = NotSupported;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = NotSupported;
    type PipelineVTable = NotSupported;

    fn id(encoding: &Self::Encoding) -> EncodingId {
        todo!()
    }

    fn encoding(array: &Self::Array) -> EncodingRef {
        todo!()
    }
}

impl ArrayVTable<LogicalVTable> for LogicalVTable {
    fn len(array: &LogicalArray) -> usize {
        array.lhs.len()
    }

    fn dtype(array: &LogicalArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &LogicalArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array)
    }
}
