// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use enum_map::{Enum, EnumMap, enum_map};
use vortex_buffer::ByteBuffer;
use vortex_compute::logical::{
    LogicalAnd, LogicalAndKleene, LogicalAndNot, LogicalOr, LogicalOrKleene,
};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::bool::BoolVector;

use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::serde::ArrayChildren;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayVTable, NotSupported, OperatorVTable, SerdeVTable, VTable, VisitorVTable,
};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef,
    DeserializeMetadata, EmptyMetadata, EncodingId, EncodingRef, Precision, vtable,
};

/// The set of operators supported by a logical array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum LogicalOperator {
    /// Logical AND
    And,
    /// Logical AND with Kleene logic
    AndKleene,
    /// Logical OR
    Or,
    /// Logical OR with Kleene logic
    OrKleene,
    /// Logical AND NOT
    AndNot,
}

vtable!(Logical);

#[derive(Debug, Clone)]
pub struct LogicalArray {
    encoding: EncodingRef,
    lhs: ArrayRef,
    rhs: ArrayRef,
    stats: ArrayStats,
}

impl LogicalArray {
    /// Create a new logical array.
    pub fn new(lhs: ArrayRef, rhs: ArrayRef, operator: LogicalOperator) -> Self {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "Logical arrays require lhs and rhs to have the same length"
        );

        // TODO(ngates): should we automatically cast non-null to nullable if required?
        assert!(matches!(lhs.dtype(), DType::Bool(_)));
        assert_eq!(lhs.dtype(), rhs.dtype());

        Self {
            encoding: ENCODINGS[operator].clone(),
            lhs,
            rhs,
            stats: ArrayStats::default(),
        }
    }

    /// Returns the operator of this logical array.
    pub fn operator(&self) -> LogicalOperator {
        self.encoding.as_::<LogicalVTable>().operator
    }
}

#[derive(Debug, Clone)]
pub struct LogicalEncoding {
    // We include the operator in the encoding so each operator is a different encoding ID.
    // This makes it easier for plugins to construct expressions and perform pushdown
    // optimizations.
    operator: LogicalOperator,
}

#[allow(clippy::mem_forget)]
static ENCODINGS: LazyLock<EnumMap<LogicalOperator, EncodingRef>> = LazyLock::new(|| {
    enum_map! {
        operator => LogicalEncoding { operator }.to_encoding(),
    }
});

impl VTable for LogicalVTable {
    type Array = LogicalArray;
    type Encoding = LogicalEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = Self;

    fn id(encoding: &Self::Encoding) -> EncodingId {
        match encoding.operator {
            LogicalOperator::And => EncodingId::from("vortex.and"),
            LogicalOperator::AndKleene => EncodingId::from("vortex.and_kleene"),
            LogicalOperator::Or => EncodingId::from("vortex.or"),
            LogicalOperator::OrKleene => EncodingId::from("vortex.or_kleene"),
            LogicalOperator::AndNot => EncodingId::from("vortex.and_not"),
        }
    }

    fn encoding(array: &Self::Array) -> EncodingRef {
        array.encoding.clone()
    }
}

impl ArrayVTable<LogicalVTable> for LogicalVTable {
    fn len(array: &LogicalArray) -> usize {
        array.lhs.len()
    }

    fn dtype(array: &LogicalArray) -> &DType {
        array.lhs.dtype()
    }

    fn stats(array: &LogicalArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &LogicalArray, state: &mut H, precision: Precision) {
        array.lhs.array_hash(state, precision);
        array.rhs.array_hash(state, precision);
    }

    fn array_eq(array: &LogicalArray, other: &LogicalArray, precision: Precision) -> bool {
        array.lhs.array_eq(&other.lhs, precision) && array.rhs.array_eq(&other.rhs, precision)
    }
}

impl VisitorVTable<LogicalVTable> for LogicalVTable {
    fn visit_buffers(_array: &LogicalArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // No buffers
    }

    fn visit_children(array: &LogicalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("lhs", array.lhs.as_ref());
        visitor.visit_child("rhs", array.rhs.as_ref());
    }
}

impl SerdeVTable<LogicalVTable> for LogicalVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &LogicalArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        encoding: &LogicalEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<LogicalArray> {
        assert!(buffers.is_empty());
        Ok(LogicalArray::new(
            children.get(0, dtype, len)?,
            children.get(1, dtype, len)?,
            encoding.operator,
        ))
    }
}

impl OperatorVTable<LogicalVTable> for LogicalVTable {
    fn bind(
        array: &LogicalArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let lhs = ctx.bind(&array.lhs, selection)?;
        let rhs = ctx.bind(&array.rhs, selection)?;

        Ok(match array.operator() {
            LogicalOperator::And => logical_kernel(lhs, rhs, |l, r| l.and(&r)),
            LogicalOperator::AndKleene => logical_kernel(lhs, rhs, |l, r| l.and_kleene(&r)),
            LogicalOperator::Or => logical_kernel(lhs, rhs, |l, r| l.or(&r)),
            LogicalOperator::OrKleene => logical_kernel(lhs, rhs, |l, r| l.or_kleene(&r)),
            LogicalOperator::AndNot => logical_kernel(lhs, rhs, |l, r| l.and_not(&r)),
        })
    }
}

/// Batch execution kernel for logical operations.
fn logical_kernel<O>(lhs: BatchKernelRef, rhs: BatchKernelRef, op: O) -> BatchKernelRef
where
    O: Fn(BoolVector, BoolVector) -> BoolVector + Send + 'static,
{
    kernel(move || {
        let lhs = lhs.execute()?.into_bool();
        let rhs = rhs.execute()?.into_bool();
        Ok(op(lhs, rhs).into())
    })
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;

    use crate::compute::arrays::logical::{LogicalArray, LogicalOperator};
    use crate::{ArrayOperator, ArrayRef, IntoArray};

    fn and_(lhs: ArrayRef, rhs: ArrayRef) -> ArrayRef {
        LogicalArray::new(lhs, rhs, LogicalOperator::And).into_array()
    }

    #[test]
    fn test_and() {
        let lhs = bitbuffer![0 1 0].into_array();
        let rhs = bitbuffer![0 1 1].into_array();
        let result = and_(lhs, rhs).execute().unwrap().into_bool();
        assert_eq!(result.bits(), &bitbuffer![0 1 0]);
    }

    #[test]
    fn test_and_selected() {
        let lhs = bitbuffer![0 1 0].into_array();
        let rhs = bitbuffer![0 1 1].into_array();

        let selection = bitbuffer![0 1 1].into_array();

        let result = and_(lhs, rhs)
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_bool();
        assert_eq!(result.bits(), &bitbuffer![1 0]);
    }
}
