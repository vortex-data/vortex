// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::sync::LazyLock;

use crate::execution::{BatchKernel, BatchKernelRef, BindCtx};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, OperatorVTable, VTable, VisitorVTable};
use crate::{
    vtable, Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, EncodingId, EncodingRef,
};
use enum_map::{enum_map, Enum, EnumMap};
use futures::try_join;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;
use vortex_vector::{BoolVector, BoolVectorMut, Vector, VectorMut, VectorMutOps, VectorOps};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum LogicalOperator {
    And,
    AndKleene,
    Or,
    OrKleene,
    AndNot,
}

vtable!(Logical);

#[derive(Debug, Clone)]
pub struct LogicalArray {
    encoding: EncodingRef,
    lhs: ArrayRef,
    rhs: ArrayRef,
    dtype: DType,
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
        assert!(matches!(lhs.dtype(), DType::Bool(_)));
        assert!(matches!(rhs.dtype(), DType::Bool(_)));
        let dtype = DType::Bool(lhs.dtype().nullability() | rhs.dtype().nullability());

        Self {
            encoding: ENCODINGS[operator].clone(),
            lhs,
            rhs,
            dtype,
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
    type SerdeVTable = NotSupported;
    type PipelineVTable = Self;

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
        &array.dtype
    }

    fn stats(array: &LogicalArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
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

impl OperatorVTable<LogicalVTable> for LogicalVTable {
    fn compute_constant(
        _array: &LogicalArray,
        children: &[&Scalar],
    ) -> VortexResult<Option<ArrayRef>> {
        let lhs = children[0].as_bool();
        let rhs = children[1].as_bool();
        todo!()
    }

    fn bind(
        array: &LogicalVTable::Array,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let lhs = ctx.bind(&array.lhs, selection)?;
        let rhs = ctx.bind(&array.rhs, selection)?;

        match array.operator() {
            LogicalOperator::And => LogicalKernel::new(lhs, rhs, |l, r| l.bitand(r)),
            LogicalOperator::AndKleene => {}
            LogicalOperator::Or => {}
            LogicalOperator::OrKleene => {}
            LogicalOperator::AndNot => {}
        }
    }
}

struct LogicalKernel<O> {
    lhs: BatchKernelRef,
    rhs: BatchKernelRef,
    op: O,
}

impl<O> LogicalKernel<O>
where
    O: Fn(&BitBuffer, &BitBuffer) -> BitBuffer,
{
    pub fn new(lhs: BatchKernelRef, rhs: BatchKernelRef, op: O) -> VortexResult<BatchKernelRef> {
        Ok(Box::new(Self { lhs, rhs, op }))
    }
}

impl<O> BatchKernel for LogicalKernel<O>
where
    O: Fn(&BitBuffer, &BitBuffer) -> BitBuffer,
{
    async fn execute(self: Box<Self>, out: VectorMut) -> VortexResult<Vector> {
        // We pass the output into the LHS and then attempt to call the mutate-in-place op.
        let rhs_out = BoolVectorMut::with_capacity(0, out.nullability());
        let (lhs, rhs) = try_join!(self.lhs.execute(out), self.rhs.execute(rhs_out.into()))?;
        let (lhs, rhs) = (BoolVector::from(lhs), BoolVector::from(rhs));

        // First, we compute the union the input validity.
        let validity = lhs.validity() | rhs.validity();

        // TODO(ngates): we should then find the threshold by which a low enough result validity
        //   means it's better for us to evaluate over scalars vs over the entire bit-buffer.

        Ok(BoolVector::new((self.op)(lhs.bits(), rhs.bits()), validity).into())
    }
}
