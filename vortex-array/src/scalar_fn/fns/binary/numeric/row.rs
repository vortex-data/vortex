// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The primitive arithmetic operators as a [`RowFn`].
//!
//! [`Binary`] keeps its ID, its options serialization, and its strictness, fallibility and validity
//! contracts, and delegates only the _execution_ of `Add`, `Sub`, `Mul` and `Div` over primitive
//! columns to [`NumericBinary`]. Delegation rather than conversion is what makes the port possible
//! at all: `Binary` also covers Kleene `And`/`Or`, which are not strict, and the six comparisons,
//! which are infallible, so no single [`RowFn`] can stand in for the whole function.
//!
//! [`NumericBinary`] is not registered and appears in no serialized expression. It is reached only
//! through the [`ScalarFnVTable::execute`] that the blanket [`RowFn`] implementation provides, so
//! it needs no rewrite rule, no ID in the registry, and no wire format of its own.
//!
//! Everything the previous hand-written implementation did around the arithmetic itself now comes
//! from the lifting: input decoding, the constant operand collapse, the all-constant fold, the
//! null-constant short circuit, output allocation, nullability widening, and masking. What is left
//! here is the per-type checked operation and the sink that carries its overflow bit.
//!
//! [`Binary`]: crate::scalar_fn::fns::binary::Binary

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::primitive::CheckedAdd;
use super::primitive::CheckedDiv;
use super::primitive::CheckedMul;
use super::primitive::CheckedPrimitiveOp;
use super::primitive::CheckedSub;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::scalar_fn::DeferredError;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

/// Execute a numeric operation between two primitive-typed arrays.
///
/// The caller has already established that both operands are primitive, of the same type, and of
/// the same length.
pub(super) fn execute_numeric_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    ScalarFnVTable::execute(&NumericBinary, &op, &args, ctx)
}

/// The four arithmetic operators of [`Binary`] over primitive columns, as one row function per
/// operator and width.
///
/// [`Binary`]: crate::scalar_fn::fns::binary::Binary
#[derive(Clone)]
struct NumericBinary;

impl RowFn for NumericBinary {
    type Options = NumericOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    /// Only the integer widths can overflow, and only integer division can divide by zero, but
    /// fallibility is declared without input dtypes. The float widths are therefore covered by the
    /// same `true`, which costs them nothing: a deferred error keeps the batch on the dense path.
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.numeric_binary");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        op: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        let ptype = operand_ptype(args)?;

        match_each_native_ptype!(ptype, |T| {
            match op {
                NumericOperator::Add => visit_checked::<T, CheckedAdd, V>(visitor),
                NumericOperator::Sub => visit_checked::<T, CheckedSub, V>(visitor),
                NumericOperator::Mul => visit_checked::<T, CheckedMul, V>(visitor),
                NumericOperator::Div => visit_checked::<T, CheckedDiv, V>(visitor),
            }
        })
    }
}

/// The width both operands are read at.
///
/// Only the left operand is inspected. `(T, T)` validates each argument against the chosen width,
/// so a right operand of a different type is rejected by the visit rather than here.
fn operand_ptype(args: &[DType]) -> VortexResult<PType> {
    let lhs = args
        .first()
        .ok_or_else(|| vortex_err!("a numeric operator takes two operands, got none"))?;

    PType::try_from(lhs)
}

/// Visit at two `T` columns, applying `Op` per row into the sink that defers its overflow bit.
///
/// The const block enforces, at monomorphization time, the width rule stated on
/// [`Failure`](super::primitive::Failure): evidence wider than the element would make the
/// OR-reduction rather than the arithmetic decide how many rows fit in a vector.
fn visit_checked<T, Op, V>(visitor: V) -> VortexResult<V::Out>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
    V: RowVisitor,
{
    const {
        assert!(
            size_of::<Op::Failure>() <= size_of::<T>(),
            "failure evidence must be no wider than the value, or it bounds the vector width"
        )
    };

    visitor.visit_prepared_into::<(T, T), CheckedSink<T, Op>, _, _>(
        |_| (),
        |&(), (lhs, rhs), output| output.write(lhs, rhs),
    )
}

/// The output column of one checked arithmetic batch, reporting failure once after the row loop.
///
/// Deferring the failure is what keeps a fallible kernel on the dense path: every row writes a
/// value unconditionally and OR-reduces its failure evidence, so the loop holds no branch and no
/// `Result` discriminant. The lifting retries a nullable batch over only its valid rows if that
/// reduction is non-zero, which is what makes an overflow behind a null row invisible.
///
/// The reduction lives in the sink rather than in the row closure's return type so that its width
/// is [`Op::Failure`](CheckedPrimitiveOp::Failure), the operator's choice, rather than one bit. That
/// is what lets unsigned multiplication report its discarded high half instead of a comparison, and
/// so stay vectorized.
///
/// **The storage is deliberately uninitialized, not zeroed.** Substituting `BufferMut::zeroed` to
/// make the sink safe was measured at **1.65 to 1.71x** the cost of allocate-and-fill, stable across
/// two runs and every batch size from 8 KiB to 2 MiB, because `alloc_zeroed` does not avoid the
/// write: below glibc's mmap threshold `calloc` recycles a dirty chunk and memsets it, and above it
/// the first touch of each fresh page faults instead. The row loop overwrites every slot regardless,
/// so that pass is pure duplicate work on the hottest kernel in the system. This is the case the
/// repository's "avoid `unsafe` unless it is necessary" rule leaves room for: the safe spelling
/// exists, and it costs a second pass over the output.
///
/// Rows are written into uninitialized storage, so this sink cannot finish a batch whose rows were
/// not all visited, and leaves [`OutputSink::SUPPORTS_SKIPPED_ROWS`] at `false`. Nothing is lost:
/// `SUPPORTS_SKIPPED_ROWS` is what makes branch-and-skip unavailable, which is the guard that keeps
/// the uninitialized slots sound. Note this is _not_ implied by the dispatch policy alone: a
/// deferred result still reaches the executor's valid-only policy whenever its arguments are not
/// dense-safe, so the `false` here is load-bearing rather than a restatement.
struct CheckedSink<T: NativePType, Op: CheckedPrimitiveOp<T>> {
    /// The result values, initialized one row at a time up to `row_count`.
    values: BufferMut<T>,

    /// The batch length, which is the capacity `values` was allocated with.
    row_count: usize,

    /// The operation applied to every row, which names the error reported by
    /// [`finish`](OutputSink::finish).
    op: PhantomData<Op>,
}

/// The uninitialized output slots of a [`CheckedSink`], borrowed once for the row loop.
struct CheckedRows<'a, T: NativePType, Op: CheckedPrimitiveOp<T>> {
    values: &'a mut [MaybeUninit<T>],
    op: PhantomData<Op>,
}

/// One output slot of a [`CheckedSink`].
struct CheckedRow<'a, T: NativePType, Op: CheckedPrimitiveOp<T>> {
    value: &'a mut MaybeUninit<T>,
    op: PhantomData<Op>,
}

impl<T: NativePType, Op: CheckedPrimitiveOp<T>> CheckedRow<'_, T, Op> {
    /// Apply `Op` to one row, writing its value and handing back its failure evidence.
    ///
    /// The value is written whether or not the operation failed, since a failing row is either
    /// masked away as null or turned into a batch error before it can be read. The evidence is
    /// returned rather than reduced here so the executor can keep the reduction in a register, and
    /// it is `Op`'s own width so the row never has to compare.
    fn write(self, lhs: T, rhs: T) -> Op::Failure {
        let (value, failure) = Op::apply(lhs, rhs);
        self.value.write(value);

        failure
    }
}

impl<T: NativePType, Op: CheckedPrimitiveOp<T>> OutputSink for CheckedSink<T, Op> {
    const ERRORS_ARE_DEFERRED: bool = true;

    type Rows<'a>
        = CheckedRows<'a, T, Op>
    where
        Self: 'a;
    type Row<'a>
        = CheckedRow<'a, T, Op>
    where
        Self: 'a;

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(T::PTYPE, Nullability::NonNullable))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            values: BufferMut::with_capacity(rows),
            row_count: rows,
            op: PhantomData,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        let row_count = self.row_count;
        CheckedRows {
            values: &mut self.values.spare_capacity_mut()[..row_count],
            op: PhantomData,
        }
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.values.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        CheckedRow {
            value: &mut rows.values[index],
            op: PhantomData,
        }
    }

    fn finish(mut self, error: DeferredError) -> VortexResult<ArrayRef> {
        if error.occurred() {
            return Err(vortex_err!(InvalidArgument: "{}", Op::ERROR));
        }

        // SAFETY: the sink reports `SUPPORTS_SKIPPED_ROWS = false`, so every path that reaches
        // `finish` without an error has written all `row_count` slots: dense execution visits
        // `0..row_count`, and the valid-row retry runs densely over a sink allocated for exactly
        // the filtered rows.
        unsafe { self.values.set_len(self.row_count) };

        Ok(PrimitiveArray::new(self.values.freeze(), Validity::NonNullable).into_array())
    }
}
