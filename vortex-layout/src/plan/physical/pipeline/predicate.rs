// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary predicates evaluated only for referenced values, shared across scan chunks.

use std::sync::OnceLock;
use std::sync::atomic::AtomicU8;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

const UNKNOWN: u8 = 0;
const FALSE: u8 = 1;
const TRUE: u8 = 2;
const NULL: u8 = 3;

/// Entries only transition from unknown to their immutable predicate result. Racing readers may
/// evaluate the same missing entry, but never block each other while executing a predicate.
pub(crate) struct PredicateCache {
    entries: Vec<AtomicU8>,
    whole: OnceLock<ArrayRef>,
    #[cfg(test)]
    evaluated: AtomicUsize,
}

impl PredicateCache {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| AtomicU8::new(UNKNOWN)).collect(),
            whole: OnceLock::new(),
            #[cfg(test)]
            evaluated: AtomicUsize::new(0),
        }
    }

    pub(crate) fn evaluate(
        &self,
        values: ArrayRef,
        codes: ArrayRef,
        expression: &BoundExpression,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(whole) = self.whole.get() {
            tracing::debug!(
                selected_rows = codes.len(),
                dictionary_values = whole.len(),
                missing_values = 0,
                evaluated_values = 0,
                dense = false,
                cached = true,
                "v2 dictionary predicate work"
            );
            return Ok(DictArray::try_new(codes, whole.clone())?.into_array());
        }
        let nullable = codes.dtype().is_nullable() || expression.dtype().is_nullable();
        let codes = codes
            .cast(DType::Primitive(PType::U64, codes.dtype().nullability()))?
            .execute::<PrimitiveArray>(ctx)?;
        let valid = codes.validity()?.execute_mask(codes.len(), ctx)?;
        let indices = codes.as_slice::<u64>();
        let mut missing = Vec::new();
        let mut check = |row: usize| -> VortexResult<()> {
            let code = usize::try_from(indices[row])?;
            vortex_ensure!(code < self.entries.len(), "Dictionary code out of bounds");
            if self.entries[code].load(Ordering::Relaxed) == UNKNOWN {
                missing.push(indices[row]);
            }
            Ok(())
        };
        match valid.indices() {
            AllOr::All => (0..codes.len()).try_for_each(&mut check)?,
            AllOr::Some(rows) => rows.iter().copied().try_for_each(&mut check)?,
            AllOr::None => {}
        }
        missing.sort_unstable();
        missing.dedup();
        let dense = missing.len().saturating_mul(4) >= values.len();
        let evaluated = if missing.is_empty() {
            0
        } else if dense {
            values.len()
        } else {
            missing.len()
        };
        if evaluated > 0 {
            let selected = if dense {
                values.clone()
            } else {
                values.take(Buffer::copy_from(&missing).into_array())?
            };
            let result = selected
                .apply_bound(expression)?
                .execute::<BoolArray>(ctx)?;
            let mut states = vec![FALSE; evaluated];
            result
                .to_bit_buffer()
                .for_each_set_index(|row| states[row] = TRUE);
            let invalid = !BoolArrayExt::validity(&result).execute_mask(result.len(), ctx)?;
            match invalid.indices() {
                AllOr::All => states.fill(NULL),
                AllOr::Some(rows) => rows.iter().for_each(|&row| states[row] = NULL),
                AllOr::None => {}
            }
            for (row, state) in states.into_iter().enumerate() {
                let index = if dense {
                    row
                } else {
                    usize::try_from(missing[row])?
                };
                self.entries[index].store(state, Ordering::Relaxed);
            }
            #[cfg(test)]
            self.evaluated.fetch_add(evaluated, Ordering::Relaxed);
            if dense {
                // Bulk evaluation has already paid for every value. Keep its packed result so
                // later batches use the normal dictionary kernels without per-code atomics.
                drop(self.whole.set(result.into_array()));
            }
        }
        tracing::debug!(
            selected_rows = codes.len(),
            dictionary_values = values.len(),
            missing_values = missing.len(),
            evaluated_values = evaluated,
            dense,
            "v2 dictionary predicate work"
        );
        if let Some(whole) = self.whole.get() {
            return Ok(DictArray::try_new(codes.into_array(), whole.clone())?.into_array());
        }
        self.gather(&codes, &valid, nullable)
    }

    fn gather(
        &self,
        codes: &PrimitiveArray,
        valid: &Mask,
        nullable: bool,
    ) -> VortexResult<ArrayRef> {
        let indices = codes.as_slice::<u64>();
        let mut states = vec![NULL; codes.len()];
        let mut read = |row: usize| -> VortexResult<()> {
            // Codes were checked before evaluation. Relaxed ordering suffices because each
            // atomic stores the complete result, with no separate data to publish.
            states[row] = self.entries[usize::try_from(indices[row])?].load(Ordering::Relaxed);
            Ok(())
        };
        match valid.indices() {
            AllOr::All => (0..codes.len()).try_for_each(&mut read)?,
            AllOr::Some(rows) => rows.iter().copied().try_for_each(&mut read)?,
            AllOr::None => {}
        }
        let bits = BitBuffer::collect_bool(states.len(), |row| states[row] == TRUE);
        let validity = if nullable {
            Validity::from(BitBuffer::collect_bool(states.len(), |row| {
                states[row] != NULL
            }))
        } else {
            Validity::NonNullable
        };
        Ok(BoolArray::new(bits, validity).into_array())
    }

    #[cfg(test)]
    pub(crate) fn evaluated_rows(&self) -> usize {
        self.evaluated.load(Ordering::Relaxed)
    }
}
