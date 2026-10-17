// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates the number of distinct values of a column while it is written.

use twox_hash::XxHash3_64;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::ToBytes;
use vortex_array::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// The number of bits of a hash that select a register.
const PRECISION: u32 = 14;
const NUM_REGISTERS: usize = 1 << PRECISION;

/// A HyperLogLog sketch of the distinct non-null values of a column.
///
/// With 2^14 registers its estimates have a standard error of about 0.8%. Query engines use
/// distinct counts to estimate the size of joins and to order them.
pub(super) struct DistinctSketch {
    registers: Box<[u8]>,
}

impl DistinctSketch {
    /// A sketch for values of `dtype`, if they can be counted.
    pub(super) fn try_new(dtype: &DType) -> Option<Self> {
        let countable = match dtype {
            DType::Extension(ext) => ext.storage_dtype(),
            dtype => dtype,
        };
        matches!(
            countable,
            DType::Bool(_) | DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
        )
        .then(|| Self {
            registers: vec![0; NUM_REGISTERS].into_boxed_slice(),
        })
    }

    /// Adds the non-null values of `array`.
    pub(super) fn insert_array(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let canonical = array.clone().execute::<Canonical>(ctx)?;
        self.insert_canonical(canonical, ctx)
    }

    fn insert_canonical(
        &mut self,
        canonical: Canonical,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match canonical {
            Canonical::Bool(array) => self.insert_bool(&array, ctx),
            Canonical::Primitive(array) => self.insert_primitive(&array, ctx),
            Canonical::VarBinView(array) => self.insert_varbin(&array, ctx),
            Canonical::Extension(array) => {
                let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
                self.insert_canonical(storage, ctx)
            }
            _ => Ok(()),
        }
    }

    fn insert_bool(&mut self, array: &BoolArray, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let bits = array.to_bit_buffer();
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        for (value, valid) in bits.iter().zip(mask.to_bit_buffer().iter()) {
            if valid {
                self.insert_bytes(&[u8::from(value)]);
            }
        }
        Ok(())
    }

    fn insert_primitive(
        &mut self,
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        match_each_native_ptype!(array.ptype(), |T| {
            self.insert_values(array.as_slice::<T>(), &mask)
        });
        Ok(())
    }

    fn insert_values<T: ToBytes>(&mut self, values: &[T], mask: &Mask) {
        match mask {
            Mask::AllTrue(_) => {
                for value in values {
                    self.insert_bytes(value.to_le_bytes().as_ref());
                }
            }
            Mask::AllFalse(_) => {}
            Mask::Values(mask) => {
                for &(start, end) in mask.slices() {
                    for value in &values[start..end] {
                        self.insert_bytes(value.to_le_bytes().as_ref());
                    }
                }
            }
        }
    }

    fn insert_varbin(
        &mut self,
        array: &VarBinViewArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let buffers = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host())
            .collect::<Vec<_>>();
        for (index, view) in array.views().iter().enumerate() {
            if !mask.value(index) {
                continue;
            }
            if view.is_inlined() {
                self.insert_bytes(view.as_inlined().value());
            } else {
                let view = view.as_view();
                self.insert_bytes(&buffers[view.buffer_index as usize][view.as_range()]);
            }
        }
        Ok(())
    }

    #[inline]
    fn insert_bytes(&mut self, bytes: &[u8]) {
        let hash = XxHash3_64::oneshot(bytes);
        let register = usize::try_from(hash >> (64 - PRECISION)).unwrap_or_default();
        // The position of the first set bit after the register bits, with a sentinel bit so that
        // the rank is at most 64 - PRECISION + 1.
        let rank = ((hash << PRECISION) | (1 << (PRECISION - 1))).leading_zeros() + 1;
        let rank = u8::try_from(rank).unwrap_or(u8::MAX);
        if rank > self.registers[register] {
            self.registers[register] = rank;
        }
    }

    /// The estimated number of distinct values added.
    pub(super) fn estimate(&self) -> u64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the register count is a small power of two"
        )]
        let registers = NUM_REGISTERS as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / registers);
        let mut harmonic_sum = 0.0;
        let mut zeros = 0usize;
        for &rank in self.registers.iter() {
            harmonic_sum += (-f64::from(rank)).exp2();
            zeros += usize::from(rank == 0);
        }
        let raw = alpha * registers * registers / harmonic_sum;
        // Linear counting is more accurate while many registers are still empty.
        let estimate = if raw <= 2.5 * registers && zeros > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the count of empty registers is small"
            )]
            let zeros = zeros as f64;
            registers * (registers / zeros).ln()
        } else {
            raw
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the estimate is a non-negative count"
        )]
        let estimate = estimate.round() as u64;
        estimate
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::DistinctSketch;

    fn assert_close(estimate: u64, exact: u64) {
        let error = estimate.abs_diff(exact) as f64 / exact.max(1) as f64;
        assert!(error < 0.03, "estimate {estimate} is not close to {exact}");
    }

    #[rstest]
    #[case(1)]
    #[case(10)]
    #[case(1_000)]
    #[case(100_000)]
    #[case(1_000_000)]
    fn estimates_distinct_integers(#[case] distinct: u64) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        // Each value appears once in every chunk.
        let distinct = i64::try_from(distinct)?;
        for chunk in 0..3 {
            let values = (0..distinct).map(|value| (value + chunk) % distinct);
            sketch.insert_array(&PrimitiveArray::from_iter(values).into_array(), &mut ctx)?;
        }
        let distinct = u64::try_from(distinct)?;
        assert_close(sketch.estimate(), distinct);
        Ok(())
    }

    #[test]
    fn skips_nulls_and_counts_strings() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Utf8(Nullability::Nullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("strings are countable");
        let values =
            (0..5_000).map(|i| (i % 3 != 0).then(|| format!("value number {}", i % 1_000)));
        sketch.insert_array(
            &VarBinViewArray::from_iter_nullable_str(values).into_array(),
            &mut ctx,
        )?;
        assert_close(sketch.estimate(), 1_000);
        Ok(())
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        assert_eq!(sketch.estimate(), 0);
    }
}
