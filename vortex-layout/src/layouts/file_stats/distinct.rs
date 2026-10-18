// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates the number of distinct values of a column while it is written.

use std::ops::BitAnd;

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
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// The number of bits of a hash that select a register.
const PRECISION: u32 = 14;
const NUM_REGISTERS: usize = 1 << PRECISION;
/// The vectors whose leading rows DuckDB samples.
const VECTOR_SIZE: u64 = 2048;

/// A HyperLogLog sketch of the distinct non-null values of a column, estimated as DuckDB estimates
/// them for its own tables.
///
/// DuckDB samples the leading rows of each vector it stores and scales the count of distinct
/// sampled values up to the whole column. Its optimizer orders joins by these estimates, so the
/// sketch samples and scales the same way: exact counts of columns whose values the samples miss
/// led DuckDB to different and slower join orders than for its own tables, as in TPC-DS q81.
/// With 2^14 registers the count of sampled values has a standard error of about 0.8%.
pub(super) struct DistinctSketch {
    registers: Box<[u8]>,
    /// How many rows of each `VECTOR_SIZE` rows the sketch samples, as DuckDB samples them.
    sample_len: u64,
    /// The rows added, and the rows sampled among them.
    rows: u64,
    sampled: u64,
    /// The byte length of the longest string or binary value added, while it is tracked.
    max_length: Option<u64>,
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
            // DuckDB samples 30% of each vector of integers and 10% of other values.
            sample_len: if matches!(dtype, DType::Primitive(ptype, _) if ptype.is_int()) {
                614
            } else {
                204
            },
            rows: 0,
            sampled: 0,
            max_length: matches!(countable, DType::Utf8(_) | DType::Binary(_)).then_some(0),
        })
    }

    /// Adds the non-null values of `array`.
    pub(super) fn insert_array(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let canonical = array.clone().execute::<Canonical>(ctx)?;
        let start = self.rows;
        let sample_len = self.sample_len;
        let sample = Mask::from_buffer(BitBuffer::collect_bool(array.len(), |idx| {
            (start + idx as u64) % VECTOR_SIZE < sample_len
        }));
        self.rows += array.len() as u64;
        self.sampled += sample.true_count() as u64;
        self.insert_canonical(canonical, &sample, ctx)
    }

    fn insert_canonical(
        &mut self,
        canonical: Canonical,
        sample: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match canonical {
            Canonical::Bool(array) => self.insert_bool(&array, sample, ctx),
            Canonical::Primitive(array) => self.insert_primitive(&array, sample, ctx),
            Canonical::VarBinView(array) => self.insert_varbin(&array, sample, ctx),
            Canonical::Extension(array) => {
                let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
                self.insert_canonical(storage, sample, ctx)
            }
            _ => Ok(()),
        }
    }

    fn insert_bool(
        &mut self,
        array: &BoolArray,
        sample: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let bits = array.to_bit_buffer();
        let mask = array
            .validity()?
            .execute_mask(array.len(), ctx)?
            .bitand(sample);
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
        sample: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mask = array
            .validity()?
            .execute_mask(array.len(), ctx)?
            .bitand(sample);
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
        sample: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let buffers = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host())
            .collect::<Vec<_>>();
        let mut max_length = 0;
        for (index, view) in array.views().iter().enumerate() {
            if !mask.value(index) {
                continue;
            }
            max_length = max_length.max(view.len());
            if !sample.value(index) {
                continue;
            }
            if view.is_inlined() {
                self.insert_bytes(view.as_inlined().value());
            } else {
                let view = view.as_view();
                self.insert_bytes(&buffers[view.buffer_index as usize][view.as_range()]);
            }
        }
        if let Some(longest) = &mut self.max_length {
            *longest = (*longest).max(u64::from(max_length));
        }
        Ok(())
    }

    /// The byte length of the longest string or binary value added.
    pub(super) fn max_length(&self) -> Option<u64> {
        self.max_length
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

    /// The estimated number of distinct values added, scaled up from the sample the way DuckDB
    /// scales the samples of its own tables, so that DuckDB plans scans of Vortex files with the
    /// estimates it is tuned for.
    pub(super) fn estimate(&self) -> u64 {
        if self.sampled == 0 {
            return 0;
        }
        #[expect(clippy::cast_precision_loss, reason = "counts are estimates")]
        let (distinct, sampled, rows) = (
            self.sample_estimate().min(self.sampled) as f64,
            self.sampled as f64,
            self.rows as f64,
        );
        // Good-Turing estimation, assuming this proportion of the sampled values occurs once.
        let once = (distinct / sampled).powi(2) * distinct;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the estimate is a non-negative count"
        )]
        let estimate = (distinct + once / sampled * (rows - sampled)) as u64;
        estimate.min(self.rows)
    }

    /// The estimated number of distinct values sampled.
    fn sample_estimate(&self) -> u64 {
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DistinctSketch;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn assert_close(estimate: u64, exact: u64) {
        let error = estimate.abs_diff(exact) as f64 / exact.max(1) as f64;
        assert!(error < 0.03, "estimate {estimate} is not close to {exact}");
    }

    #[rstest]
    #[case(1)]
    #[case(10)]
    #[case(500)]
    fn counts_values_that_every_sample_sees(#[case] distinct: i64) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        let values = (0..100_000i64).map(|row| row % distinct);
        sketch.insert_array(&PrimitiveArray::from_iter(values).into_array(), &mut ctx)?;
        assert_close(sketch.estimate(), u64::try_from(distinct)?);
        Ok(())
    }

    #[test]
    fn scales_unique_values_to_the_row_count() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        for chunk in 0..3i64 {
            let values = (chunk * 100_000..(chunk + 1) * 100_000).map(|row| row * 7);
            sketch.insert_array(&PrimitiveArray::from_iter(values).into_array(), &mut ctx)?;
        }
        assert_close(sketch.estimate(), 300_000);
        Ok(())
    }

    /// Like DuckDB, the sketch samples only the leading rows of each vector.
    #[test]
    fn samples_the_leading_rows_of_each_vector() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        let values = (0..20_000i64).map(|row| if row % 2048 < 614 { 7 } else { row });
        sketch.insert_array(&PrimitiveArray::from_iter(values).into_array(), &mut ctx)?;
        assert_eq!(sketch.estimate(), 1);
        Ok(())
    }

    #[test]
    fn skips_nulls_and_counts_strings() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::Nullable);
        let mut sketch = DistinctSketch::try_new(&dtype).expect("strings are countable");
        let values = (0..5_000).map(|i| (i % 3 != 0).then(|| format!("value number {}", i % 100)));
        sketch.insert_array(
            &VarBinViewArray::from_iter_nullable_str(values).into_array(),
            &mut ctx,
        )?;
        // The 612 sampled rows of the 5000 hold all 100 values, which Good-Turing estimation
        // scales up to 100 + (100 / 612)^2 * 100 / 612 * (5000 - 612).
        assert_close(sketch.estimate(), 119);
        Ok(())
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let sketch = DistinctSketch::try_new(&dtype).expect("integers are countable");
        assert_eq!(sketch.estimate(), 0);
    }
}
