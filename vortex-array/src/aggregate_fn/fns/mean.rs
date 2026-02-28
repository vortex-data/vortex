// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::arrays::PrimitiveArray;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;

/// Computes the arithmetic mean of numeric values.
#[derive(Clone)]
pub struct Mean;

impl AggregateFnVTable for Mean {
    type Options = EmptyOptions;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.mean")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        if !input_dtype.is_int() && !input_dtype.is_float() {
            vortex_bail!("Mean requires numeric input, got {}", input_dtype);
        }
        Ok(DType::Primitive(PType::F64, Nullability::Nullable))
    }

    fn state_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        if !input_dtype.is_int() && !input_dtype.is_float() {
            vortex_bail!("Mean requires numeric input, got {}", input_dtype);
        }
        Ok(DType::Struct(
            StructFields::from_iter([
                (
                    "sum",
                    DType::Primitive(PType::F64, Nullability::NonNullable),
                ),
                (
                    "count",
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                ),
            ]),
            Nullability::Nullable,
        ))
    }

    fn accumulator(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>> {
        if !input_dtype.is_int() && !input_dtype.is_float() {
            vortex_bail!("Mean requires numeric input, got {}", input_dtype);
        }
        Ok(Box::new(MeanAccumulator::new()))
    }
}

struct MeanAccumulator {
    sum: f64,
    count: u64,
    results: Vec<Option<f64>>,
}

impl MeanAccumulator {
    fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            results: Vec::new(),
        }
    }
}

/// Accumulate all-valid values of type `T` into `sum` and `count`.
fn accumulate_all_valid<T: NativePType>(values: &[T], sum: &mut f64, count: &mut u64) {
    for v in values {
        *sum += v.to_f64().unwrap_or(0.0);
        *count += 1;
    }
}

/// Accumulate partially-valid values of type `T` into `sum` and `count`.
fn accumulate_with_mask<T: NativePType>(
    values: &[T],
    mask: &vortex_mask::MaskValues,
    sum: &mut f64,
    count: &mut u64,
) {
    for (val, valid) in values.iter().zip(mask.bit_buffer().iter()) {
        if valid {
            *sum += val.to_f64().unwrap_or(0.0);
            *count += 1;
        }
    }
}

impl Accumulator for MeanAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        let primitive = batch.to_primitive();
        let validity = primitive.validity_mask()?;

        match_each_native_ptype!(primitive.ptype(), |T| {
            let values = primitive.as_slice::<T>();
            match &validity {
                Mask::AllTrue(_) => accumulate_all_valid(values, &mut self.sum, &mut self.count),
                Mask::AllFalse(_) => {}
                Mask::Values(v) => accumulate_with_mask(values, v, &mut self.sum, &mut self.count),
            }
        });

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }

        let s = state.as_struct();
        let Some(sum_scalar) = s.field_by_idx(0) else {
            vortex_bail!("Mean state struct missing sum field at index 0");
        };
        let Some(count_scalar) = s.field_by_idx(1) else {
            vortex_bail!("Mean state struct missing count field at index 1");
        };

        self.sum += sum_scalar
            .as_primitive()
            .typed_value::<f64>()
            .unwrap_or(0.0);
        self.count += count_scalar
            .as_primitive()
            .typed_value::<u64>()
            .unwrap_or(0);
        Ok(())
    }

    fn flush(&mut self) -> VortexResult<()> {
        if self.count == 0 {
            self.results.push(None);
        } else {
            self.results.push(Some(self.sum / self.count as f64));
        }
        self.sum = 0.0;
        self.count = 0;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(self.results).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::fns::mean::Mean;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::validity::Validity;

    fn run_mean(batch: &ArrayRef) -> VortexResult<ArrayRef> {
        let mut acc = Mean.accumulator(&EmptyOptions, batch.dtype())?;
        acc.accumulate(batch)?;
        acc.flush()?;
        acc.finish()
    }

    fn get_f64_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<f64>> {
        let scalar = array.scalar_at(idx)?;
        Ok(scalar.as_primitive().typed_value::<f64>())
    }

    #[test]
    fn mean_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = run_mean(&arr)?;
        assert_eq!(get_f64_value(&result, 0)?, Some(2.5));
        Ok(())
    }

    #[test]
    fn mean_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        let result = run_mean(&arr)?;
        assert_eq!(get_f64_value(&result, 0)?, Some(2.0));
        Ok(())
    }

    #[test]
    fn mean_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = run_mean(&arr)?;
        assert_eq!(get_f64_value(&result, 0)?, Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = run_mean(&arr)?;
        assert_eq!(get_f64_value(&result, 0)?, None);
        Ok(())
    }

    #[test]
    fn mean_empty_flush() -> VortexResult<()> {
        let mut acc = Mean.accumulator(
            &EmptyOptions,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_f64_value(&result, 0)?, None);
        Ok(())
    }

    #[test]
    fn mean_multi_group() -> VortexResult<()> {
        let mut acc = Mean.accumulator(
            &EmptyOptions,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;
        acc.flush()?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;
        acc.flush()?;

        let result = acc.finish()?;
        assert_eq!(get_f64_value(&result, 0)?, Some(15.0));
        assert_eq!(get_f64_value(&result, 1)?, Some(6.0));
        Ok(())
    }

    #[test]
    fn mean_merge() -> VortexResult<()> {
        let mut acc = Mean.accumulator(
            &EmptyOptions,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;

        let state_dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "sum",
                    DType::Primitive(PType::F64, Nullability::NonNullable),
                ),
                (
                    "count",
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                ),
            ]),
            Nullability::Nullable,
        );

        let state = Scalar::struct_(
            state_dtype.clone(),
            vec![
                Scalar::primitive(30.0f64, Nullability::NonNullable),
                Scalar::primitive(3u64, Nullability::NonNullable),
            ],
        );
        acc.merge(&state)?;

        let state2 = Scalar::struct_(
            state_dtype,
            vec![
                Scalar::primitive(20.0f64, Nullability::NonNullable),
                Scalar::primitive(2u64, Nullability::NonNullable),
            ],
        );
        acc.merge(&state2)?;

        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_f64_value(&result, 0)?, Some(10.0));
        Ok(())
    }
}
