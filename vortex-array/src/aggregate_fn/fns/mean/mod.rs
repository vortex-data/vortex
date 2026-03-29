// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::PrimitiveArray;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Compute the arithmetic mean of an array.
///
/// See [`Mean`] for details.
pub fn mean(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(Mean, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Compute the arithmetic mean of an array, returning `f64`.
///
/// Applies to boolean and primitive numeric types. Returns a nullable `f64`.
/// Internally tracks sum and count, returning `sum / count` on finalize.
/// If there are no valid elements, returns null.
///
/// The partial state is a struct `{sum: f64, count: u64}` so that partials from
/// different accumulators can be correctly combined via weighted addition.
#[derive(Clone, Debug)]
pub struct Mean;

/// Internal accumulation state for [`Mean`].
pub struct MeanPartial {
    sum: f64,
    count: u64,
}

fn partial_struct_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            [FieldName::from("sum"), FieldName::from("count")].into(),
            vec![
                DType::Primitive(PType::F64, Nullability::NonNullable),
                DType::Primitive(PType::U64, Nullability::NonNullable),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for Mean {
    type Options = EmptyOptions;
    type Partial = MeanPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.mean")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &vortex_session::VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Bool(_) | DType::Primitive(..) => {
                Some(DType::Primitive(PType::F64, Nullability::Nullable))
            }
            _ => None,
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Bool(_) | DType::Primitive(..) => Some(partial_struct_dtype()),
            _ => None,
        }
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(MeanPartial { sum: 0.0, count: 0 })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let s = other.as_struct();
        let sum_scalar = s
            .field("sum")
            .vortex_expect("mean partial must have sum field");
        let count_scalar = s
            .field("count")
            .vortex_expect("mean partial must have count field");

        partial.sum += sum_scalar
            .as_primitive()
            .typed_value::<f64>()
            .vortex_expect("sum field should not be null");
        partial.count += count_scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count field should not be null");
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if partial.count == 0 {
            Ok(Scalar::null(partial_struct_dtype()))
        } else {
            Ok(Scalar::struct_(
                partial_struct_dtype(),
                vec![
                    Scalar::primitive(partial.sum, Nullability::NonNullable),
                    Scalar::primitive(partial.count, Nullability::NonNullable),
                ],
            ))
        }
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.sum = 0.0;
        partial.count = 0;
    }

    #[inline]
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(c) => {
                if !c.scalar().is_null() {
                    let val = scalar_to_f64(c.scalar())?;
                    partial.sum += val * c.len() as f64;
                    partial.count += c.len() as u64;
                }
            }
            Columnar::Canonical(canonical) => match canonical {
                Canonical::Primitive(prim) => {
                    let mask = prim.validity_mask()?;
                    match_each_native_ptype!(prim.ptype(), |T| {
                        accumulate_values(partial, prim.as_slice::<T>(), &mask);
                    });
                }
                Canonical::Bool(bool_arr) => {
                    let mask = bool_arr.validity_mask()?;
                    let bits = bool_arr.to_bit_buffer();
                    match &mask {
                        Mask::AllTrue(_) => {
                            partial.sum += bits.true_count() as f64;
                            partial.count += bool_arr.len() as u64;
                        }
                        Mask::AllFalse(_) => {}
                        Mask::Values(validity) => {
                            let valid_count = validity.true_count();
                            let valid_and_true = (&bits & validity.bit_buffer()).true_count();
                            partial.sum += valid_and_true as f64;
                            partial.count += valid_count as u64;
                        }
                    }
                }
                _ => vortex_bail!("Unsupported canonical type for mean: {}", batch.dtype()),
            },
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        let struct_arr = partials.to_struct();
        let sums = struct_arr.unmasked_field_by_name("sum")?;
        let counts = struct_arr.unmasked_field_by_name("count")?;
        let validity_mask = struct_arr.validity_mask()?;

        let sum_prim = sums.to_primitive();
        let count_prim = counts.to_primitive();
        let sum_values = sum_prim.as_slice::<f64>();
        let count_values = count_prim.as_slice::<u64>();

        let means: vortex_buffer::Buffer<f64> = sum_values
            .iter()
            .zip(count_values.iter())
            .map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 })
            .collect();

        // A mean is valid when the group itself was valid AND had at least one
        // non-null element (count > 0).
        let validity = Validity::from_iter(
            count_values
                .iter()
                .enumerate()
                .map(|(i, c)| validity_mask.value(i) && *c > 0),
        );

        Ok(PrimitiveArray::new(means, validity).into_array())
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if partial.count == 0 {
            Ok(Scalar::null(DType::Primitive(
                PType::F64,
                Nullability::Nullable,
            )))
        } else {
            Ok(Scalar::primitive(
                partial.sum / partial.count as f64,
                Nullability::Nullable,
            ))
        }
    }
}

fn scalar_to_f64(scalar: &Scalar) -> VortexResult<f64> {
    match scalar.dtype() {
        DType::Bool(_) => {
            let v = scalar.as_bool().value().vortex_expect("checked non-null");
            Ok(if v { 1.0 } else { 0.0 })
        }
        DType::Primitive(..) => f64::try_from(scalar),
        _ => vortex_bail!("Cannot convert {} to f64 for mean", scalar.dtype()),
    }
}

fn accumulate_values<T: ToPrimitive>(partial: &mut MeanPartial, values: &[T], mask: &Mask) {
    match mask {
        Mask::AllTrue(_) => {
            partial.count += values.len() as u64;
            for v in values {
                partial.sum += v.to_f64().unwrap_or(0.0);
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(v) => {
            for (val, valid) in values.iter().zip(v.bit_buffer().iter()) {
                if valid {
                    partial.count += 1;
                    partial.sum += val.to_f64().unwrap_or(0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::mean::Mean;
    use crate::aggregate_fn::fns::mean::mean;
    use crate::aggregate_fn::fns::mean::partial_struct_dtype;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn mean_all_valid() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable)
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(2.0f64), None, Some(4.0)]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<f64, _>([None, None, None]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn mean_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Mean, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn mean_integers() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.0));
        Ok(())
    }

    #[test]
    fn mean_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Mean, EmptyOptions, dtype)?;

        let batch1 =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![4.0f64, 5.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Mean, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![2.0f64, 4.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().as_::<f64>(), Some(3.0));

        let batch2 =
            PrimitiveArray::new(buffer![10.0f64, 20.0, 30.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().as_::<f64>(), Some(20.0));
        Ok(())
    }

    #[test]
    fn mean_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut state = Mean.empty_partial(&EmptyOptions, &dtype)?;

        // Partition 1: mean of [2, 4] → sum=6, count=2
        let partial1 = Scalar::struct_(
            partial_struct_dtype(),
            vec![
                Scalar::primitive(6.0f64, Nullability::NonNullable),
                Scalar::primitive(2u64, Nullability::NonNullable),
            ],
        );
        Mean.combine_partials(&mut state, partial1)?;

        // Partition 2: mean of [10, 20, 30] → sum=60, count=3
        let partial2 = Scalar::struct_(
            partial_struct_dtype(),
            vec![
                Scalar::primitive(60.0f64, Nullability::NonNullable),
                Scalar::primitive(3u64, Nullability::NonNullable),
            ],
        );
        Mean.combine_partials(&mut state, partial2)?;

        // Combined: (6 + 60) / (2 + 3) = 13.2
        let result = Mean.finalize_scalar(&state)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(13.2));
        Ok(())
    }

    #[test]
    fn mean_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(5.0f64, 4);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn mean_constant_null() -> VortexResult<()> {
        let array = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable)),
            10,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn mean_bool() -> VortexResult<()> {
        let array: BoolArray = [true, false, true, true].into_iter().collect();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(0.75));
        Ok(())
    }

    #[test]
    fn mean_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(3.0)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(5.0f64), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&chunked.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }
}
