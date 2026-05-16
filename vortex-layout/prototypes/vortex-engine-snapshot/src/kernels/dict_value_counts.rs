//! Dict-aware `ValueCounts` kernel.
//!
//! For a `DictArray`, the codes already index into `[0, K)`, so
//! value-counts is one walk over the codes building a `Vec<u64>` of
//! length `K` — no `HashMap` needed. Each non-zero, non-null bucket
//! becomes a `(values[k], freq[k])` pair in the output histogram.
//!
//! Output shape matches the generic [`ValueCounts`] partial:
//! `Struct { values: List<T>, counts: List<u64> }`. Since codes are
//! non-negative integers and we iterate `k` in increasing order, the
//! emitted pairs are naturally sorted by code, which means sorted by
//! `values[k]`'s positional order in the dictionary (not by value).
//! Generic-`ValueCounts` consumers shouldn't depend on a specific
//! key order; the dict kernel chooses the cheapest one available.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use std::sync::Arc;

use crate::kernels::ValueCounts;

#[derive(Debug)]
pub struct DictValueCountsKernel;

impl DynAggregateKernel for DictValueCountsKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<ValueCounts>() {
            return Ok(None);
        }
        let Some(dict_view) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };
        let codes = dict_view.codes();
        let values = dict_view.values();

        // Only primitive-valued dicts for V1 — matches `ValueCounts`
        // itself.
        let values_ptype = match values.dtype() {
            DType::Primitive(p, _) => *p,
            _ => return Ok(None),
        };
        let values_nullability = match values.dtype() {
            DType::Primitive(_, n) => *n,
            _ => return Ok(None),
        };

        let codes_canonical = codes.clone().execute::<Canonical>(ctx)?;
        let values_canonical = values.clone().execute::<Canonical>(ctx)?;
        let Canonical::Primitive(codes_primitive) = codes_canonical else {
            return Ok(None);
        };
        let Canonical::Primitive(values_primitive) = values_canonical else {
            return Ok(None);
        };
        let n_values = values_primitive.len();
        let codes_validity = codes_primitive
            .validity()?
            .execute_mask(codes_primitive.len(), ctx)?;
        let values_validity = values_primitive
            .validity()?
            .execute_mask(n_values, ctx)?;

        // Frequency histogram over codes.
        let mut freq = vec![0u64; n_values];
        match codes_primitive.ptype() {
            PType::U8 => accumulate_freq(codes_primitive.as_slice::<u8>(), &codes_validity, &mut freq),
            PType::U16 => accumulate_freq(codes_primitive.as_slice::<u16>(), &codes_validity, &mut freq),
            PType::U32 => accumulate_freq(codes_primitive.as_slice::<u32>(), &codes_validity, &mut freq),
            PType::U64 => accumulate_freq(codes_primitive.as_slice::<u64>(), &codes_validity, &mut freq),
            PType::I8 => accumulate_freq(codes_primitive.as_slice::<i8>(), &codes_validity, &mut freq),
            PType::I16 => accumulate_freq(codes_primitive.as_slice::<i16>(), &codes_validity, &mut freq),
            PType::I32 => accumulate_freq(codes_primitive.as_slice::<i32>(), &codes_validity, &mut freq),
            PType::I64 => accumulate_freq(codes_primitive.as_slice::<i64>(), &codes_validity, &mut freq),
            _ => return Ok(None),
        }

        // Emit (values[k], freq[k]) for non-zero buckets where the
        // value isn't null. Output is in dictionary-positional order
        // — sorted by `k` not by value, which is fine for any
        // `ValueCounts` consumer that doesn't assume a specific
        // ordering.
        let mut emitted_values: Vec<Scalar> = Vec::new();
        let mut emitted_counts: Vec<Scalar> = Vec::new();
        for k in 0..n_values {
            let f = freq[k];
            if f == 0 || !validity_bit(&values_validity, k) {
                continue;
            }
            let value_scalar = primitive_to_scalar(
                &values_primitive,
                k,
                values_ptype,
                values_nullability,
            );
            emitted_values.push(value_scalar);
            emitted_counts.push(Scalar::primitive(f, Nullability::NonNullable));
        }

        // Build the struct-of-lists Scalar.
        let value_elem_dtype = DType::Primitive(values_ptype, values_nullability);
        let count_elem_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let values_list =
            Scalar::list(value_elem_dtype.clone(), emitted_values, Nullability::NonNullable);
        let counts_list =
            Scalar::list(count_elem_dtype.clone(), emitted_counts, Nullability::NonNullable);
        let struct_dtype = DType::Struct(
            StructFields::new(
                FieldNames::from(vec![FieldName::from("values"), FieldName::from("counts")]),
                vec![
                    DType::List(Arc::new(value_elem_dtype), Nullability::NonNullable),
                    DType::List(Arc::new(count_elem_dtype), Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );
        Ok(Some(Scalar::struct_(struct_dtype, vec![values_list, counts_list])))
    }
}

fn accumulate_freq<P>(codes: &[P], validity: &Mask, freq: &mut [u64])
where
    P: Copy + TryInto<usize>,
{
    match validity.bit_buffer() {
        AllOr::All => {
            for &c in codes {
                if let Ok(idx) = c.try_into()
                    && idx < freq.len()
                {
                    freq[idx] = freq[idx].saturating_add(1);
                }
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for i in mask.set_indices() {
                if let Ok(idx) = codes[i].try_into()
                    && idx < freq.len()
                {
                    freq[idx] = freq[idx].saturating_add(1);
                }
            }
        }
    }
}

fn validity_bit(validity: &Mask, index: usize) -> bool {
    match validity.bit_buffer() {
        AllOr::All => true,
        AllOr::None => false,
        AllOr::Some(m) => m.value(index),
    }
}

fn primitive_to_scalar(
    primitive: &vortex_array::arrays::PrimitiveArray,
    index: usize,
    ptype: PType,
    nullability: Nullability,
) -> Scalar {
    match ptype {
        PType::U8 => Scalar::primitive(primitive.as_slice::<u8>()[index], nullability),
        PType::U16 => Scalar::primitive(primitive.as_slice::<u16>()[index], nullability),
        PType::U32 => Scalar::primitive(primitive.as_slice::<u32>()[index], nullability),
        PType::U64 => Scalar::primitive(primitive.as_slice::<u64>()[index], nullability),
        PType::I8 => Scalar::primitive(primitive.as_slice::<i8>()[index], nullability),
        PType::I16 => Scalar::primitive(primitive.as_slice::<i16>()[index], nullability),
        PType::I32 => Scalar::primitive(primitive.as_slice::<i32>()[index], nullability),
        PType::I64 => Scalar::primitive(primitive.as_slice::<i64>()[index], nullability),
        PType::F32 => Scalar::primitive(primitive.as_slice::<f32>()[index], nullability),
        PType::F64 => Scalar::primitive(primitive.as_slice::<f64>()[index], nullability),
        PType::F16 => unreachable!("dict_value_counts rejects f16"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    fn dict<C, V>(codes: &[C], values: &[V]) -> DictArray
    where
        C: vortex_array::dtype::NativePType,
        V: vortex_array::dtype::NativePType,
    {
        let codes_arr =
            PrimitiveArray::new(Buffer::copy_from(codes), Validity::NonNullable).into_array();
        let values_arr =
            PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable).into_array();
        DictArray::try_new(codes_arr, values_arr).expect("dict")
    }

    fn run_kernel(dict_array: DictArray) -> Scalar {
        use vortex::VortexSessionDefault;
        let session = VortexSession::default();
        let mut exec = session.create_execution_ctx();
        let agg = ValueCounts.bind(EmptyOptions);
        DictValueCountsKernel
            .aggregate(&agg, &dict_array.into_array(), &mut exec)
            .expect("kernel ok")
            .expect("kernel produced partial")
    }

    fn extract_i64(scalar: &Scalar) -> (Vec<i64>, Vec<u64>) {
        let view = scalar.as_struct();
        let values_field = view.field_by_idx(0).expect("values");
        let counts_field = view.field_by_idx(1).expect("counts");
        let values = values_field.as_list();
        let counts = counts_field.as_list();
        let mut vs: Vec<i64> = Vec::with_capacity(values.len());
        let mut cs: Vec<u64> = Vec::with_capacity(counts.len());
        for i in 0..values.len() {
            vs.push(
                values
                    .element(i)
                    .unwrap()
                    .as_primitive()
                    .typed_value::<i64>()
                    .unwrap(),
            );
            cs.push(
                counts
                    .element(i)
                    .unwrap()
                    .as_primitive()
                    .typed_value::<u64>()
                    .unwrap(),
            );
        }
        (vs, cs)
    }

    #[test]
    fn counts_referenced_values() {
        // codes=[0,1,2,3,0,1] values=[10,20,30,40]
        // freq = [2, 2, 1, 1]; emit pairs in code order.
        let d = dict::<u32, i64>(&[0, 1, 2, 3, 0, 1], &[10, 20, 30, 40]);
        let scalar = run_kernel(d);
        let (vs, cs) = extract_i64(&scalar);
        assert_eq!(vs, vec![10, 20, 30, 40]);
        assert_eq!(cs, vec![2, 2, 1, 1]);
    }

    #[test]
    fn unreferenced_values_excluded() {
        // codes=[0] values=[1, 999]; only the referenced value
        // appears in the output.
        let d = dict::<u32, i64>(&[0], &[1, 999]);
        let scalar = run_kernel(d);
        let (vs, cs) = extract_i64(&scalar);
        assert_eq!(vs, vec![1]);
        assert_eq!(cs, vec![1]);
    }

    #[test]
    fn empty_codes_empty_output() {
        let d = dict::<u32, i64>(&[], &[10, 20, 30]);
        let scalar = run_kernel(d);
        let (vs, cs) = extract_i64(&scalar);
        assert!(vs.is_empty());
        assert!(cs.is_empty());
    }
}
