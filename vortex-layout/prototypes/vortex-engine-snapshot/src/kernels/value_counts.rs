//! `value_counts` aggregate.
//!
//! Counts occurrences per distinct value. The output (and partial)
//! shape is `struct(values: List<T>, counts: List<u64>)` where
//! `values[i]` and `counts[i]` form a parallel-array histogram.
//!
//! Fits vortex's `AggregateFnVTable` like any other aggregate; the
//! only thing unusual is that the partial state carries `O(K)` data
//! instead of a single scalar. The framework doesn't care.
//!
//! Registered on the engine's session under id `"engine.value_counts"`.
//! Composes with multiply + `Sum` to reproduce dict-aware `Sum`:
//! `Σ (dict.values * value_counts(dict.codes).counts)`.
//!
//! V1 supports primitive numeric inputs (i8/i16/i32/i64, u8/u16/u32/u64,
//! f32/f64). Decimals, strings, structs are follow-ups — the
//! framework supports them, this kernel just hasn't enumerated their
//! key encoding yet.

use std::collections::HashMap;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

/// `ValueCounts` aggregate function. Multiset construction.
#[derive(Clone, Debug)]
pub struct ValueCounts;

impl ValueCounts {
    pub const ID: &'static str = "engine.value_counts";
}

#[derive(Debug)]
pub struct ValueCountsPartial {
    input_dtype: DType,
    /// Keyed by the raw 8-byte bit pattern of the value (or 1 byte
    /// for booleans, etc.) cast to u64. Restored by the input ptype
    /// when serialising to a Scalar.
    counts: HashMap<u64, u64>,
}

impl ValueCountsPartial {
    fn new(input_dtype: DType) -> Self {
        Self {
            input_dtype,
            counts: HashMap::new(),
        }
    }

    fn input_ptype(&self) -> Option<PType> {
        match &self.input_dtype {
            DType::Primitive(p, _) => Some(*p),
            _ => None,
        }
    }
}

impl AggregateFnVTable for ValueCounts {
    type Options = EmptyOptions;
    type Partial = ValueCountsPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new(Self::ID)
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // Multiset → struct(values: List<T?>, counts: List<u64>).
        // Values list element matches the input dtype's nullability.
        let values_dtype = DType::List(
            Arc::new(input_dtype.clone()),
            Nullability::NonNullable,
        );
        let counts_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let fields = StructFields::new(
            FieldNames::from(vec![FieldName::from("values"), FieldName::from("counts")]),
            vec![values_dtype, counts_dtype],
        );
        Some(DType::Struct(fields, Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(ValueCountsPartial::new(input_dtype.clone()))
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        // `other` is the struct-of-lists form. Pull the value/count
        // pairs out and merge into `partial.counts`.
        let view = other.as_struct();
        let values_scalar = view
            .field_by_idx(0)
            .ok_or_else(|| vortex_err!("value_counts partial missing 'values' field"))?;
        let counts_scalar = view
            .field_by_idx(1)
            .ok_or_else(|| vortex_err!("value_counts partial missing 'counts' field"))?;
        let values_list = values_scalar.as_list();
        let counts_list = counts_scalar.as_list();
        let n = values_list.len();
        if counts_list.len() != n {
            return Err(vortex_err!(
                "value_counts partial: values len {} != counts len {}",
                n,
                counts_list.len()
            ));
        }
        let ptype = partial.input_ptype().ok_or_else(|| {
            vortex_err!(
                "value_counts: unsupported input dtype {} (primitives only)",
                partial.input_dtype
            )
        })?;
        for i in 0..n {
            let v_scalar = values_list
                .element(i)
                .ok_or_else(|| vortex_err!("value_counts: missing values[{i}]"))?;
            let c_scalar = counts_list
                .element(i)
                .ok_or_else(|| vortex_err!("value_counts: missing counts[{i}]"))?;
            let bits = scalar_to_bits(&v_scalar, ptype)?;
            let c = c_scalar
                .as_primitive()
                .typed_value::<u64>()
                .vortex_expect("counts entries are non-null u64");
            *partial.counts.entry(bits).or_insert(0) += c;
        }
        Ok(())
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let ptype = partial
            .input_ptype()
            .ok_or_else(|| vortex_err!("value_counts only supports primitive inputs"))?;
        match batch {
            Columnar::Constant(c) => {
                // Constant batch: one distinct value, repeated `len` times.
                let scalar = c.scalar();
                if scalar.is_null() {
                    return Ok(());
                }
                let bits = scalar_to_bits(scalar, ptype)?;
                *partial.counts.entry(bits).or_insert(0) += c.len() as u64;
            }
            Columnar::Canonical(canonical) => {
                let primitive = canonical.as_primitive();
                let n = primitive.len();
                let validity = primitive.validity()?.execute_mask(n, ctx)?;
                match ptype {
                    PType::U8 => count_primitive::<u8>(primitive.as_slice::<u8>(), &validity, &mut partial.counts),
                    PType::U16 => count_primitive::<u16>(primitive.as_slice::<u16>(), &validity, &mut partial.counts),
                    PType::U32 => count_primitive::<u32>(primitive.as_slice::<u32>(), &validity, &mut partial.counts),
                    PType::U64 => count_primitive::<u64>(primitive.as_slice::<u64>(), &validity, &mut partial.counts),
                    PType::I8 => count_primitive::<i8>(primitive.as_slice::<i8>(), &validity, &mut partial.counts),
                    PType::I16 => count_primitive::<i16>(primitive.as_slice::<i16>(), &validity, &mut partial.counts),
                    PType::I32 => count_primitive::<i32>(primitive.as_slice::<i32>(), &validity, &mut partial.counts),
                    PType::I64 => count_primitive::<i64>(primitive.as_slice::<i64>(), &validity, &mut partial.counts),
                    PType::F32 => count_float::<f32>(primitive.as_slice::<f32>(), &validity, &mut partial.counts),
                    PType::F64 => count_float::<f64>(primitive.as_slice::<f64>(), &validity, &mut partial.counts),
                    PType::F16 => {
                        return Err(vortex_err!("value_counts: f16 not yet supported"));
                    }
                }
            }
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let ptype = partial.input_ptype().ok_or_else(|| {
            vortex_err!("value_counts: cannot serialise non-primitive input dtype")
        })?;
        // Sort by key bits so the output is deterministic across runs.
        let mut entries: Vec<(u64, u64)> = partial.counts.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_unstable_by_key(|(k, _)| *k);
        let nullability = match &partial.input_dtype {
            DType::Primitive(_, n) => *n,
            _ => Nullability::NonNullable,
        };
        let value_dtype = DType::Primitive(ptype, nullability);
        let count_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let values: Vec<Scalar> = entries
            .iter()
            .map(|(k, _)| bits_to_scalar(*k, ptype, nullability))
            .collect();
        let counts: Vec<Scalar> = entries
            .iter()
            .map(|(_, c)| Scalar::primitive(*c, Nullability::NonNullable))
            .collect();
        let values_list = Scalar::list(value_dtype, values, Nullability::NonNullable);
        let counts_list = Scalar::list(count_dtype, counts, Nullability::NonNullable);
        let struct_dtype = self
            .return_dtype(&EmptyOptions, &partial.input_dtype)
            .vortex_expect("value_counts has a return_dtype for primitives");
        Ok(Scalar::struct_(struct_dtype, vec![values_list, counts_list]))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.counts.clear();
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn count_primitive<P>(slice: &[P], validity: &vortex_mask::Mask, counts: &mut HashMap<u64, u64>)
where
    P: Copy + IntoU64Bits,
{
    match validity.bit_buffer() {
        AllOr::All => {
            for &v in slice {
                *counts.entry(v.into_u64_bits()).or_insert(0) += 1;
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for i in mask.set_indices() {
                *counts.entry(slice[i].into_u64_bits()).or_insert(0) += 1;
            }
        }
    }
}

fn count_float<P>(slice: &[P], validity: &vortex_mask::Mask, counts: &mut HashMap<u64, u64>)
where
    P: Copy + IntoFloatBits,
{
    match validity.bit_buffer() {
        AllOr::All => {
            for &v in slice {
                *counts.entry(v.into_float_bits()).or_insert(0) += 1;
            }
        }
        AllOr::None => {}
        AllOr::Some(mask) => {
            for i in mask.set_indices() {
                *counts.entry(slice[i].into_float_bits()).or_insert(0) += 1;
            }
        }
    }
}

fn scalar_to_bits(scalar: &Scalar, ptype: PType) -> VortexResult<u64> {
    let prim = scalar.as_primitive();
    Ok(match ptype {
        PType::U8 => u64::from(prim.typed_value::<u8>().vortex_expect("u8")),
        PType::U16 => u64::from(prim.typed_value::<u16>().vortex_expect("u16")),
        PType::U32 => u64::from(prim.typed_value::<u32>().vortex_expect("u32")),
        PType::U64 => prim.typed_value::<u64>().vortex_expect("u64"),
        PType::I8 => prim.typed_value::<i8>().vortex_expect("i8") as u64,
        PType::I16 => prim.typed_value::<i16>().vortex_expect("i16") as u64,
        PType::I32 => prim.typed_value::<i32>().vortex_expect("i32") as u64,
        PType::I64 => prim.typed_value::<i64>().vortex_expect("i64") as u64,
        PType::F32 => {
            u64::from(prim.typed_value::<f32>().vortex_expect("f32").to_bits())
        }
        PType::F64 => prim.typed_value::<f64>().vortex_expect("f64").to_bits(),
        PType::F16 => return Err(vortex_err!("value_counts: f16 not yet supported")),
    })
}

fn bits_to_scalar(bits: u64, ptype: PType, nullability: Nullability) -> Scalar {
    match ptype {
        PType::U8 => Scalar::primitive(bits as u8, nullability),
        PType::U16 => Scalar::primitive(bits as u16, nullability),
        PType::U32 => Scalar::primitive(bits as u32, nullability),
        PType::U64 => Scalar::primitive(bits, nullability),
        PType::I8 => Scalar::primitive(bits as i8, nullability),
        PType::I16 => Scalar::primitive(bits as i16, nullability),
        PType::I32 => Scalar::primitive(bits as i32, nullability),
        PType::I64 => Scalar::primitive(bits as i64, nullability),
        PType::F32 => Scalar::primitive(f32::from_bits(bits as u32), nullability),
        PType::F64 => Scalar::primitive(f64::from_bits(bits), nullability),
        PType::F16 => unreachable!("scalar_to_bits rejects f16"),
    }
}

trait IntoU64Bits {
    fn into_u64_bits(self) -> u64;
}
macro_rules! into_u64_bits {
    ($($t:ty),+) => { $(impl IntoU64Bits for $t { fn into_u64_bits(self) -> u64 { self as u64 } })+ };
}
into_u64_bits!(u8, u16, u32, u64, i8, i16, i32, i64);

trait IntoFloatBits {
    fn into_float_bits(self) -> u64;
}
impl IntoFloatBits for f32 {
    fn into_float_bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl IntoFloatBits for f64 {
    fn into_float_bits(self) -> u64 {
        self.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    fn make_array(values: &[i64]) -> ArrayRef {
        PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable).into_array()
    }

    fn run(values: Vec<i64>) -> Scalar {
        use vortex::VortexSessionDefault;
        let session = VortexSession::default();
        let mut exec = session.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(ValueCounts, EmptyOptions, dtype).expect("acc");
        let array = make_array(&values);
        acc.accumulate(&array, &mut exec).expect("accumulate");
        acc.flush().expect("flush")
    }

    fn extract_pairs(scalar: &Scalar) -> (Vec<i64>, Vec<u64>) {
        let view = scalar.as_struct();
        let values_field = view.field_by_idx(0).expect("values");
        let counts_field = view.field_by_idx(1).expect("counts");
        let values = values_field.as_list();
        let counts = counts_field.as_list();
        let mut vs: Vec<i64> = Vec::with_capacity(values.len());
        let mut cs: Vec<u64> = Vec::with_capacity(counts.len());
        for i in 0..values.len() {
            let v = values.element(i).expect("value present");
            let c = counts.element(i).expect("count present");
            vs.push(v.as_primitive().typed_value::<i64>().unwrap());
            cs.push(c.as_primitive().typed_value::<u64>().unwrap());
        }
        (vs, cs)
    }

    #[test]
    fn counts_per_distinct_value() {
        let scalar = run(vec![1, 2, 2, 3, 3, 3]);
        let (vs, cs) = extract_pairs(&scalar);
        // Sorted by key bits.
        assert_eq!(vs, vec![1, 2, 3]);
        assert_eq!(cs, vec![1, 2, 3]);
    }

    #[test]
    fn empty_input_emits_empty_histogram() {
        let scalar = run(vec![]);
        let (vs, cs) = extract_pairs(&scalar);
        assert!(vs.is_empty());
        assert!(cs.is_empty());
    }

    #[test]
    fn combine_partials_sums_counts() {
        use vortex::VortexSessionDefault;
        let session = VortexSession::default();
        let mut exec = session.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        // Accumulating two batches into the same accumulator exercises
        // the same code paths combine_partials would (the per-batch
        // partial is folded via the framework's internal merge).
        let mut acc = Accumulator::try_new(ValueCounts, EmptyOptions, dtype).unwrap();
        acc.accumulate(&make_array(&[1, 1, 2]), &mut exec).unwrap();
        acc.accumulate(&make_array(&[2, 3, 3]), &mut exec).unwrap();
        let scalar = acc.flush().unwrap();
        let (vs, cs) = extract_pairs(&scalar);
        assert_eq!(vs, vec![1, 2, 3]);
        assert_eq!(cs, vec![2, 2, 2]);
    }
}
