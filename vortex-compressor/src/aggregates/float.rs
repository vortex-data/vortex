// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bitwise distinct floating-point values for compression and dictionary preparation.
//!
//! Nulls are excluded, while signed zeros and distinct NaN payloads remain separate values.
//! The retained native set keeps the serial dictionary order used by the compressor.

use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use rustc_hash::FxBuildHasher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateDTypes;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::half::f16;
use vortex_array::expr::list_length;
use vortex_array::expr::root;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use super::runs::RunSummary;
use super::runs::RunSummaryPartial;

/// Exact bitwise float distinctness, with cardinality as the final result.
#[derive(Clone, Debug)]
pub struct FloatDistinct;

/// Native float values and the numeric run summary collected in the same scan.
#[derive(Clone, Debug)]
pub struct FloatDistribution<T> {
    /// Bitwise distinct valid values, including signed zeros and NaN payloads.
    values: HashSet<NativeValue<T>, FxBuildHasher>,
    /// Numeric run counts and endpoints from the same valid values.
    runs: RunSummaryPartial,
}

impl<T> FloatDistribution<T> {
    /// Returns bitwise distinct valid native values.
    pub fn values(&self) -> &HashSet<NativeValue<T>, FxBuildHasher> {
        &self.values
    }

    /// Returns the ordered numeric run summary collected with the distinct values.
    pub fn run_summary(&self) -> &RunSummaryPartial {
        &self.runs
    }
}

/// Float distributions retained for dictionary preparation and run estimates.
#[derive(Clone, Debug)]
pub enum FloatDistinctPartial {
    /// Distinct values and runs of half-precision values.
    F16(FloatDistribution<f16>),
    /// Distinct values and runs of single-precision values.
    F32(FloatDistribution<f32>),
    /// Distinct values and runs of double-precision values.
    F64(FloatDistribution<f64>),
}

/// Visit the native float distribution variant.
macro_rules! with_distinct {
    ($partial:expr, | $values:ident | $body:expr) => {
        match $partial {
            FloatDistinctPartial::F16($values) => $body,
            FloatDistinctPartial::F32($values) => $body,
            FloatDistinctPartial::F64($values) => $body,
        }
    };
}

impl FloatDistinctPartial {
    /// Returns the ordered run summary collected during distinct-value accumulation.
    pub fn run_summary(&self) -> &RunSummaryPartial {
        with_distinct!(self, |values| &values.runs)
    }

    /// Returns the number of bitwise distinct valid values.
    ///
    /// # Panics
    ///
    /// Panics if the distinct cardinality exceeds the supported `u32` range.
    pub fn distinct_count(&self) -> u32 {
        with_distinct!(self, |values| {
            u32::try_from(values.values.len())
                .vortex_expect("float distinct cardinality fits in u32")
        })
    }
}

/// Erase each native float distribution without rebuilding its set.
macro_rules! impl_from_distinct {
    ($($variant:ident: $T:ty),+ $(,)?) => {
        $(impl From<FloatDistribution<$T>> for FloatDistinctPartial {
            fn from(values: FloatDistribution<$T>) -> Self {
                Self::$variant(values)
            }
        })+
    };
}

impl_from_distinct!(F16: f16, F32: f32, F64: f64);

impl AggregateFnVTable for FloatDistinct {
    type Options = EmptyOptions;
    type Partial = FloatDistinctPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.compressor.float_distinct");
        *ID
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        input_dtype
            .is_float()
            .then(|| DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if !input_dtype.is_float() {
            return None;
        }

        Some(DType::Struct(
            StructFields::from_iter([
                (
                    "values",
                    DType::List(
                        Arc::new(input_dtype.as_nonnullable()),
                        Nullability::NonNullable,
                    ),
                ),
                (
                    "runs",
                    RunSummary.partial_dtype(&EmptyOptions, input_dtype)?,
                ),
            ]),
            Nullability::NonNullable,
        ))
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        match_each_float_ptype!(args.dtype.as_ptype(), |T| {
            Ok(FloatDistribution {
                values: HashSet::<NativeValue<T>, FxBuildHasher>::with_capacity_and_hasher(
                    0,
                    FxBuildHasher,
                ),
                runs: RunSummaryPartial::default(),
            }
            .into())
        })
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        if scalar.dtype() != args.partial_dtype {
            vortex_bail!(
                "Expected float distinct partial dtype {}, got {}",
                args.partial_dtype,
                scalar.dtype()
            );
        }

        let fields = scalar.as_struct();
        let values = fields
            .field_by_idx(0)
            .vortex_expect("distribution has values");
        let runs = fields
            .field_by_idx(1)
            .vortex_expect("distribution has a run summary");
        let run_dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, args.dtype.clone())?;
        let runs = RunSummary.partial_from_scalar(run_dtypes.args(&EmptyOptions), runs)?;
        u32::try_from(runs.valid_count())?;

        match_each_float_ptype!(args.dtype.as_ptype(), |T| {
            let values = parse_distinct::<T>(&values)?;
            if values.len() as u64 > runs.valid_count()
                || values.is_empty() != (runs.valid_count() == 0)
            {
                vortex_bail!(
                    "Float distribution distinct values must match the run summary valid count"
                );
            }
            Ok(FloatDistribution { values, runs }.into())
        })
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        /// Merge matching native float distribution variants.
        macro_rules! merge {
            ($($variant:ident),+) => {
                match (first, second) {
                    $((FloatDistinctPartial::$variant(first), FloatDistinctPartial::$variant(second)) => {
                        Ok(merge_distribution(first, second)?.into())
                    })+
                    _ => vortex_bail!("Float distinct partials must have matching primitive types"),
                }
            };
        }

        merge!(F16, F32, F64)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let fields = args.partial_dtype.as_struct_fields();
        let values_dtype = fields
            .field_by_index(0)
            .vortex_expect("distribution has values");
        let element_dtype = values_dtype
            .as_list_element_opt()
            .vortex_expect("distinct values are a list");
        let entries = with_distinct!(partial, |values| {
            values
                .values
                .iter()
                .map(|value| Scalar::primitive(value.0, Nullability::NonNullable))
                .collect()
        });
        let values = Scalar::list(Arc::clone(element_dtype), entries, Nullability::NonNullable);
        let run_dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, args.dtype.clone())?;
        let runs = RunSummary.to_scalar(run_dtypes.args(&EmptyOptions), partial.run_summary())?;

        Ok(Scalar::struct_(args.partial_dtype.clone(), [values, runs]))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let next: Self::Partial = match batch {
            Columnar::Canonical(Canonical::Primitive(array)) => {
                match_each_float_ptype!(array.ptype(), |T| {
                    typed_distinct::<T>(array, ctx)?.into()
                })
            }
            Columnar::Constant(array) => {
                if array.is_empty() || array.scalar().is_null() {
                    return Ok(());
                }
                u32::try_from(array.len())?;

                match_each_float_ptype!(args.dtype.as_ptype(), |T| {
                    let mut values = HashSet::with_capacity_and_hasher(1, FxBuildHasher);
                    let value = T::try_from(array.scalar())?;
                    values.insert(NativeValue(value));
                    let count = array.len() as u64;
                    let transitions = if value.is_nan() { count - 1 } else { 0 };
                    let runs = RunSummaryPartial::from_parts(
                        count,
                        transitions,
                        Some(value.into()),
                        Some(value.into()),
                    )?;
                    FloatDistribution { values, runs }.into()
                })
            }
            _ => vortex_bail!(
                "Float distinct requires primitive float input, got {}",
                batch.dtype()
            ),
        };

        let previous = std::mem::replace(state, self.empty_partial(args)?);
        *state = self.merge_partials(args, previous, next)?;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        states.get_item("values")?.apply(&list_length(root()))
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            u64::from(partial.distinct_count()),
            Nullability::NonNullable,
        ))
    }
}

/// Parse bitwise distinct valid native values.
fn parse_distinct<T>(scalar: &Scalar) -> VortexResult<HashSet<NativeValue<T>, FxBuildHasher>>
where
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    NativeValue<T>: Eq + Hash,
{
    let entries = scalar
        .as_list()
        .elements()
        .vortex_expect("float distinct partial is non-nullable");
    u32::try_from(entries.len())?;
    let mut values = HashSet::with_capacity_and_hasher(entries.len(), FxBuildHasher);
    for entry in entries {
        if !values.insert(NativeValue(T::try_from(&entry)?)) {
            vortex_bail!("Float distinct entries must have bitwise distinct values");
        }
    }

    Ok(values)
}

/// Unite distinct values and merge the ordered run boundary.
fn merge_distribution<T>(
    first: FloatDistribution<T>,
    second: FloatDistribution<T>,
) -> VortexResult<FloatDistribution<T>>
where
    NativeValue<T>: Eq + Hash,
{
    if first.values.is_empty() {
        return Ok(second);
    }
    if second.values.is_empty() {
        return Ok(first);
    }

    let runs = first.runs.merge(second.runs)?;
    u32::try_from(runs.valid_count())?;
    let mut values = first.values;
    values.extend(second.values);
    Ok(FloatDistribution { values, runs })
}

/// Collect bitwise values and numeric runs in one valid-value scan.
fn typed_distinct<T>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FloatDistribution<T>>
where
    T: NativePType + Into<PValue>,
    NativeValue<T>: Eq + Hash,
{
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    u32::try_from(validity.true_count())?;
    u32::try_from(validity.false_count())?;
    let Some(head_idx) = validity.first() else {
        return Ok(FloatDistribution {
            values: HashSet::with_capacity_and_hasher(0, FxBuildHasher),
            runs: RunSummaryPartial::default(),
        });
    };

    let mut values = HashSet::with_capacity_and_hasher(array.len() / 2, FxBuildHasher);
    let buffer = array.to_buffer::<T>();
    let first = buffer[head_idx];
    let mut last = first;
    let mut transitions = 0u64;
    values.insert(NativeValue(first));
    let valid_values = buffer.slice(head_idx + 1..array.len());
    match validity.bit_buffer() {
        AllOr::All => {
            for value in valid_values {
                values.insert(NativeValue(value));
                transitions += u64::from(value != last);
                last = value;
            }
        }
        AllOr::None => unreachable!("the input has a valid value"),
        AllOr::Some(valid) => {
            for (&value, valid) in valid_values
                .iter()
                .zip_eq(valid.slice(head_idx + 1..array.len()).iter())
            {
                if valid {
                    values.insert(NativeValue(value));
                    transitions += u64::from(value != last);
                    last = value;
                }
            }
        }
    }

    let runs = RunSummaryPartial::from_parts(
        validity.true_count() as u64,
        transitions,
        Some(first.into()),
        Some(last.into()),
    )?;
    Ok(FloatDistribution { values, runs })
}
