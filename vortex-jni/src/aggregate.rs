// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ungrouped aggregates over one Vortex file, for query engines that push aggregation into a scan.
//!
//! Each aggregate is answered from the footer statistics when they state it exactly, which costs
//! no read beyond the footer. An aggregate the footer cannot answer is computed by a scan of only
//! the columns it needs, through Vortex's encoding-aware aggregate kernels, or reported as
//! unanswered when the caller forbids that scan.
//!
//! Floating-point extrema follow SQL engines such as Spark, which order NaN above every number:
//! a column holding a NaN has a NaN maximum, and only a column of nothing but NaN has a NaN
//! minimum. Integer sums wrap on overflow, which is exact modulo 2^64 however the rows are
//! grouped, so a caller can combine per-file sums without a separate overflow rule.

use std::ptr;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::Field;
use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jboolean;
use jni::sys::jlong;
use vortex::aggregate_fn::Accumulator;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::DynAccumulator;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::is_sorted::IsSorted;
use vortex::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::nan_count;
use vortex::aggregate_fn::fns::sum::sum;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::stats::Stat;
use vortex::file::VortexFile;
use vortex::file::multi::open_cached;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::RUNTIME;
use crate::errors::try_or_throw;
use crate::io::java_readable;
use crate::io::read_coalesce_config;
use crate::session::session_ref;

/// The aggregates a caller may request. The tags match `dev.vortex.api.FileAggregate.Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AggregateKind {
    /// The number of rows.
    CountStar,
    /// The number of non-null values of a column.
    Count,
    /// The smallest non-null value of a column.
    Min,
    /// The largest non-null value of a column.
    Max,
    /// The wrapping sum of the non-null values of an integer column.
    Sum,
    /// Whether a column is sorted ascending with no nulls, as the footer states. Never computed.
    IsSorted,
}

impl AggregateKind {
    fn from_tag(tag: u8) -> VortexResult<Self> {
        Ok(match tag {
            0 => Self::CountStar,
            1 => Self::Count,
            2 => Self::Min,
            3 => Self::Max,
            4 => Self::Sum,
            5 => Self::IsSorted,
            other => vortex_bail!("unknown aggregate kind tag: {other}"),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::CountStar => "COUNT(*)",
            Self::Count => "COUNT",
            Self::Min => "MIN",
            Self::Max => "MAX",
            Self::Sum => "SUM",
            Self::IsSorted => "IS_SORTED",
        }
    }
}

/// One requested aggregate, with its column resolved against the file schema.
struct AggregateRequest {
    kind: AggregateKind,
    /// The top-level field the aggregate reads. [`AggregateKind::CountStar`] reads none.
    field: Option<usize>,
}

/// What is known about one column. A field is `None` while it is unknown.
#[derive(Default)]
struct ColumnSummary {
    null_count: Option<u64>,
    /// The number of NaN values, known as zero for a non-float column.
    nan_count: Option<u64>,
    /// The smallest value that is neither null nor NaN, or `Some(None)` when there is none.
    min: Option<Option<Scalar>>,
    /// The largest value that is neither null nor NaN, or `Some(None)` when there is none.
    max: Option<Option<Scalar>>,
    /// The sum of the non-null values, before it wraps to 64 bits.
    sum: Option<i128>,
    /// Whether the column is sorted ascending, which the footer only states for null-free columns.
    sorted: Option<bool>,
}

/// Which aggregates a scan must compute for one column.
#[derive(Default, Clone, Copy)]
struct ColumnNeeds {
    extrema: bool,
    sum: bool,
}

/// The dtype of an aggregate's result. Every result is nullable, so an unanswered aggregate can
/// be represented as null.
fn result_dtype(request: &AggregateRequest, fields: &[DType]) -> VortexResult<DType> {
    Ok(match request.kind {
        AggregateKind::CountStar | AggregateKind::Count => {
            DType::Primitive(PType::I64, Nullability::Nullable)
        }
        AggregateKind::Min | AggregateKind::Max => field_dtype(request, fields)?.as_nullable(),
        AggregateKind::IsSorted => {
            field_dtype(request, fields)?;
            DType::Bool(Nullability::Nullable)
        }
        AggregateKind::Sum => {
            let dtype = field_dtype(request, fields)?;
            if !dtype.is_int() {
                vortex_bail!("SUM can only be answered for an integer column, not {dtype}");
            }
            DType::Primitive(PType::I64, Nullability::Nullable)
        }
    })
}

fn field_dtype<'a>(request: &AggregateRequest, fields: &'a [DType]) -> VortexResult<&'a DType> {
    request
        .field
        .map(|field| &fields[field])
        .ok_or_else(|| vortex_err!("{} requires a column", request.kind.name()))
}

/// The answer to `request`, or `None` when `summary` does not determine it.
///
/// The returned scalar has the nullable result dtype of the request.
fn answer(
    request: &AggregateRequest,
    summary: Option<&ColumnSummary>,
    dtype: &DType,
    row_count: u64,
) -> VortexResult<Option<Scalar>> {
    let count = |value: u64| -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            i64::try_from(value)?,
            Nullability::Nullable,
        ))
    };

    if request.kind == AggregateKind::CountStar {
        return count(row_count).map(Some);
    }
    let Some(summary) = summary else {
        return Ok(None);
    };
    if request.kind == AggregateKind::IsSorted {
        return Ok(summary
            .sorted
            .map(|sorted| Scalar::bool(sorted, Nullability::Nullable)));
    }
    let Some(null_count) = summary.null_count else {
        return Ok(None);
    };
    let valid_count = row_count.saturating_sub(null_count);
    if request.kind == AggregateKind::Count {
        return count(valid_count).map(Some);
    }
    if valid_count == 0 {
        return Ok(Some(Scalar::null(dtype.clone())));
    }

    match request.kind {
        AggregateKind::Min | AggregateKind::Max => {
            let Some(nan_count) = summary.nan_count else {
                return Ok(None);
            };
            let is_max = request.kind == AggregateKind::Max;
            // NaN orders above every number, so one NaN is the maximum, and only a column of
            // nothing but NaN has a NaN minimum.
            if (is_max && nan_count > 0) || nan_count == valid_count {
                return Ok(Some(nan_scalar(dtype)?));
            }
            let extremum = if is_max { &summary.max } else { &summary.min };
            extremum
                .as_ref()
                .map(|value| {
                    value
                        .as_ref()
                        .ok_or_else(|| vortex_err!("no extremum for a column holding values"))?
                        .cast(dtype)
                })
                .transpose()
        }
        AggregateKind::Sum => Ok(summary
            .sum
            .map(|total| Scalar::primitive(wrap_to_i64(total), Nullability::Nullable))),
        AggregateKind::CountStar | AggregateKind::Count | AggregateKind::IsSorted => {
            vortex_bail!("{} is answered before the extrema", request.kind.name())
        }
    }
}

/// The exact sum `total` wrapped to 64 bits, as Spark's own `LONG` addition wraps.
#[expect(
    clippy::cast_possible_truncation,
    reason = "wrapping to 64 bits is the intended semantics"
)]
fn wrap_to_i64(total: i128) -> i64 {
    total as i64
}

fn nan_scalar(dtype: &DType) -> VortexResult<Scalar> {
    if !dtype.is_float() {
        vortex_bail!("NaN count is positive for the non-float dtype {dtype}");
    }
    let nan = match dtype.as_ptype() {
        PType::F32 => Scalar::primitive(f32::NAN, Nullability::Nullable),
        PType::F64 => Scalar::primitive(f64::NAN, Nullability::Nullable),
        PType::F16 => Scalar::primitive(f64::NAN, Nullability::Nullable),
        other => vortex_bail!("unexpected float ptype {other}"),
    };
    nan.cast(dtype)
}

/// The exact value the footer states for `stat` of a field.
fn exact_stat(stats: &AggregateStats, stat: Stat) -> Option<Scalar> {
    stats.get(&stat.aggregate_fn()?).as_exact()
}

/// The exact count statistic `stat` of a field, if the footer states one.
fn exact_count(stats: &AggregateStats, stat: Stat) -> Option<u64> {
    exact_stat(stats, stat)?.as_primitive().as_::<u64>()
}

/// The exact value statistic `stat` of a field of `dtype`, if the footer states one.
fn exact_value(stats: &AggregateStats, stat: Stat, dtype: &DType) -> Option<Scalar> {
    let stat_dtype = stat.dtype(dtype)?;
    Scalar::try_new(stat_dtype, exact_stat(stats, stat)?.into_value()).ok()
}

/// What the footer states exactly about one field.
fn footer_summary(stats: &AggregateStats, dtype: &DType) -> ColumnSummary {
    let nan_count = if dtype.is_float() {
        exact_count(stats, Stat::NaNCount)
    } else {
        Some(0)
    };
    // `Stat::Sum` widens every integer to 64 bits, which is the type a wrapping sum needs.
    let sum = dtype
        .is_int()
        .then(|| exact_value(stats, Stat::Sum, dtype))
        .flatten()
        .and_then(|total| integer_value(&total));

    let null_count = exact_count(stats, Stat::NullCount);
    // `IsSorted` admits leading nulls, and NaN has no place in its order, so only a column free of
    // both counts as sorted.
    let sorted = stats
        .get(&IsSorted.bind(IsSortedOptions { strict: false }))
        .as_exact()
        .and_then(|sorted| sorted.as_bool().value())
        .filter(|_| null_count == Some(0) && nan_count == Some(0));

    ColumnSummary {
        sorted,
        null_count,
        nan_count,
        min: exact_value(stats, Stat::Min, dtype).map(Some),
        max: exact_value(stats, Stat::Max, dtype).map(Some),
        sum,
    }
}

/// A non-null signed or unsigned integer scalar, widened to `i128`.
fn integer_value(scalar: &Scalar) -> Option<i128> {
    let primitive = scalar.as_primitive_opt()?;
    match primitive.ptype() {
        PType::U64 => primitive.typed_value::<u64>().map(i128::from),
        _ => primitive.as_::<i64>().map(i128::from),
    }
}

/// Accumulates a [`ColumnSummary`] over the chunks of one column.
struct ColumnAccumulator {
    needs: ColumnNeeds,
    null_count: u64,
    nan_count: u64,
    min: Option<Accumulator<Min>>,
    max: Option<Accumulator<Max>>,
    sum: i128,
}

impl ColumnAccumulator {
    fn try_new(dtype: &DType, needs: ColumnNeeds) -> VortexResult<Self> {
        // The NaN-skipping extrema; NaN is accounted for through the NaN count.
        let options = NumericalAggregateOpts::default();
        let (min, max) = if needs.extrema {
            (
                Some(Accumulator::try_new(Min, options, dtype.clone())?),
                Some(Accumulator::try_new(Max, options, dtype.clone())?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            needs,
            null_count: 0,
            nan_count: 0,
            min,
            max,
            sum: 0,
        })
    }

    fn accumulate(&mut self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let valid_count = chunk.valid_count(ctx)?;
        self.null_count += u64::try_from(chunk.len() - valid_count)?;
        if valid_count == 0 {
            return Ok(());
        }

        if let Some(min) = self.min.as_mut() {
            min.accumulate(chunk, ctx)?;
        }
        if let Some(max) = self.max.as_mut() {
            max.accumulate(chunk, ctx)?;
        }
        if self.needs.extrema && chunk.dtype().is_float() {
            self.nan_count += u64::try_from(nan_count(chunk, ctx)?)?;
        }
        if self.needs.sum {
            self.sum = self.sum.wrapping_add(chunk_sum(chunk, ctx)?);
        }
        Ok(())
    }

    fn finish(mut self) -> VortexResult<ColumnSummary> {
        let extremum = |scalar: Scalar| (!scalar.is_null()).then_some(scalar);
        Ok(ColumnSummary {
            null_count: Some(self.null_count),
            nan_count: Some(self.nan_count),
            min: self
                .min
                .as_mut()
                .map(|min| min.finish())
                .transpose()?
                .map(extremum),
            max: self
                .max
                .as_mut()
                .map(|max| max.finish())
                .transpose()?
                .map(extremum),
            sum: self.needs.sum.then_some(self.sum),
            // Sortedness is the footer's to state; a scan does not prove it.
            sorted: None,
        })
    }
}

/// The sum of the valid values of an integer chunk.
fn chunk_sum(chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<i128> {
    // The kernel answers from cached statistics and encodings, such as a constant or a
    // frame-of-reference array, without decoding. It reports an overflow of 64 bits as null.
    if let Some(total) = integer_value(&sum(chunk, ctx)?) {
        return Ok(total);
    }

    let primitive = chunk.clone().execute::<PrimitiveArray>(ctx)?;
    let mask = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    match_each_integer_ptype!(primitive.ptype(), |T| {
        Ok(primitive
            .as_slice::<T>()
            .iter()
            .enumerate()
            .filter(|(index, _)| mask.value(*index))
            .fold(0i128, |total, (_, value)| total + i128::from(*value)))
    })
}

/// Scans the fields in `needs` and summarizes each one.
fn compute_summaries(
    file: &VortexFile,
    needs: &HashMap<usize, ColumnNeeds>,
    session: &VortexSession,
) -> VortexResult<HashMap<usize, ColumnSummary>> {
    let file_fields = file
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("aggregates require a struct file dtype"))?;

    let mut indices: Vec<usize> = needs.keys().copied().collect();
    indices.sort_unstable();
    let names: Vec<FieldName> = indices
        .iter()
        .map(|&index| file_fields.field_name(index).cloned())
        .collect::<Option<_>>()
        .ok_or_else(|| vortex_err!("aggregate field out of range"))?;

    let mut accumulators = indices
        .iter()
        .map(|index| {
            let dtype = file_fields
                .field_by_index(*index)
                .ok_or_else(|| vortex_err!("aggregate field {index} out of range"))?;
            ColumnAccumulator::try_new(&dtype, needs[index])
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let projection = select(FieldNames::from(names), root()).bind(file.dtype())?;
    let mut stream = Box::pin(
        file.scan()?
            .with_projection(projection)
            .into_array_stream()?,
    );
    let mut ctx = session.create_execution_ctx();
    RUNTIME.block_on(async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?.execute::<StructArray>(&mut ctx)?;
            for (position, accumulator) in accumulators.iter_mut().enumerate() {
                accumulator.accumulate(chunk.unmasked_field(position), &mut ctx)?;
            }
        }
        VortexResult::Ok(())
    })?;

    indices
        .into_iter()
        .zip(accumulators)
        .map(|(index, accumulator)| Ok((index, accumulator.finish()?)))
        .collect()
}

/// Answers `requests` over `file`. A `None` answer was neither stated by the footer nor computed.
fn aggregate_file(
    file: &VortexFile,
    requests: &[AggregateRequest],
    compute_missing: bool,
    session: &VortexSession,
) -> VortexResult<(Vec<DType>, Vec<Option<Scalar>>)> {
    let file_fields = file
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("aggregates require a struct file dtype"))?;
    let fields: Vec<DType> = file_fields.fields().collect();
    let row_count = file.row_count();
    let dtypes = requests
        .iter()
        .map(|request| result_dtype(request, &fields))
        .collect::<VortexResult<Vec<_>>>()?;

    let mut summaries: HashMap<usize, ColumnSummary> = HashMap::default();
    if let Some(stats) = file.file_stats() {
        for field in requests.iter().filter_map(|request| request.field) {
            let path = FieldPath::from_name(file_fields.names()[field].clone());
            if let Some((aggregates, _)) = stats.get_by_path(&path) {
                summaries
                    .entry(field)
                    .or_insert_with(|| footer_summary(aggregates, &fields[field]));
            }
        }
    }

    let mut answers = requests
        .iter()
        .zip(&dtypes)
        .map(|(request, dtype)| {
            let summary = request.field.and_then(|field| summaries.get(&field));
            answer(request, summary, dtype, row_count)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    if !compute_missing || answers.iter().all(Option::is_some) {
        return Ok((dtypes, answers));
    }

    let mut needs: HashMap<usize, ColumnNeeds> = HashMap::default();
    for (request, known) in requests.iter().zip(&answers) {
        let (Some(field), None) = (request.field, known) else {
            continue;
        };
        let entry = needs.entry(field).or_default();
        match request.kind {
            AggregateKind::Min | AggregateKind::Max => entry.extrema = true,
            AggregateKind::Sum => entry.sum = true,
            AggregateKind::Count | AggregateKind::CountStar | AggregateKind::IsSorted => {}
        }
    }

    let computed = compute_summaries(file, &needs, session)?;
    for ((request, dtype), slot) in requests.iter().zip(&dtypes).zip(answers.iter_mut()) {
        if slot.is_none() {
            let summary = request.field.and_then(|field| computed.get(&field));
            *slot = answer(request, summary, dtype, row_count)?;
        }
    }
    Ok((dtypes, answers))
}

/// A one-row struct array holding `answers`, with null in place of an unanswered aggregate.
fn answers_array(dtypes: &[DType], answers: Vec<Option<Scalar>>) -> VortexResult<ArrayRef> {
    let names: Vec<String> = (0..answers.len())
        .map(|index| format!("agg_{index}"))
        .collect();
    let columns = answers
        .into_iter()
        .zip(dtypes)
        .map(|(answer, dtype)| {
            ConstantArray::new(answer.unwrap_or_else(|| Scalar::null(dtype.clone())), 1)
                .into_array()
        })
        .collect::<Vec<_>>();
    Ok(StructArray::try_new(
        FieldNames::from(names.iter().map(String::as_str).collect::<Vec<_>>()),
        columns,
        1,
        Validity::NonNullable,
    )?
    .into_array())
}

/// Compute ungrouped aggregates over one Vortex file read through a `dev.vortex.io.NativeReadable`.
///
/// `kinds` holds one `AggregateKind` tag per aggregate, and `columns` the top-level column each one
/// reads, null for `COUNT(*)`. With `compute_missing`, an aggregate the footer does not state
/// exactly is computed by scanning its column. Without it, the aggregate is left unanswered.
///
/// The answers are exported as a one-row struct, one field per aggregate, into the Arrow C Data
/// Interface structs at `array_addr` and `schema_addr`. `answered[i]` is set to 1 when aggregate
/// `i` was answered and 0 when its field holds a placeholder null.
///
/// `name` keys the file in the session footer cache, so a scan of the same readable shares the
/// footer read.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFiles_aggregateFromReadable(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    readable: JObject,
    name: JString,
    length: jlong,
    kinds: JByteArray,
    columns: JObjectArray,
    compute_missing: jboolean,
    answered: JLongArray,
    array_addr: jlong,
    schema_addr: jlong,
) {
    try_or_throw(&mut env, |env| {
        if array_addr == 0 || schema_addr == 0 {
            throw_runtime!("null arrow export address");
        }
        if readable.is_null() {
            throw_runtime!("null readable");
        }
        let session = unsafe { session_ref(session_ptr) };
        let name: String = name.try_to_string(env)?;
        let length =
            u64::try_from(length).map_err(|_| vortex_err!("negative readable length: {length}"))?;

        let kinds = env.convert_byte_array(&kinds)?;
        let count = columns.len(env)?;
        if kinds.len() != count || answered.len(env)? != count {
            throw_runtime!("aggregate kinds, columns, and answers must have equal length");
        }
        let mut column_names = Vec::with_capacity(count);
        for index in 0..count {
            let column = columns.get_element(env, index)?;
            column_names.push(if column.is_null() {
                None
            } else {
                Some(env.cast_local::<JString>(column)?.try_to_string(env)?)
            });
        }

        let vm = env.get_java_vm()?;
        let coalesce_config = read_coalesce_config(env, &readable)?;
        let readable = Arc::new(env.new_global_ref(&readable)?);
        let source = java_readable(vm, readable, length, coalesce_config, session.handle());
        // Normalized as the scan path keys Java readables, so both share one footer read.
        let cache_key = name.trim_start_matches('/');
        let file = RUNTIME.block_on(open_cached(
            session,
            Some(cache_key),
            source,
            Some(length),
            &|options| options,
        ))?;

        let file_fields = file
            .dtype()
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("aggregates require a struct file dtype"))?;
        let requests = kinds
            .iter()
            .zip(column_names)
            .map(|(tag, column)| {
                let kind = AggregateKind::from_tag(*tag)?;
                let field = match (kind, column) {
                    (AggregateKind::CountStar, _) => None,
                    (_, None) => vortex_bail!("{} requires a column", kind.name()),
                    (_, Some(column)) => Some(
                        file_fields
                            .find(&column)
                            .ok_or_else(|| vortex_err!("{name} has no column {column}"))?,
                    ),
                };
                Ok(AggregateRequest { kind, field })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (dtypes, answers) = aggregate_file(&file, &requests, compute_missing, session)?;
        let flags: Vec<jlong> = answers
            .iter()
            .map(|answer| jlong::from(answer.is_some()))
            .collect();
        let array = answers_array(&dtypes, answers)?;

        let schema = session.arrow().to_arrow_schema(array.dtype())?;
        let target = Field::new_struct("", schema.fields().clone(), false);
        let mut ctx = session.create_execution_ctx();
        let arrow = array.execute_arrow(Some(&target), &mut ctx)?;
        let data = arrow.to_data();
        let ffi_array = FFI_ArrowArray::new(&data);
        let ffi_schema = FFI_ArrowSchema::try_from(&target)?;

        answered.set_region(env, 0, &flags)?;
        unsafe {
            ptr::write(array_addr as *mut FFI_ArrowArray, ffi_array);
            ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::scalar::Scalar;

    use super::AggregateKind;
    use super::AggregateRequest;
    use super::ColumnSummary;
    use super::answer;

    fn f64_dtype() -> DType {
        DType::Primitive(PType::F64, Nullability::Nullable)
    }

    fn summary(nan_count: u64) -> ColumnSummary {
        ColumnSummary {
            null_count: Some(1),
            nan_count: Some(nan_count),
            min: Some(Some(Scalar::primitive(-1.5f64, Nullability::NonNullable))),
            max: Some(Some(Scalar::primitive(2.5f64, Nullability::NonNullable))),
            sum: None,
            sorted: None,
        }
    }

    fn request(kind: AggregateKind) -> AggregateRequest {
        AggregateRequest {
            kind,
            field: Some(0),
        }
    }

    #[rstest]
    #[case::no_nan_min(AggregateKind::Min, 0, Some(-1.5))]
    #[case::no_nan_max(AggregateKind::Max, 0, Some(2.5))]
    #[case::some_nan_min(AggregateKind::Min, 2, Some(-1.5))]
    #[case::some_nan_max(AggregateKind::Max, 2, None)]
    #[case::all_nan_min(AggregateKind::Min, 9, None)]
    fn float_extrema_order_nan_last(
        #[case] kind: AggregateKind,
        #[case] nan_count: u64,
        #[case] expected: Option<f64>,
    ) -> VortexResult<()> {
        let result = answer(&request(kind), Some(&summary(nan_count)), &f64_dtype(), 10)?
            .expect("summary determines the answer");
        let value = result
            .as_primitive()
            .typed_value::<f64>()
            .expect("non-null");
        match expected {
            Some(expected) => assert_eq!(value, expected),
            None => assert!(value.is_nan()),
        }
        Ok(())
    }

    #[test]
    fn all_null_column_answers_null_extrema_and_zero_count() -> VortexResult<()> {
        let summary = ColumnSummary {
            null_count: Some(4),
            ..ColumnSummary::default()
        };
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);

        let min = answer(&request(AggregateKind::Min), Some(&summary), &dtype, 4)?;
        assert!(min.expect("answered").is_null());

        let count = answer(&request(AggregateKind::Count), Some(&summary), &dtype, 4)?;
        assert_eq!(
            count.expect("answered").as_primitive().typed_value::<i64>(),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn unknown_statistics_leave_the_aggregate_unanswered() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let unknown = ColumnSummary::default();
        assert!(answer(&request(AggregateKind::Max), Some(&unknown), &dtype, 4)?.is_none());
        assert!(answer(&request(AggregateKind::Count), None, &dtype, 4)?.is_none());
        Ok(())
    }

    #[test]
    fn integer_sum_wraps_to_64_bits() -> VortexResult<()> {
        let summary = ColumnSummary {
            null_count: Some(0),
            nan_count: Some(0),
            sum: Some(i128::from(i64::MAX) + 2),
            ..ColumnSummary::default()
        };
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let sum = answer(&request(AggregateKind::Sum), Some(&summary), &dtype, 2)?;
        assert_eq!(
            sum.expect("answered").as_primitive().typed_value::<i64>(),
            Some(i64::MIN + 1)
        );
        Ok(())
    }
}
