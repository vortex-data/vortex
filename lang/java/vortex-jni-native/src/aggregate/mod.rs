// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-level aggregates exposed through the Java API.

use std::collections::BTreeMap;
use std::ptr;

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
use vortex::aggregate_fn::AccumulatorRef;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::count::Count;
use vortex::aggregate_fn::fns::is_sorted::IsSorted;
use vortex::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::aggregate_fn::fns::sum::Sum;
use vortex::aggregate_fn::fns::sum_v2::SumV2;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt as _;
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
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::Entry;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::RUNTIME;
use crate::errors::try_or_throw;
use crate::file::open_readable_file;
use crate::session::session_ref;
use crate::statistics::checked_jlong;

enum FileAggregate {
    RowCount,
    Column(FieldName, AggregateFnRef),
}

impl FileAggregate {
    // Tags are defined in dev.vortex.api.FileAggregate.Kind.
    fn from_tag(tag: u8, column: Option<FieldName>) -> VortexResult<Self> {
        let function = match tag {
            0 if column.is_none() => return Ok(Self::RowCount),
            0 => vortex_bail!("COUNT(*) takes no column"),
            1 => Count.bind(NumericalAggregateOpts::include_nans()),
            2 => Min.bind(NumericalAggregateOpts::default()),
            3 => Max.bind(NumericalAggregateOpts::default()),
            4 => SumV2.bind(NumericalAggregateOpts::default()),
            5 => IsSorted.bind(IsSortedOptions { strict: false }),
            _ => vortex_bail!("unknown aggregate tag {tag}"),
        };
        let column = column.ok_or_else(|| vortex_err!("{function} requires a column"))?;
        Ok(Self::Column(column, function))
    }
}

fn footer_answer(
    function: &AggregateFnRef,
    dtype: &DType,
    stats: Option<&AggregateStats>,
    rows: u64,
) -> VortexResult<Option<Scalar>> {
    // AggregateStats::get omits nulls. A recorded exact null is an answered Java result.
    if let Some(value) = stats.and_then(|stats| {
        stats
            .iter()
            .find(|stat| stat.aggregate_fn() == function)
            .and_then(|stat| stat.value().clone().as_exact())
    }) {
        return Ok(Some(value));
    }
    if function.is::<IsSorted>() {
        return Ok(None);
    }

    // Default footers record NullCount. It can answer Count and prove that the aggregate
    // received no valid inputs, including in legacy files without aggregate partials.
    let nulls = if dtype.is_nullable() {
        stats
            .and_then(|stats| stats.get(&NullCount.bind(EmptyOptions)).as_exact())
            .and_then(|s| s.as_primitive().typed_value::<u64>())
    } else {
        Some(0)
    };
    if rows == 0 || nulls == Some(rows) {
        return Ok(Some(function.accumulator(dtype)?.final_scalar()?));
    }
    // Older writers could record Sum, whose empty-input value is zero. Its non-null result
    // agrees with SumV2 once the footer proves at least one valid integer contributed.
    if function.is::<SumV2>() && nulls.is_some_and(|n| n < rows) {
        return Ok(stats.and_then(|stats| {
            stats
                .get(&Sum.bind(NumericalAggregateOpts::default()))
                .as_exact()
        }));
    }
    if function.is::<Count>()
        && let Some(nulls) = nulls
    {
        let count = rows
            .checked_sub(nulls)
            .ok_or_else(|| vortex_err!("null count {nulls} exceeds row count {rows}"))?;
        return Ok(Some(Scalar::from(count)));
    }
    Ok(None)
}

/// Resolve a Java result from native aggregates, with NaNs ordered above numbers.
fn column_answer(
    function: &AggregateFnRef,
    dtype: &DType,
    rows: u64,
    mut get: impl FnMut(&AggregateFnRef) -> VortexResult<Option<Scalar>>,
) -> VortexResult<Option<Scalar>> {
    if function.is::<IsSorted>() {
        let sorted = get(function)?;
        // Spark can consume this ordering only when its null and NaN placement also agrees.
        let mut zero = |count| -> VortexResult<bool> {
            Ok(get(&count)?.as_ref().map(u64::try_from).transpose()? == Some(0))
        };
        if sorted.is_none()
            || (dtype.is_nullable() && !zero(NullCount.bind(EmptyOptions))?)
            || (dtype.is_float() && !zero(NanCount.bind(EmptyOptions))?)
        {
            return Ok(None);
        }
        return Ok(sorted);
    }
    if !dtype.is_float() || !(function.is::<Min>() || function.is::<Max>()) {
        return get(function);
    }
    let minimum = function.is::<Min>();
    let extremum = get(function)?;
    if minimum && extremum.as_ref().is_some_and(|value| !value.is_null()) {
        return Ok(extremum);
    }
    let Some(nans) = get(&NanCount.bind(EmptyOptions))? else {
        return Ok(None);
    };
    let nans = u64::try_from(&nans)?;
    if nans == 0 {
        return Ok(extremum);
    }
    if !minimum || extremum.as_ref().is_some_and(Scalar::is_null) || nans == rows {
        return Scalar::from(f64::NAN).cast(&dtype.as_nullable()).map(Some);
    }
    // Legacy footers may omit a null minimum. If every non-null value is NaN, NaN is the minimum.
    if let Some(count) = get(&Count.bind(NumericalAggregateOpts::include_nans()))?
        && u64::try_from(&count)? == nans
    {
        return Scalar::from(f64::NAN).cast(&dtype.as_nullable()).map(Some);
    }
    Ok(None)
}

type PendingColumns = BTreeMap<FieldName, HashMap<AggregateFnRef, AccumulatorRef>>;

fn scan_missing(
    file: &VortexFile,
    mut pending: PendingColumns,
    session: &VortexSession,
) -> VortexResult<HashMap<(FieldName, AggregateFnRef), Scalar>> {
    // get_item propagates the validity of a nullable root into each projected column.
    let projection = pack(
        pending
            .keys()
            .map(|name| (name.clone(), get_item(name.clone(), root()))),
        Nullability::NonNullable,
    )
    .bind(file.dtype())?;
    let mut stream = Box::pin(
        file.scan()?
            .with_projection(projection)
            .into_array_stream()?,
    );
    let mut ctx = session.create_execution_ctx();
    RUNTIME.block_on(async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?.execute::<StructArray>(&mut ctx)?;
            for (field, aggregates) in chunk.iter_unmasked_fields().zip(pending.values_mut()) {
                for accumulator in aggregates.values_mut() {
                    accumulator.accumulate(field, &mut ctx)?;
                }
            }
        }
        VortexResult::Ok(())
    })?;
    let mut values = HashMap::default();
    for (name, aggregates) in pending {
        for (function, accumulator) in aggregates {
            values.insert((name.clone(), function), accumulator.final_scalar()?);
        }
    }
    Ok(values)
}

fn aggregate_file(
    file: &VortexFile,
    requests: &[FileAggregate],
    compute_missing: bool,
    session: &VortexSession,
) -> VortexResult<(Vec<DType>, Vec<Option<Scalar>>)> {
    let fields = file
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("aggregates require a struct file dtype"))?;
    let rows = file.row_count();
    let mut dtypes = Vec::with_capacity(requests.len());
    let mut answers = Vec::with_capacity(requests.len());
    let mut pending = PendingColumns::new();
    let mut unresolved = Vec::new();
    for (slot, request) in requests.iter().enumerate() {
        let FileAggregate::Column(name, function) = request else {
            let count = Scalar::from(checked_jlong(rows, "row count")?).into_nullable();
            dtypes.push(count.dtype().clone());
            answers.push(Some(count));
            continue;
        };
        let dtype = fields
            .field(name)
            .ok_or_else(|| vortex_err!("file has no column {name}"))?
            .union_nullability(file.dtype().nullability());
        if function.is::<SumV2>() && !dtype.is_int() {
            vortex_bail!("SUM requires an integer column, got {dtype}");
        }
        let result_dtype = if function.is::<Count>() {
            DType::Primitive(PType::I64, Nullability::Nullable)
        } else {
            function
                .return_dtype(&dtype)
                .ok_or_else(|| vortex_err!("{function} does not support {dtype}"))?
                .as_nullable()
        };
        let stats = file
            .file_stats()
            .and_then(|stats| stats.get_by_path(&FieldPath::from_name(name.clone())))
            .map(|(stats, _)| stats);
        let mut missing = Vec::new();
        let answer = column_answer(function, &dtype, rows, |f| {
            let value = footer_answer(f, &dtype, stats, rows)?;
            if value.is_none() {
                missing.push(f.clone());
            }
            Ok(value)
        })?;
        if answer.is_none() && compute_missing && !function.is::<IsSorted>() {
            let column = pending.entry(name.clone()).or_default();
            for f in missing {
                if let Entry::Vacant(entry) = column.entry(f) {
                    let accumulator = entry.key().accumulator(&dtype)?;
                    entry.insert(accumulator);
                }
            }
            unresolved.push((slot, name, function, dtype, stats));
        }
        dtypes.push(result_dtype);
        answers.push(answer);
    }
    if !pending.is_empty() {
        let scanned = scan_missing(file, pending, session)?;
        for (slot, name, function, dtype, stats) in unresolved {
            answers[slot] = column_answer(function, &dtype, rows, |f| {
                if let Some(value) = scanned.get(&(name.clone(), f.clone())) {
                    return Ok(Some(value.clone()));
                }
                footer_answer(f, &dtype, stats, rows)
            })?;
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
            let value = match answer {
                Some(scalar) => scalar.cast(dtype)?,
                None => Scalar::null(dtype.clone()),
            };
            Ok(ConstantArray::new(value, 1).into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;
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
/// `kinds` holds one Java aggregate tag per aggregate, and `columns` the top-level column each one
/// reads, null for `COUNT(*)`. With `compute_missing`, an aggregate the footer does not state
/// exactly can be computed by scanning its column. Sortedness always requires a footer statistic.
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
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        // SAFETY: Java keeps the Session that owns this pointer reachable through the call.
        let session = unsafe { session_ref(session_ptr) };
        let file = open_readable_file(env, session, &readable, &name, length, false)?;
        let kinds = env.convert_byte_array(&kinds)?;
        let count = columns.len(env)?;
        if kinds.len() != count || answered.len(env)? != count {
            throw_runtime!("aggregate kinds, columns, and answers must have equal length");
        }
        let mut requests = Vec::with_capacity(count);
        for (index, tag) in kinds.into_iter().enumerate() {
            let column = columns.get_element(env, index)?;
            let name = if column.is_null() {
                None
            } else {
                Some(FieldName::from(
                    env.cast_local::<JString>(column)?.try_to_string(env)?,
                ))
            };
            requests.push(FileAggregate::from_tag(tag, name)?);
        }

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
        // SAFETY: Java supplied writable Arrow C Data structures and takes ownership on success.
        unsafe {
            ptr::write(array_addr as *mut FFI_ArrowArray, ffi_array);
            ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests;
