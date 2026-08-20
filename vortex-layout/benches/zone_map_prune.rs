// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for [`ZoneMap::prune`].
//!
//! Each case pre-falsifies its predicate so the timed region covers exactly what `prune` does:
//! lowering the stats placeholders against the zone map, then evaluating the result per zone.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpressionRef;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::lit;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::array_session().with::<LayoutSession>());

/// Zone counts to sweep. The small case exposes per-call fixed cost, the large case exposes
/// per-zone evaluation cost.
const ZONE_COUNTS: &[usize] = &[16, 1024, 8192];

const ZONE_LEN: u64 = 8192;

/// Deterministic pseudo-random values, so both branches benchmark identical data.
fn pseudo_random(len: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed | 1;
    (0..len).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    })
}

fn i32_stats(num_zones: usize) -> (Vec<i32>, Vec<i32>) {
    let mins: Vec<i32> = pseudo_random(num_zones, 0x5eed)
        .map(|v| (v % 10_000) as i32)
        .collect();
    let maxs = mins.iter().map(|min| min + 100).collect();
    (mins, maxs)
}

struct ZoneCounts {
    null: Vec<u64>,
    nan: Option<Vec<u64>>,
    has_min_max: Vec<bool>,
}

fn zone_counts(column_dtype: &DType, num_zones: usize) -> ZoneCounts {
    let row_counts = (0..num_zones)
        .map(|zone| {
            if zone + 1 == num_zones {
                ZONE_LEN / 2
            } else {
                ZONE_LEN
            }
        })
        .collect::<Vec<_>>();
    let null = pseudo_random(num_zones, 0xc0ffee)
        .zip(&row_counts)
        .map(|(value, row_count)| value % (row_count + 1))
        .collect::<Vec<_>>();
    let nan = column_dtype.is_float().then(|| {
        pseudo_random(num_zones, 0xfeed)
            .zip(row_counts.iter().zip(&null))
            .map(|(value, (row_count, null_count))| value % (row_count - null_count + 1))
            .collect::<Vec<_>>()
    });
    let has_min_max = row_counts
        .iter()
        .enumerate()
        .map(|(zone, row_count)| {
            null[zone] + nan.as_ref().map_or(0, |counts| counts[zone]) < *row_count
        })
        .collect();

    ZoneCounts {
        null,
        nan,
        has_min_max,
    }
}

/// The aggregates the zoned writer stores by default for a numeric column. `nan_count` only has a
/// state dtype for floats, so it is omitted for integers.
fn min_max_fields(
    column_dtype: &DType,
    num_zones: usize,
    has_min_max: &[bool],
) -> Vec<(String, ArrayRef)> {
    let (mins, maxs) = i32_stats(num_zones);
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let min_validity = Validity::from_iter(has_min_max.iter().copied());
    let max_validity = Validity::from_iter(has_min_max.iter().copied());

    let (min_array, max_array) = if column_dtype.is_float() {
        (
            PrimitiveArray::new(
                mins.iter().map(|v| f64::from(*v)).collect::<Buffer<f64>>(),
                min_validity,
            )
            .into_array(),
            PrimitiveArray::new(
                maxs.iter().map(|v| f64::from(*v)).collect::<Buffer<f64>>(),
                max_validity,
            )
            .into_array(),
        )
    } else {
        (
            PrimitiveArray::new(mins.iter().copied().collect::<Buffer<i32>>(), min_validity)
                .into_array(),
            PrimitiveArray::new(maxs.iter().copied().collect::<Buffer<i32>>(), max_validity)
                .into_array(),
        )
    };

    vec![(max.to_string(), max_array), (min.to_string(), min_array)]
}

fn count_fields(counts: &ZoneCounts) -> Vec<(String, ArrayRef)> {
    let mut fields = Vec::new();
    if let Some(nan) = &counts.nan {
        fields.push((
            NanCount.bind(EmptyOptions).to_string(),
            PrimitiveArray::new(
                nan.iter().copied().collect::<Buffer<_>>(),
                Validity::AllValid,
            )
            .into_array(),
        ));
    }
    fields.push((
        NullCount.bind(EmptyOptions).to_string(),
        PrimitiveArray::new(
            counts.null.iter().copied().collect::<Buffer<_>>(),
            Validity::AllValid,
        )
        .into_array(),
    ));
    fields
}

/// Key identifying a cached zone map: column dtype, number of zones, and whether min/max are
/// omitted.
type ZoneMapKey = (DType, usize, bool);

/// Divan calls a benchmark function once per sample, so zone maps are cached to keep construction
/// out of both the reported times and the profile.
static ZONE_MAPS: LazyLock<Mutex<HashMap<ZoneMapKey, ZoneMap>>> = LazyLock::new(Mutex::default);

fn zone_map(column_dtype: DType, num_zones: usize, counts_only: bool) -> ZoneMap {
    ZONE_MAPS
        .lock()
        .entry((column_dtype.clone(), num_zones, counts_only))
        .or_insert_with(|| {
            let counts = zone_counts(&column_dtype, num_zones);
            let mut fields = if counts_only {
                Vec::new()
            } else {
                min_max_fields(&column_dtype, num_zones, &counts.has_min_max)
            };
            fields.extend(count_fields(&counts));
            build(column_dtype.clone(), fields, num_zones)
        })
        .clone()
}

/// A zone map carrying every aggregate the zoned writer stores by default.
fn numeric_zone_map(column_dtype: DType, num_zones: usize) -> ZoneMap {
    zone_map(column_dtype, num_zones, false)
}

/// A zone map carrying only the count aggregates, so min/max proofs find no stat to bind.
fn counts_only_zone_map(column_dtype: DType, num_zones: usize) -> ZoneMap {
    zone_map(column_dtype, num_zones, true)
}

fn build(column_dtype: DType, fields: Vec<(String, ArrayRef)>, num_zones: usize) -> ZoneMap {
    let aggregate_fns: Arc<[AggregateFnRef]> = [
        Max.bind(NumericalAggregateOpts::skip_nans()),
        Min.bind(NumericalAggregateOpts::skip_nans()),
        NanCount.bind(EmptyOptions),
        NullCount.bind(EmptyOptions),
    ]
    .into_iter()
    .filter(|aggregate_fn| {
        fields
            .iter()
            .any(|(name, _)| name == &aggregate_fn.to_string())
    })
    .collect();

    let stats = StructArray::from_fields(
        &fields
            .iter()
            .map(|(name, array)| (name.as_str(), array.clone()))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    // A trailing short zone, which is the common shape and forces the run-end row-count array.
    let row_count = ZONE_LEN * (num_zones as u64 - 1) + ZONE_LEN / 2;
    ZoneMap::try_new(column_dtype, stats, aggregate_fns, ZONE_LEN, row_count).unwrap()
}

fn i32_dtype() -> DType {
    DType::Primitive(PType::I32, Nullability::Nullable)
}

fn f64_dtype() -> DType {
    DType::Primitive(PType::F64, Nullability::Nullable)
}

fn falsify(expr: Expression, column_dtype: &DType) -> BoundExpressionRef {
    expr.bind(column_dtype)
        .unwrap()
        .falsify(&SESSION)
        .unwrap()
        .unwrap()
}

fn run(bencher: Bencher, zone_map: ZoneMap, predicate: BoundExpressionRef) {
    bencher.bench(|| {
        divan::black_box(
            zone_map
                .prune(divan::black_box(&predicate), &SESSION)
                .unwrap(),
        )
    });
}

/// Integer range predicate: binds to `max` only, no row count, no NaN guard.
#[divan::bench(args = ZONE_COUNTS)]
fn int_gt(bencher: Bencher, num_zones: usize) {
    static PREDICATE: LazyLock<BoundExpressionRef> =
        LazyLock::new(|| falsify(gt(root(), lit(5_000i32)), &i32_dtype()));
    run(
        bencher,
        numeric_zone_map(i32_dtype(), num_zones),
        PREDICATE.clone(),
    );
}

/// Float range predicate: the NaN-guarded and unguarded rules both fire, and on a zone map that
/// stores `nan_count` they lower to the same expression.
#[divan::bench(args = ZONE_COUNTS)]
fn float_gt(bencher: Bencher, num_zones: usize) {
    static PREDICATE: LazyLock<BoundExpressionRef> =
        LazyLock::new(|| falsify(gt(root(), lit(5_000f64)), &f64_dtype()));
    run(
        bencher,
        numeric_zone_map(f64_dtype(), num_zones),
        PREDICATE.clone(),
    );
}

/// Null predicate: lowers to `null_count == row_count`, exercising the row-count path.
#[divan::bench(args = ZONE_COUNTS)]
fn is_not_null_pred(bencher: Bencher, num_zones: usize) {
    static PREDICATE: LazyLock<BoundExpressionRef> =
        LazyLock::new(|| falsify(is_not_null(root()), &i32_dtype()));
    run(
        bencher,
        numeric_zone_map(i32_dtype(), num_zones),
        PREDICATE.clone(),
    );
}

/// A 16-term `OR` chain, which is where lowering cost grows relative to evaluation cost.
#[divan::bench(args = ZONE_COUNTS)]
fn or_chain(bencher: Bencher, num_zones: usize) {
    static PREDICATE: LazyLock<BoundExpressionRef> = LazyLock::new(|| {
        let expr = (0..16i32)
            .map(|i| eq(root(), lit(i * 500)))
            .reduce(or)
            .unwrap();
        falsify(expr, &i32_dtype())
    });
    run(
        bencher,
        numeric_zone_map(i32_dtype(), num_zones),
        PREDICATE.clone(),
    );
}

/// The zone map lacks min/max, so every proof binds to a null literal and the lowered predicate is
/// constant.
#[divan::bench(args = ZONE_COUNTS)]
fn missing_stats(bencher: Bencher, num_zones: usize) {
    static PREDICATE: LazyLock<BoundExpressionRef> =
        LazyLock::new(|| falsify(gt(root(), lit(5_000i32)), &i32_dtype()));
    run(
        bencher,
        counts_only_zone_map(i32_dtype(), num_zones),
        PREDICATE.clone(),
    );
}
