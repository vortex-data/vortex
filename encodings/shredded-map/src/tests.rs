// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property tests comparing shredded and canonical maps against a plain Rust model.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::collections::BTreeSet;

use proptest::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ShredOptions;
use crate::ShreddedMapArray;
use crate::array::ShreddedMapArraySlotsExt;
use crate::labels::LabelMapBuilder;
use crate::labels::LabelValue;
use crate::labels::Utf8MapBuilder;
use crate::ops;
use crate::shred;

type Row = Option<Vec<(String, Option<LabelValue>)>>;

/// Keys with decreasing frequency, including empty, long (non-inlined) and non-ASCII keys.
const VOCAB: &[&str] = &[
    "__name__",
    "job",
    "instance",
    "container_label_com_docker_compose_service",
    "cpu",
    "",
    "device",
    "le",
    "zone_\u{e9}t\u{e9}",
    "a_rather_long_label_name_that_is_not_inlined",
    "z",
];

fn value_strategy(key: usize, nullable: bool) -> BoxedStrategy<Option<LabelValue>> {
    let any_value = prop_oneof![
        "[a-z0-9/._-]{0,20}".prop_map(LabelValue::Str),
        any::<i64>().prop_map(LabelValue::Int),
        (-1e6f64..1e6).prop_map(LabelValue::Float),
        any::<bool>().prop_map(LabelValue::Bool),
    ];
    // Most keys mostly hold one variant so typed shredding kicks in.
    let preferred = match key % 4 {
        0 => "[a-z]{0,16}".prop_map(LabelValue::Str).boxed(),
        1 => (0i64..64).prop_map(LabelValue::Int).boxed(),
        2 => (0.0f64..1.0).prop_map(LabelValue::Float).boxed(),
        _ => any::<bool>().prop_map(LabelValue::Bool).boxed(),
    };
    let value = prop_oneof![9 => preferred, 1 => any_value];
    if nullable {
        prop_oneof![9 => value.prop_map(Some), 1 => Just(None)].boxed()
    } else {
        value.prop_map(Some).boxed()
    }
}

fn row_strategy(nullable_values: bool, nullable_rows: bool) -> BoxedStrategy<Row> {
    let entries = VOCAB
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let present = 1.0 - (i as f64 + 0.5) / VOCAB.len() as f64;
            (
                proptest::bool::weighted(present),
                proptest::bool::weighted(0.05),
                value_strategy(i, nullable_values),
                value_strategy(i, nullable_values),
            )
        })
        .collect::<Vec<_>>();
    let row = entries.prop_map(|choices| {
        let mut row: Vec<(String, Option<LabelValue>)> = Vec::new();
        for (i, (present, duplicate, value, dup_value)) in choices.into_iter().enumerate() {
            if present {
                row.push((VOCAB[i].to_string(), value));
                if duplicate {
                    row.push((VOCAB[i].to_string(), dup_value));
                }
            }
        }
        row.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        row
    });
    if nullable_rows {
        prop_oneof![9 => row.prop_map(Some), 1 => Just(None)].boxed()
    } else {
        row.prop_map(Some).boxed()
    }
}

#[derive(Debug, Clone)]
struct Case {
    rows: Vec<Row>,
    /// Rows built by sharing the previous row's entries.
    repeats: Vec<bool>,
    /// Build a `Map<Utf8, Utf8>` instead of a label union map.
    utf8: bool,
    nullable_values: bool,
    nullable_rows: bool,
    options: ShredOptions,
}

fn case_strategy() -> impl Strategy<Value = Case> {
    (any::<bool>(), any::<bool>(), any::<bool>()).prop_flat_map(
        |(nullable_values, nullable_rows, utf8)| {
            (
                prop::collection::vec(
                    (
                        row_strategy(nullable_values, nullable_rows),
                        proptest::bool::weighted(0.5),
                        any::<bool>(),
                    ),
                    0..48,
                ),
                0.0f64..1.0,
                0usize..8,
                any::<bool>(),
                any::<bool>(),
                0.0f64..=1.0,
                0.0f64..=1.0,
            )
                .prop_map(move |(rows, min_frequency, max_sparse_columns, typed, dictionary, max_distinct_rows, sparse_below)| {
                    let mut out: Vec<Row> = Vec::with_capacity(rows.len());
                    let mut repeats = Vec::with_capacity(rows.len());
                    for (i, (row, repeat, share)) in rows.into_iter().enumerate() {
                        let repeat = repeat && i > 0;
                        let row = if repeat { out[i - 1].clone() } else { row };
                        // A repeated row is either shared with the previous row or copied.
                        let repeat = repeat && share;
                        // A string map stores every value as its label string.
                        let row = if utf8 && !(i > 0 && repeat) {
                            row.map(|entries| {
                                entries
                                    .into_iter()
                                    .map(|(k, v)| {
                                        (k, v.map(|v| LabelValue::Str(v.to_label_string())))
                                    })
                                    .collect()
                            })
                        } else {
                            row
                        };
                        out.push(row);
                        repeats.push(repeat);
                    }
                    Case {
                        rows: out,
                        repeats,
                        utf8,
                        nullable_values,
                        nullable_rows,
                        options: ShredOptions {
                            min_frequency,
                            max_sparse_columns,
                            typed,
                            dictionary,
                            max_distinct_rows,
                            sparse_below,
                        },
                    }
                })
        },
    )
}

fn nullability(nullable: bool) -> Nullability {
    if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

fn build(case: &Case) -> VortexResult<MapArray> {
    let values = nullability(case.nullable_values);
    let rows = nullability(case.nullable_rows);
    if case.utf8 {
        let mut builder = Utf8MapBuilder::new(values, rows);
        for (row, &repeat) in case.rows.iter().zip(&case.repeats) {
            match (repeat, row) {
                (true, _) => builder.repeat_last_row(),
                (false, Some(entries)) => builder.push_row(
                    entries
                        .iter()
                        .map(|(k, v)| (k, v.as_ref().map(LabelValue::to_label_string))),
                ),
                (false, None) => builder.push_null(),
            }
        }
        return builder.finish();
    }
    let mut builder = LabelMapBuilder::new(values, rows);
    for (row, &repeat) in case.rows.iter().zip(&case.repeats) {
        match (repeat, row) {
            (true, _) => builder.repeat_last_row(),
            (false, Some(entries)) => builder.push_row(entries.iter().map(|(k, v)| (k, v.as_ref()))),
            (false, None) => builder.push_null(),
        }
    }
    builder.finish()
}

fn label_value(scalar: &Scalar) -> Option<LabelValue> {
    if let Some(s) = scalar.as_utf8_opt() {
        return s.value().map(|v| LabelValue::Str(v.to_string()));
    }
    let union = scalar.as_union();
    let child = union.child()?;
    Some(match union.variant_name()?.as_ref() {
        "str" => LabelValue::Str(child.as_utf8().value()?.to_string()),
        "int" => LabelValue::Int(child.as_primitive().typed_value::<i64>()?),
        "float" => LabelValue::Float(child.as_primitive().typed_value::<f64>()?),
        "bool" => LabelValue::Bool(child.as_bool().value()?),
        other => unreachable!("unknown variant {other}"),
    })
}

/// Reads a label map back through scalar access, independently of the shredding code paths.
fn read_rows(map: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<Row>> {
    (0..map.len())
        .map(|i| {
            let scalar = map.execute_scalar(i, ctx)?;
            let map = scalar.as_map();
            if map.is_null() {
                return Ok(None);
            }
            Ok(Some(
                map.entries()
                    .map(|(k, v)| (k.as_utf8().value().unwrap().to_string(), label_value(&v)))
                    .collect(),
            ))
        })
        .collect()
}

/// Reads a `Map<Utf8, Utf8>` back through scalar access.
fn read_string_rows(
    map: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Option<Vec<(String, Option<String>)>>>> {
    (0..map.len())
        .map(|i| {
            let scalar = map.execute_scalar(i, ctx)?;
            let map = scalar.as_map();
            if map.is_null() {
                return Ok(None);
            }
            Ok(Some(
                map.entries()
                    .map(|(k, v)| {
                        (
                            k.as_utf8().value().unwrap().to_string(),
                            v.as_utf8().value().map(|s| s.to_string()),
                        )
                    })
                    .collect(),
            ))
        })
        .collect()
}

fn read_strings(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<Option<String>>> {
    (0..array.len())
        .map(|i| {
            Ok(array
                .execute_scalar(i, ctx)?
                .as_utf8()
                .value()
                .map(|s| s.to_string()))
        })
        .collect()
}

fn read_string_lists(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Option<Vec<String>>>> {
    (0..array.len())
        .map(|i| {
            let scalar = array.execute_scalar(i, ctx)?;
            let list = scalar.as_list();
            if list.is_null() {
                return Ok(None);
            }
            Ok(Some(
                list.elements()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_utf8().value().unwrap().to_string())
                    .collect(),
            ))
        })
        .collect()
}

fn expected_label(rows: &[Row], key: &str) -> Vec<Option<String>> {
    rows.iter()
        .map(|row| {
            row.as_ref()?
                .iter()
                .find(|(k, _)| k == key)?
                .1
                .as_ref()
                .map(LabelValue::to_label_string)
        })
        .collect()
}

fn expected_names(rows: &[Row]) -> Vec<Option<Vec<String>>> {
    rows.iter()
        .map(|row| Some(row.as_ref()?.iter().map(|(k, _)| k.clone()).collect()))
        .collect()
}

fn expected_strings(rows: &[Row]) -> Vec<Option<Vec<(String, Option<String>)>>> {
    rows.iter()
        .map(|row| {
            Some(
                row.as_ref()?
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_ref().map(LabelValue::to_label_string)))
                    .collect(),
            )
        })
        .collect()
}

fn expected_project(rows: &[Row], keys: &[&str]) -> Vec<Row> {
    rows.iter()
        .map(|row| {
            Some(
                row.as_ref()?
                    .iter()
                    .filter(|(k, _)| keys.contains(&k.as_str()))
                    .cloned()
                    .collect(),
            )
        })
        .collect()
}

fn expected_distinct(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .flatten()
        .flat_map(|row| row.iter().map(|(k, _)| k.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn btrblocks_array(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let session = array_session();
    vortex_btrblocks::BtrBlocksCompressorBuilder::from_session(&session)
        .unrestricted()
        .build()
        .compress(array, &mut session.create_execution_ctx())
}

fn btrblocks(shredded: &ShreddedMapArray) -> VortexResult<ShreddedMapArray> {
    crate::compress_shredded(shredded, btrblocks_array)
}

/// Every key of every row through the retained probes, in a shuffled order so reads revisit
/// decoded blocks out of order.
fn check_probes(
    rows: &[Row],
    shredded: &ShreddedMapArray,
    map: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mut shredded_probe = crate::point::ShreddedProbe::new(shredded);
    let mut map_probe = crate::point::MapProbe::new(map, ctx)?;
    let order = (0..rows.len()).map(|i| (i * 7919) % rows.len().max(1));
    for row in order.take(rows.len()) {
        for key in VOCAB.iter().chain(&["missing"]) {
            let expected = rows[row]
                .as_ref()
                .and_then(|r| r.iter().find(|(k, _)| k == key))
                .and_then(|(_, v)| v.as_ref().map(LabelValue::to_label_string));
            let got = shredded_probe.get_label(key, row, ctx)?;
            assert_eq!(got, expected, "ShreddedProbe {key:?} row {row}");
            let got = map_probe.get_label(key, row, ctx)?;
            assert_eq!(got, expected, "MapProbe {key:?} row {row}");
        }
    }
    Ok(())
}

fn check_case(case: &Case, projection: &[&str], selection: &[usize]) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let ctx = &mut ctx;
    let map = build(case)?.into_array();
    assert_eq!(read_rows(&map, ctx)?, case.rows, "builder round trip");

    let shredded: ShreddedMapArray = shred(&map, &case.options, ctx)?;
    let sparse = ShreddedMapArraySlotsExt::columns(&shredded)
        .iter()
        .filter(|c| c.is::<vortex_sparse::Sparse>())
        .count();
    assert!(sparse <= case.options.max_sparse_columns);
    let shredded_ref = shredded.clone().into_array();

    // Decompress, both through the encoding's execute and the explicit op.
    assert_eq!(read_rows(&shredded_ref, ctx)?, case.rows, "scalar_at");
    let decoded = ops::shredded::to_map(&shredded, ctx)?.into_array();
    assert_eq!(read_rows(&decoded, ctx)?, case.rows, "to_map");
    let executed = shredded_ref.clone().execute::<MapArray>(ctx)?.into_array();
    assert_eq!(read_rows(&executed, ctx)?, case.rows, "execute");

    // Label names per row and distinct.
    let names = expected_names(&case.rows);
    let got = ops::shredded::label_names(&shredded, ctx)?.into_array();
    assert_eq!(read_string_lists(&got, ctx)?, names, "shredded label_names");
    let got = ops::map::label_names(&map, ctx)?.into_array();
    assert_eq!(read_string_lists(&got, ctx)?, names, "map label_names");
    let distinct = expected_distinct(&case.rows);
    assert_eq!(ops::shredded::distinct_label_names(&shredded, ctx)?, distinct);
    assert_eq!(ops::map::distinct_label_names(&map, ctx)?, distinct);

    // Single label extraction, including a key that never occurs.
    for key in VOCAB.iter().chain(&["missing"]) {
        let expected = expected_label(&case.rows, key);
        let got = ops::shredded::get_label_utf8(&shredded, key, ctx)?;
        assert_eq!(read_strings(&got, ctx)?, expected, "shredded get_label {key:?}");
        let got = ops::map::get_label_utf8(&map, key, ctx)?;
        assert_eq!(read_strings(&got, ctx)?, expected, "map get_label {key:?}");
    }

    // Point lookups read one label of one row.
    for (row, expected_row) in case.rows.iter().enumerate().step_by(3) {
        for key in VOCAB.iter().chain(&["missing"]) {
            let expected = expected_row
                .as_ref()
                .and_then(|r| r.iter().find(|(k, _)| k == key))
                .and_then(|(_, v)| v.as_ref().map(LabelValue::to_label_string));
            let got = crate::point::label_at(&shredded, key, row, ctx)?;
            assert_eq!(got, expected, "label_at {key:?} row {row}");
        }
    }
    check_probes(&case.rows, &shredded, &map, ctx)?;
    let compressed = btrblocks(&shredded)?;
    let compressed_map = btrblocks_array(&map)?;
    check_probes(&case.rows, &compressed, &compressed_map, ctx)?;

    // Filter on one label's value, then project labels of the matching rows.
    let filter_key = VOCAB[selection.first().copied().unwrap_or(0) % VOCAB.len()];
    let filter_labels = expected_label(&case.rows, filter_key);
    if let Some(value) = filter_labels.iter().flatten().next().cloned() {
        let matching: Vec<usize> = (0..filter_labels.len())
            .filter(|&r| filter_labels[r].as_deref() == Some(value.as_str()))
            .collect();
        let results = [
            ops::query::shredded(&shredded, filter_key, &value, projection, ctx)?,
            ops::query::shredded(&compressed, filter_key, &value, projection, ctx)?,
            ops::query::map(&map, filter_key, &value, projection, ctx)?,
            ops::query::map(&compressed_map, filter_key, &value, projection, ctx)?,
        ];
        for (i, got) in results.into_iter().enumerate() {
            for (key, array) in projection.iter().zip(got) {
                let all = expected_label(&case.rows, key);
                let expected: Vec<Option<String>> = matching.iter().map(|&r| all[r].clone()).collect();
                assert_eq!(read_strings(&array.into_array(), ctx)?, expected, "query {i} {key}");
            }
        }
    }

    // All values as strings.
    let strings = expected_strings(&case.rows);
    let got = ops::shredded::to_utf8_map(&shredded, ctx)?.into_array();
    assert_eq!(read_string_rows(&got, ctx)?, strings, "shredded to_utf8_map");
    let got = ops::map::to_utf8_map(&map, ctx)?.into_array();
    assert_eq!(read_string_rows(&got, ctx)?, strings, "map to_utf8_map");

    // Projection onto a subset of keys.
    let projected = expected_project(&case.rows, projection);
    let got = ops::shredded::project(&shredded, projection, ctx)?;
    let got = ops::shredded::to_map(&got, ctx)?.into_array();
    assert_eq!(read_rows(&got, ctx)?, projected, "shredded project");
    let got = ops::map::project(&map, projection, ctx)?.into_array();
    assert_eq!(read_rows(&got, ctx)?, projected, "map project");

    // The row-dictionary encoding and its operations.
    let encoded = crate::encode(&map, &case.options, ctx)?;
    assert_eq!(read_rows(&encoded, ctx)?, case.rows, "encoded scalar_at");
    let got = ops::encoded::to_map(&encoded, ctx)?.into_array();
    assert_eq!(read_rows(&got, ctx)?, case.rows, "encoded to_map");
    let got = ops::encoded::label_names(&encoded, ctx)?.into_array();
    assert_eq!(read_string_lists(&got, ctx)?, names, "encoded label_names");
    assert_eq!(ops::encoded::distinct_label_names(&encoded, ctx)?, distinct);
    for key in VOCAB.iter().chain(&["missing"]) {
        let got = ops::encoded::get_label_utf8(&encoded, key, ctx)?;
        assert_eq!(
            read_strings(&got, ctx)?,
            expected_label(&case.rows, key),
            "encoded get_label {key:?}"
        );
    }
    let got = ops::encoded::to_utf8_map(&encoded, ctx)?.into_array();
    assert_eq!(read_string_rows(&got, ctx)?, strings, "encoded to_utf8_map");
    let got = ops::encoded::project(&encoded, projection, ctx)?;
    let got = ops::encoded::to_map(&got, ctx)?.into_array();
    assert_eq!(read_rows(&got, ctx)?, projected, "encoded project");

    // The key-set encoding, with and without value deduplication.
    for dedup_values in [false, true] {
        let ks = crate::keyset::keyset_encode(
            &map,
            crate::keyset::KeySetOptions { dedup_values },
            ctx,
        )?
        .into_array();
        assert_eq!(read_rows(&ks, ctx)?, case.rows, "keyset scalar_at");
        let got = ops::encoded::to_map(&ks, ctx)?.into_array();
        assert_eq!(read_rows(&got, ctx)?, case.rows, "keyset to_map");
        let got = ops::encoded::label_names(&ks, ctx)?.into_array();
        assert_eq!(read_string_lists(&got, ctx)?, names, "keyset label_names");
        assert_eq!(ops::encoded::distinct_label_names(&ks, ctx)?, distinct);
        for key in VOCAB.iter().chain(&["missing"]) {
            let got = ops::encoded::get_label_utf8(&ks, key, ctx)?;
            assert_eq!(
                read_strings(&got, ctx)?,
                expected_label(&case.rows, key),
                "keyset get_label {key:?}"
            );
        }
        let got = ops::encoded::project(&ks, projection, ctx)?;
        let got = ops::encoded::to_map(&got, ctx)?.into_array();
        assert_eq!(read_rows(&got, ctx)?, projected, "keyset project");
        let len = case.rows.len();
        if len > 0 {
            let (start, end) = (selection[0] % len, selection[1] % (len + 1));
            let (start, end) = (start.min(end), start.max(end));
            let got = ks.slice(start..end)?;
            assert_eq!(read_rows(&got, ctx)?, case.rows[start..end].to_vec(), "keyset slice");
            let indices: Vec<u64> = selection.iter().map(|&i| (i % len) as u64).collect();
            let take = PrimitiveArray::new(Buffer::from(indices.clone()), Validity::NonNullable);
            let got = ks.take(take.into_array())?;
            let expected: Vec<Row> =
                indices.iter().map(|&i| case.rows[i as usize].clone()).collect();
            assert_eq!(read_rows(&got, ctx)?, expected, "keyset take");
        }
    }

    // Slice, take and filter keep the children aligned.
    let len = case.rows.len();
    if len > 0 {
        let (start, end) = (selection[0] % len, selection[1] % (len + 1));
        let (start, end) = (start.min(end), start.max(end));
        let got = shredded_ref.slice(start..end)?;
        assert_eq!(read_rows(&got, ctx)?, case.rows[start..end].to_vec(), "slice");

        let indices: Vec<u64> = selection.iter().map(|&i| (i % len) as u64).collect();
        let take = PrimitiveArray::new(Buffer::from(indices.clone()), Validity::NonNullable);
        let got = shredded_ref.take(take.into_array())?;
        let expected: Vec<Row> = indices.iter().map(|&i| case.rows[i as usize].clone()).collect();
        assert_eq!(read_rows(&got, ctx)?, expected, "take");

        let mask = Mask::from_iter((0..len).map(|i| selection.contains(&i)));
        let got = shredded_ref.filter(mask)?;
        let expected: Vec<Row> = (0..len)
            .filter(|i| selection.contains(i))
            .map(|i| case.rows[i].clone())
            .collect();
        assert_eq!(read_rows(&got, ctx)?, expected, "filter");
    }
    Ok(())
}

fn projection_strategy() -> impl Strategy<Value = Vec<&'static str>> {
    prop::sample::subsequence(VOCAB.iter().copied().chain(["missing"]).collect::<Vec<_>>(), 0..6)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn shredded_map_matches_model(
        case in case_strategy(),
        projection in projection_strategy(),
        selection in prop::collection::vec(0usize..64, 2..12),
    ) {
        check_case(&case, &projection, &selection).unwrap();
    }
}

#[test]
fn infers_label_value_types() {
    for (s, expected) in [
        ("200", LabelValue::Int(200)),
        ("-3", LabelValue::Int(-3)),
        ("0.5", LabelValue::Float(0.5)),
        ("true", LabelValue::Bool(true)),
        ("False", LabelValue::Str("False".into())),
        ("007", LabelValue::Str("007".into())),
        ("1e-05", LabelValue::Str("1e-05".into())),
        ("+Inf", LabelValue::Str("+Inf".into())),
        ("cadvisor:8080", LabelValue::Str("cadvisor:8080".into())),
    ] {
        assert_eq!(LabelValue::infer(s), expected, "{s}");
        assert_eq!(LabelValue::infer(s).to_label_string(), s);
    }
}

#[test]
fn shreds_frequent_keys_into_typed_columns() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let mut builder = LabelMapBuilder::new(Nullability::NonNullable, Nullability::NonNullable);
    for i in 0..100i64 {
        let job = LabelValue::Str("node".into());
        let cpu = LabelValue::Int(i % 4);
        let rare = LabelValue::Str(format!("r{i}"));
        let mut row = vec![("cpu", Some(&cpu)), ("job", Some(&job))];
        if i % 10 == 0 {
            row.push(("rare", Some(&rare)));
        }
        builder.push_row(row);
    }
    let map = builder.finish()?.into_array();
    let options = ShredOptions {
        min_frequency: 0.5,
        ..ShredOptions::default()
    };
    let shredded = shred(&map, &options, &mut ctx)?;
    let columns = shredded.data().columns();
    assert_eq!(columns.len(), 2);
    assert_eq!((columns[0].key.as_ref(), columns[0].variant), ("cpu", Some(1)));
    assert_eq!((columns[1].key.as_ref(), columns[1].variant), ("job", Some(0)));
    Ok(())
}

/// Probes over a BtrBlocks-compressed map large enough to produce run-end codes, chunked sparse
/// columns and a dictionary-coded residual.
#[test]
fn probes_read_compressed_columns() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let mut builder = Utf8MapBuilder::new(Nullability::Nullable, Nullability::NonNullable);
    let mut rows: Vec<Row> = Vec::new();
    for i in 0..20_000usize {
        let mut row = vec![("host".to_string(), Some(format!("h{}", (i / 37) % 300)))];
        if i % 10 == 3 {
            row.push(("err".to_string(), Some(format!("e{}", i % 7))));
        }
        if i % 53 == 0 {
            row.push((format!("rare{}", i % 5), (i % 2 == 0).then(|| format!("r{i}"))));
        }
        if i % 3 == 0 {
            let n = (i * 31) % 6000;
            row.push(("msg".to_string(), Some(format!("message {n} from the probe test"))));
        }
        row.sort();
        builder.push_row(row.iter().map(|(k, v)| (k, v.as_ref())));
        rows.push(Some(
            row.into_iter()
                .map(|(k, v)| (k, v.map(LabelValue::Str)))
                .collect(),
        ));
    }
    let map = builder.finish()?.into_array();
    let shredded = shred(&map, &ShredOptions::default(), &mut ctx)?;
    let compressed = btrblocks(&shredded)?;
    let compressed_map = btrblocks_array(&map)?;
    let keys = ["host", "err", "msg", "rare0", "rare1", "rare4", "missing"];
    let mut shredded_probe = crate::point::ShreddedProbe::new(&compressed);
    let mut map_probe = crate::point::MapProbe::new(&compressed_map, &mut ctx)?;
    for row in (0..rows.len()).map(|i| (i * 7919) % rows.len()) {
        for key in keys {
            let expected = rows[row]
                .as_ref()
                .and_then(|r| r.iter().find(|(k, _)| k == key))
                .and_then(|(_, v)| v.as_ref().map(LabelValue::to_label_string));
            assert_eq!(shredded_probe.get_label(key, row, &mut ctx)?, expected, "{key} {row}");
            assert_eq!(map_probe.get_label(key, row, &mut ctx)?, expected, "{key} {row}");
        }
    }

    // Selective filters reference few entries of the large `msg` dictionary, which the query
    // gathers block by block rather than decoding whole.
    let project = ["msg", "err", "rare0", "host", "missing"];
    for value in ["h5", "h299"] {
        let host = expected_label(&rows, "host");
        let matching: Vec<usize> =
            (0..rows.len()).filter(|&r| host[r].as_deref() == Some(value)).collect();
        let results = [
            ops::query::shredded(&compressed, "host", value, &project, &mut ctx)?,
            ops::query::map(&compressed_map, "host", value, &project, &mut ctx)?,
        ];
        for got in results {
            for (key, array) in project.iter().zip(got) {
                let all = expected_label(&rows, key);
                let expected: Vec<Option<String>> = matching.iter().map(|&r| all[r].clone()).collect();
                assert_eq!(read_strings(&array.into_array(), &mut ctx)?, expected, "{value} {key}");
            }
        }
    }
    Ok(())
}
