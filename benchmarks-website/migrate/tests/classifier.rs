// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Classifier behavior pinned by representative v2 names from each
//! group in `benchmarks-website/server.js`'s `getGroup`.

use rstest::rstest;
use serde_json::json;
use vortex_bench_migrate::classifier::Outcome;
use vortex_bench_migrate::classifier::Skip;
use vortex_bench_migrate::classifier::V3Bin;
use vortex_bench_migrate::classifier::classify;
use vortex_bench_migrate::classifier::classify_outcome;
use vortex_bench_migrate::classifier::format_query;
use vortex_bench_migrate::classifier::rename_engine;
use vortex_bench_migrate::v2::V2Record;

fn record(name: &str) -> V2Record {
    V2Record {
        name: name.to_string(),
        commit_id: Some("deadbeef".into()),
        unit: Some("ns".into()),
        value: Some(json!(123)),
        storage: None,
        dataset: None,
        all_runtimes: None,
        env_triple: None,
    }
}

fn record_with_storage_and_sf(name: &str, storage: &str, suite: &str, sf: &str) -> V2Record {
    let mut r = record(name);
    r.storage = Some(storage.into());
    r.dataset = Some(json!({ suite: { "scale_factor": sf } }));
    r
}

#[rstest]
#[case::clickbench(
    "clickbench_q07/datafusion:parquet",
    V3Bin::Query {
        dataset: "clickbench".into(),
        dataset_variant: None,
        scale_factor: None,
        query_idx: 7,
        storage: "nvme".into(),
        engine: "datafusion".into(),
        format: "parquet".into(),
    },
)]
#[case::clickbench_vortex_renamed(
    "clickbench_q12/datafusion:vortex-file-compressed",
    V3Bin::Query {
        dataset: "clickbench".into(),
        dataset_variant: None,
        scale_factor: None,
        query_idx: 12,
        storage: "nvme".into(),
        engine: "datafusion".into(),
        format: "vortex-file-compressed".into(),
    },
)]
#[case::statpopgen(
    "statpopgen_q3/datafusion:parquet",
    V3Bin::Query {
        dataset: "statpopgen".into(),
        dataset_variant: None,
        scale_factor: None,
        query_idx: 3,
        storage: "nvme".into(),
        engine: "datafusion".into(),
        format: "parquet".into(),
    },
)]
#[case::polarsignals(
    "polarsignals_q1/duckdb:parquet",
    V3Bin::Query {
        dataset: "polarsignals".into(),
        dataset_variant: None,
        scale_factor: None,
        query_idx: 1,
        storage: "nvme".into(),
        engine: "duckdb".into(),
        format: "parquet".into(),
    },
)]
fn non_fan_out_query_records(#[case] name: &str, #[case] expected: V3Bin) {
    let r = record(name);
    assert_eq!(classify(&r), Some(expected));
}

#[rstest]
#[case::tpch_s3_sf100(
    "tpch_q01/datafusion:parquet",
    "S3",
    "tpch",
    "100",
    V3Bin::Query {
        dataset: "tpch".into(),
        dataset_variant: None,
        scale_factor: Some("100".into()),
        query_idx: 1,
        storage: "s3".into(),
        engine: "datafusion".into(),
        format: "parquet".into(),
    },
)]
#[case::tpch_nvme_sf1(
    "tpch_q22/duckdb:vortex-file-compressed",
    "NVMe",
    "tpch",
    "1",
    V3Bin::Query {
        dataset: "tpch".into(),
        dataset_variant: None,
        scale_factor: Some("1".into()),
        query_idx: 22,
        storage: "nvme".into(),
        engine: "duckdb".into(),
        format: "vortex-file-compressed".into(),
    },
)]
#[case::tpcds_nvme_sf10(
    "tpcds_q05/datafusion:vortex-file-compressed",
    "NVMe",
    "tpcds",
    "10",
    V3Bin::Query {
        dataset: "tpcds".into(),
        dataset_variant: None,
        scale_factor: Some("10".into()),
        query_idx: 5,
        storage: "nvme".into(),
        engine: "datafusion".into(),
        format: "vortex-file-compressed".into(),
    },
)]
fn fan_out_query_records(
    #[case] name: &str,
    #[case] storage: &str,
    #[case] suite: &str,
    #[case] sf: &str,
    #[case] expected: V3Bin,
) {
    let r = record_with_storage_and_sf(name, storage, suite, sf);
    assert_eq!(classify(&r), Some(expected));
}

#[rstest]
#[case::random_access_4_part(
    "random-access/taxi/take/parquet-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "taxi/take".into(),
        format: "parquet".into(),
    },
)]
#[case::random_access_4_part_vortex(
    "random-access/chimp/take/vortex-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "chimp/take".into(),
        format: "vortex-file-compressed".into(),
    },
)]
#[case::random_access_4_part_lance(
    "random-access/taxi/take/lance-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "taxi/take".into(),
        format: "lance".into(),
    },
)]
fn random_access_records(#[case] name: &str, #[case] expected: V3Bin) {
    let r = record(name);
    assert_eq!(classify(&r), Some(expected));
}

#[rstest]
#[case::compress_time_vortex(
    "compress time/clickbench",
    V3Bin::CompressionTime {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "vortex-file-compressed".into(),
        op: "encode".into(),
    },
)]
#[case::decompress_time_vortex(
    "decompress time/tpch_lineitem",
    V3Bin::CompressionTime {
        dataset: "tpch_lineitem".into(),
        dataset_variant: None,
        format: "vortex-file-compressed".into(),
        op: "decode".into(),
    },
)]
#[case::parquet_compress(
    "parquet_rs-zstd compress time/clickbench",
    V3Bin::CompressionTime {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "parquet".into(),
        op: "encode".into(),
    },
)]
#[case::lance_decompress(
    "lance decompress time/clickbench",
    V3Bin::CompressionTime {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "lance".into(),
        op: "decode".into(),
    },
)]
fn compression_time_records(#[case] name: &str, #[case] expected: V3Bin) {
    let r = record(name);
    assert_eq!(classify(&r), Some(expected));
}

#[rstest]
#[case::vortex_size(
    "vortex size/clickbench",
    V3Bin::CompressionSize {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "vortex-file-compressed".into(),
    },
)]
#[case::vortex_file_compressed_size_normalizes(
    "vortex-file-compressed size/clickbench",
    V3Bin::CompressionSize {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "vortex-file-compressed".into(),
    },
)]
#[case::parquet_size(
    "parquet size/clickbench",
    V3Bin::CompressionSize {
        dataset: "clickbench".into(),
        dataset_variant: None,
        format: "parquet".into(),
    },
)]
#[case::lance_size(
    "lance size/tpch_lineitem",
    V3Bin::CompressionSize {
        dataset: "tpch_lineitem".into(),
        dataset_variant: None,
        format: "lance".into(),
    },
)]
fn compression_size_records(#[case] name: &str, #[case] expected: V3Bin) {
    let r = record(name);
    assert_eq!(classify(&r), Some(expected));
}

#[rstest]
#[case::ratio_vortex_parquet("vortex:parquet-zstd ratio compress time/clickbench")]
#[case::ratio_vortex_lance("vortex:lance ratio decompress time/clickbench")]
#[case::ratio_size_vortex_parquet("vortex:parquet-zstd size/clickbench")]
#[case::ratio_size_vortex_raw("vortex:raw size/clickbench")]
#[case::throughput("compress throughput/clickbench")]
#[case::nonsense_prefix("not-a-known-bench/series")]
#[case::random_access_3_part("random-access/taxi/parquet-tokio-local-disk")]
fn unmapped_records_yield_none(#[case] name: &str) {
    let r = record(name);
    assert_eq!(
        classify(&r),
        None,
        "expected {name:?} to classify as None (drop)",
    );
}

#[rstest]
#[case::parquet_2_part(
    "random-access/parquet-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "taxi".into(),
        format: "parquet".into(),
    },
)]
#[case::vortex_2_part(
    "random-access/vortex-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "taxi".into(),
        format: "vortex-file-compressed".into(),
    },
)]
#[case::lance_2_part(
    "random-access/lance-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "taxi".into(),
        format: "lance".into(),
    },
)]
fn random_access_2_part_legacy_recovered_as_taxi(#[case] name: &str, #[case] expected: V3Bin) {
    // The 2-part shape `random-access/<format>-tokio-local-disk` is
    // emitted by `random-access-bench`'s legacy taxi run (no
    // `AccessPattern`, see `measurement_name` in
    // `benchmarks/random-access-bench/src/main.rs`). The live v3
    // emitter writes `dataset="taxi"` for those measurements, so the
    // historical 2-part records on S3 must land in the same v3
    // chart instead of being dropped as `UnsupportedShape`.
    let r = record(name);
    assert_eq!(
        classify(&r),
        Some(expected),
        "2-part legacy random-access must recover as dataset=taxi"
    );
}

#[rstest]
#[case::parquet_footer("random-access/parquet-tokio-local-disk-footer")]
#[case::vortex_footer("random-access/vortex-tokio-local-disk-footer")]
#[case::lance_footer("random-access/lance-tokio-local-disk-footer")]
fn random_access_2_part_footer_is_deprecated(#[case] name: &str) {
    // The reopen-mode `-footer` variant is a different access pattern
    // (file is reopened per take). The live v3 emitter passes the
    // bare `format.name()` for both reopen and cached, so it can't
    // distinguish them on the wire. Keep migration consistent with
    // that by routing `-footer` 2-part records to Skip::Deprecated
    // (they don't strip clean to a v3-allowlisted format).
    let r = record(name);
    assert!(
        matches!(classify_outcome(&r), Outcome::Skip(Skip::Deprecated)),
        "2-part `-footer` random-access must be Skip::Deprecated"
    );
}

#[rstest]
#[case::parquet_footer("random-access/taxi/correlated/parquet-tokio-local-disk-footer")]
#[case::vortex_footer("random-access/feature-vectors/uniform/vortex-tokio-local-disk-footer")]
#[case::lance_footer("random-access/nested-structs/correlated/lance-tokio-local-disk-footer")]
fn random_access_4_part_footer_is_deprecated(#[case] name: &str) {
    // Same reasoning as 2-part `-footer`: the format string ends in
    // `-tokio-local-disk-footer`, the strip_suffix doesn't match, and
    // the unstripped value fails the V3_FORMATS allowlist.
    let r = record(name);
    assert!(
        matches!(classify_outcome(&r), Outcome::Skip(Skip::Deprecated)),
        "4-part `-footer` random-access must be Skip::Deprecated"
    );
}

#[test]
fn parquet_zstd_size_is_deprecated() {
    // `parquet-zstd` is not on the v3 emitter's format allowlist, so
    // historical `parquet-zstd size/...` records bucket under
    // Skip::Deprecated and don't render as orphan charts in v3.
    let r = record("parquet-zstd size/clickbench");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Skip(Skip::Deprecated)
    ));
}

#[test]
fn vortex_parquet_zstd_ratio_is_intentional_skip() {
    let r = record("vortex:parquet-zstd ratio compress time/clickbench");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Skip(Skip::DerivedRatio)
    ));
}

#[test]
fn vortex_parquet_zst_typo_ratio_is_intentional_skip() {
    // `parquet-zst` (no trailing `d`) was emitted by some v2 runs.
    // Both spellings should classify as derived ratios.
    for name in [
        "vortex:parquet-zst ratio compress time/clickbench",
        "vortex:parquet-zst ratio decompress time/clickbench",
    ] {
        let r = record(name);
        assert!(
            matches!(classify_outcome(&r), Outcome::Skip(Skip::DerivedRatio)),
            "{name:?} should be DerivedRatio",
        );
    }
}

#[test]
fn throughput_is_intentional_skip() {
    let r = record("compress throughput/clickbench");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Skip(Skip::Throughput)
    ));
}

#[test]
fn unknown_prefix_is_unknown() {
    let r = record("not-a-known-bench/series");
    assert!(matches!(classify_outcome(&r), Outcome::Unknown));
}

#[test]
fn gharchive_q00_is_deprecated() {
    // gharchive isn't on the v3 query-suite allowlist, so historical
    // gharchive query records bucket as Skip::Deprecated.
    let r = record("gharchive_q00/datafusion:parquet");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Skip(Skip::Deprecated)
    ));
}

#[test]
fn fineweb_q00_classifies() {
    // fineweb is on V3_QUERY_SUITES (still emitted by v3 CI per
    // .github/workflows/sql-benchmarks.yml's `fineweb` matrix entry),
    // so historical fineweb records ingest like any other suite.
    let r = record("fineweb_q00/datafusion:parquet");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Bin(V3Bin::Query { .. })
    ));
}

#[test]
fn memory_record_is_historical_memory_skip() {
    // v2 emitted `<suite>_q<NN>_memory/<engine>:<format>` records that
    // carry top-level memory fields V2Record doesn't deserialize.
    // Skip them with a known variant so they don't trip the 5% gate.
    let r = record("clickbench_q07_memory/datafusion:parquet");
    assert!(matches!(
        classify_outcome(&r),
        Outcome::Skip(Skip::HistoricalMemory)
    ));
}

#[test]
fn tpch_compression_size_carries_scale_factor() {
    // The data.json.gz "vortex size/tpch" path needs to derive
    // dataset_variant from the v2 record's `dataset` object, the same
    // way the file-sizes path does. Otherwise SF=10 rows from the two
    // sources never collide on `mid` and produce duplicate rows.
    let mut r = record("vortex size/tpch");
    r.dataset = Some(serde_json::json!({ "tpch": { "scale_factor": "10" } }));
    let outcome = classify_outcome(&r);
    let Outcome::Bin(V3Bin::CompressionSize {
        dataset,
        dataset_variant,
        format,
    }) = outcome
    else {
        panic!("expected Bin(CompressionSize), got {outcome:?}");
    };
    assert_eq!(dataset, "tpch");
    assert_eq!(dataset_variant, Some("10".into()));
    assert_eq!(format, "vortex-file-compressed");
}

#[test]
fn tpch_compression_size_drops_default_scale_factor() {
    // SF "1.0" matches the file-sizes path's filter and collapses to
    // dataset_variant: None.
    let mut r = record("vortex size/tpch");
    r.dataset = Some(serde_json::json!({ "tpch": { "scale_factor": "1.0" } }));
    let outcome = classify_outcome(&r);
    let Outcome::Bin(V3Bin::CompressionSize {
        dataset_variant, ..
    }) = outcome
    else {
        panic!("expected Bin(CompressionSize), got {outcome:?}");
    };
    assert_eq!(dataset_variant, None);
}

#[rstest]
// SF=1 is the implicit default; both spellings must drop to None so
// `bin_compression_size` and `migrate_file_sizes` agree.
#[case::int_one("1", None)]
#[case::float_one("1.0", None)]
// SF=10 must produce the same canonical string regardless of spelling.
#[case::int_ten("10", Some("10".into()))]
#[case::float_ten("10.0", Some("10".into()))]
#[case::float_fractional("0.1", Some("0.1".into()))]
#[case::whitespace("  10  ", Some("10".into()))]
#[case::empty("", None)]
fn compression_size_scale_factor_canonicalizes(
    #[case] raw_sf: &str,
    #[case] expected: Option<String>,
) {
    let mut r = record("vortex size/tpch");
    r.dataset = Some(serde_json::json!({ "tpch": { "scale_factor": raw_sf } }));
    let outcome = classify_outcome(&r);
    let Outcome::Bin(V3Bin::CompressionSize {
        dataset_variant, ..
    }) = outcome
    else {
        panic!("expected Bin(CompressionSize) for sf={raw_sf:?}, got {outcome:?}");
    };
    assert_eq!(dataset_variant, expected, "sf={raw_sf:?}");
}

#[test]
fn engine_casing_lowercased() {
    // Older v2 records emitted display-case engines like `DataFusion`
    // and `DuckDB`. The classifier lowercases at push time so dedup
    // collapses display-case rows into the canonical lowercase ones.
    let r = record("clickbench_q07/DataFusion:parquet");
    let outcome = classify_outcome(&r);
    let Outcome::Bin(V3Bin::Query { engine, format, .. }) = outcome else {
        panic!("expected Bin(Query), got {outcome:?}");
    };
    assert_eq!(engine, "datafusion");
    assert_eq!(format, "parquet");
}

#[test]
fn rename_engine_pins_canonical_outputs() {
    assert_eq!(rename_engine("vortex-tokio-local-disk"), "vortex-nvme");
    assert_eq!(
        rename_engine("datafusion:vortex-file-compressed"),
        "datafusion:vortex"
    );
    assert_eq!(rename_engine("LANCE"), "lance");
}

#[test]
fn format_query_pins_v2_display() {
    assert_eq!(format_query("clickbench_q00"), "CLICKBENCH Q0");
    assert_eq!(format_query("tpch_q22"), "TPC-H Q22");
    assert_eq!(format_query("tpcds_q42"), "TPC-DS Q42");
    assert_eq!(format_query("polarsignals_q1"), "POLARSIGNALS Q1");
    // Names that don't match a suite fall back to upper + " " replace.
    assert_eq!(
        format_query("vortex-file-compressed size"),
        "VORTEX FILE COMPRESSED SIZE"
    );
}
