// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Classifier behavior pinned by representative v2 names from each
//! group in `benchmarks-website/server.js`'s `getGroup`.

use rstest::rstest;
use serde_json::json;
use vortex_bench_migrate::classifier::V3Bin;
use vortex_bench_migrate::classifier::classify;
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
        format: "vortex".into(),
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
        format: "vortex".into(),
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
        format: "vortex".into(),
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
        format: "parquet-nvme".into(),
    },
)]
#[case::random_access_4_part_vortex(
    "random-access/chimp/take/vortex-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "chimp/take".into(),
        format: "vortex-nvme".into(),
    },
)]
#[case::random_access_2_part_legacy(
    "random-access/parquet-tokio-local-disk",
    V3Bin::RandomAccess {
        dataset: "random access".into(),
        format: "parquet-nvme".into(),
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
#[case::fineweb_skipped("fineweb_q01/datafusion:parquet")]
#[case::nonsense_prefix("not-a-known-bench/series")]
fn unmapped_records_yield_none(#[case] name: &str) {
    let r = record(name);
    assert_eq!(
        classify(&r),
        None,
        "expected {name:?} to classify as None (drop)",
    );
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
