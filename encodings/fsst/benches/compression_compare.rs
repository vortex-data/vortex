// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comprehensive benchmarks comparing FSST, FSST-12, Zstd, and Snappy
//! across diverse string datasets.
//!
//! Includes datasets with 10k+ rows and long shared substrings to test where
//! FSST-12 (with its larger symbol table) outperforms FSST-8.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::VarBinArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::fsst12;
use vortex_fsst::test_utils;

fn main() {
    warm_up_vtables();
    print_compression_summary();
    divan::main();
}

// ---------------------------------------------------------------------------
// Dataset generators
// ---------------------------------------------------------------------------

const NUM_STRINGS: usize = 50_000;

/// 1. Short emails (~25 bytes avg)
fn gen_emails(n: usize) -> Vec<String> {
    test_utils::generate_emails(n)
}

/// 2. Medium URLs (~50 bytes avg)
fn gen_urls(n: usize) -> Vec<String> {
    test_utils::generate_short_urls(n)
}

/// 3. Long log lines (~150 bytes avg)
fn gen_logs(n: usize) -> Vec<String> {
    test_utils::generate_log_lines(n)
}

/// 4. Highly repetitive JSON (~80 bytes avg, template-based)
fn gen_json(n: usize) -> Vec<String> {
    test_utils::generate_json_strings(n)
}

/// 5. File paths (~40 bytes avg, hierarchical)
fn gen_paths(n: usize) -> Vec<String> {
    test_utils::generate_file_paths(n)
}

/// 6. ClickBench URLs (~100 bytes avg, long with query params)
fn gen_clickbench_urls(n: usize) -> Vec<String> {
    test_utils::generate_clickbench_urls(n)
}

/// 7. UUIDs - high cardinality, low compressibility
fn gen_uuids(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(777);
    (0..n)
        .map(|_| {
            let bytes: [u8; 16] = rng.random();
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
                u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
                u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
                u64::from_le_bytes({
                    let mut buf = [0u8; 8];
                    buf[..6].copy_from_slice(&bytes[10..16]);
                    buf
                }),
            )
        })
        .collect()
}

/// 8. Enum-like status strings - very low cardinality
fn gen_status_strings(n: usize) -> Vec<String> {
    let statuses = [
        "PENDING",
        "ACTIVE",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "IN_PROGRESS",
        "WAITING_FOR_APPROVAL",
        "ARCHIVED",
    ];
    let mut rng = StdRng::seed_from_u64(888);
    (0..n)
        .map(|_| statuses[rng.random_range(0..statuses.len())].to_string())
        .collect()
}

/// 9. Natural language (English-like sentences)
fn gen_english_text(n: usize) -> Vec<String> {
    let subjects = [
        "The quick brown fox",
        "A lazy dog",
        "The system administrator",
        "An unexpected error",
        "The database connection",
        "A new user",
        "The API endpoint",
        "Our monitoring system",
    ];
    let verbs = [
        "jumped over",
        "encountered",
        "processed",
        "failed to connect to",
        "successfully completed",
        "was unable to handle",
        "quickly resolved",
        "reported issues with",
    ];
    let objects = [
        "the production server",
        "multiple requests",
        "the authentication module",
        "the network interface",
        "several database queries",
        "the configuration file",
        "incoming traffic spikes",
        "the deployment pipeline",
    ];
    let mut rng = StdRng::seed_from_u64(999);
    (0..n)
        .map(|_| {
            format!(
                "{} {} {} at {:02}:{:02}:{:02}.",
                subjects[rng.random_range(0..subjects.len())],
                verbs[rng.random_range(0..verbs.len())],
                objects[rng.random_range(0..objects.len())],
                rng.random_range(0..24u32),
                rng.random_range(0..60u32),
                rng.random_range(0..60u32),
            )
        })
        .collect()
}

/// 10. Base64-encoded data - high entropy
fn gen_base64(n: usize) -> Vec<String> {
    let charset = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rng = StdRng::seed_from_u64(1010);
    (0..n)
        .map(|_| {
            let len = rng.random_range(20..60);
            let s: String = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())] as char)
                .collect();
            format!("{s}==")
        })
        .collect()
}

/// 11. Rich JSON with many distinct field names and nested structure. FSST-12's larger symbol table helps here.
fn gen_rich_json(n: usize) -> Vec<String> {
    let field_names = [
        "user_id",
        "username",
        "display_name",
        "email_address",
        "phone_number",
        "street_address",
        "city_name",
        "state_code",
        "zip_code",
        "country",
        "created_at",
        "updated_at",
        "last_login",
        "is_verified",
        "is_active",
        "subscription_type",
        "payment_method",
        "billing_address",
        "shipping_address",
        "order_count",
        "total_spent",
        "loyalty_points",
        "referral_code",
        "preferences",
        "notification_settings",
        "privacy_level",
        "two_factor_enabled",
        "profile_image_url",
        "cover_image_url",
        "bio_text",
        "website_url",
        "company_name",
        "job_title",
        "department",
        "team_name",
        "manager_id",
    ];
    let values = [
        "\"premium_enterprise\"",
        "\"standard_basic\"",
        "\"trial_30day\"",
        "\"free_tier\"",
        "true",
        "false",
        "null",
        "12345",
        "67890",
        "\"2024-03-15T10:30:00Z\"",
        "\"2024-01-01T00:00:00Z\"",
        "\"active\"",
        "\"suspended\"",
        "\"pending_review\"",
        "\"credit_card\"",
        "\"paypal\"",
        "\"bank_transfer\"",
    ];
    let mut rng = StdRng::seed_from_u64(1111);
    (0..n)
        .map(|_| {
            let num_fields = rng.random_range(8..20);
            let mut fields = Vec::with_capacity(num_fields);
            for _ in 0..num_fields {
                let name = field_names[rng.random_range(0..field_names.len())];
                let val = values[rng.random_range(0..values.len())];
                fields.push(format!("\"{name}\":{val}"));
            }
            format!("{{{}}}", fields.join(","))
        })
        .collect()
}

/// 12. XML-like data with many distinct tags and attributes. Verbose repeated patterns ideal for symbol table compression.
fn gen_xml_records(n: usize) -> Vec<String> {
    let tags = [
        "record", "entry", "item", "row", "element", "node", "data", "field",
    ];
    let attrs = [
        "id",
        "type",
        "class",
        "name",
        "value",
        "status",
        "priority",
        "category",
        "timestamp",
        "version",
        "source",
        "target",
        "format",
        "encoding",
        "locale",
    ];
    let attr_values = [
        "primary",
        "secondary",
        "active",
        "inactive",
        "pending",
        "archived",
        "high",
        "medium",
        "low",
        "critical",
        "normal",
        "debug",
        "utf-8",
        "ascii",
        "en-US",
        "de-DE",
        "ja-JP",
        "2024-03-15",
    ];
    let mut rng = StdRng::seed_from_u64(1212);
    (0..n)
        .map(|_| {
            let tag = tags[rng.random_range(0..tags.len())];
            let num_attrs = rng.random_range(3..8);
            let attr_str: String = (0..num_attrs)
                .map(|_| {
                    let attr_name = attrs[rng.random_range(0..attrs.len())];
                    let attr_val = attr_values[rng.random_range(0..attr_values.len())];
                    format!(" {attr_name}=\"{attr_val}\"")
                })
                .collect();
            let inner_tags = rng.random_range(1..4);
            let inner: String = (0..inner_tags)
                .map(|_| {
                    let itag = tags[rng.random_range(0..tags.len())];
                    let val = attr_values[rng.random_range(0..attr_values.len())];
                    format!("<{itag}>{val}</{itag}>")
                })
                .collect();
            format!("<{tag}{attr_str}>{inner}</{tag}>")
        })
        .collect()
}

/// 13. Long URLs with many distinct query parameter patterns (>100 chars each) and long shared substrings.
fn gen_long_query_urls(n: usize) -> Vec<String> {
    let domains = [
        "analytics.example.com",
        "tracking.adnetwork.io",
        "api.platform.dev",
        "cdn.content-delivery.net",
        "events.telemetry.cloud",
    ];
    let param_names = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
        "session_id",
        "user_id",
        "device_id",
        "app_version",
        "os_version",
        "screen_width",
        "screen_height",
        "language",
        "country",
        "region",
        "event_type",
        "event_name",
        "event_category",
        "event_label",
        "event_value",
        "timestamp",
        "request_id",
        "correlation_id",
        "trace_id",
        "span_id",
        "ab_test_group",
        "feature_flag",
        "experiment_id",
        "variant_id",
    ];
    let param_values = [
        "google",
        "facebook",
        "twitter",
        "email",
        "organic",
        "direct",
        "referral",
        "spring_sale_2024",
        "summer_promo",
        "holiday_special",
        "black_friday",
        "banner_v2",
        "sidebar_v1",
        "popup_v3",
        "interstitial",
        "click",
        "view",
        "purchase",
        "signup",
        "login",
        "search",
        "filter",
        "en-US",
        "de-DE",
        "fr-FR",
        "ja-JP",
        "ko-KR",
        "zh-CN",
    ];
    let mut rng = StdRng::seed_from_u64(1313);
    (0..n)
        .map(|_| {
            let domain = domains[rng.random_range(0..domains.len())];
            let num_params = rng.random_range(8..20);
            let params: String = (0..num_params)
                .map(|i| {
                    let name = param_names[rng.random_range(0..param_names.len())];
                    let val = param_values[rng.random_range(0..param_values.len())];
                    let sep = if i == 0 { "?" } else { "&" };
                    format!("{sep}{name}={val}")
                })
                .collect();
            format!("https://{domain}/collect{params}")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dataset wrapper
// ---------------------------------------------------------------------------

struct Dataset {
    name: &'static str,
    raw_bytes: Vec<Vec<u8>>,
    total_raw_size: usize,
}

impl Dataset {
    fn new(name: &'static str, strings: Vec<String>) -> Self {
        let raw_bytes: Vec<Vec<u8>> = strings.into_iter().map(|s| s.into_bytes()).collect();
        let total_raw_size: usize = raw_bytes.iter().map(|b| b.len()).sum();
        Self {
            name,
            raw_bytes,
            total_raw_size,
        }
    }

    fn as_refs(&self) -> Vec<&[u8]> {
        self.raw_bytes.iter().map(|v| v.as_slice()).collect()
    }

    fn to_varbin(&self) -> VarBinArray {
        VarBinArray::from_iter(
            self.raw_bytes
                .iter()
                .map(|s| Some(s.clone().into_boxed_slice())),
            DType::Utf8(Nullability::NonNullable),
        )
    }
}

static DATASETS: LazyLock<Vec<Dataset>> = LazyLock::new(|| {
    vec![
        Dataset::new("emails", gen_emails(NUM_STRINGS)),
        Dataset::new("urls", gen_urls(NUM_STRINGS)),
        Dataset::new("logs", gen_logs(NUM_STRINGS)),
        Dataset::new("json", gen_json(NUM_STRINGS)),
        Dataset::new("paths", gen_paths(NUM_STRINGS)),
        Dataset::new("clickbench_urls", gen_clickbench_urls(NUM_STRINGS)),
        Dataset::new("uuids", gen_uuids(NUM_STRINGS)),
        Dataset::new("status_strings", gen_status_strings(NUM_STRINGS)),
        Dataset::new("english_text", gen_english_text(NUM_STRINGS)),
        Dataset::new("base64", gen_base64(NUM_STRINGS)),
        // New datasets with long shared substrings
        Dataset::new("rich_json", gen_rich_json(NUM_STRINGS)),
        Dataset::new("xml_records", gen_xml_records(NUM_STRINGS)),
        Dataset::new("long_query_urls", gen_long_query_urls(NUM_STRINGS)),
    ]
});

// ---------------------------------------------------------------------------
// Compression summary
// ---------------------------------------------------------------------------

fn print_compression_summary() {
    println!("\n{}", "=".repeat(90));
    println!("COMPRESSION RATIO SUMMARY (compressed/original, lower is better)");
    println!(
        "{:>20} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Dataset", "RawSize", "Rows", "FSST", "FSST-12", "Zstd", "Snappy"
    );
    println!("{}", "-".repeat(90));

    for ds in DATASETS.iter() {
        let raw_size = ds.total_raw_size;
        let num_rows = ds.raw_bytes.len();

        // FSST
        let varbin = ds.to_varbin();
        let compressor = fsst_train_compressor(&varbin);
        let mut fsst_size = 0;
        for bytes in &ds.raw_bytes {
            fsst_size += compressor.compress(bytes).len();
        }
        fsst_size += compressor.symbol_table().len() * 8 + compressor.symbol_lengths().len();

        // FSST-12
        let refs = ds.as_refs();
        let compressor12 = fsst12::Compressor12::train(&refs);
        let mut fsst12_size = 0;
        for bytes in &ds.raw_bytes {
            fsst12_size += compressor12.compress(bytes).len();
        }
        fsst12_size += compressor12.symbols().len() * 9;

        // Zstd (level 3)
        let all_data: Vec<u8> = ds
            .raw_bytes
            .iter()
            .flat_map(|b| {
                let len = (b.len() as u32).to_le_bytes();
                len.iter()
                    .copied()
                    .chain(b.iter().copied())
                    .collect::<Vec<u8>>()
            })
            .collect();
        let zstd_compressed = zstd::bulk::compress(&all_data, 3).unwrap();
        let zstd_size = zstd_compressed.len();

        // Snappy
        let mut snappy_encoder = snap::raw::Encoder::new();
        let snappy_compressed = snappy_encoder.compress_vec(&all_data).unwrap();
        let snappy_size = snappy_compressed.len();

        println!(
            "{:>20} {:>9} {:>8} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            ds.name,
            format_size(raw_size),
            num_rows,
            fsst_size as f64 / raw_size as f64,
            fsst12_size as f64 / raw_size as f64,
            zstd_size as f64 / raw_size as f64,
            snappy_size as f64 / raw_size as f64,
        );
    }
    println!("{}", "=".repeat(90));
    println!();
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

// ---------------------------------------------------------------------------
// Dataset index for divan parametrization
// ---------------------------------------------------------------------------

const DATASET_INDICES: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

// ---------------------------------------------------------------------------
// FSST benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();
    let compressor = fsst_train_compressor(&varbin);

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| ds.raw_bytes.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for bytes in data.iter() {
                total += compressor.compress(bytes).len();
            }
            total
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();
    let compressor = fsst_train_compressor(&varbin);
    let decompressor = compressor.decompressor();
    let compressed: Vec<Vec<u8>> = ds
        .raw_bytes
        .iter()
        .map(|b| compressor.compress(b))
        .collect();

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for c in data.iter() {
                total += decompressor.decompress(c).len();
            }
            total
        });
}

// ---------------------------------------------------------------------------
// FSST-12 benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();
    let compressor = fsst12::Compressor12::train(&refs);

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| ds.raw_bytes.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for bytes in data.iter() {
                total += compressor.compress(bytes).len();
            }
            total
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();
    let compressor = fsst12::Compressor12::train(&refs);
    let decompressor = compressor.decompressor();
    let compressed: Vec<Vec<u8>> = ds
        .raw_bytes
        .iter()
        .map(|b| compressor.compress(b))
        .collect();

    bencher
        .counter(divan::counter::BytesCount::of_iter(
            ds.raw_bytes.iter().map(|b| b.len()),
        ))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut total = 0;
            for c in data.iter() {
                total += decompressor.decompress(c).len();
            }
            total
        });
}

// ---------------------------------------------------------------------------
// Zstd benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn zstd_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| all_data.clone())
        .bench_refs(|data| zstd::bulk::compress(data, 3).unwrap());
}

#[divan::bench(args = DATASET_INDICES)]
fn zstd_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();
    let compressed = zstd::bulk::compress(&all_data, 3).unwrap();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| zstd::bulk::decompress(data, all_data.len() * 2).unwrap());
}

// ---------------------------------------------------------------------------
// Snappy benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn snappy_compress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| all_data.clone())
        .bench_refs(|data| {
            let mut encoder = snap::raw::Encoder::new();
            encoder.compress_vec(data).unwrap()
        });
}

#[divan::bench(args = DATASET_INDICES)]
fn snappy_decompress(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let all_data: Vec<u8> = ds
        .raw_bytes
        .iter()
        .flat_map(|b| {
            let len = (b.len() as u32).to_le_bytes();
            len.iter()
                .copied()
                .chain(b.iter().copied())
                .collect::<Vec<u8>>()
        })
        .collect();
    let mut encoder = snap::raw::Encoder::new();
    let compressed = encoder.compress_vec(&all_data).unwrap();

    bencher
        .counter(divan::counter::BytesCount::new(all_data.len()))
        .with_inputs(|| compressed.clone())
        .bench_refs(|data| {
            let mut decoder = snap::raw::Decoder::new();
            decoder.decompress_vec(data).unwrap()
        });
}

// ---------------------------------------------------------------------------
// Training benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(args = DATASET_INDICES)]
fn fsst12_train(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let refs = ds.as_refs();

    bencher
        .with_inputs(|| refs.clone())
        .bench_refs(|data| fsst12::Compressor12::train(data));
}

#[divan::bench(args = DATASET_INDICES)]
fn fsst_train(bencher: Bencher, idx: usize) {
    let ds = &DATASETS[idx];
    let varbin = ds.to_varbin();

    bencher
        .with_inputs(|| &varbin)
        .bench_refs(|vb| fsst_train_compressor(vb));
}
