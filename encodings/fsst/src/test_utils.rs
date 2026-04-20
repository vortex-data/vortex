// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;

use crate::FSSTArray;
use crate::fsst_compress;
use crate::fsst_train_compressor;

pub fn gen_fsst_test_data(
    len: usize,
    avg_str_len: usize,
    unique_chars: u8,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut strings = Vec::with_capacity(len);

    for _ in 0..len {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_str_len * rng.random_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.random_range(b'a'..(b'a' + unique_chars)))
                .collect::<Vec<u8>>(),
        ));
    }

    let varbin = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);

    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx).into_array()
}

pub fn gen_dict_fsst_test_data<T: NativePType>(
    len: usize,
    unique_values: usize,
    str_len: usize,
    unique_char_count: u8,
    ctx: &mut ExecutionCtx,
) -> DictArray {
    let values = gen_fsst_test_data(len, str_len, unique_char_count, ctx);
    let mut rng = StdRng::seed_from_u64(0);
    let codes = (0..len)
        .map(|_| T::from(rng.random_range(0..unique_values)).unwrap())
        .collect::<PrimitiveArray>();
    DictArray::try_new(codes.into_array(), values)
        .vortex_expect("DictArray::try_new should succeed for test data")
}

// ---------------------------------------------------------------------------
// Benchmark dataset generators
// ---------------------------------------------------------------------------

pub const NUM_STRINGS: usize = 100_000;

// ---------------------------------------------------------------------------
// URL generator (ClickBench-style weighted domains)
// ---------------------------------------------------------------------------

pub const HIGH_MATCH_DOMAIN: &str = "smeshariki.ru";
pub const LOW_MATCH_DOMAIN: &str = "rare-example-domain.com";

pub const URL_DOMAINS: &[(&str, u32)] = &[
    ("smeshariki.ru", 500),
    ("auto.ru", 150),
    ("komme.ru", 100),
    ("yandex.ru", 80),
    ("mail.ru", 60),
    ("livejournal.com", 40),
    ("vk.com", 30),
    ("avito.ru", 20),
    ("kinopoisk.ru", 10),
    ("rare-example-domain.com", 10),
];

pub const URL_PATHS: &[&str] = &[
    "/GameMain.aspx",
    "/index.php",
    "/catalog/item",
    "/search",
    "/news/article",
    "/user/profile",
    "/collection/view",
    "/cars/used/sale",
    "/forum/thread",
    "/photo/album",
    "/video/watch",
    "/download/file",
    "/api/v1/resource",
    "/shop/product",
    "/blog/post",
];

pub fn generate_url_data() -> VarBinArray {
    generate_url_data_n(NUM_STRINGS)
}

pub fn generate_url_data_n(n: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let total_weight: u32 = URL_DOMAINS.iter().map(|(_, w)| w).sum();
    let urls: Vec<Option<Box<[u8]>>> = (0..n)
        .map(|_| {
            let domain_roll = rng.random_range(0..total_weight);
            let mut cumulative = 0u32;
            let mut domain = URL_DOMAINS[0].0;
            for &(d, w) in URL_DOMAINS {
                cumulative += w;
                if domain_roll < cumulative {
                    domain = d;
                    break;
                }
            }
            let path = URL_PATHS[rng.random_range(0..URL_PATHS.len())];
            let query_id: u32 = rng.random_range(1..100_000);
            let tab: u16 = rng.random_range(1..20);
            let url = format!("http://{domain}{path}?id={query_id}&tab={tab}#ref={query_id}");
            Some(url.into_bytes().into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(urls, DType::Utf8(Nullability::NonNullable))
}

pub fn make_fsst_urls(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let varbin = generate_url_data_n(n);
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// ClickBench-style URL generator (longer URLs with query params, fragments)
// ---------------------------------------------------------------------------

const CB_DOMAINS: &[&str] = &[
    "www.google.com",
    "yandex.ru",
    "mail.ru",
    "vk.com",
    "www.youtube.com",
    "www.facebook.com",
    "ok.ru",
    "go.mail.ru",
    "www.avito.ru",
    "pogoda.yandex.ru",
    "news.yandex.ru",
    "maps.yandex.ru",
    "market.yandex.ru",
    "afisha.yandex.ru",
    "auto.ru",
    "www.kinopoisk.ru",
    "www.ozon.ru",
    "www.wildberries.ru",
    "aliexpress.ru",
    "lenta.ru",
];

const CB_PATHS: &[&str] = &[
    "/search",
    "/catalog/electronics/smartphones",
    "/product/item/123456789",
    "/news/2024/03/15/article-about-technology",
    "/user/profile/settings/notifications",
    "/api/v2/catalog/search",
    "/checkout/cart/summary",
    "/blog/2024/how-to-optimize-database-queries-for-better-performance",
    "/category/home-and-garden/furniture/tables",
    "/",
];

const CB_PARAMS: &[&str] = &[
    "?utm_source=google&utm_medium=cpc&utm_campaign=spring_sale_2024&utm_content=banner_v2",
    "?q=buy+smartphone+online+cheap+free+shipping&category=electronics&sort=price_asc&page=3",
    "?ref=main_page_carousel_block_position_4&sessionid=abc123def456",
    "?from=tabbar&clid=2270455&text=weather+forecast+tomorrow",
    "?lr=213&msid=1234567890.12345&suggest_reqid=abcdef&csg=12345",
    "",
    "",
    "",
    "?page=1&per_page=20",
    "?source=serp&forceshow=1",
];

const CB_FRAGMENTS: &[&str] = &[
    "",
    "",
    "",
    "#section-reviews",
    "#comments",
    "#price-history",
    "",
    "",
    "",
    "",
];

pub fn generate_clickbench_urls(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(123);
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.7) {
                "https"
            } else {
                "http"
            };
            let domain = CB_DOMAINS[rng.random_range(0..CB_DOMAINS.len())];
            let path = CB_PATHS[rng.random_range(0..CB_PATHS.len())];
            let params = CB_PARAMS[rng.random_range(0..CB_PARAMS.len())];
            let fragment = CB_FRAGMENTS[rng.random_range(0..CB_FRAGMENTS.len())];
            format!("{scheme}://{domain}{path}{params}{fragment}")
        })
        .collect()
}

pub fn make_fsst_clickbench_urls(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let urls = generate_clickbench_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// Short URL generator (simple URLs for contains benchmarks)
// ---------------------------------------------------------------------------

const SHORT_URL_DOMAINS: &[&str] = &[
    "google.com",
    "facebook.com",
    "github.com",
    "stackoverflow.com",
    "amazon.com",
    "reddit.com",
    "twitter.com",
    "youtube.com",
    "wikipedia.org",
    "microsoft.com",
    "apple.com",
    "netflix.com",
    "linkedin.com",
    "cloudflare.com",
    "google.co.uk",
    "docs.google.com",
    "mail.google.com",
    "maps.google.com",
    "news.ycombinator.com",
    "arxiv.org",
];

const SHORT_URL_PATHS: &[&str] = &[
    "/index.html",
    "/about",
    "/search?q=vortex",
    "/user/profile/settings",
    "/api/v2/data",
    "/blog/2024/post",
    "/products/item/12345",
    "/docs/reference/guide",
    "/login",
    "/dashboard/analytics",
];

pub fn generate_short_urls(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.8) {
                "https"
            } else {
                "http"
            };
            let domain = SHORT_URL_DOMAINS[rng.random_range(0..SHORT_URL_DOMAINS.len())];
            let path = SHORT_URL_PATHS[rng.random_range(0..SHORT_URL_PATHS.len())];
            format!("{scheme}://{domain}{path}")
        })
        .collect()
}

pub fn make_fsst_short_urls(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let urls = generate_short_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// Log lines generator (Apache/nginx-style access logs)
// ---------------------------------------------------------------------------

const LOG_METHODS: &[&str] = &["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"];
const LOG_PATHS: &[&str] = &[
    "/api/v1/users",
    "/api/v2/products/search",
    "/healthcheck",
    "/static/js/app.bundle.min.js",
    "/favicon.ico",
    "/login",
    "/dashboard/analytics",
    "/api/v1/orders/12345/status",
    "/graphql",
    "/metrics",
];
const LOG_STATUS: &[u16] = &[
    200, 200, 200, 200, 200, 201, 301, 302, 400, 403, 404, 500, 502,
];
const LOG_IPS: &[&str] = &[
    "192.168.1.1",
    "10.0.0.42",
    "172.16.0.100",
    "203.0.113.50",
    "198.51.100.23",
    "8.8.8.8",
    "1.1.1.1",
    "74.125.200.100",
    "151.101.1.69",
    "93.184.216.34",
];
const LOG_UAS: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "curl/7.81.0",
    "python-requests/2.28.1",
    "Go-http-client/1.1",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
];

pub fn generate_log_lines(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(456);
    (0..n)
        .map(|_| {
            let ip = LOG_IPS[rng.random_range(0..LOG_IPS.len())];
            let method = LOG_METHODS[rng.random_range(0..LOG_METHODS.len())];
            let path = LOG_PATHS[rng.random_range(0..LOG_PATHS.len())];
            let status = LOG_STATUS[rng.random_range(0..LOG_STATUS.len())];
            let size = rng.random_range(100..50000);
            let ua = LOG_UAS[rng.random_range(0..LOG_UAS.len())];
            format!(
                r#"{ip} - - [15/Mar/2024:10:{:02}:{:02} +0000] "{method} {path} HTTP/1.1" {status} {size} "-" "{ua}""#,
                rng.random_range(0..60u32),
                rng.random_range(0..60u32),
            )
        })
        .collect()
}

pub fn make_fsst_log_lines(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let lines = generate_log_lines(n);
    let varbin = VarBinArray::from_iter(
        lines.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// JSON strings generator (typical API response payloads)
// ---------------------------------------------------------------------------

const JSON_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack",
];
const JSON_CITIES: &[&str] = &[
    "New York",
    "London",
    "Tokyo",
    "Berlin",
    "Sydney",
    "Toronto",
    "Paris",
    "Mumbai",
    "São Paulo",
    "Seoul",
];
const JSON_TAGS: &[&str] = &[
    "premium",
    "verified",
    "admin",
    "moderator",
    "subscriber",
    "trial",
    "enterprise",
    "developer",
];

pub fn generate_json_strings(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(789);
    (0..n)
        .map(|_| {
            let name = JSON_NAMES[rng.random_range(0..JSON_NAMES.len())];
            let city = JSON_CITIES[rng.random_range(0..JSON_CITIES.len())];
            let age = rng.random_range(18..80u32);
            let tag1 = JSON_TAGS[rng.random_range(0..JSON_TAGS.len())];
            let tag2 = JSON_TAGS[rng.random_range(0..JSON_TAGS.len())];
            let id = rng.random_range(10000..99999u32);
            format!(
                r#"{{"id":{id},"name":"{name}","age":{age},"city":"{city}","tags":["{tag1}","{tag2}"],"active":true}}"#
            )
        })
        .collect()
}

pub fn make_fsst_json_strings(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let jsons = generate_json_strings(n);
    let varbin = VarBinArray::from_iter(
        jsons.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// File paths generator (Unix-style paths with various depths)
// ---------------------------------------------------------------------------

const PATH_ROOTS: &[&str] = &[
    "/home/user",
    "/var/log",
    "/etc",
    "/usr/local/bin",
    "/opt/app",
    "/tmp",
    "/srv/www",
    "/data/warehouse",
];
const PATH_DIRS: &[&str] = &[
    "src",
    "build",
    "dist",
    "node_modules",
    "target/release",
    "config",
    ".cache",
    "logs/2024",
    "backups/daily",
    "migrations",
];
const PATH_FILES: &[&str] = &[
    "main.rs",
    "index.ts",
    "config.yaml",
    "Dockerfile",
    "schema.sql",
    "app.log",
    "data.parquet",
    "model.onnx",
    "README.md",
    "package.json",
];

pub fn generate_file_paths(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(321);
    (0..n)
        .map(|_| {
            let root = PATH_ROOTS[rng.random_range(0..PATH_ROOTS.len())];
            let dir = PATH_DIRS[rng.random_range(0..PATH_DIRS.len())];
            let file = PATH_FILES[rng.random_range(0..PATH_FILES.len())];
            let depth = rng.random_range(0..3u32);
            let mut path = format!("{root}/{dir}");
            for _ in 0..depth {
                let subdir = PATH_DIRS[rng.random_range(0..PATH_DIRS.len())];
                path.push('/');
                path.push_str(subdir);
            }
            path.push('/');
            path.push_str(file);
            path
        })
        .collect()
}

pub fn make_fsst_file_paths(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let paths = generate_file_paths(n);
    let varbin = VarBinArray::from_iter(
        paths.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// Email addresses generator
// ---------------------------------------------------------------------------

const EMAIL_USERS: &[&str] = &[
    "john.doe",
    "jane.smith",
    "admin",
    "support",
    "no-reply",
    "sales.team",
    "dev+test",
    "marketing",
    "info",
    "contact.us",
];
const EMAIL_DOMAINS: &[&str] = &[
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "company.io",
    "example.org",
    "mail.ru",
    "protonmail.com",
    "fastmail.com",
    "icloud.com",
    "hey.com",
];

pub fn generate_emails(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(654);
    (0..n)
        .map(|_| {
            let user = EMAIL_USERS[rng.random_range(0..EMAIL_USERS.len())];
            let domain = EMAIL_DOMAINS[rng.random_range(0..EMAIL_DOMAINS.len())];
            let suffix = rng.random_range(0..1000u32);
            format!("{user}{suffix}@{domain}")
        })
        .collect()
}

pub fn make_fsst_emails(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let emails = generate_emails(n);
    let varbin = VarBinArray::from_iter(
        emails.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}

// ---------------------------------------------------------------------------
// Rare match strings generator
// ---------------------------------------------------------------------------

pub const RARE_NEEDLE: &[u8] = b"xyzzy";

pub fn generate_rare_match_strings(n: usize, match_rate: f64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(999);
    let charset: &[u8] = b"abcdefghijklmnopqrstuvwABCDEFGHIJKLMNOPQRSTUVW0123456789-_.:/";
    (0..n)
        .map(|_| {
            let len = rng.random_range(30..60);
            let mut s: String = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())] as char)
                .collect();
            if rng.random_bool(match_rate) {
                let pos = rng.random_range(0..s.len().saturating_sub(RARE_NEEDLE.len()) + 1);
                s.replace_range(
                    pos..pos + RARE_NEEDLE.len().min(s.len() - pos),
                    std::str::from_utf8(RARE_NEEDLE).unwrap(),
                );
            }
            s
        })
        .collect()
}

pub fn make_fsst_rare_match(n: usize, ctx: &mut ExecutionCtx) -> FSSTArray {
    let strings = generate_rare_match_strings(n, 0.00001);
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor, ctx)
}
