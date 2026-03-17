// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use rand::Rng;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
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

pub fn gen_fsst_test_data(len: usize, avg_str_len: usize, unique_chars: u8) -> ArrayRef {
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

    fsst_compress(varbin, &compressor).into_array()
}

pub fn gen_dict_fsst_test_data<T: NativePType>(
    len: usize,
    unique_values: usize,
    str_len: usize,
    unique_char_count: u8,
) -> DictArray {
    let values = gen_fsst_test_data(len, str_len, unique_char_count);
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

pub fn make_fsst_urls(n: usize) -> FSSTArray {
    let varbin = generate_url_data_n(n);
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_clickbench_urls(n: usize) -> FSSTArray {
    let urls = generate_clickbench_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_short_urls(n: usize) -> FSSTArray {
    let urls = generate_short_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_log_lines(n: usize) -> FSSTArray {
    let lines = generate_log_lines(n);
    let varbin = VarBinArray::from_iter(
        lines.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_json_strings(n: usize) -> FSSTArray {
    let jsons = generate_json_strings(n);
    let varbin = VarBinArray::from_iter(
        jsons.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_file_paths(n: usize) -> FSSTArray {
    let paths = generate_file_paths(n);
    let varbin = VarBinArray::from_iter(
        paths.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_emails(n: usize) -> FSSTArray {
    let emails = generate_emails(n);
    let varbin = VarBinArray::from_iter(
        emails.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
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

pub fn make_fsst_rare_match(n: usize) -> FSSTArray {
    let strings = generate_rare_match_strings(n, 0.00001);
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

// ---------------------------------------------------------------------------
// SQL queries generator (high keyword repetition)
// ---------------------------------------------------------------------------

const SQL_TABLES: &[&str] = &[
    "users",
    "orders",
    "products",
    "customers",
    "invoices",
    "payments",
    "sessions",
    "events",
    "accounts",
    "transactions",
];
const SQL_COLUMNS: &[&str] = &[
    "id",
    "name",
    "email",
    "created_at",
    "updated_at",
    "status",
    "amount",
    "quantity",
    "price",
    "description",
    "category",
    "region",
];
const SQL_STATUSES: &[&str] = &[
    "'active'",
    "'inactive'",
    "'pending'",
    "'completed'",
    "'cancelled'",
];
const SQL_OPERATORS: &[&str] = &["=", ">", "<", ">=", "<=", "!=", "LIKE", "IN"];

pub fn generate_sql_queries(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1001);
    (0..n)
        .map(|_| {
            let kind = rng.random_range(0..3u32);
            let table = SQL_TABLES[rng.random_range(0..SQL_TABLES.len())];
            match kind {
                0 => {
                    // SELECT
                    let ncols = rng.random_range(2..6usize);
                    let cols: Vec<&str> = (0..ncols)
                        .map(|_| SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())])
                        .collect();
                    let nconds = rng.random_range(1..4usize);
                    let mut where_parts = Vec::with_capacity(nconds);
                    for _ in 0..nconds {
                        let col = SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())];
                        let op = SQL_OPERATORS[rng.random_range(0..SQL_OPERATORS.len())];
                        let val = if op == "LIKE" {
                            format!("'%pattern_{}'", rng.random_range(0..100u32))
                        } else if op == "IN" {
                            format!(
                                "({}, {}, {})",
                                rng.random_range(1..1000u32),
                                rng.random_range(1..1000u32),
                                rng.random_range(1..1000u32)
                            )
                        } else {
                            let status = SQL_STATUSES[rng.random_range(0..SQL_STATUSES.len())];
                            status.to_string()
                        };
                        where_parts.push(format!("{col} {op} {val}"));
                    }
                    let order_col = SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())];
                    let dir = if rng.random_bool(0.5) { "ASC" } else { "DESC" };
                    let limit = rng.random_range(10..1000u32);
                    format!(
                        "SELECT {} FROM {} WHERE {} ORDER BY {} {} LIMIT {}",
                        cols.join(", "),
                        table,
                        where_parts.join(" AND "),
                        order_col,
                        dir,
                        limit,
                    )
                }
                1 => {
                    // INSERT
                    let ncols = rng.random_range(3..7usize);
                    let cols: Vec<&str> = (0..ncols)
                        .map(|_| SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())])
                        .collect();
                    let vals: Vec<String> = (0..ncols)
                        .map(|_| {
                            if rng.random_bool(0.5) {
                                format!(
                                    "'{}'",
                                    SQL_STATUSES[rng.random_range(0..SQL_STATUSES.len())]
                                )
                            } else {
                                rng.random_range(1..100000u32).to_string()
                            }
                        })
                        .collect();
                    format!(
                        "INSERT INTO {} ({}) VALUES ({})",
                        table,
                        cols.join(", "),
                        vals.join(", "),
                    )
                }
                _ => {
                    // UPDATE
                    let nsets = rng.random_range(1..4usize);
                    let set_parts: Vec<String> = (0..nsets)
                        .map(|_| {
                            let col = SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())];
                            let val = SQL_STATUSES[rng.random_range(0..SQL_STATUSES.len())];
                            format!("{col} = {val}")
                        })
                        .collect();
                    let cond_col = SQL_COLUMNS[rng.random_range(0..SQL_COLUMNS.len())];
                    let cond_val = rng.random_range(1..100000u32);
                    format!(
                        "UPDATE {} SET {} WHERE {} = {}",
                        table,
                        set_parts.join(", "),
                        cond_col,
                        cond_val,
                    )
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// XML fragments generator (nested tags, attributes, namespaces)
// ---------------------------------------------------------------------------

const XML_TAGS: &[&str] = &[
    "record", "entry", "item", "element", "node", "data", "field", "row", "value", "property",
];
const XML_NAMESPACES: &[&str] = &["ns1", "ns2", "xsi", "xsd", "soap", "app"];
const XML_ATTRS: &[&str] = &[
    "id", "type", "name", "class", "version", "status", "priority", "lang", "encoding", "format",
];
const XML_ATTR_VALUES: &[&str] = &[
    "primary",
    "secondary",
    "active",
    "deprecated",
    "1.0",
    "2.0",
    "utf-8",
    "json",
    "xml",
    "default",
];

pub fn generate_xml_fragments(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1002);
    (0..n)
        .map(|_| {
            let mut xml = String::with_capacity(256);
            let depth = rng.random_range(2..5usize);
            let mut tags_stack = Vec::with_capacity(depth);
            for _ in 0..depth {
                let ns = XML_NAMESPACES[rng.random_range(0..XML_NAMESPACES.len())];
                let tag = XML_TAGS[rng.random_range(0..XML_TAGS.len())];
                let nattrs = rng.random_range(1..4usize);
                xml.push('<');
                xml.push_str(ns);
                xml.push(':');
                xml.push_str(tag);
                for _ in 0..nattrs {
                    let attr = XML_ATTRS[rng.random_range(0..XML_ATTRS.len())];
                    let val = XML_ATTR_VALUES[rng.random_range(0..XML_ATTR_VALUES.len())];
                    xml.push(' ');
                    xml.push_str(attr);
                    xml.push_str("=\"");
                    xml.push_str(val);
                    xml.push('"');
                }
                xml.push('>');
                tags_stack.push(format!("{ns}:{tag}"));
            }
            // inner text content
            let inner_id = rng.random_range(1000..99999u32);
            xml.push_str(&format!("value_{inner_id}"));
            // close tags in reverse
            for qualified in tags_stack.iter().rev() {
                xml.push_str("</");
                xml.push_str(qualified);
                xml.push('>');
            }
            xml
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Repeated binary patterns generator
// ---------------------------------------------------------------------------

pub fn generate_repeated_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(1003);
    // Pre-generate a pool of patterns
    let num_patterns = rng.random_range(4..7usize);
    let patterns: Vec<Vec<u8>> = (0..num_patterns)
        .map(|_| {
            let len = rng.random_range(8..33usize);
            (0..len).map(|_| rng.random::<u8>()).collect()
        })
        .collect();

    (0..n)
        .map(|_| {
            let num_segments = rng.random_range(3..9usize);
            let mut record = Vec::with_capacity(256);
            for _ in 0..num_segments {
                let pattern = &patterns[rng.random_range(0..patterns.len())];
                record.extend_from_slice(pattern);
                // small random separator (1-4 bytes)
                let sep_len = rng.random_range(1..5usize);
                for _ in 0..sep_len {
                    record.push(rng.random::<u8>());
                }
            }
            record
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CSV rows generator (fixed schema with repeated categorical values)
// ---------------------------------------------------------------------------

const CSV_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Karen",
    "Leo", "Mia", "Nathan", "Olivia",
];
const CSV_CITIES: &[&str] = &[
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
];
const CSV_STATUSES: &[&str] = &["active", "inactive", "pending", "suspended", "trial"];

pub fn generate_csv_rows(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1004);
    (0..n)
        .map(|_| {
            let name = CSV_NAMES[rng.random_range(0..CSV_NAMES.len())];
            let age = rng.random_range(18..75u32);
            let city = CSV_CITIES[rng.random_range(0..CSV_CITIES.len())];
            let status = CSV_STATUSES[rng.random_range(0..CSV_STATUSES.len())];
            let score = rng.random_range(0..1000u32);
            format!("{name},{age},{city},{status},{score}")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Key-value config lines generator (shared prefixes)
// ---------------------------------------------------------------------------

const CONFIG_SECTIONS: &[&str] = &[
    "database",
    "server",
    "logging",
    "cache",
    "security",
    "messaging",
    "monitoring",
];
const CONFIG_SUBSECTIONS: &[&str] = &[
    "connection",
    "pool",
    "timeout",
    "retry",
    "buffer",
    "auth",
    "tls",
    "metrics",
];
const CONFIG_KEYS: &[&str] = &[
    "max_size",
    "min_size",
    "idle_timeout",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "enabled",
    "level",
    "host",
    "port",
    "interval",
    "threshold",
];
const CONFIG_VALUES: &[&str] = &[
    "true",
    "false",
    "10",
    "50",
    "100",
    "256",
    "1024",
    "5000",
    "30000",
    "localhost",
    "0.0.0.0",
    "/var/log/app.log",
    "INFO",
    "DEBUG",
    "WARN",
];

pub fn generate_key_value_config(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1005);
    (0..n)
        .map(|_| {
            let section = CONFIG_SECTIONS[rng.random_range(0..CONFIG_SECTIONS.len())];
            let subsection = CONFIG_SUBSECTIONS[rng.random_range(0..CONFIG_SUBSECTIONS.len())];
            let key = CONFIG_KEYS[rng.random_range(0..CONFIG_KEYS.len())];
            let value = CONFIG_VALUES[rng.random_range(0..CONFIG_VALUES.len())];
            format!("{section}.{subsection}.{key} = {value}")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Timestamps with prefix generator (highly repetitive prefix pattern)
// ---------------------------------------------------------------------------

const TS_LEVELS: &[&str] = &["INFO", "DEBUG", "WARN", "ERROR", "TRACE"];
const TS_SERVICES: &[&str] = &[
    "api-gateway",
    "user-service",
    "order-service",
    "payment-service",
    "auth-service",
    "notification-service",
];
const TS_MESSAGES: &[&str] = &[
    "Request processed successfully",
    "Connection established to upstream",
    "Cache miss for key lookup",
    "Retrying failed operation attempt",
    "Rate limit threshold exceeded",
    "Health check passed",
    "Configuration reloaded from disk",
    "Session token refreshed",
    "Background job completed",
    "Metric batch flushed to sink",
    "Circuit breaker state changed",
    "Graceful shutdown initiated",
];

pub fn generate_timestamps_with_prefix(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1006);
    (0..n)
        .map(|_| {
            let month = rng.random_range(1..13u32);
            let day = rng.random_range(1..29u32);
            let hour = rng.random_range(0..24u32);
            let minute = rng.random_range(0..60u32);
            let second = rng.random_range(0..60u32);
            let millis = rng.random_range(0..1000u32);
            let level = TS_LEVELS[rng.random_range(0..TS_LEVELS.len())];
            let service = TS_SERVICES[rng.random_range(0..TS_SERVICES.len())];
            let message = TS_MESSAGES[rng.random_range(0..TS_MESSAGES.len())];
            let req_id = rng.random_range(10000..99999u32);
            format!(
                "2024-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z {level} [{service}] {message} req_id={req_id}"
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HTTP headers generator (request/response header lines)
// ---------------------------------------------------------------------------

const HTTP_HEADER_NAMES: &[&str] = &[
    "Content-Type",
    "Accept-Encoding",
    "X-Request-Id",
    "Authorization",
    "Cache-Control",
    "Content-Length",
    "X-Forwarded-For",
    "User-Agent",
    "Accept",
    "Host",
    "Connection",
    "X-Correlation-Id",
    "Set-Cookie",
    "ETag",
    "Vary",
];

const HTTP_HEADER_VALUES: &[(&str, &[&str])] = &[
    (
        "Content-Type",
        &[
            "application/json",
            "text/html; charset=utf-8",
            "application/xml",
            "text/plain",
            "application/octet-stream",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
            "application/grpc",
            "image/png",
            "text/css",
        ],
    ),
    (
        "Accept-Encoding",
        &[
            "gzip, deflate",
            "gzip, deflate, br",
            "br",
            "identity",
            "gzip",
            "deflate",
            "zstd, gzip, deflate",
            "gzip, br",
            "*",
            "compress",
        ],
    ),
    (
        "Cache-Control",
        &[
            "no-cache",
            "no-store",
            "max-age=3600",
            "max-age=86400",
            "public, max-age=31536000",
            "private, no-cache",
            "must-revalidate",
            "no-cache, no-store, must-revalidate",
            "s-maxage=600",
            "max-age=0",
        ],
    ),
    (
        "Accept",
        &[
            "application/json",
            "text/html",
            "*/*",
            "application/xml",
            "text/plain",
            "application/json, text/plain, */*",
            "text/html, application/xhtml+xml",
            "image/webp, image/apng, */*",
            "application/signed-exchange",
            "application/grpc",
        ],
    ),
    (
        "Connection",
        &[
            "keep-alive",
            "close",
            "upgrade",
            "keep-alive",
            "keep-alive",
            "close",
            "keep-alive",
            "keep-alive",
            "close",
            "keep-alive",
        ],
    ),
    (
        "Vary",
        &[
            "Accept-Encoding",
            "Accept",
            "Origin",
            "Accept-Encoding, Accept",
            "Accept-Encoding, Origin",
            "Cookie",
            "Accept-Language",
            "User-Agent",
            "Accept-Encoding, Accept-Language",
            "Origin, Accept",
        ],
    ),
];

/// Generates HTTP request/response header lines like `"Content-Type: application/json"`.
pub fn generate_http_headers(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2001);

    let generic_values: &[&str] = &[
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "192.168.1.42",
        "api.example.com",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
        "session_id=abc123def456; Path=/; HttpOnly",
        "W/\"5e8c4a6b\"",
        "1024",
        "4096",
        "keep-alive",
        "en-US,en;q=0.9",
    ];

    (0..n)
        .map(|_| {
            let header_name = HTTP_HEADER_NAMES[rng.random_range(0..HTTP_HEADER_NAMES.len())];

            // Try to find a specific value pool for this header
            let value = if let Some((_, vals)) = HTTP_HEADER_VALUES
                .iter()
                .find(|(name, _)| *name == header_name)
            {
                vals[rng.random_range(0..vals.len())].to_string()
            } else {
                // For headers without specific pools, generate contextual values
                match header_name {
                    "X-Request-Id" | "X-Correlation-Id" => {
                        format!(
                            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                            rng.random_range(0..u32::MAX),
                            rng.random_range(0..u16::MAX),
                            rng.random_range(0..u16::MAX),
                            rng.random_range(0..u16::MAX),
                            rng.random_range(0..u64::MAX) & 0xFFFF_FFFF_FFFF,
                        )
                    }
                    "Content-Length" => {
                        format!("{}", rng.random_range(0..1_000_000u32))
                    }
                    "X-Forwarded-For" => {
                        format!(
                            "{}.{}.{}.{}",
                            rng.random_range(1..255u32),
                            rng.random_range(0..256u32),
                            rng.random_range(0..256u32),
                            rng.random_range(1..255u32),
                        )
                    }
                    _ => generic_values[rng.random_range(0..generic_values.len())].to_string(),
                }
            };

            format!("{header_name}: {value}")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// IPv4 CIDR firewall/ACL rules generator
// ---------------------------------------------------------------------------

const CIDR_ACTIONS: &[&str] = &["allow", "deny"];
const CIDR_PROTOCOLS: &[&str] = &["tcp", "udp", "icmp"];
const CIDR_PORTS: &[u16] = &[
    22, 53, 80, 443, 993, 1433, 3306, 3389, 5432, 6379, 8080, 8443,
];
const CIDR_DEST_RANGES: &[&str] = &[
    "10.0.0.0/8",
    "10.0.1.0/24",
    "10.1.0.0/16",
    "10.10.0.0/16",
    "172.16.0.0/12",
    "172.16.1.0/24",
    "172.20.0.0/16",
    "192.168.0.0/16",
    "192.168.1.0/24",
    "192.168.10.0/24",
    "192.168.100.0/24",
    "0.0.0.0/0",
];
const CIDR_SRC_RANGES: &[&str] = &[
    "192.168.1.0/24",
    "192.168.0.0/16",
    "10.0.0.0/8",
    "10.0.1.0/24",
    "172.16.0.0/12",
    "172.16.5.0/24",
    "203.0.113.0/24",
    "198.51.100.0/24",
    "any",
    "any",
    "any",
];

/// Generates firewall/ACL rules like `"allow 10.0.0.0/8 tcp 443 from 192.168.1.0/24"`.
pub fn generate_ipv4_cidr_rules(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2002);
    (0..n)
        .map(|_| {
            let action = CIDR_ACTIONS[rng.random_range(0..CIDR_ACTIONS.len())];
            let dest = CIDR_DEST_RANGES[rng.random_range(0..CIDR_DEST_RANGES.len())];
            let proto = CIDR_PROTOCOLS[rng.random_range(0..CIDR_PROTOCOLS.len())];
            let src = CIDR_SRC_RANGES[rng.random_range(0..CIDR_SRC_RANGES.len())];

            if proto == "icmp" {
                format!("{action} {dest} {proto} from {src}")
            } else {
                let port = CIDR_PORTS[rng.random_range(0..CIDR_PORTS.len())];
                format!("{action} {dest} {proto} {port} from {src}")
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Prometheus exposition format metrics generator
// ---------------------------------------------------------------------------

const PROM_METRIC_NAMES: &[&str] = &[
    "http_requests_total",
    "http_request_duration_seconds",
    "process_cpu_seconds_total",
    "process_resident_memory_bytes",
    "go_goroutines",
    "node_cpu_seconds_total",
    "up",
    "scrape_duration_seconds",
    "grpc_server_handled_total",
    "api_errors_total",
];

const PROM_LABEL_KEYS: &[&str] = &[
    "method", "handler", "status", "instance", "job", "mode", "le", "quantile",
];

const PROM_LABEL_VALUES: &[&str] = &[
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "/api/v1/users",
    "/api/v1/orders",
    "/api/v2/products",
    "/healthz",
    "/metrics",
    "200",
    "201",
    "404",
    "500",
    "localhost:9090",
    "prometheus",
    "node-exporter",
    "idle",
    "user",
    "system",
    "0.5",
    "0.9",
    "0.99",
];

/// Generates Prometheus exposition format lines like
/// `http_requests_total{method="GET",handler="/api/v1/users",status="200"} 1234 1678901234`.
pub fn generate_prometheus_metrics(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2003);
    (0..n)
        .map(|_| {
            let metric = PROM_METRIC_NAMES[rng.random_range(0..PROM_METRIC_NAMES.len())];
            let num_labels = rng.random_range(1..5usize);

            let mut labels = Vec::with_capacity(num_labels);
            for _ in 0..num_labels {
                let key = PROM_LABEL_KEYS[rng.random_range(0..PROM_LABEL_KEYS.len())];
                let val = PROM_LABEL_VALUES[rng.random_range(0..PROM_LABEL_VALUES.len())];
                labels.push(format!("{key}=\"{val}\""));
            }

            let value = rng.random_range(0..1_000_000u64);
            let timestamp = rng.random_range(1_678_000_000..1_710_000_000u64);

            format!(
                "{metric}{{{labels}}} {value} {timestamp}",
                labels = labels.join(",")
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DNS wire format binary generator (simplified)
// ---------------------------------------------------------------------------

const DNS_DOMAIN_COMPONENTS: &[&str] = &[
    "www",
    "mail",
    "ns1",
    "ns2",
    "api",
    "cdn",
    "static",
    "img",
    "docs",
    "blog",
    "google",
    "yahoo",
    "amazon",
    "cloudflare",
    "github",
    "example",
    "com",
    "net",
    "org",
    "io",
];

const DNS_QUERY_TYPES: &[u16] = &[
    1,  // A
    2,  // NS
    5,  // CNAME
    15, // MX
    16, // TXT
    28, // AAAA
];

/// Generates simplified DNS wire format packets.
///
/// Each packet contains: 2-byte transaction ID, 2-byte flags (`0x0100` for query
/// or `0x8180` for response), 2-byte question count, then one or more questions
/// encoded as length-prefixed labels (e.g., `[3]www[6]google[3]com[0]`) followed
/// by 2-byte type and 2-byte class.
pub fn generate_dns_wire_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(2004);
    (0..n)
        .map(|_| {
            let mut pkt = Vec::with_capacity(64);

            // Transaction ID (2 bytes)
            let txn_id: u16 = rng.random_range(0..u16::MAX);
            pkt.extend_from_slice(&txn_id.to_be_bytes());

            // Flags (2 bytes): query or response
            let flags: u16 = if rng.random_bool(0.6) { 0x0100 } else { 0x8180 };
            pkt.extend_from_slice(&flags.to_be_bytes());

            // Question count (2 bytes)
            let qcount: u16 = rng.random_range(1..4);
            pkt.extend_from_slice(&qcount.to_be_bytes());

            // Encode each question
            for _ in 0..qcount {
                // Build domain name from 2-4 random components
                let num_labels = rng.random_range(2..5usize);
                for _ in 0..num_labels {
                    let component =
                        DNS_DOMAIN_COMPONENTS[rng.random_range(0..DNS_DOMAIN_COMPONENTS.len())];
                    let label_bytes = component.as_bytes();
                    pkt.push(u8::try_from(label_bytes.len()).unwrap());
                    pkt.extend_from_slice(label_bytes);
                }
                // Null terminator for domain name
                pkt.push(0);

                // Query type (2 bytes)
                let qtype = DNS_QUERY_TYPES[rng.random_range(0..DNS_QUERY_TYPES.len())];
                pkt.extend_from_slice(&qtype.to_be_bytes());

                // Query class (2 bytes): IN = 1
                pkt.extend_from_slice(&1u16.to_be_bytes());
            }

            pkt
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Parquet-like column metadata binary generator
// ---------------------------------------------------------------------------

const PARQUET_COL_NAMES: &[&str] = &[
    "user_id",
    "timestamp",
    "event_type",
    "amount",
    "currency",
    "session_id",
    "page_url",
    "referrer",
    "duration_ms",
    "status_code",
    "country",
    "device_type",
    "browser",
    "ip_address",
    "payload",
];

/// Generates simplified Parquet-like column metadata binary records.
///
/// Each record contains: 4-byte magic `"PAR1"`, then 3-8 column chunks where each
/// chunk has a 1-byte type tag (0-6), 4-byte LE offset, 4-byte LE size, and a
/// length-prefixed column name drawn from a pool of ~15 analytics column names.
pub fn generate_parquet_footer_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(3001);
    (0..n)
        .map(|_| {
            let mut buf = Vec::with_capacity(128);
            // 4-byte magic
            buf.extend_from_slice(b"PAR1");
            let num_chunks = rng.random_range(3..9usize);
            for _ in 0..num_chunks {
                // 1-byte type tag (0=bool, 1=int32, 2=int64, 3=float, 4=double, 5=string, 6=binary)
                buf.push(rng.random_range(0..7u8));
                // 4-byte LE offset
                let offset: u32 = rng.random_range(0..1_000_000);
                buf.extend_from_slice(&offset.to_le_bytes());
                // 4-byte LE size
                let size: u32 = rng.random_range(64..65536);
                buf.extend_from_slice(&size.to_le_bytes());
                // length-prefixed column name
                let name = PARQUET_COL_NAMES[rng.random_range(0..PARQUET_COL_NAMES.len())];
                let name_bytes = name.as_bytes();
                #[allow(clippy::cast_possible_truncation)]
                buf.push(name_bytes.len() as u8); // column names are all <256 bytes
                buf.extend_from_slice(name_bytes);
            }
            buf
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Spark query plan strings generator
// ---------------------------------------------------------------------------

const SPARK_OPERATORS: &[&str] = &[
    "HashAggregate",
    "Sort",
    "Exchange",
    "Project",
    "Filter",
    "FileScan parquet",
    "BroadcastHashJoin",
    "ShuffledHashJoin",
    "SortMergeJoin",
    "Window",
];

const SPARK_COLUMNS: &[&str] = &[
    "user_id",
    "amount",
    "timestamp",
    "event_type",
    "session_id",
    "country",
    "product_id",
    "quantity",
    "price",
    "category",
    "order_id",
    "status",
];

const SPARK_AGG_FUNCTIONS: &[&str] = &[
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "first",
    "collect_list",
    "approx_count_distinct",
];

/// Generates Spark-style physical query plan fragments.
///
/// Each plan starts with `"== Physical Plan =="` followed by 3-6 operators at
/// increasing indentation with `+- ` connectors. Operators include `HashAggregate`,
/// `Sort`, `Exchange`, `FileScan parquet`, etc., with column references in `name#id`
/// notation.
pub fn generate_spark_plan_strings(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(3002);
    (0..n)
        .map(|_| {
            let mut plan = String::with_capacity(256);
            plan.push_str("== Physical Plan ==\n");
            let depth = rng.random_range(3..7usize);
            for level in 0..depth {
                if level > 0 {
                    for _ in 0..level {
                        plan.push_str("   ");
                    }
                    plan.push_str("+- ");
                }
                let op = SPARK_OPERATORS[rng.random_range(0..SPARK_OPERATORS.len())];
                plan.push_str(op);
                let ncols = rng.random_range(1..4usize);
                plan.push('(');
                match op {
                    "HashAggregate" => {
                        plan.push_str("keys=[");
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let col = SPARK_COLUMNS[rng.random_range(0..SPARK_COLUMNS.len())];
                            let col_id = rng.random_range(100..999u32);
                            plan.push_str(&format!("{col}#{col_id}"));
                        }
                        plan.push_str("], functions=[");
                        let nfuncs = rng.random_range(1..3usize);
                        for i in 0..nfuncs {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let func =
                                SPARK_AGG_FUNCTIONS[rng.random_range(0..SPARK_AGG_FUNCTIONS.len())];
                            let col = SPARK_COLUMNS[rng.random_range(0..SPARK_COLUMNS.len())];
                            let col_id = rng.random_range(100..999u32);
                            plan.push_str(&format!("{func}({col}#{col_id})"));
                        }
                        plan.push(']');
                    }
                    "Exchange" => {
                        let col = SPARK_COLUMNS[rng.random_range(0..SPARK_COLUMNS.len())];
                        let col_id = rng.random_range(100..999u32);
                        let partitions = rng.random_range(50..400u32);
                        plan.push_str(&format!("hashpartitioning({col}#{col_id}, {partitions})"));
                    }
                    "FileScan parquet" => {
                        plan.push('[');
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push(',');
                            }
                            let col = SPARK_COLUMNS[rng.random_range(0..SPARK_COLUMNS.len())];
                            let col_id = rng.random_range(100..999u32);
                            plan.push_str(&format!("{col}#{col_id}"));
                        }
                        plan.push(']');
                    }
                    _ => {
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let col = SPARK_COLUMNS[rng.random_range(0..SPARK_COLUMNS.len())];
                            let col_id = rng.random_range(100..999u32);
                            plan.push_str(&format!("{col}#{col_id}"));
                        }
                    }
                }
                plan.push(')');
                if level < depth - 1 {
                    plan.push('\n');
                }
            }
            plan
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Arrow IPC-like binary generator
// ---------------------------------------------------------------------------

const ARROW_FIELD_NAMES: &[&str] = &[
    "id",
    "name",
    "timestamp",
    "value",
    "label",
    "payload",
    "score",
    "count",
    "flag",
    "data",
    "key",
    "index",
];

/// Generates simplified Arrow IPC-like binary records.
///
/// Each record contains: 4-byte continuation marker `0xFFFFFFFF`, 4-byte LE metadata
/// length, then flatbuffer-style field descriptors (1-byte type, 2-byte name offset,
/// 1-byte nullable flag) for 3-8 fields, followed by a 4-byte LE body length.
pub fn generate_arrow_ipc_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(3003);
    (0..n)
        .map(|_| {
            let mut buf = Vec::with_capacity(128);
            // 4-byte continuation marker
            buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
            // Placeholder for metadata length
            let meta_len_pos = buf.len();
            buf.extend_from_slice(&0u32.to_le_bytes());
            let meta_start = buf.len();
            let num_fields = rng.random_range(3..9usize);
            for _ in 0..num_fields {
                // 1-byte type (0=null, 1=int, 2=float, 3=utf8, 4=binary)
                buf.push(rng.random_range(0..5u8));
                // 2-byte name_offset
                let name_idx = rng.random_range(0..ARROW_FIELD_NAMES.len());
                #[allow(clippy::cast_possible_truncation)]
                let name_offset = (name_idx * 16) as u16; // 12 names * 16 fits in u16
                buf.extend_from_slice(&name_offset.to_le_bytes());
                // 1-byte nullable flag
                buf.push(if rng.random_bool(0.3) { 1 } else { 0 });
            }
            // Patch metadata length
            #[allow(clippy::cast_possible_truncation)]
            let meta_len = (buf.len() - meta_start) as u32; // metadata is at most ~32 bytes
            buf[meta_len_pos..meta_len_pos + 4].copy_from_slice(&meta_len.to_le_bytes());
            // 4-byte body length
            let body_len: u32 = rng.random_range(1024..1_048_576);
            buf.extend_from_slice(&body_len.to_le_bytes());
            buf
        })
        .collect()
}

// ---------------------------------------------------------------------------
// JSONL (JSON Lines) generator
// ---------------------------------------------------------------------------

const JSONL_EVENT_TYPES: &[&str] = &[
    "page_view",
    "click",
    "purchase",
    "signup",
    "logout",
    "search",
    "add_to_cart",
    "error",
];

const JSONL_PROP_KEYS: &[&str] = &[
    "page",
    "ref",
    "device",
    "browser",
    "os",
    "campaign",
    "source",
    "medium",
    "duration",
    "scroll_depth",
];

const JSONL_PROP_VALUES: &[&str] = &[
    "/home",
    "/products",
    "/checkout",
    "/search",
    "/profile",
    "google",
    "facebook",
    "twitter",
    "email",
    "direct",
    "mobile",
    "desktop",
    "tablet",
    "chrome",
    "firefox",
];

/// Generates JSONL (one JSON object per line) records resembling event tracking data.
///
/// Each line is a complete JSON object with an event type, user ID, timestamp,
/// a nested `"props"` object with 3-6 key-value pairs, and 0-3 additional top-level
/// fields (`amount`, `success`, `v`).
pub fn generate_json_lines(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(3004);
    (0..n)
        .map(|_| {
            let mut line = String::with_capacity(256);
            line.push('{');
            let event = JSONL_EVENT_TYPES[rng.random_range(0..JSONL_EVENT_TYPES.len())];
            line.push_str(&format!("\"event\":\"{event}\""));
            let user_id = rng.random_range(10000..99999u32);
            line.push_str(&format!(",\"user\":\"u_{user_id}\""));
            let ts = rng.random_range(1_678_000_000..1_700_000_000u64);
            line.push_str(&format!(",\"ts\":{ts}"));
            let nprops = rng.random_range(3..7usize);
            line.push_str(",\"props\":{");
            for i in 0..nprops {
                if i > 0 {
                    line.push(',');
                }
                let key = JSONL_PROP_KEYS[rng.random_range(0..JSONL_PROP_KEYS.len())];
                let val = JSONL_PROP_VALUES[rng.random_range(0..JSONL_PROP_VALUES.len())];
                line.push_str(&format!("\"{key}\":\"{val}\""));
            }
            line.push('}');
            let extra_fields = rng.random_range(0..4usize);
            if extra_fields > 0 {
                let amount = rng.random_range(1..10000u32);
                line.push_str(&format!(",\"amount\":{amount}"));
            }
            if extra_fields > 1 {
                let success = if rng.random_bool(0.8) {
                    "true"
                } else {
                    "false"
                };
                line.push_str(&format!(",\"success\":{success}"));
            }
            if extra_fields > 2 {
                let version = rng.random_range(1..10u32);
                line.push_str(&format!(",\"v\":{version}"));
            }
            line.push('}');
            line
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Markdown fragments generator (headers, lists, code blocks, links)
// ---------------------------------------------------------------------------

const MD_HEADER_WORDS: &[&str] = &[
    "Introduction",
    "Overview",
    "Architecture",
    "Configuration",
    "Deployment",
    "Performance",
    "Security",
    "Troubleshooting",
    "Reference",
    "Migration",
    "API",
    "Guide",
    "Tutorial",
    "Quickstart",
    "Advanced",
];
const MD_BODY_WORDS: &[&str] = &[
    "the",
    "system",
    "processes",
    "incoming",
    "requests",
    "using",
    "configured",
    "middleware",
    "pipeline",
    "ensure",
    "proper",
    "authentication",
    "before",
    "forwarding",
    "to",
    "upstream",
    "services",
    "with",
    "retries",
    "enabled",
];
const MD_CODE_LANGS: &[&str] = &["rust", "python", "javascript", "go"];
const MD_LINK_TITLES: &[&str] = &[
    "documentation",
    "source code",
    "issue tracker",
    "release notes",
    "changelog",
    "contributing guide",
    "license",
    "examples",
    "benchmarks",
    "FAQ",
];
const MD_LINK_URLS: &[&str] = &[
    "https://docs.example.com/guide",
    "https://github.com/org/repo",
    "https://github.com/org/repo/issues",
    "https://example.com/releases/v2.0",
    "https://wiki.example.com/setup",
    "https://docs.example.com/api/reference",
    "https://example.com/blog/best-practices",
    "https://crates.io/crates/example",
];
const MD_CODE_SNIPPETS: &[&str] = &[
    "let config = Config::from_env();",
    "def handle_request(req):",
    "const server = createServer(handler);",
    "func main() { log.Println(\"starting\") }",
    "fn process(input: &[u8]) -> Result<()>",
    "import asyncio",
    "export default function App() {}",
    "ctx, cancel := context.WithTimeout(ctx, 5*time.Second)",
];

/// Generates markdown text fragments with headers, bullet lists, code blocks, and links.
pub fn generate_markdown_fragments(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4001);
    (0..n)
        .map(|_| {
            let num_lines = rng.random_range(2..6usize);
            let mut lines = Vec::with_capacity(num_lines);
            for i in 0..num_lines {
                let kind = if i == 0 {
                    0 // always start with a header
                } else {
                    rng.random_range(1..4u32)
                };
                match kind {
                    0 => {
                        let depth = rng.random_range(1..4usize);
                        let hashes: String = "#".repeat(depth);
                        let w1 = MD_HEADER_WORDS[rng.random_range(0..MD_HEADER_WORDS.len())];
                        let w2 = MD_HEADER_WORDS[rng.random_range(0..MD_HEADER_WORDS.len())];
                        lines.push(format!("{hashes} {w1} {w2}"));
                    }
                    1 => {
                        let nwords = rng.random_range(4..8usize);
                        let words: Vec<&str> = (0..nwords)
                            .map(|_| MD_BODY_WORDS[rng.random_range(0..MD_BODY_WORDS.len())])
                            .collect();
                        lines.push(format!("- {}", words.join(" ")));
                    }
                    2 => {
                        let lang = MD_CODE_LANGS[rng.random_range(0..MD_CODE_LANGS.len())];
                        let snippet = MD_CODE_SNIPPETS[rng.random_range(0..MD_CODE_SNIPPETS.len())];
                        lines.push(format!("```{lang}\n{snippet}\n```"));
                    }
                    _ => {
                        let prefix_words: Vec<&str> = (0..rng.random_range(2..5usize))
                            .map(|_| MD_BODY_WORDS[rng.random_range(0..MD_BODY_WORDS.len())])
                            .collect();
                        let title = MD_LINK_TITLES[rng.random_range(0..MD_LINK_TITLES.len())];
                        let url = MD_LINK_URLS[rng.random_range(0..MD_LINK_URLS.len())];
                        lines.push(format!("{} [{title}]({url})", prefix_words.join(" ")));
                    }
                }
            }
            lines.join("\n")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stack traces generator (Java/Python-style)
// ---------------------------------------------------------------------------

const ST_JAVA_PACKAGES: &[&str] = &[
    "com.example.service",
    "com.example.controller",
    "com.example.repository",
    "com.example.config",
    "org.springframework.web",
    "org.springframework.security",
    "org.apache.catalina.core",
    "org.apache.tomcat.util",
    "io.netty.channel",
    "io.grpc.internal",
    "com.google.common.util",
    "com.zaxxer.hikari",
];
const ST_CLASSES: &[&str] = &[
    "UserService",
    "OrderController",
    "PaymentGateway",
    "AuthFilter",
    "SessionManager",
    "CacheProvider",
    "DatabasePool",
    "RequestHandler",
    "MessageBroker",
    "ConfigLoader",
    "MetricsCollector",
    "HealthChecker",
    "RateLimiter",
    "CircuitBreaker",
    "EventDispatcher",
];
const ST_METHODS: &[&str] = &[
    "getUser",
    "processOrder",
    "validateToken",
    "handleRequest",
    "executeQuery",
    "sendMessage",
    "loadConfig",
    "checkHealth",
    "refreshCache",
    "authenticate",
    "serialize",
    "dispatch",
    "connect",
    "initialize",
    "shutdown",
    "retry",
    "transform",
    "aggregate",
    "publish",
    "subscribe",
];
const ST_PYTHON_MODULES: &[&str] = &[
    "/app/handlers/auth.py",
    "/app/handlers/api.py",
    "/app/models/user.py",
    "/app/services/payment.py",
    "/app/middleware/logging.py",
    "/app/utils/crypto.py",
    "/app/core/config.py",
    "/app/db/session.py",
    "/usr/lib/python3.11/asyncio/tasks.py",
    "/usr/lib/python3.11/concurrent/futures/thread.py",
    "/site-packages/fastapi/routing.py",
    "/site-packages/sqlalchemy/engine/base.py",
];

/// Generates Java/Python-style stack traces with package paths, class names, and line numbers.
pub fn generate_stack_traces(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4002);
    (0..n)
        .map(|_| {
            let is_java = rng.random_bool(0.5);
            let depth = rng.random_range(3..7usize);
            let mut frames = Vec::with_capacity(depth);
            for _ in 0..depth {
                if is_java {
                    let pkg = ST_JAVA_PACKAGES[rng.random_range(0..ST_JAVA_PACKAGES.len())];
                    let class = ST_CLASSES[rng.random_range(0..ST_CLASSES.len())];
                    let method = ST_METHODS[rng.random_range(0..ST_METHODS.len())];
                    let line = rng.random_range(10..500u32);
                    frames.push(format!("\tat {pkg}.{class}.{method}({class}.java:{line})"));
                } else {
                    let module = ST_PYTHON_MODULES[rng.random_range(0..ST_PYTHON_MODULES.len())];
                    let method = ST_METHODS[rng.random_range(0..ST_METHODS.len())];
                    let line = rng.random_range(10..500u32);
                    frames.push(format!("  File \"{module}\", line {line}, in {method}"));
                }
            }
            if is_java {
                let class = ST_CLASSES[rng.random_range(0..ST_CLASSES.len())];
                format!(
                    "java.lang.RuntimeException: Operation failed in {class}\n{}",
                    frames.join("\n")
                )
            } else {
                let class = ST_CLASSES[rng.random_range(0..ST_CLASSES.len())];
                format!(
                    "Traceback (most recent call last):\n{}\n{class}Error: operation failed",
                    frames.join("\n")
                )
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CSS rules generator (selectors, properties, values)
// ---------------------------------------------------------------------------

const CSS_SELECTORS: &[&str] = &[
    ".container",
    "#main",
    ".btn-primary",
    "nav > ul",
    ".sidebar",
    ".card",
    "header",
    "footer",
    ".modal-overlay",
    ".form-group",
    ".text-muted",
    "section > div",
    ".grid-item",
    ".dropdown-menu",
    "input[type=\"text\"]",
];
const CSS_PROPERTIES: &[&str] = &[
    "display",
    "justify-content",
    "align-items",
    "padding",
    "margin",
    "border",
    "background-color",
    "color",
    "font-size",
    "font-weight",
    "width",
    "height",
    "max-width",
    "position",
    "overflow",
    "gap",
    "border-radius",
    "box-shadow",
    "opacity",
    "z-index",
];
const CSS_VALUES_DISPLAY: &[&str] = &["flex", "grid", "block", "inline-block", "none"];
const CSS_VALUES_JUSTIFY: &[&str] = &[
    "center",
    "flex-start",
    "flex-end",
    "space-between",
    "space-around",
];
const CSS_VALUES_SPACING: &[&str] = &[
    "0", "4px", "8px", "12px", "16px", "24px", "32px", "1rem", "1.5rem", "2rem",
];
const CSS_VALUES_COLOR: &[&str] = &[
    "#333333",
    "#ffffff",
    "#e0e0e0",
    "#1a73e8",
    "#f44336",
    "rgba(0, 0, 0, 0.1)",
    "transparent",
    "inherit",
];
const CSS_VALUES_BORDER: &[&str] = &[
    "none",
    "1px solid #e0e0e0",
    "2px solid #1a73e8",
    "1px dashed #ccc",
    "1px solid transparent",
];
const CSS_VALUES_FONT: &[&str] = &[
    "12px", "14px", "16px", "18px", "24px", "0.875rem", "1rem", "1.25rem",
];
const CSS_VALUES_WEIGHT: &[&str] = &["normal", "bold", "400", "500", "600", "700"];
const CSS_VALUES_MISC: &[&str] = &[
    "auto", "100%", "50%", "200px", "300px", "relative", "absolute", "fixed", "hidden", "1",
];

/// Generates CSS rule blocks with selectors, properties, and contextual values.
pub fn generate_css_rules(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4003);
    (0..n)
        .map(|_| {
            let selector = CSS_SELECTORS[rng.random_range(0..CSS_SELECTORS.len())];
            let num_props = rng.random_range(2..6usize);
            let mut declarations = Vec::with_capacity(num_props);
            for _ in 0..num_props {
                let prop = CSS_PROPERTIES[rng.random_range(0..CSS_PROPERTIES.len())];
                let value = match prop {
                    "display" => CSS_VALUES_DISPLAY[rng.random_range(0..CSS_VALUES_DISPLAY.len())],
                    "justify-content" | "align-items" => {
                        CSS_VALUES_JUSTIFY[rng.random_range(0..CSS_VALUES_JUSTIFY.len())]
                    }
                    "padding" | "margin" | "gap" => {
                        CSS_VALUES_SPACING[rng.random_range(0..CSS_VALUES_SPACING.len())]
                    }
                    "background-color" | "color" => {
                        CSS_VALUES_COLOR[rng.random_range(0..CSS_VALUES_COLOR.len())]
                    }
                    "border" => CSS_VALUES_BORDER[rng.random_range(0..CSS_VALUES_BORDER.len())],
                    "font-size" => CSS_VALUES_FONT[rng.random_range(0..CSS_VALUES_FONT.len())],
                    "font-weight" => {
                        CSS_VALUES_WEIGHT[rng.random_range(0..CSS_VALUES_WEIGHT.len())]
                    }
                    _ => CSS_VALUES_MISC[rng.random_range(0..CSS_VALUES_MISC.len())],
                };
                declarations.push(format!("{prop}: {value};"));
            }
            format!("{selector} {{ {} }}", declarations.join(" "))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shell commands generator (kubectl, docker, git, curl, etc.)
// ---------------------------------------------------------------------------

const SHELL_COMMANDS: &[&str] = &[
    "kubectl",
    "docker",
    "git",
    "curl",
    "aws",
    "terraform",
    "cargo",
    "npm",
    "ssh",
    "rsync",
];
const SHELL_KUBECTL_SUB: &[&str] = &[
    "get pods",
    "get deployments",
    "get services",
    "describe pod",
    "logs",
    "apply -f",
    "delete pod",
    "rollout status",
    "scale deployment",
    "exec -it",
];
const SHELL_DOCKER_SUB: &[&str] = &[
    "run -d",
    "build -t",
    "compose up -d",
    "ps -a",
    "exec -it",
    "pull",
    "push",
    "logs --tail=100",
    "inspect",
    "network create",
];
const SHELL_FLAGS: &[&str] = &[
    "-n production",
    "--selector=app=api",
    "-o json",
    "--rm",
    "--name=redis",
    "-p 6379:6379",
    "-v /data:/data",
    "--timeout=30s",
    "--recursive",
    "-L",
    "--follow",
    "--all-namespaces",
    "--format='{{.ID}}'",
    "--no-cache",
    "-e NODE_ENV=production",
];
const SHELL_ARGS: &[&str] = &[
    "redis:7-alpine",
    "nginx:latest",
    "postgres:15",
    "node:20-slim",
    "user@bastion.example.com",
    "s3://my-bucket/data/",
    "./deploy/manifests/",
    "https://api.example.com/v2/health",
    "main.tf",
    "origin/main",
];
const SHELL_PIPES: &[&str] = &[
    "| jq '.items[].metadata.name'",
    "| grep -v Completed",
    "| wc -l",
    "| head -20",
    "| sort -k2 -rn",
    "| tee output.log",
    "| xargs -I{} echo {}",
    ">> /var/log/deploy.log 2>&1",
];

/// Generates shell command lines with subcommands, flags, arguments, and optional pipes.
pub fn generate_shell_commands(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4004);
    (0..n)
        .map(|_| {
            let cmd = SHELL_COMMANDS[rng.random_range(0..SHELL_COMMANDS.len())];
            let mut parts = Vec::with_capacity(6);
            parts.push(cmd.to_string());
            match cmd {
                "kubectl" => {
                    parts.push(
                        SHELL_KUBECTL_SUB[rng.random_range(0..SHELL_KUBECTL_SUB.len())].to_string(),
                    );
                }
                "docker" => {
                    parts.push(
                        SHELL_DOCKER_SUB[rng.random_range(0..SHELL_DOCKER_SUB.len())].to_string(),
                    );
                }
                _ => {}
            }
            let num_flags = rng.random_range(1..4usize);
            for _ in 0..num_flags {
                parts.push(SHELL_FLAGS[rng.random_range(0..SHELL_FLAGS.len())].to_string());
            }
            let arg = SHELL_ARGS[rng.random_range(0..SHELL_ARGS.len())];
            parts.push(arg.to_string());
            let mut command = parts.join(" ");
            if rng.random_bool(0.4) {
                let pipe = SHELL_PIPES[rng.random_range(0..SHELL_PIPES.len())];
                command.push(' ');
                command.push_str(pipe);
            }
            command
        })
        .collect()
}
