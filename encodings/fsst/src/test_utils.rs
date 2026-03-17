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
