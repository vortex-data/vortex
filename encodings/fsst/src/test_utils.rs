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

/// Generate a random alphanumeric word of given length range.
fn random_word(rng: &mut StdRng, min_len: usize, max_len: usize) -> String {
    let len = rng.random_range(min_len..=max_len);
    let charset = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (0..len)
        .map(|_| charset[rng.random_range(0..charset.len())] as char)
        .collect()
}

/// Generate a random lowercase alphabetic word.
fn random_alpha_word(rng: &mut StdRng, min_len: usize, max_len: usize) -> String {
    let len = rng.random_range(min_len..=max_len);
    (0..len)
        .map(|_| (b'a' + rng.random_range(0..26u8)) as char)
        .collect()
}

/// Generate a random hex string of given byte count.
fn random_hex(rng: &mut StdRng, bytes: usize) -> String {
    (0..bytes)
        .map(|_| format!("{:02x}", rng.random_range(0..256u32)))
        .collect()
}

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

const CB_TLDS: &[&str] = &[
    "com", "ru", "org", "net", "io", "co.uk", "de", "fr", "jp", "br", "in", "au", "ca", "es", "it",
    "nl", "se", "ch", "pl", "cz",
];

const CB_PATH_SEGMENTS: &[&str] = &[
    "search",
    "catalog",
    "product",
    "news",
    "user",
    "api",
    "checkout",
    "blog",
    "category",
    "settings",
    "profile",
    "dashboard",
    "admin",
    "docs",
    "help",
    "download",
    "upload",
    "stream",
    "analytics",
    "report",
    "feed",
    "notifications",
    "messages",
    "orders",
    "cart",
    "wishlist",
    "compare",
    "reviews",
    "support",
    "faq",
    "about",
    "contact",
    "terms",
    "privacy",
    "sitemap",
    "robots",
    "health",
    "status",
    "metrics",
    "v1",
    "v2",
    "v3",
];

const CB_PARAM_KEYS: &[&str] = &[
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "q",
    "category",
    "sort",
    "page",
    "per_page",
    "ref",
    "sessionid",
    "from",
    "clid",
    "text",
    "lr",
    "msid",
    "suggest_reqid",
    "csg",
    "source",
    "forceshow",
    "lang",
    "region",
    "currency",
    "format",
    "callback",
    "token",
    "sig",
    "ts",
    "v",
    "debug",
    "preview",
    "draft",
    "filter",
    "tag",
    "id",
    "offset",
    "limit",
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
            // Generate varied domain: optional subdomain + random name + tld
            let subdomain = match rng.random_range(0..4u32) {
                0 => "www.".to_string(),
                1 => format!("{}.", random_alpha_word(&mut rng, 2, 6)),
                _ => String::new(),
            };
            let domain_name = random_alpha_word(&mut rng, 3, 12);
            let tld = CB_TLDS[rng.random_range(0..CB_TLDS.len())];

            // Generate path with 1-5 segments, mixing fixed and random
            let depth = rng.random_range(1..6usize);
            let path: String = (0..depth)
                .map(|_| {
                    if rng.random_bool(0.6) {
                        CB_PATH_SEGMENTS[rng.random_range(0..CB_PATH_SEGMENTS.len())].to_string()
                    } else {
                        random_word(&mut rng, 3, 15)
                    }
                })
                .collect::<Vec<_>>()
                .join("/");

            // Generate 0-5 query params
            let num_params = rng.random_range(0..6usize);
            let params = if num_params > 0 {
                let pairs: Vec<String> = (0..num_params)
                    .map(|_| {
                        let key = CB_PARAM_KEYS[rng.random_range(0..CB_PARAM_KEYS.len())];
                        let val = random_word(&mut rng, 1, 20);
                        format!("{key}={val}")
                    })
                    .collect();
                format!("?{}", pairs.join("&"))
            } else {
                String::new()
            };

            // Optional fragment
            let fragment = if rng.random_bool(0.15) {
                format!("#{}", random_word(&mut rng, 3, 15))
            } else {
                String::new()
            };

            format!("{scheme}://{subdomain}{domain_name}.{tld}/{path}{params}{fragment}")
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

const LOG_METHODS: &[&str] = &["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];
const LOG_STATUS: &[u16] = &[
    200, 200, 200, 200, 200, 201, 204, 301, 302, 304, 400, 401, 403, 404, 405, 408, 429, 500, 502,
    503,
];
const LOG_UA_PREFIXES: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 14)",
];
const LOG_UA_ENGINES: &[&str] = &["AppleWebKit/537.36 (KHTML, like Gecko)", "Gecko/20100101"];
const LOG_UA_BROWSERS: &[&str] = &[
    "Chrome/120.0.0.0 Safari/537.36",
    "Chrome/119.0.0.0 Safari/537.36",
    "Firefox/121.0",
    "Firefox/120.0",
    "Safari/605.1.15",
    "Edge/120.0.0.0",
];
const LOG_BOT_UAS: &[&str] = &[
    "curl/7.81.0",
    "curl/8.4.0",
    "python-requests/2.28.1",
    "python-requests/2.31.0",
    "Go-http-client/1.1",
    "Go-http-client/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "Bingbot/2.0 (+http://www.bing.com/bingbot.htm)",
    "Apache-HttpClient/4.5.14",
    "okhttp/4.12.0",
];

pub fn generate_log_lines(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(456);
    (0..n)
        .map(|_| {
            // Random IP address
            let ip = format!(
                "{}.{}.{}.{}",
                rng.random_range(1..224u32),
                rng.random_range(0..256u32),
                rng.random_range(0..256u32),
                rng.random_range(1..255u32),
            );
            let method = LOG_METHODS[rng.random_range(0..LOG_METHODS.len())];
            // Random path with 1-4 segments
            let depth = rng.random_range(1..5usize);
            let path_segments: Vec<String> = (0..depth)
                .map(|_| random_alpha_word(&mut rng, 2, 12))
                .collect();
            let path = format!("/{}", path_segments.join("/"));
            // Optional query string
            let query = if rng.random_bool(0.4) {
                format!("?{}={}", random_alpha_word(&mut rng, 2, 8), rng.random_range(1..100000u32))
            } else {
                String::new()
            };
            let status = LOG_STATUS[rng.random_range(0..LOG_STATUS.len())];
            let size = rng.random_range(50..500000u32);
            let day = rng.random_range(1..29u32);
            let month = rng.random_range(1..13u32);
            let hour = rng.random_range(0..24u32);
            let minute = rng.random_range(0..60u32);
            let second = rng.random_range(0..60u32);
            // Varied user agents
            let ua = if rng.random_bool(0.3) {
                LOG_BOT_UAS[rng.random_range(0..LOG_BOT_UAS.len())].to_string()
            } else {
                let prefix = LOG_UA_PREFIXES[rng.random_range(0..LOG_UA_PREFIXES.len())];
                let engine = LOG_UA_ENGINES[rng.random_range(0..LOG_UA_ENGINES.len())];
                let browser = LOG_UA_BROWSERS[rng.random_range(0..LOG_UA_BROWSERS.len())];
                format!("{prefix} {engine} {browser}")
            };
            // Varied referrer
            let referrer = if rng.random_bool(0.5) {
                "-".to_string()
            } else {
                format!("https://{}.com/{}", random_alpha_word(&mut rng, 4, 10), random_alpha_word(&mut rng, 3, 8))
            };
            format!(
                r#"{ip} - - [{day:02}/{month:02}/2024:{hour:02}:{minute:02}:{second:02} +0000] "{method} {path}{query} HTTP/1.1" {status} {size} "{referrer}" "{ua}""#,
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

const JSON_FIELD_NAMES: &[&str] = &[
    "id",
    "name",
    "email",
    "age",
    "city",
    "country",
    "status",
    "role",
    "tags",
    "active",
    "created_at",
    "updated_at",
    "score",
    "balance",
    "plan",
    "org",
    "team",
    "department",
    "title",
    "phone",
    "address",
    "zip",
    "state",
    "lat",
    "lng",
    "verified",
    "premium",
    "notes",
    "description",
    "url",
    "avatar",
    "locale",
    "timezone",
    "currency",
];

pub fn generate_json_strings(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(789);
    (0..n)
        .map(|_| {
            let mut json = String::with_capacity(256);
            json.push('{');
            // Vary the number of fields per record (3-8)
            let nfields = rng.random_range(3..9usize);
            for i in 0..nfields {
                if i > 0 {
                    json.push(',');
                }
                let field = JSON_FIELD_NAMES[rng.random_range(0..JSON_FIELD_NAMES.len())];
                json.push('"');
                json.push_str(field);
                json.push_str("\":");
                // Vary value types
                match rng.random_range(0..5u32) {
                    0 => {
                        // string value - random word
                        json.push('"');
                        json.push_str(&random_alpha_word(&mut rng, 3, 15));
                        json.push('"');
                    }
                    1 => {
                        // integer
                        let v = rng.random_range(0..1_000_000u32);
                        json.push_str(&v.to_string());
                    }
                    2 => {
                        // float
                        let v = rng.random_range(0..100000u32) as f64 / 100.0;
                        json.push_str(&format!("{v:.2}"));
                    }
                    3 => {
                        // boolean
                        json.push_str(if rng.random_bool(0.5) {
                            "true"
                        } else {
                            "false"
                        });
                    }
                    _ => {
                        // array of strings
                        let arr_len = rng.random_range(1..4usize);
                        json.push('[');
                        for j in 0..arr_len {
                            if j > 0 {
                                json.push(',');
                            }
                            json.push('"');
                            json.push_str(&random_alpha_word(&mut rng, 3, 10));
                            json.push('"');
                        }
                        json.push(']');
                    }
                }
            }
            // Sometimes add a nested object
            if rng.random_bool(0.4) {
                json.push_str(",\"meta\":{\"src\":\"");
                json.push_str(&random_alpha_word(&mut rng, 4, 10));
                json.push_str("\",\"v\":");
                json.push_str(&rng.random_range(1..100u32).to_string());
                json.push('}');
            }
            json.push('}');
            json
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
    "/home",
    "/var/log",
    "/etc",
    "/usr/local",
    "/opt",
    "/tmp",
    "/srv",
    "/data",
    "/mnt",
    "/run",
];
const PATH_EXTENSIONS: &[&str] = &[
    "rs",
    "ts",
    "js",
    "py",
    "go",
    "java",
    "c",
    "h",
    "cpp",
    "yaml",
    "yml",
    "json",
    "toml",
    "xml",
    "sql",
    "log",
    "txt",
    "md",
    "csv",
    "parquet",
    "avro",
    "proto",
    "html",
    "css",
    "scss",
    "conf",
    "cfg",
    "ini",
    "sh",
    "dockerfile",
];

pub fn generate_file_paths(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(321);
    (0..n)
        .map(|_| {
            let root = PATH_ROOTS[rng.random_range(0..PATH_ROOTS.len())];
            let depth = rng.random_range(2..7usize);
            let mut path = root.to_string();
            for _ in 0..depth {
                path.push('/');
                path.push_str(&random_alpha_word(&mut rng, 2, 12));
            }
            path.push('/');
            path.push_str(&random_alpha_word(&mut rng, 3, 15));
            path.push('.');
            path.push_str(PATH_EXTENSIONS[rng.random_range(0..PATH_EXTENSIONS.len())]);
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

const EMAIL_TLDS: &[&str] = &[
    "com", "org", "net", "io", "co", "dev", "app", "ru", "uk", "de", "fr", "jp",
];
const EMAIL_SEPARATORS: &[&str] = &[".", "-", "_", "+", ""];

pub fn generate_emails(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(654);
    (0..n)
        .map(|_| {
            // Random local part: 1-3 segments with random separators
            let num_parts = rng.random_range(1..4usize);
            let local: Vec<String> = (0..num_parts)
                .map(|_| random_alpha_word(&mut rng, 2, 10))
                .collect();
            let sep = EMAIL_SEPARATORS[rng.random_range(0..EMAIL_SEPARATORS.len())];
            let local_part = local.join(sep);
            // Optional numeric suffix
            let suffix = if rng.random_bool(0.4) {
                rng.random_range(1..9999u32).to_string()
            } else {
                String::new()
            };
            // Random domain
            let domain_name = random_alpha_word(&mut rng, 3, 12);
            let tld = EMAIL_TLDS[rng.random_range(0..EMAIL_TLDS.len())];
            format!("{local_part}{suffix}@{domain_name}.{tld}")
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

const SQL_OPERATORS: &[&str] = &[
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "LIKE",
    "IN",
    "IS NOT NULL",
    "IS NULL",
    "BETWEEN",
];
const SQL_FUNCTIONS: &[&str] = &[
    "COUNT", "SUM", "AVG", "MAX", "MIN", "COALESCE", "NULLIF", "CAST", "TRIM", "LOWER", "UPPER",
];
const SQL_JOIN_TYPES: &[&str] = &[
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "LEFT OUTER JOIN",
    "CROSS JOIN",
];

pub fn generate_sql_queries(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1001);
    (0..n)
        .map(|_| {
            let table = random_alpha_word(&mut rng, 4, 15);
            let kind = rng.random_range(0..5u32);
            match kind {
                0 | 1 => {
                    // SELECT (most common)
                    let ncols = rng.random_range(2..8usize);
                    let cols: Vec<String> = (0..ncols)
                        .map(|_| {
                            if rng.random_bool(0.2) {
                                let func = SQL_FUNCTIONS[rng.random_range(0..SQL_FUNCTIONS.len())];
                                let col = random_alpha_word(&mut rng, 3, 12);
                                format!("{func}({col})")
                            } else {
                                random_alpha_word(&mut rng, 3, 12)
                            }
                        })
                        .collect();
                    let nconds = rng.random_range(1..5usize);
                    let mut where_parts = Vec::with_capacity(nconds);
                    for _ in 0..nconds {
                        let col = random_alpha_word(&mut rng, 3, 10);
                        let op = SQL_OPERATORS[rng.random_range(0..SQL_OPERATORS.len())];
                        let val = match op {
                            "LIKE" => format!("'%{}%'", random_alpha_word(&mut rng, 3, 8)),
                            "IN" => {
                                let cnt = rng.random_range(2..5usize);
                                let vals: Vec<String> = (0..cnt)
                                    .map(|_| rng.random_range(1..100000u32).to_string())
                                    .collect();
                                format!("({})", vals.join(", "))
                            }
                            "IS NOT NULL" | "IS NULL" | "BETWEEN" => String::new(),
                            _ => {
                                if rng.random_bool(0.5) {
                                    format!("'{}'", random_alpha_word(&mut rng, 3, 10))
                                } else {
                                    rng.random_range(1..1000000u32).to_string()
                                }
                            }
                        };
                        if val.is_empty() {
                            where_parts.push(format!("{col} {op}"));
                        } else {
                            where_parts.push(format!("{col} {op} {val}"));
                        }
                    }
                    // Optional JOIN
                    let join = if rng.random_bool(0.4) {
                        let jtype = SQL_JOIN_TYPES[rng.random_range(0..SQL_JOIN_TYPES.len())];
                        let jtable = random_alpha_word(&mut rng, 4, 12);
                        let jcol1 = random_alpha_word(&mut rng, 3, 10);
                        let jcol2 = random_alpha_word(&mut rng, 3, 10);
                        format!(" {jtype} {jtable} ON {table}.{jcol1} = {jtable}.{jcol2}")
                    } else {
                        String::new()
                    };
                    let order_col = random_alpha_word(&mut rng, 3, 10);
                    let dir = if rng.random_bool(0.5) { "ASC" } else { "DESC" };
                    let limit = rng.random_range(10..10000u32);
                    format!(
                        "SELECT {} FROM {}{} WHERE {} ORDER BY {} {} LIMIT {}",
                        cols.join(", "),
                        table,
                        join,
                        where_parts.join(" AND "),
                        order_col,
                        dir,
                        limit,
                    )
                }
                2 => {
                    // INSERT
                    let ncols = rng.random_range(3..8usize);
                    let cols: Vec<String> = (0..ncols)
                        .map(|_| random_alpha_word(&mut rng, 3, 12))
                        .collect();
                    let vals: Vec<String> = (0..ncols)
                        .map(|_| {
                            if rng.random_bool(0.5) {
                                format!("'{}'", random_alpha_word(&mut rng, 3, 15))
                            } else {
                                rng.random_range(1..1000000u32).to_string()
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
                3 => {
                    // UPDATE
                    let nsets = rng.random_range(1..5usize);
                    let set_parts: Vec<String> = (0..nsets)
                        .map(|_| {
                            let col = random_alpha_word(&mut rng, 3, 10);
                            let val = if rng.random_bool(0.5) {
                                format!("'{}'", random_alpha_word(&mut rng, 3, 12))
                            } else {
                                rng.random_range(1..1000000u32).to_string()
                            };
                            format!("{col} = {val}")
                        })
                        .collect();
                    let cond_col = random_alpha_word(&mut rng, 3, 10);
                    let cond_val = rng.random_range(1..1000000u32);
                    format!(
                        "UPDATE {} SET {} WHERE {} = {}",
                        table,
                        set_parts.join(", "),
                        cond_col,
                        cond_val,
                    )
                }
                _ => {
                    // DELETE
                    let col = random_alpha_word(&mut rng, 3, 10);
                    let val = rng.random_range(1..1000000u32);
                    format!("DELETE FROM {} WHERE {} = {}", table, col, val)
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// XML fragments generator (nested tags, attributes, namespaces)
// ---------------------------------------------------------------------------

const XML_NAMESPACES: &[&str] = &[
    "ns1", "ns2", "xsi", "xsd", "soap", "app", "svc", "core", "ext", "cfg",
];
const XML_ATTR_NAMES: &[&str] = &[
    "id", "type", "name", "class", "version", "status", "priority", "lang", "encoding", "format",
    "scope", "ref", "key", "source", "target", "mode", "level", "enabled",
];

pub fn generate_xml_fragments(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1002);
    (0..n)
        .map(|_| {
            let mut xml = String::with_capacity(256);
            let depth = rng.random_range(2..6usize);
            let mut tags_stack = Vec::with_capacity(depth);
            for d in 0..depth {
                let use_ns = rng.random_bool(0.6);
                let tag = random_alpha_word(&mut rng, 3, 12);
                let qualified = if use_ns {
                    let ns = XML_NAMESPACES[rng.random_range(0..XML_NAMESPACES.len())];
                    format!("{ns}:{tag}")
                } else {
                    tag
                };
                xml.push('<');
                xml.push_str(&qualified);
                let nattrs = rng.random_range(0..4usize);
                for _ in 0..nattrs {
                    let attr = XML_ATTR_NAMES[rng.random_range(0..XML_ATTR_NAMES.len())];
                    xml.push(' ');
                    xml.push_str(attr);
                    xml.push_str("=\"");
                    // Random attribute value
                    xml.push_str(&random_word(&mut rng, 2, 15));
                    xml.push('"');
                }
                xml.push('>');
                // Add text content at intermediate levels sometimes
                if d < depth - 1 && rng.random_bool(0.3) {
                    xml.push_str(&random_alpha_word(&mut rng, 5, 20));
                }
                tags_stack.push(qualified);
            }
            // inner text content
            xml.push_str(&random_word(&mut rng, 5, 30));
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
// CSV rows generator (varied schemas and values)
// ---------------------------------------------------------------------------

pub fn generate_csv_rows(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1004);
    // Generate random column headers (reused across rows for this "table")
    let ncols = rng.random_range(5..10usize);
    let headers: Vec<String> = (0..ncols)
        .map(|_| random_alpha_word(&mut rng, 3, 12))
        .collect();
    let mut rows = Vec::with_capacity(n);
    // First row is the header
    rows.push(headers.join(","));
    for _ in 1..n {
        let values: Vec<String> = (0..ncols)
            .map(|_| {
                match rng.random_range(0..4u32) {
                    0 => rng.random_range(0..1000000u32).to_string(),
                    1 => format!("{:.2}", rng.random_range(0..100000u32) as f64 / 100.0),
                    2 => random_alpha_word(&mut rng, 3, 15),
                    _ => {
                        // Quoted string with possible spaces
                        let w1 = random_alpha_word(&mut rng, 3, 8);
                        let w2 = random_alpha_word(&mut rng, 3, 8);
                        format!("\"{w1} {w2}\"")
                    }
                }
            })
            .collect();
        rows.push(values.join(","));
    }
    rows
}

// ---------------------------------------------------------------------------
// Key-value config lines generator (shared prefixes)
// ---------------------------------------------------------------------------

pub fn generate_key_value_config(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1005);
    (0..n)
        .map(|_| {
            // Build a dotted key path with 2-4 random segments
            let depth = rng.random_range(2..5usize);
            let key_path: Vec<String> = (0..depth)
                .map(|_| random_alpha_word(&mut rng, 3, 12))
                .collect();
            // Random value of various types
            let value = match rng.random_range(0..5u32) {
                0 => rng.random_range(0..65536u32).to_string(),
                1 => if rng.random_bool(0.5) {
                    "true"
                } else {
                    "false"
                }
                .to_string(),
                2 => format!(
                    "{}.{}.{}.{}",
                    rng.random_range(0..256u32),
                    rng.random_range(0..256u32),
                    rng.random_range(0..256u32),
                    rng.random_range(0..256u32)
                ),
                3 => format!(
                    "/{}/{}/{}",
                    random_alpha_word(&mut rng, 3, 8),
                    random_alpha_word(&mut rng, 3, 8),
                    random_alpha_word(&mut rng, 3, 8)
                ),
                _ => random_alpha_word(&mut rng, 3, 15),
            };
            // Vary the separator style
            let sep = match rng.random_range(0..3u32) {
                0 => " = ",
                1 => "=",
                _ => ": ",
            };
            format!("{}{sep}{value}", key_path.join("."))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Timestamps with prefix generator
// ---------------------------------------------------------------------------

const TS_LEVELS: &[&str] = &["INFO", "DEBUG", "WARN", "ERROR", "TRACE", "FATAL"];

pub fn generate_timestamps_with_prefix(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(1006);
    (0..n)
        .map(|_| {
            let year = rng.random_range(2020..2025u32);
            let month = rng.random_range(1..13u32);
            let day = rng.random_range(1..29u32);
            let hour = rng.random_range(0..24u32);
            let minute = rng.random_range(0..60u32);
            let second = rng.random_range(0..60u32);
            let millis = rng.random_range(0..1000u32);
            let micro = rng.random_range(0..1000u32);
            let level = TS_LEVELS[rng.random_range(0..TS_LEVELS.len())];
            // Random service name like "order-processor" or "auth-svc"
            let svc = format!("{}-{}", random_alpha_word(&mut rng, 3, 10), random_alpha_word(&mut rng, 2, 8));
            // Random message: 3-8 words
            let nwords = rng.random_range(3..9usize);
            let msg: Vec<String> = (0..nwords)
                .map(|_| random_alpha_word(&mut rng, 2, 10))
                .collect();
            let req_id = random_hex(&mut rng, 8);
            // Vary the format
            match rng.random_range(0..3u32) {
                0 => format!(
                    "{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z {level} [{svc}] {} req_id={req_id}",
                    msg.join(" ")
                ),
                1 => format!(
                    "{year}/{month:02}/{day:02} {hour:02}:{minute:02}:{second:02}.{millis:03}{micro:03} [{level}] {svc}: {} trace={req_id}",
                    msg.join(" ")
                ),
                _ => format!(
                    "[{level}] {year}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} {svc} - {} (id={req_id})",
                    msg.join(" ")
                ),
            }
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
    "X-Frame-Options",
    "X-XSS-Protection",
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "Access-Control-Allow-Origin",
    "Access-Control-Allow-Methods",
    "X-Powered-By",
    "X-RateLimit-Remaining",
    "X-RateLimit-Limit",
    "Retry-After",
    "Location",
    "Server",
    "Date",
    "Expires",
    "Pragma",
    "Transfer-Encoding",
    "X-Trace-Id",
];

const HTTP_CONTENT_TYPES: &[&str] = &[
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
    "application/javascript",
    "application/pdf",
    "image/svg+xml",
    "text/csv",
];

/// Generates HTTP request/response header lines.
pub fn generate_http_headers(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2001);
    (0..n)
        .map(|_| {
            let header_name = HTTP_HEADER_NAMES[rng.random_range(0..HTTP_HEADER_NAMES.len())];
            let value = match header_name {
                "Content-Type" | "Accept" => {
                    HTTP_CONTENT_TYPES[rng.random_range(0..HTTP_CONTENT_TYPES.len())].to_string()
                }
                "X-Request-Id" | "X-Correlation-Id" | "X-Trace-Id" | "ETag" => {
                    let len = rng.random_range(8..20usize);
                    random_hex(&mut rng, len)
                }
                "Content-Length"
                | "X-RateLimit-Remaining"
                | "X-RateLimit-Limit"
                | "Retry-After" => rng.random_range(0..10_000_000u32).to_string(),
                "X-Forwarded-For" => {
                    let n_ips = rng.random_range(1..4usize);
                    let ips: Vec<String> = (0..n_ips)
                        .map(|_| {
                            format!(
                                "{}.{}.{}.{}",
                                rng.random_range(1..224u32),
                                rng.random_range(0..256u32),
                                rng.random_range(0..256u32),
                                rng.random_range(1..255u32)
                            )
                        })
                        .collect();
                    ips.join(", ")
                }
                "Authorization" => {
                    format!("Bearer {}", random_hex(&mut rng, 32))
                }
                "Host" | "Server" | "X-Powered-By" => {
                    format!(
                        "{}.{}",
                        random_alpha_word(&mut rng, 3, 10),
                        random_alpha_word(&mut rng, 2, 4)
                    )
                }
                "Set-Cookie" => {
                    let name = random_alpha_word(&mut rng, 3, 10);
                    let val = random_hex(&mut rng, 16);
                    let domain = random_alpha_word(&mut rng, 4, 10);
                    format!("{name}={val}; Domain={domain}.com; Path=/; HttpOnly; Secure")
                }
                "Cache-Control" => {
                    let max_age = rng.random_range(0..86400u32);
                    if rng.random_bool(0.3) {
                        "no-cache, no-store, must-revalidate".to_string()
                    } else {
                        format!("public, max-age={max_age}")
                    }
                }
                "User-Agent" => {
                    let ver = rng.random_range(90..130u32);
                    format!(
                        "Mozilla/5.0 ({}) AppleWebKit/537.36 Chrome/{ver}.0.0.0",
                        random_alpha_word(&mut rng, 5, 20)
                    )
                }
                "Location" | "Access-Control-Allow-Origin" => {
                    format!(
                        "https://{}.com/{}",
                        random_alpha_word(&mut rng, 4, 10),
                        random_alpha_word(&mut rng, 3, 12)
                    )
                }
                _ => random_word(&mut rng, 5, 30),
            };
            format!("{header_name}: {value}")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// IPv4 CIDR firewall/ACL rules generator
// ---------------------------------------------------------------------------

const CIDR_ACTIONS: &[&str] = &["allow", "deny", "reject", "drop", "accept", "log"];
const CIDR_PROTOCOLS: &[&str] = &["tcp", "udp", "icmp", "sctp", "gre", "any"];

/// Generates firewall/ACL rules with random CIDR ranges.
pub fn generate_ipv4_cidr_rules(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2002);
    (0..n)
        .map(|_| {
            let action = CIDR_ACTIONS[rng.random_range(0..CIDR_ACTIONS.len())];
            let proto = CIDR_PROTOCOLS[rng.random_range(0..CIDR_PROTOCOLS.len())];
            // Random CIDR ranges
            let dest = format!(
                "{}.{}.{}.0/{}",
                rng.random_range(1..224u32),
                rng.random_range(0..256u32),
                rng.random_range(0..256u32),
                rng.random_range(8..33u32),
            );
            let src = if rng.random_bool(0.2) {
                "any".to_string()
            } else {
                format!(
                    "{}.{}.{}.0/{}",
                    rng.random_range(1..224u32),
                    rng.random_range(0..256u32),
                    rng.random_range(0..256u32),
                    rng.random_range(8..33u32),
                )
            };
            // Optional port (random, not from fixed list)
            if proto == "icmp" || proto == "gre" || proto == "any" {
                format!("{action} {dest} {proto} from {src}")
            } else {
                let port = rng.random_range(1..65536u32);
                // Optional comment
                let comment = if rng.random_bool(0.3) {
                    format!(" # {}", random_alpha_word(&mut rng, 4, 15))
                } else {
                    String::new()
                };
                format!("{action} {dest} {proto} {port} from {src}{comment}")
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Prometheus exposition format metrics generator
// ---------------------------------------------------------------------------

const PROM_METRIC_SUFFIXES: &[&str] = &[
    "_total",
    "_duration_seconds",
    "_bytes",
    "_count",
    "_sum",
    "_bucket",
    "_info",
    "_created",
    "_ratio",
];

const PROM_LABEL_KEYS: &[&str] = &[
    "method",
    "handler",
    "status",
    "instance",
    "job",
    "mode",
    "le",
    "quantile",
    "namespace",
    "pod",
    "container",
    "node",
    "endpoint",
    "service",
    "code",
    "version",
    "region",
    "cluster",
    "env",
    "tenant",
];

/// Generates Prometheus exposition format lines.
pub fn generate_prometheus_metrics(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(2003);
    (0..n)
        .map(|_| {
            // Random metric name: prefix_suffix
            let prefix = random_alpha_word(&mut rng, 3, 12);
            let suffix = PROM_METRIC_SUFFIXES[rng.random_range(0..PROM_METRIC_SUFFIXES.len())];
            let metric = format!("{prefix}{suffix}");

            let num_labels = rng.random_range(1..6usize);
            let mut labels = Vec::with_capacity(num_labels);
            for _ in 0..num_labels {
                let key = PROM_LABEL_KEYS[rng.random_range(0..PROM_LABEL_KEYS.len())];
                // Random label value
                let val = if rng.random_bool(0.3) {
                    rng.random_range(0..1000u32).to_string()
                } else {
                    random_alpha_word(&mut rng, 2, 15)
                };
                labels.push(format!("{key}=\"{val}\""));
            }

            let value = if rng.random_bool(0.3) {
                format!("{:.6}", rng.random_range(0..1000000u32) as f64 / 1000.0)
            } else {
                rng.random_range(0..10_000_000u64).to_string()
            };
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

const DNS_QUERY_TYPES: &[u16] = &[1, 2, 5, 6, 12, 15, 16, 28, 33, 35, 43, 44, 46, 48, 52, 65];

/// Generates simplified DNS wire format packets with random domain labels.
pub fn generate_dns_wire_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(2004);
    (0..n)
        .map(|_| {
            let mut pkt = Vec::with_capacity(96);

            // Transaction ID (2 bytes)
            let txn_id: u16 = rng.random_range(0..u16::MAX);
            pkt.extend_from_slice(&txn_id.to_be_bytes());

            // Flags (2 bytes)
            let flags: u16 = if rng.random_bool(0.6) { 0x0100 } else { 0x8180 };
            pkt.extend_from_slice(&flags.to_be_bytes());

            // Question count
            let qcount: u16 = rng.random_range(1..4);
            pkt.extend_from_slice(&qcount.to_be_bytes());

            for _ in 0..qcount {
                // Random domain labels (2-5 labels, each 2-12 chars)
                let num_labels = rng.random_range(2..6usize);
                for _ in 0..num_labels {
                    let label = random_alpha_word(&mut rng, 2, 12);
                    let label_bytes = label.as_bytes();
                    #[allow(clippy::cast_possible_truncation)]
                    pkt.push(label_bytes.len() as u8);
                    pkt.extend_from_slice(label_bytes);
                }
                pkt.push(0);
                let qtype = DNS_QUERY_TYPES[rng.random_range(0..DNS_QUERY_TYPES.len())];
                pkt.extend_from_slice(&qtype.to_be_bytes());
                pkt.extend_from_slice(&1u16.to_be_bytes());
            }

            // For responses, add some random answer data
            if flags == 0x8180 && rng.random_bool(0.5) {
                let ans_count = rng.random_range(1..4usize);
                for _ in 0..ans_count {
                    // Pointer + type + class + TTL + rdlength + rdata
                    pkt.extend_from_slice(&0xC00Cu16.to_be_bytes());
                    pkt.extend_from_slice(&1u16.to_be_bytes()); // type A
                    pkt.extend_from_slice(&1u16.to_be_bytes()); // class IN
                    let ttl: u32 = rng.random_range(60..86400);
                    pkt.extend_from_slice(&ttl.to_be_bytes());
                    pkt.extend_from_slice(&4u16.to_be_bytes()); // rdlength
                    for _ in 0..4 {
                        pkt.push(rng.random_range(1..255u8));
                    }
                }
            }

            pkt
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Parquet-like column metadata binary generator
// ---------------------------------------------------------------------------

/// Generates simplified Parquet-like column metadata binary records with random column names.
pub fn generate_parquet_footer_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(3001);
    (0..n)
        .map(|_| {
            let mut buf = Vec::with_capacity(256);
            // 4-byte magic
            buf.extend_from_slice(b"PAR1");
            // Version byte
            buf.push(rng.random_range(1..4u8));
            let num_chunks = rng.random_range(3..12usize);
            for _ in 0..num_chunks {
                // 1-byte type tag
                buf.push(rng.random_range(0..7u8));
                // 4-byte LE offset
                let offset: u32 = rng.random_range(0..100_000_000);
                buf.extend_from_slice(&offset.to_le_bytes());
                // 4-byte LE size
                let size: u32 = rng.random_range(64..1_000_000);
                buf.extend_from_slice(&size.to_le_bytes());
                // 4-byte LE num_values
                let num_vals: u32 = rng.random_range(100..10_000_000);
                buf.extend_from_slice(&num_vals.to_le_bytes());
                // 1-byte encoding (0=plain, 1=rle, 2=delta, 3=dict)
                buf.push(rng.random_range(0..4u8));
                // 1-byte compression (0=none, 1=snappy, 2=gzip, 3=zstd)
                buf.push(rng.random_range(0..4u8));
                // length-prefixed random column name
                let name = random_alpha_word(&mut rng, 3, 20);
                let name_bytes = name.as_bytes();
                #[allow(clippy::cast_possible_truncation)]
                buf.push(name_bytes.len() as u8);
                buf.extend_from_slice(name_bytes);
                // Optional statistics block
                if rng.random_bool(0.5) {
                    // min/max values (8 bytes each)
                    for _ in 0..16 {
                        buf.push(rng.random::<u8>());
                    }
                }
            }
            // Footer size + magic
            #[allow(clippy::cast_possible_truncation)]
            let footer_size = buf.len() as u32; // footer is at most ~1KB
            buf.extend_from_slice(&footer_size.to_le_bytes());
            buf.extend_from_slice(b"PAR1");
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
    "GlobalLimit",
    "LocalLimit",
    "Union",
    "Expand",
    "Generate",
    "SubqueryAlias",
    "Coalesce",
    "ReusedExchange",
    "AdaptiveSparkPlan",
];

const SPARK_AGG_FUNCTIONS: &[&str] = &[
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "first",
    "last",
    "collect_list",
    "collect_set",
    "approx_count_distinct",
    "stddev",
    "variance",
];

/// Generates Spark-style physical query plan fragments with random column names.
pub fn generate_spark_plan_strings(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(3002);
    (0..n)
        .map(|_| {
            let mut plan = String::with_capacity(512);
            plan.push_str("== Physical Plan ==\n");
            let depth = rng.random_range(3..8usize);
            for level in 0..depth {
                if level > 0 {
                    for _ in 0..level {
                        plan.push_str("   ");
                    }
                    plan.push_str("+- ");
                }
                let op = SPARK_OPERATORS[rng.random_range(0..SPARK_OPERATORS.len())];
                plan.push_str(op);
                let ncols = rng.random_range(1..5usize);
                plan.push('(');
                match op {
                    "HashAggregate" => {
                        plan.push_str("keys=[");
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let col = random_alpha_word(&mut rng, 3, 15);
                            let col_id = rng.random_range(100..9999u32);
                            plan.push_str(&format!("{col}#{col_id}"));
                        }
                        plan.push_str("], functions=[");
                        let nfuncs = rng.random_range(1..4usize);
                        for i in 0..nfuncs {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let func =
                                SPARK_AGG_FUNCTIONS[rng.random_range(0..SPARK_AGG_FUNCTIONS.len())];
                            let col = random_alpha_word(&mut rng, 3, 12);
                            let col_id = rng.random_range(100..9999u32);
                            plan.push_str(&format!("{func}({col}#{col_id})"));
                        }
                        plan.push(']');
                    }
                    "Exchange" => {
                        let col = random_alpha_word(&mut rng, 3, 12);
                        let col_id = rng.random_range(100..9999u32);
                        let partitions = rng.random_range(50..1000u32);
                        plan.push_str(&format!("hashpartitioning({col}#{col_id}, {partitions})"));
                    }
                    "FileScan parquet" => {
                        let table = random_alpha_word(&mut rng, 5, 15);
                        plan.push_str(&format!("[{table}] "));
                        plan.push('[');
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push(',');
                            }
                            let col = random_alpha_word(&mut rng, 3, 12);
                            let col_id = rng.random_range(100..9999u32);
                            plan.push_str(&format!("{col}#{col_id}"));
                        }
                        plan.push(']');
                    }
                    "Filter" => {
                        let col = random_alpha_word(&mut rng, 3, 12);
                        let col_id = rng.random_range(100..9999u32);
                        let op_str = if rng.random_bool(0.5) {
                            "isnotnull"
                        } else {
                            "=="
                        };
                        if op_str == "isnotnull" {
                            plan.push_str(&format!("isnotnull({col}#{col_id})"));
                        } else {
                            let val = rng.random_range(0..10000u32);
                            plan.push_str(&format!("({col}#{col_id} == {val})"));
                        }
                    }
                    _ => {
                        for i in 0..ncols {
                            if i > 0 {
                                plan.push_str(", ");
                            }
                            let col = random_alpha_word(&mut rng, 3, 12);
                            let col_id = rng.random_range(100..9999u32);
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

/// Generates simplified Arrow IPC-like binary records with random field names.
pub fn generate_arrow_ipc_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(3003);
    (0..n)
        .map(|_| {
            let mut buf = Vec::with_capacity(256);
            // 4-byte continuation marker
            buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
            // Placeholder for metadata length
            let meta_len_pos = buf.len();
            buf.extend_from_slice(&0u32.to_le_bytes());
            let meta_start = buf.len();
            // Schema: num_fields (2 bytes) + field descriptors
            let num_fields = rng.random_range(3..12usize);
            #[allow(clippy::cast_possible_truncation)]
            buf.extend_from_slice(&(num_fields as u16).to_le_bytes());
            for _ in 0..num_fields {
                // 1-byte type (0=null, 1=int8, 2=int16, ..., 7=utf8, 8=binary)
                buf.push(rng.random_range(0..9u8));
                // 1-byte bit_width
                buf.push(match rng.random_range(0..4u8) {
                    0 => 8,
                    1 => 16,
                    2 => 32,
                    _ => 64,
                });
                // 1-byte nullable flag
                buf.push(if rng.random_bool(0.3) { 1 } else { 0 });
                // length-prefixed random field name
                let name = random_alpha_word(&mut rng, 3, 18);
                let name_bytes = name.as_bytes();
                #[allow(clippy::cast_possible_truncation)]
                buf.push(name_bytes.len() as u8);
                buf.extend_from_slice(name_bytes);
                // Optional dictionary encoding info
                if rng.random_bool(0.2) {
                    buf.push(1); // has_dictionary
                    let dict_id: u16 = rng.random_range(0..1000);
                    buf.extend_from_slice(&dict_id.to_le_bytes());
                } else {
                    buf.push(0);
                }
            }
            // Patch metadata length
            #[allow(clippy::cast_possible_truncation)]
            let meta_len = (buf.len() - meta_start) as u32;
            buf[meta_len_pos..meta_len_pos + 4].copy_from_slice(&meta_len.to_le_bytes());
            // 4-byte body length
            let body_len: u32 = rng.random_range(1024..10_000_000);
            buf.extend_from_slice(&body_len.to_le_bytes());
            buf
        })
        .collect()
}

// ---------------------------------------------------------------------------
// JSONL (JSON Lines) generator
// ---------------------------------------------------------------------------

/// Generates JSONL records with varied schemas and random field names/values.
pub fn generate_json_lines(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(3004);
    (0..n)
        .map(|_| {
            let mut line = String::with_capacity(256);
            line.push('{');
            // Random event type
            line.push_str("\"event\":\"");
            line.push_str(&random_alpha_word(&mut rng, 4, 15));
            line.push('"');
            // Random user id
            let user_id = random_hex(&mut rng, 8);
            line.push_str(&format!(",\"user\":\"{user_id}\""));
            let ts = rng.random_range(1_600_000_000..1_710_000_000u64);
            line.push_str(&format!(",\"ts\":{ts}"));
            // Nested props with random keys and values
            let nprops = rng.random_range(2..8usize);
            line.push_str(",\"props\":{");
            for i in 0..nprops {
                if i > 0 {
                    line.push(',');
                }
                let key = random_alpha_word(&mut rng, 2, 10);
                line.push('"');
                line.push_str(&key);
                line.push_str("\":");
                // Vary value types
                match rng.random_range(0..3u32) {
                    0 => {
                        line.push('"');
                        line.push_str(&random_alpha_word(&mut rng, 3, 15));
                        line.push('"');
                    }
                    1 => {
                        line.push_str(&rng.random_range(0..100000u32).to_string());
                    }
                    _ => {
                        line.push_str(if rng.random_bool(0.5) {
                            "true"
                        } else {
                            "false"
                        });
                    }
                }
            }
            line.push('}');
            // Random extra top-level fields
            let extras = rng.random_range(0..4usize);
            for _ in 0..extras {
                let key = random_alpha_word(&mut rng, 3, 10);
                line.push_str(",\"");
                line.push_str(&key);
                line.push_str("\":");
                if rng.random_bool(0.5) {
                    line.push_str(&rng.random_range(0..100000u32).to_string());
                } else {
                    line.push('"');
                    line.push_str(&random_alpha_word(&mut rng, 3, 12));
                    line.push('"');
                }
            }
            line.push('}');
            line
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Markdown fragments generator (headers, lists, code blocks, links)
// ---------------------------------------------------------------------------

const MD_CODE_LANGS: &[&str] = &[
    "rust",
    "python",
    "javascript",
    "go",
    "java",
    "typescript",
    "bash",
    "sql",
    "yaml",
    "json",
];

/// Generates markdown text fragments with random content.
pub fn generate_markdown_fragments(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4001);
    (0..n)
        .map(|_| {
            let num_lines = rng.random_range(2..7usize);
            let mut lines = Vec::with_capacity(num_lines);
            for i in 0..num_lines {
                let kind = if i == 0 {
                    0 // always start with a header
                } else {
                    rng.random_range(1..5u32)
                };
                match kind {
                    0 => {
                        let depth = rng.random_range(1..5usize);
                        let hashes: String = "#".repeat(depth);
                        let nwords = rng.random_range(2..5usize);
                        let words: Vec<String> = (0..nwords)
                            .map(|_| {
                                let mut w = random_alpha_word(&mut rng, 3, 12);
                                // Capitalize first letter
                                if let Some(c) = w.get_mut(0..1) {
                                    c.make_ascii_uppercase();
                                }
                                w
                            })
                            .collect();
                        lines.push(format!("{hashes} {}", words.join(" ")));
                    }
                    1 => {
                        // Bullet list item
                        let nwords = rng.random_range(4..12usize);
                        let words: Vec<String> = (0..nwords)
                            .map(|_| random_alpha_word(&mut rng, 2, 10))
                            .collect();
                        let bullet = if rng.random_bool(0.5) { "-" } else { "*" };
                        lines.push(format!("{bullet} {}", words.join(" ")));
                    }
                    2 => {
                        // Code block
                        let lang = MD_CODE_LANGS[rng.random_range(0..MD_CODE_LANGS.len())];
                        let nwords = rng.random_range(3..8usize);
                        let snippet: Vec<String> = (0..nwords)
                            .map(|_| random_alpha_word(&mut rng, 2, 12))
                            .collect();
                        lines.push(format!("```{lang}\n{}\n```", snippet.join(" ")));
                    }
                    3 => {
                        // Link
                        let nwords = rng.random_range(2..5usize);
                        let prefix: Vec<String> = (0..nwords)
                            .map(|_| random_alpha_word(&mut rng, 2, 10))
                            .collect();
                        let title = random_alpha_word(&mut rng, 4, 15);
                        let domain = random_alpha_word(&mut rng, 4, 12);
                        let path = random_alpha_word(&mut rng, 3, 10);
                        lines.push(format!(
                            "{} [{title}](https://{domain}.com/{path})",
                            prefix.join(" ")
                        ));
                    }
                    _ => {
                        // Plain paragraph
                        let nwords = rng.random_range(6..15usize);
                        let words: Vec<String> = (0..nwords)
                            .map(|_| random_alpha_word(&mut rng, 2, 12))
                            .collect();
                        lines.push(words.join(" "));
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

const ST_JAVA_PKG_PREFIXES: &[&str] = &["com", "org", "io", "net", "dev", "co"];
const ST_EXCEPTION_TYPES: &[&str] = &[
    "RuntimeException",
    "NullPointerException",
    "IllegalStateException",
    "IOException",
    "TimeoutException",
    "SecurityException",
    "IllegalArgumentException",
    "UnsupportedOperationException",
    "ConcurrentModificationException",
    "OutOfMemoryError",
];

/// Generates Java/Python-style stack traces with random package/class/method names.
pub fn generate_stack_traces(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4002);
    (0..n)
        .map(|_| {
            let is_java = rng.random_bool(0.5);
            let depth = rng.random_range(3..10usize);
            let mut frames = Vec::with_capacity(depth);
            for _ in 0..depth {
                if is_java {
                    // Random package: com.example.randomword.randomword
                    let prefix =
                        ST_JAVA_PKG_PREFIXES[rng.random_range(0..ST_JAVA_PKG_PREFIXES.len())];
                    let org = random_alpha_word(&mut rng, 4, 10);
                    let pkg = random_alpha_word(&mut rng, 4, 12);
                    let mut class = random_alpha_word(&mut rng, 5, 15);
                    if let Some(c) = class.get_mut(0..1) {
                        c.make_ascii_uppercase();
                    }
                    let method = random_alpha_word(&mut rng, 4, 15);
                    let line = rng.random_range(10..2000u32);
                    frames.push(format!(
                        "\tat {prefix}.{org}.{pkg}.{class}.{method}({class}.java:{line})"
                    ));
                } else {
                    // Random Python module path
                    let depth_parts = rng.random_range(2..5usize);
                    let parts: Vec<String> = (0..depth_parts)
                        .map(|_| random_alpha_word(&mut rng, 3, 10))
                        .collect();
                    let module = format!("/{}.py", parts.join("/"));
                    let method = random_alpha_word(&mut rng, 4, 15);
                    let line = rng.random_range(10..2000u32);
                    frames.push(format!("  File \"{module}\", line {line}, in {method}"));
                }
            }
            if is_java {
                let exc = ST_EXCEPTION_TYPES[rng.random_range(0..ST_EXCEPTION_TYPES.len())];
                let msg = random_alpha_word(&mut rng, 5, 20);
                format!("java.lang.{exc}: {msg}\n{}", frames.join("\n"))
            } else {
                let mut exc_class = random_alpha_word(&mut rng, 5, 15);
                if let Some(c) = exc_class.get_mut(0..1) {
                    c.make_ascii_uppercase();
                }
                let msg = random_alpha_word(&mut rng, 5, 25);
                format!(
                    "Traceback (most recent call last):\n{}\n{exc_class}Error: {msg}",
                    frames.join("\n")
                )
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CSS rules generator (selectors, properties, values)
// ---------------------------------------------------------------------------

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
    "min-height",
    "position",
    "overflow",
    "gap",
    "border-radius",
    "box-shadow",
    "opacity",
    "z-index",
    "transition",
    "transform",
    "cursor",
    "text-decoration",
    "text-align",
    "line-height",
    "letter-spacing",
    "flex",
    "grid-template-columns",
    "grid-gap",
    "white-space",
    "text-overflow",
    "visibility",
    "outline",
    "background",
    "top",
    "left",
    "right",
    "bottom",
];
const CSS_SELECTOR_PREFIXES: &[&str] = &[".", "#", "", ".", "#", ".", ".", ""];

/// Generates CSS rule blocks with random selectors and properties.
pub fn generate_css_rules(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4003);
    (0..n)
        .map(|_| {
            // Random selector
            let prefix = CSS_SELECTOR_PREFIXES[rng.random_range(0..CSS_SELECTOR_PREFIXES.len())];
            let name = random_alpha_word(&mut rng, 3, 15);
            let selector = if rng.random_bool(0.3) {
                // Compound selector
                let prefix2 =
                    CSS_SELECTOR_PREFIXES[rng.random_range(0..CSS_SELECTOR_PREFIXES.len())];
                let name2 = random_alpha_word(&mut rng, 3, 12);
                let combinator = match rng.random_range(0..3u32) {
                    0 => " > ",
                    1 => " ",
                    _ => " ~ ",
                };
                format!("{prefix}{name}{combinator}{prefix2}{name2}")
            } else if rng.random_bool(0.2) {
                // Pseudo-class
                let pseudo = match rng.random_range(0..4u32) {
                    0 => ":hover",
                    1 => ":focus",
                    2 => ":active",
                    _ => "::before",
                };
                format!("{prefix}{name}{pseudo}")
            } else {
                format!("{prefix}{name}")
            };

            let num_props = rng.random_range(2..7usize);
            let mut declarations = Vec::with_capacity(num_props);
            for _ in 0..num_props {
                let prop = CSS_PROPERTIES[rng.random_range(0..CSS_PROPERTIES.len())];
                // Generate random but plausible values
                let value = match prop {
                    "display" => match rng.random_range(0..5u32) {
                        0 => "flex",
                        1 => "grid",
                        2 => "block",
                        3 => "inline-block",
                        _ => "none",
                    }
                    .to_string(),
                    "color" | "background-color" | "border-color" => {
                        format!("#{:06x}", rng.random_range(0..0xFFFFFFu32))
                    }
                    "padding" | "margin" | "gap" | "top" | "left" | "right" | "bottom" => {
                        format!("{}px", rng.random_range(0..64u32))
                    }
                    "font-size" | "line-height" | "letter-spacing" => {
                        format!("{:.1}rem", rng.random_range(5..30u32) as f64 / 10.0)
                    }
                    "width" | "height" | "max-width" | "min-height" => {
                        if rng.random_bool(0.5) {
                            format!("{}px", rng.random_range(10..1200u32))
                        } else {
                            format!("{}%", rng.random_range(10..101u32))
                        }
                    }
                    "z-index" => rng.random_range(1..1000u32).to_string(),
                    "opacity" => format!("{:.2}", rng.random_range(0..100u32) as f64 / 100.0),
                    "border-radius" => format!("{}px", rng.random_range(0..24u32)),
                    "box-shadow" => format!(
                        "{}px {}px {}px rgba(0, 0, 0, {:.2})",
                        rng.random_range(0..10u32),
                        rng.random_range(0..10u32),
                        rng.random_range(0..20u32),
                        rng.random_range(5..50u32) as f64 / 100.0,
                    ),
                    _ => random_alpha_word(&mut rng, 3, 12),
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

const SHELL_COMMANDS_LIST: &[&str] = &[
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
    "helm",
    "make",
    "go",
    "python3",
    "pip",
    "gradle",
    "mvn",
    "systemctl",
    "journalctl",
    "find",
    "tar",
    "gzip",
    "chmod",
    "chown",
    "ln",
    "cp",
    "mv",
];

const SHELL_KUBECTL_SUB: &[&str] = &[
    "get pods",
    "get deployments",
    "get services",
    "get nodes",
    "get namespaces",
    "describe pod",
    "describe service",
    "logs",
    "apply -f",
    "delete pod",
    "rollout status",
    "rollout restart",
    "scale deployment",
    "exec -it",
    "port-forward",
    "top pods",
    "get events",
    "get configmaps",
];

const SHELL_DOCKER_SUB: &[&str] = &[
    "run -d",
    "build -t",
    "compose up -d",
    "compose down",
    "ps -a",
    "exec -it",
    "pull",
    "push",
    "logs --tail=100",
    "inspect",
    "network create",
    "volume create",
    "system prune",
    "image ls",
];

/// Generates shell command lines with random subcommands, flags, and arguments.
pub fn generate_shell_commands(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(4004);
    (0..n)
        .map(|_| {
            let cmd = SHELL_COMMANDS_LIST[rng.random_range(0..SHELL_COMMANDS_LIST.len())];
            let mut parts = Vec::with_capacity(8);
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
                _ => {
                    // Random subcommand
                    parts.push(random_alpha_word(&mut rng, 2, 10));
                }
            }
            // Random flags (1-5)
            let num_flags = rng.random_range(1..6usize);
            for _ in 0..num_flags {
                if rng.random_bool(0.5) {
                    // Long flag
                    let flag_name = random_alpha_word(&mut rng, 3, 12);
                    if rng.random_bool(0.5) {
                        let val = random_word(&mut rng, 2, 15);
                        parts.push(format!("--{flag_name}={val}"));
                    } else {
                        parts.push(format!("--{flag_name}"));
                    }
                } else {
                    // Short flag
                    let flag_char = (b'a' + rng.random_range(0..26u8)) as char;
                    if rng.random_bool(0.4) {
                        let val = random_word(&mut rng, 2, 10);
                        parts.push(format!("-{flag_char} {val}"));
                    } else {
                        parts.push(format!("-{flag_char}"));
                    }
                }
            }
            // Random argument(s)
            let nargs = rng.random_range(1..3usize);
            for _ in 0..nargs {
                match rng.random_range(0..4u32) {
                    0 => {
                        // URL
                        let domain = random_alpha_word(&mut rng, 4, 10);
                        let path = random_alpha_word(&mut rng, 3, 8);
                        parts.push(format!("https://{domain}.com/{path}"));
                    }
                    1 => {
                        // File path
                        let dir = random_alpha_word(&mut rng, 3, 8);
                        let file = random_alpha_word(&mut rng, 3, 10);
                        parts.push(format!("./{dir}/{file}"));
                    }
                    2 => {
                        // Docker image
                        let img = random_alpha_word(&mut rng, 3, 10);
                        let tag = random_alpha_word(&mut rng, 2, 6);
                        parts.push(format!("{img}:{tag}"));
                    }
                    _ => {
                        parts.push(random_word(&mut rng, 3, 15));
                    }
                }
            }
            let mut command = parts.join(" ");
            // Optional pipe
            if rng.random_bool(0.35) {
                let pipe_cmd = match rng.random_range(0..5u32) {
                    0 => format!("| jq '.{}'", random_alpha_word(&mut rng, 3, 10)),
                    1 => format!("| grep {}", random_alpha_word(&mut rng, 3, 8)),
                    2 => "| wc -l".to_string(),
                    3 => format!("| head -{}", rng.random_range(5..100u32)),
                    _ => format!("| sort -k{} -rn", rng.random_range(1..5u32)),
                };
                command.push(' ');
                command.push_str(&pipe_cmd);
            }
            command
        })
        .collect()
}
