//! FSST Compression Analysis Tool
//!
//! Generates diverse string datasets (URLs, logs, JSON, emails, etc.),
//! compresses each with FSST, and outputs detailed symbol table statistics
//! and escape code distributions as CSV for plotting.
//!
//! Run: cargo run --example fsst_compress_analyze -p vortex-fsst

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::use_debug,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::min_ident_chars
)]

use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Zipf sampler
// ---------------------------------------------------------------------------

struct ZipfSampler {
    cdf: Vec<f64>,
}

impl ZipfSampler {
    fn new(n: usize) -> Self {
        let harmonic: f64 = (1..=n).map(|r| 1.0 / r as f64).sum();
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for r in 1..=n {
            acc += 1.0 / (r as f64 * harmonic);
            cdf.push(acc);
        }
        Self { cdf }
    }

    fn sample(&self, rng: &mut StdRng) -> usize {
        let u: f64 = rng.random();
        self.cdf.partition_point(|&p| p < u).min(self.cdf.len() - 1)
    }
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

fn generate_english_prose(n: usize, rng: &mut StdRng) -> Vec<String> {
    let words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "implementation",
        "performance",
        "algorithm",
        "database",
        "compression",
        "distributed",
        "infrastructure",
        "optimization",
        "architecture",
        "framework",
    ];
    let sampler = ZipfSampler::new(words.len());
    (0..n)
        .map(|_| {
            let word_count = rng.random_range(5..30);
            let mut sentence = String::with_capacity(word_count * 6);
            for i in 0..word_count {
                if i > 0 {
                    sentence.push(' ');
                }
                sentence.push_str(words[sampler.sample(rng)]);
            }
            sentence
        })
        .collect()
}

fn generate_urls(n: usize, rng: &mut StdRng) -> Vec<String> {
    let domains = [
        "google.com",
        "facebook.com",
        "youtube.com",
        "amazon.com",
        "wikipedia.org",
        "twitter.com",
        "instagram.com",
        "linkedin.com",
        "reddit.com",
        "netflix.com",
        "github.com",
        "stackoverflow.com",
        "medium.com",
        "nytimes.com",
        "bbc.co.uk",
        "cnn.com",
        "example.com",
        "myapp.io",
        "api.stripe.com",
        "cdn.cloudflare.net",
    ];
    let paths = [
        "/",
        "/index.html",
        "/about",
        "/contact",
        "/api/v1/users",
        "/api/v2/products",
        "/search",
        "/login",
        "/dashboard",
        "/settings",
        "/blog/2024/01/hello-world",
        "/docs/getting-started",
        "/static/css/main.css",
        "/assets/img/logo.png",
        "/feed.xml",
    ];
    let dom_sampler = ZipfSampler::new(domains.len());
    let path_sampler = ZipfSampler::new(paths.len());
    (0..n)
        .map(|_| {
            let proto = if rng.random_range(0..10u32) < 9 {
                "https://"
            } else {
                "http://"
            };
            let domain = domains[dom_sampler.sample(rng)];
            let path = paths[path_sampler.sample(rng)];
            let mut url = format!("{proto}{domain}{path}");
            if rng.random_range(0..10u32) < 3 {
                let _ = write!(
                    url,
                    "?id={}&utm_source=google",
                    rng.random_range(1..100000u32)
                );
            }
            url
        })
        .collect()
}

fn generate_http_logs(n: usize, rng: &mut StdRng) -> Vec<String> {
    let ips = [
        "192.168.1.1",
        "10.0.0.1",
        "172.16.0.1",
        "203.0.113.50",
        "198.51.100.23",
        "8.8.8.8",
        "1.1.1.1",
        "192.0.2.1",
        "100.64.0.1",
        "169.254.1.1",
    ];
    let methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"];
    let paths = [
        "/api/v1/users",
        "/api/v1/products",
        "/api/v2/orders",
        "/health",
        "/metrics",
        "/static/app.js",
        "/login",
        "/logout",
        "/search?q=test",
        "/dashboard",
    ];
    let statuses = [
        200u16, 200, 200, 200, 301, 302, 304, 400, 401, 403, 404, 500,
    ];
    let agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "curl/7.88.1",
        "python-requests/2.31.0",
    ];
    let ip_sampler = ZipfSampler::new(ips.len());
    let method_sampler = ZipfSampler::new(methods.len());
    let path_sampler = ZipfSampler::new(paths.len());
    let agent_sampler = ZipfSampler::new(agents.len());
    (0..n)
        .map(|_| {
            format!(
                "{} - - [10/Oct/2024:13:55:{:02} +0000] \"{} {} HTTP/1.1\" {} {} \"-\" \"{}\"",
                ips[ip_sampler.sample(rng)],
                rng.random_range(0..60u32),
                methods[method_sampler.sample(rng)],
                paths[path_sampler.sample(rng)],
                statuses[rng.random_range(0..statuses.len())],
                rng.random_range(100..50000u32),
                agents[agent_sampler.sample(rng)],
            )
        })
        .collect()
}

fn generate_uuids(n: usize, rng: &mut StdRng) -> Vec<String> {
    (0..n)
        .map(|_| {
            format!(
                "{:08x}-{:04x}-4{:03x}-{}{:03x}-{:012x}",
                rng.random::<u32>(),
                rng.random::<u16>(),
                rng.random_range(0..0x1000u16),
                ['8', '9', 'a', 'b'][rng.random_range(0..4usize)],
                rng.random_range(0..0x1000u16),
                rng.random::<u64>() & 0xFFFF_FFFF_FFFF,
            )
        })
        .collect()
}

fn generate_json(n: usize, rng: &mut StdRng) -> Vec<String> {
    let names = [
        "Alice Johnson",
        "Bob Smith",
        "Charlie Brown",
        "Diana Prince",
        "Eve Adams",
        "Frank Miller",
        "Grace Hopper",
        "Hank Pym",
        "Iris West",
        "Jack Ryan",
        "Karen Page",
        "Leo Messi",
    ];
    let statuses = ["active", "inactive", "pending", "suspended"];
    let cities = [
        "New York",
        "San Francisco",
        "London",
        "Tokyo",
        "Berlin",
        "Paris",
        "Sydney",
        "Toronto",
    ];
    let name_sampler = ZipfSampler::new(names.len());
    let city_sampler = ZipfSampler::new(cities.len());
    (0..n)
        .map(|_| {
            format!(
                "{{\"id\":{},\"name\":\"{}\",\"email\":\"user{}@example.com\",\"status\":\"{}\",\"city\":\"{}\",\"score\":{:.1}}}",
                rng.random_range(1..1000000u32),
                names[name_sampler.sample(rng)],
                rng.random_range(1..10000u32),
                statuses[rng.random_range(0..statuses.len())],
                cities[city_sampler.sample(rng)],
                rng.random_range(0..1000u32) as f64 / 10.0,
            )
        })
        .collect()
}

fn generate_emails(n: usize, rng: &mut StdRng) -> Vec<String> {
    let users = [
        "alice", "bob", "charlie", "diana", "eve", "frank", "grace", "hank", "iris", "jack",
        "admin", "info", "support", "sales", "dev", "noreply", "contact", "help", "billing",
        "team",
    ];
    let domains = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "icloud.com",
        "proton.me",
        "company.com",
        "example.org",
        "university.edu",
        "gov.us",
    ];
    let user_sampler = ZipfSampler::new(users.len());
    let domain_sampler = ZipfSampler::new(domains.len());
    (0..n)
        .map(|_| {
            let user = users[user_sampler.sample(rng)];
            let domain = domains[domain_sampler.sample(rng)];
            if rng.random_range(0..3u32) == 0 {
                format!("{}.{}@{}", user, rng.random_range(1..100u32), domain)
            } else {
                format!("{user}@{domain}")
            }
        })
        .collect()
}

fn generate_file_paths(n: usize, rng: &mut StdRng) -> Vec<String> {
    let dirs = [
        "home",
        "user",
        "src",
        "lib",
        "bin",
        "etc",
        "var",
        "tmp",
        "opt",
        "data",
        "app",
        "config",
        "logs",
        "docs",
        "test",
        "build",
        "dist",
        "node_modules",
        "target",
        "out",
    ];
    let files = [
        "main.rs",
        "lib.rs",
        "mod.rs",
        "index.js",
        "app.py",
        "config.yaml",
        "Cargo.toml",
        "README.md",
        "Makefile",
        ".gitignore",
        "test.go",
        "server.ts",
    ];
    let dir_sampler = ZipfSampler::new(dirs.len());
    let file_sampler = ZipfSampler::new(files.len());
    (0..n)
        .map(|_| {
            let depth = rng.random_range(2..6usize);
            let mut path = String::from("/");
            for i in 0..depth {
                path.push_str(dirs[dir_sampler.sample(rng)]);
                if i < depth - 1 {
                    path.push('/');
                }
            }
            path.push('/');
            path.push_str(files[file_sampler.sample(rng)]);
            path
        })
        .collect()
}

fn generate_timestamps(n: usize, rng: &mut StdRng) -> Vec<String> {
    (0..n)
        .map(|_| {
            format!(
                "2024-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
                rng.random_range(1..13u32),
                rng.random_range(1..29u32),
                rng.random_range(0..24u32),
                rng.random_range(0..60u32),
                rng.random_range(0..60u32),
                rng.random_range(0..1000u32),
            )
        })
        .collect()
}

fn generate_base64(n: usize, rng: &mut StdRng) -> Vec<String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    (0..n)
        .map(|_| {
            let len = rng.random_range(20..80usize);
            let mut s = String::with_capacity(len + 2);
            for _ in 0..len {
                s.push(CHARS[rng.random_range(0..64usize)] as char);
            }
            if len % 3 == 1 {
                s.push_str("==");
            } else if len % 3 == 2 {
                s.push('=');
            }
            s
        })
        .collect()
}

fn generate_code_identifiers(n: usize, rng: &mut StdRng) -> Vec<String> {
    let words = [
        "get", "set", "create", "delete", "update", "find", "parse", "build", "check", "validate",
        "process", "handle", "init", "load", "save", "read", "write", "open", "close", "start",
        "stop", "run", "exec", "config", "data", "result", "error", "status", "value", "type",
        "name", "path", "file", "user", "item", "list", "map", "node", "tree", "buffer",
    ];
    let sampler = ZipfSampler::new(words.len());
    (0..n)
        .map(|_| {
            let parts = rng.random_range(2..5usize);
            let use_snake = rng.random_range(0..2u32) == 0;
            let mut ident = String::with_capacity(parts * 8);
            for i in 0..parts {
                let word = words[sampler.sample(rng)];
                if use_snake {
                    if i > 0 {
                        ident.push('_');
                    }
                    ident.push_str(word);
                } else if i == 0 {
                    ident.push_str(word);
                } else {
                    let mut chars = word.chars();
                    if let Some(c) = chars.next() {
                        ident.push(c.to_ascii_uppercase());
                        ident.extend(chars);
                    }
                }
            }
            ident
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct AnalysisResult {
    name: String,
    n_strings: usize,
    avg_len: f64,
    total_bytes: usize,
    n_symbols: usize,
    mean_sym_len: f64,
    compression_ratio: f64,
    escape_rate: f64,
    codes_per_string: f64,
    entropy: f64,
    effective_alphabet: f64,
    sym_len_histogram: [usize; 9], // lengths 1..=8
    code_freq: Vec<(u8, usize)>,   // top-10 most frequent codes
}

fn compress_and_analyze(name: &str, strings: &[String]) -> AnalysisResult {
    let bytes_vec: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
    let total_bytes: usize = bytes_vec.iter().map(|b| b.len()).sum();
    let avg_len = total_bytes as f64 / strings.len() as f64;

    // Train compressor
    let compressor = fsst::Compressor::train(&bytes_vec);

    // Compress all strings
    let compressed: Vec<Vec<u8>> = bytes_vec.iter().map(|b| compressor.compress(b)).collect();
    let total_codes: usize = compressed.iter().map(|c| c.len()).sum();

    // Symbol table analysis
    let symbols = compressor.symbol_table();
    let symbol_lengths = compressor.symbol_lengths();
    let n_symbols = symbols.len();

    let mean_sym_len = if n_symbols > 0 {
        symbol_lengths[..n_symbols]
            .iter()
            .map(|&l| l as f64)
            .sum::<f64>()
            / n_symbols as f64
    } else {
        1.0
    };

    let compression_ratio = total_bytes as f64 / total_codes as f64;

    // Symbol length histogram
    let mut sym_len_histogram = [0usize; 9];
    for &len in &symbol_lengths[..n_symbols] {
        if (1..=8).contains(&len) {
            sym_len_histogram[len as usize] += 1;
        }
    }

    // Count escape codes and code frequencies
    let escape_code = 255u8;
    let mut code_counts = [0u64; 256];
    let mut escape_count = 0u64;
    for codes in &compressed {
        let mut i = 0;
        while i < codes.len() {
            if codes[i] == escape_code {
                escape_count += 1;
                i += 2; // skip the literal byte
            } else {
                code_counts[codes[i] as usize] += 1;
                i += 1;
            }
        }
    }

    let total_symbols_used: u64 = code_counts.iter().sum::<u64>() + escape_count;
    let escape_rate = if total_symbols_used > 0 {
        escape_count as f64 / total_symbols_used as f64
    } else {
        0.0
    };

    // Shannon entropy of code distribution
    let entropy = {
        let total = total_symbols_used as f64;
        if total == 0.0 {
            0.0
        } else {
            let mut h = 0.0;
            for &count in code_counts.iter() {
                if count > 0 {
                    let p = count as f64 / total;
                    h -= p * p.log2();
                }
            }
            if escape_count > 0 {
                let p = escape_count as f64 / total;
                h -= p * p.log2();
            }
            h
        }
    };

    let effective_alphabet = 2.0f64.powf(entropy);

    let codes_per_string = total_codes as f64 / strings.len() as f64;

    // Top-10 most frequent codes
    let mut code_freq: Vec<(u8, usize)> = code_counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| (i as u8, c as usize))
        .collect();
    code_freq.sort_by(|a, b| b.1.cmp(&a.1));
    code_freq.truncate(10);

    // Print symbol details
    println!("\n## {name}");
    println!(
        "  Strings: {} | Avg len: {:.1} | Total: {:.1} KB",
        strings.len(),
        avg_len,
        total_bytes as f64 / 1024.0
    );
    println!(
        "  Symbols: {} | Mean sym len: {:.2} | Compression: {:.2}x",
        n_symbols, mean_sym_len, compression_ratio
    );
    println!(
        "  Escape rate: {:.1}% | Codes/string: {:.1} | Entropy: {:.2} bits",
        escape_rate * 100.0,
        codes_per_string,
        entropy
    );
    println!(
        "  Effective alphabet: {:.1} | Sym len dist: {:?}",
        effective_alphabet,
        &sym_len_histogram[1..]
    );

    // Print top-5 symbols with their byte representation
    println!("  Top-5 symbols:");
    for (i, (sym, &len)) in symbols[..n_symbols]
        .iter()
        .zip(&symbol_lengths[..n_symbols])
        .enumerate()
        .take(5)
    {
        let sym_bytes = sym.to_u64().to_le_bytes();
        let bytes = &sym_bytes[..len as usize];
        let display = String::from_utf8_lossy(bytes);
        println!("    [{i}] len={len} bytes={bytes:?} \"{display}\"");
    }

    AnalysisResult {
        name: name.to_string(),
        n_strings: strings.len(),
        avg_len,
        total_bytes,
        n_symbols,
        mean_sym_len,
        compression_ratio,
        escape_rate,
        codes_per_string,
        entropy,
        effective_alphabet,
        sym_len_histogram,
        code_freq,
    }
}

// ---------------------------------------------------------------------------
// Noise sweep: mix random bytes into English text
// ---------------------------------------------------------------------------

fn generate_noisy_text(n: usize, noise_pct: f64, rng: &mut StdRng) -> Vec<String> {
    let base = generate_english_prose(n, rng);
    if noise_pct == 0.0 {
        return base;
    }
    base.into_iter()
        .map(|s| {
            s.bytes()
                .map(|b| {
                    if rng.random::<f64>() < noise_pct {
                        rng.random_range(0..=255u8) as char
                    } else {
                        b as char
                    }
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

fn write_csv(results: &[AnalysisResult], path: &str) {
    let mut f = fs::File::create(path).expect("failed to create CSV");
    writeln!(
        f,
        "dataset,n_strings,avg_len,total_bytes,n_symbols,mean_sym_len,compression_ratio,escape_rate,codes_per_string,entropy,effective_alphabet,sym_1,sym_2,sym_3,sym_4,sym_5,sym_6,sym_7,sym_8"
    )
    .unwrap();
    for r in results {
        writeln!(
            f,
            "{},{},{:.2},{},{},{:.3},{:.3},{:.4},{:.2},{:.3},{:.2},{},{},{},{},{},{},{},{}",
            r.name,
            r.n_strings,
            r.avg_len,
            r.total_bytes,
            r.n_symbols,
            r.mean_sym_len,
            r.compression_ratio,
            r.escape_rate,
            r.codes_per_string,
            r.entropy,
            r.effective_alphabet,
            r.sym_len_histogram[1],
            r.sym_len_histogram[2],
            r.sym_len_histogram[3],
            r.sym_len_histogram[4],
            r.sym_len_histogram[5],
            r.sym_len_histogram[6],
            r.sym_len_histogram[7],
            r.sym_len_histogram[8],
        )
        .unwrap();
    }
    eprintln!("Wrote {path}");
}

fn write_noise_csv(sweep: &[(f64, AnalysisResult)], path: &str) {
    let mut f = fs::File::create(path).expect("failed to create CSV");
    writeln!(
        f,
        "noise_pct,n_symbols,mean_sym_len,compression_ratio,escape_rate,codes_per_string,entropy,effective_alphabet"
    )
    .unwrap();
    for (noise, r) in sweep {
        writeln!(
            f,
            "{:.2},{},{:.3},{:.3},{:.4},{:.2},{:.3},{:.2}",
            noise,
            r.n_symbols,
            r.mean_sym_len,
            r.compression_ratio,
            r.escape_rate,
            r.codes_per_string,
            r.entropy,
            r.effective_alphabet,
        )
        .unwrap();
    }
    eprintln!("Wrote {path}");
}

fn write_code_freq_csv(results: &[AnalysisResult], path: &str) {
    let mut f = fs::File::create(path).expect("failed to create CSV");
    writeln!(f, "dataset,code,frequency,rank").unwrap();
    for r in results {
        for (rank, &(code, freq)) in r.code_freq.iter().enumerate() {
            writeln!(f, "{},{},{},{}", r.name, code, freq, rank + 1).unwrap();
        }
    }
    eprintln!("Wrote {path}");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n = 50_000;
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("# FSST Compression Analysis");
    println!("Generating {} strings per dataset...\n", n);

    // Create output directory
    let out_dir = "encodings/fsst/data";
    fs::create_dir_all(out_dir).ok();

    // Generate all datasets
    let datasets: Vec<(&str, Vec<String>)> = vec![
        ("english_prose", generate_english_prose(n, &mut rng)),
        ("urls", generate_urls(n, &mut rng)),
        ("http_logs", generate_http_logs(n, &mut rng)),
        ("uuids", generate_uuids(n, &mut rng)),
        ("json", generate_json(n, &mut rng)),
        ("emails", generate_emails(n, &mut rng)),
        ("file_paths", generate_file_paths(n, &mut rng)),
        ("timestamps", generate_timestamps(n, &mut rng)),
        ("base64", generate_base64(n, &mut rng)),
        ("code_identifiers", generate_code_identifiers(n, &mut rng)),
    ];

    // Analyze each
    let mut results = Vec::new();
    for (name, strings) in &datasets {
        results.push(compress_and_analyze(name, strings));
    }

    // Summary table
    println!("\n\n# Summary");
    println!(
        "| Dataset | Symbols | Mean Sym Len | Compression | Escape % | Entropy | Eff. Alphabet |"
    );
    println!(
        "|---------|---------|-------------|-------------|----------|---------|---------------|"
    );
    for r in &results {
        println!(
            "| {} | {} | {:.2} | {:.2}x | {:.1}% | {:.2} | {:.1} |",
            r.name,
            r.n_symbols,
            r.mean_sym_len,
            r.compression_ratio,
            r.escape_rate * 100.0,
            r.entropy,
            r.effective_alphabet,
        );
    }

    // Write CSVs
    write_csv(&results, &format!("{out_dir}/compress_analysis.csv"));
    write_code_freq_csv(&results, &format!("{out_dir}/code_freq.csv"));

    // Noise sweep
    println!("\n\n# Noise Sweep (English prose + random bytes)");
    let noise_levels = [
        0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];
    let mut sweep = Vec::new();
    for &noise in &noise_levels {
        let strings = generate_noisy_text(n, noise, &mut rng);
        let result = compress_and_analyze(&format!("noise_{:.0}pct", noise * 100.0), &strings);
        sweep.push((noise, result));
    }
    write_noise_csv(&sweep, &format!("{out_dir}/noise_sweep.csv"));

    println!("\n\nDone! CSVs written to {out_dir}/");
}
