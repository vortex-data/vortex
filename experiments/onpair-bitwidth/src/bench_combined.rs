use onpair_rs::{
    OnPair, OnPair4Tier, OnPairCombined, OnPairFcDict, OnPairFcDictParams, OnPairOpt,
    OnPairOptParams,
};
use std::env;
use std::fs;
use std::time::Instant;

fn run_dataset(path: &str) {
    let raw = fs::read_to_string(path).expect("read");
    let strings: Vec<&str> = raw.lines().filter(|s| !s.is_empty()).collect();
    let total_bytes: usize = strings.iter().map(|s| s.len()).sum();
    let thr = ((total_bytes as f64 / (1024.0 * 1024.0)).log2() as u16).max(2);
    println!("\n=== {} ({} strings, {} B, thr {}) ===", path, strings.len(), total_bytes, thr);

    // Baseline OnPair
    let mut op = OnPair::with_capacity(thr, strings.len(), total_bytes);
    let t0 = Instant::now();
    op.compress_strings(&strings);
    let dt0 = t0.elapsed().as_secs_f64();
    let base_used = op.space_used();
    let base_ratio = total_bytes as f64 / base_used as f64;
    println!("  {:32} | {:>9}B | ratio {:.4}x | {:.2}s",
        "OnPair (baseline)", base_used, base_ratio, dt0);

    // OnPairOpt (Picky + 2-tier)
    let p = OnPairOptParams { threshold: thr, ..Default::default() };
    let mut o = OnPairOpt::new(p);
    let t0 = Instant::now();
    o.compress_strings(&strings);
    let dt1 = t0.elapsed().as_secs_f64();
    let used1 = o.space_used();
    let r1 = total_bytes as f64 / used1 as f64;
    println!("  {:32} | {:>9}B | ratio {:.4}x ({:+.2}%) | {:.2}s",
        "OnPairOpt (2-tier)", used1, r1, (r1 / base_ratio - 1.0) * 100.0, dt1);

    // OnPair4Tier
    let mut t = OnPair4Tier::new(p);
    let t0 = Instant::now();
    t.compress_strings(&strings);
    let dt2 = t0.elapsed().as_secs_f64();
    let used2 = t.space_used();
    let r2 = total_bytes as f64 / used2 as f64;
    println!("  {:32} | {:>9}B | ratio {:.4}x ({:+.2}%) | {:.2}s",
        "OnPair4Tier", used2, r2, (r2 / base_ratio - 1.0) * 100.0, dt2);

    // OnPairFcDict (front-coded dict + 2-tier stream)
    let pfc = OnPairFcDictParams { threshold: thr, ..Default::default() };
    let mut fc = OnPairFcDict::new(pfc);
    let t0 = Instant::now();
    fc.compress_strings(&strings);
    let dt3 = t0.elapsed().as_secs_f64();
    let used3 = fc.space_used();
    let r3 = total_bytes as f64 / used3 as f64;
    println!("  {:32} | {:>9}B | ratio {:.4}x ({:+.2}%) | {:.2}s",
        "OnPairFcDict (FC + 2-tier)", used3, r3, (r3 / base_ratio - 1.0) * 100.0, dt3);

    // OnPairCombined (FC + 4-tier)
    let mut c = OnPairCombined::new(p);
    let t0 = Instant::now();
    c.compress_strings(&strings);
    let dt4 = t0.elapsed().as_secs_f64();
    let used4 = c.space_used();
    let r4 = total_bytes as f64 / used4 as f64;
    println!("  {:32} | {:>9}B | ratio {:.4}x ({:+.2}%) | {:.2}s",
        "OnPairCombined (FC + 4-tier)", used4, r4, (r4 / base_ratio - 1.0) * 100.0, dt4);
    println!("    widths={:?}  tier_sizes={:?}  bucket={}", c.widths(), c.tier_sizes(), c.bucket_size());

    // Roundtrip
    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0) + 32;
    let mut buf = vec![0u8; max_len];
    for (i, s) in strings.iter().enumerate() {
        let n = c.decompress_string(i, &mut buf);
        if &buf[..n] != s.as_bytes() {
            println!("  ROUNDTRIP FAIL at {}: expected {:?}, got {:?}", i, s, String::from_utf8_lossy(&buf[..n]));
            return;
        }
    }
    println!("  Roundtrip: PASS");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let datasets: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        vec![
            "/tmp/words.txt".to_string(),
            "/tmp/domains_200k.txt".to_string(),
            "/tmp/imdb_titles.txt".to_string(),
            "/tmp/wiki_titles.txt".to_string(),
        ]
    };
    for d in &datasets {
        run_dataset(d);
    }
}
