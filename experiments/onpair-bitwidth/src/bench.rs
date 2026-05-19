use onpair_rs::{OnPair, OnPairOpt, OnPairOptParams};
use std::env;
use std::fs;
use std::time::Instant;

fn run_baseline(strings: &[&str], thr: u16, total_bytes: usize) -> f64 {
    let mut onpair = OnPair::with_capacity(thr, strings.len(), total_bytes);
    let t0 = Instant::now();
    onpair.compress_strings(strings);
    let dt = t0.elapsed().as_secs_f64();
    let used = onpair.space_used();
    let ratio = total_bytes as f64 / used as f64;
    println!("  {:40} | total {:>8}B | ratio {:.4}x | {:.2}s",
        "OnPair baseline", used, ratio, dt);
    ratio
}

fn run_opt(strings: &[&str], total_bytes: usize, params: OnPairOptParams, label: &str, baseline: f64) -> (f64, OnPairOpt) {
    let mut opt = OnPairOpt::new(params);
    let t0 = Instant::now();
    opt.compress_strings(strings);
    let dt = t0.elapsed().as_secs_f64();
    let used = opt.space_used();
    let ratio = total_bytes as f64 / used as f64;
    let delta = (ratio - baseline) / baseline * 100.0;
    println!("  {:40} | total {:>8}B | ratio {:.4}x | {:+.2}% | k=2^{} b1={} b2={} N={} | {:.2}s",
        label, used, ratio, delta, opt.log2_k(), opt.b1(), opt.b2(), opt.n_tokens(), dt);
    (ratio, opt)
}

fn roundtrip(strings: &[&str], opt: &OnPairOpt) -> Result<(), String> {
    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0) + 32;
    let mut buf = vec![0u8; max_len];
    for (i, s) in strings.iter().enumerate() {
        let n = opt.decompress_string(i, &mut buf);
        if &buf[..n] != s.as_bytes() {
            return Err(format!("mismatch at index {}: expected {:?}, got {:?}",
                i, s, String::from_utf8_lossy(&buf[..n])));
        }
    }
    Ok(())
}

fn run_dataset(path: &str) {
    let raw = fs::read_to_string(path).expect("read file");
    let strings: Vec<&str> = raw.lines().filter(|s| !s.is_empty()).collect();
    let total_bytes: usize = strings.iter().map(|s| s.len()).sum();
    let thr = ((total_bytes as f64 / (1024.0 * 1024.0)).log2() as u16).max(2);

    println!("\n=== {} ({} strings, {} B, thr {}) ===", path, strings.len(), total_bytes, thr);

    let base = run_baseline(&strings, thr, total_bytes);

    // OnPairOpt with no Picky (tau_num=0 disables eviction): pure two-tier bit-pack effect.
    let p = OnPairOptParams { threshold: thr, tau_num: 0, ..Default::default() };
    let (_, opt) = run_opt(&strings, total_bytes, p, "Opt (no Picky, auto-k)", base);
    if let Err(e) = roundtrip(&strings, &opt) { println!("  ROUNDTRIP FAIL: {}", e); return; }

    // OnPairOpt with Picky, single pass, sweep tau.
    for (n, d) in [(70u32, 100), (80, 100), (90, 100)] {
        let label = format!("Opt picky τ={:.2}", n as f64 / d as f64);
        let p = OnPairOptParams { threshold: thr, tau_num: n, tau_den: d, ..Default::default() };
        let (_, opt) = run_opt(&strings, total_bytes, p, &label, base);
        roundtrip(&strings, &opt).unwrap();
    }
    // Multipass too:
    for &passes in &[2u32, 3] {
        let label = format!("Opt picky τ=0.80 P={}", passes);
        let p = OnPairOptParams { threshold: thr, tau_num: 80, tau_den: 100, passes, ..Default::default() };
        let (_, opt) = run_opt(&strings, total_bytes, p, &label, base);
        roundtrip(&strings, &opt).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let datasets: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        vec!["/tmp/words.txt".to_string(), "/tmp/domains_200k.txt".to_string()]
    };
    for d in &datasets {
        run_dataset(d);
    }
}
