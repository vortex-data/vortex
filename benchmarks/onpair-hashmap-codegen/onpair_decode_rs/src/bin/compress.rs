//! Compression-time benchmark for both Rust OnPair implementations.
//! usage: compress DATASET.txt [bits] [iterations]

use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset = &args[1];
    let bits: u8 = args.get(2).map_or(16, |a| a.parse().unwrap());
    let iterations: usize = args.get(3).map_or(5, |a| a.parse().unwrap());

    let text = std::fs::read(dataset).unwrap();
    let mut data: Vec<u8> = Vec::with_capacity(text.len());
    let mut offsets: Vec<u32> = vec![0];
    for line in text.split(|&b| b == b'\n') {
        data.extend_from_slice(line);
        offsets.push(data.len() as u32);
    }
    offsets.pop();
    let rows = offsets.len() - 1;
    data.truncate(offsets[rows] as usize);
    let mib = data.len() as f64 / 1048576.0;

    {
        let cfg = onpair::Config {
            max_dict_bits: onpair::MaxDictBits::new(bits).unwrap(),
            threshold: onpair::Threshold::new(0.15).unwrap(),
            seed: Some(42),
        };
        let mut train = Vec::new();
        let mut parse = Vec::new();
        let mut codes = 0;
        for _ in 0..iterations {
            let t0 = Instant::now();
            let parser = onpair::Parser::train(&data, &offsets, cfg).unwrap();
            let t1 = Instant::now();
            let column = parser.parse(&data, &offsets).unwrap();
            let t2 = Instant::now();
            train.push((t1 - t0).as_secs_f64() * 1000.0);
            parse.push((t2 - t1).as_secs_f64() * 1000.0);
            codes = column.view().codes.len();
        }
        let (tr, pa) = (median(train), median(parse));
        println!(
            "compress,impl=spiraldb_onpair,dataset={dataset},bits={bits},mib={mib:.2},codes={codes},\
             train_ms={tr:.2},parse_ms={pa:.2},train_mibs={:.1},parse_mibs={:.1},total_mibs={:.1}",
            mib / (tr / 1000.0), mib / (pa / 1000.0), mib / ((tr + pa) / 1000.0));
    }

    if bits == 16 {
        let ends: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();
        for variant in ["onpair_rs_OnPair16", "onpair_rs_OnPair"] {
            let mut total = Vec::new();
            for _ in 0..iterations {
                let t0 = Instant::now();
                if variant == "onpair_rs_OnPair16" {
                    let mut c = onpair_rs::OnPair16::new(5);
                    c.compress_bytes(&data, &ends);
                    std::hint::black_box(c.space_used());
                } else {
                    let mut c = onpair_rs::OnPair::new(5);
                    c.compress_bytes(&data, &ends);
                    std::hint::black_box(c.space_used());
                }
                total.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let t = median(total);
            println!(
                "compress,impl={variant},dataset={dataset},bits={bits},mib={mib:.2},codes=0,\
                 train_ms=0.00,parse_ms=0.00,train_mibs=0.0,parse_mibs=0.0,total_mibs={:.1}",
                mib / (t / 1000.0));
        }
    }
}
