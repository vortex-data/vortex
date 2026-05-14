//! Measures the AOT code size of `BitPacking::unpack` alone, for one type
//! per binary build. Pass the type on the command line: `u8`, `u16`, `u32`,
//! or `u64`.

use aot_size::force_unpack;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "u32".to_string());
    let width: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    match arg.as_str() {
        "u8" => {
            let input = vec![0u8; 1024];
            let mut output = vec![0u8; 1024];
            force_unpack(width, &input, &mut output);
            black_print(output.iter().map(|&x| x as u64).sum::<u64>());
        }
        "u16" => {
            let input = vec![0u16; 1024];
            let mut output = vec![0u16; 1024];
            force_unpack(width, &input, &mut output);
            black_print(output.iter().map(|&x| x as u64).sum::<u64>());
        }
        "u32" => {
            let input = vec![0u32; 1024];
            let mut output = vec![0u32; 1024];
            force_unpack(width, &input, &mut output);
            black_print(output.iter().map(|&x| x as u64).sum::<u64>());
        }
        "u64" => {
            let input = vec![0u64; 1024];
            let mut output = vec![0u64; 1024];
            force_unpack(width, &input, &mut output);
            black_print(output.iter().sum::<u64>());
        }
        other => panic!("unknown type: {other}"),
    }
}

fn black_print(v: u64) {
    println!("{}", std::hint::black_box(v));
}
