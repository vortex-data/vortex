//! Measures the AOT code size of `unpack_cmp` for all 6 comparison ops on
//! `u32`. Six distinct closure call-sites force six independent sets of
//! width-monomorphizations.

use aot_size::force_cmp;

fn main() {
    let width: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let input = vec![0u32; 1024];
    let mut output = [false; 1024];

    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a == b, 0u32);
    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a != b, 0u32);
    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a < b, 0u32);
    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a <= b, 0u32);
    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a > b, 0u32);
    force_cmp::<u32, u32, _>(width, &input, &mut output, |a, b| a >= b, 0u32);

    println!("{}", std::hint::black_box(output.iter().filter(|&&b| b).count()));
}
