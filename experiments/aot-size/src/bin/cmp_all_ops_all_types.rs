//! Measures the AOT code size of `unpack_cmp` for all 6 ops across all 4
//! integer types — the maximum cartesian product that ships in fastlanes.
//! Widths exercised: u8: 1..=7, u16: 1..=15, u32: 1..=31, u64: 1..=63
//! ⇒ (7+15+31+63) × 6 = 696 kernel monomorphizations.

use aot_size::force_cmp;

fn main() {
    let width: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    macro_rules! exercise {
        ($T:ty) => {{
            let input = vec![<$T>::default(); 1024];
            let mut output = [false; 1024];
            let c: $T = 0;
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a == b, c);
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a != b, c);
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a < b, c);
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a <= b, c);
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a > b, c);
            force_cmp::<$T, $T, _>(width, &input, &mut output, |a, b| a >= b, c);
            output.iter().filter(|&&b| b).count()
        }};
    }

    let n = exercise!(u8) + exercise!(u16) + exercise!(u32) + exercise!(u64);
    println!("{}", std::hint::black_box(n));
}
