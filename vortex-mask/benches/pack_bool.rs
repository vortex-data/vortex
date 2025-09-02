// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    divan::main();
}

#[inline]
pub fn collect_bool<F: FnMut(usize) -> bool>(output: &mut [u64], mut f: F) {
    let len = output.len();
    let chunks = len;
    for chunk in 0..chunks {
        let mut packed = 0;
        for bit_idx in 0..64 {
            let i = bit_idx + chunk * 64;
            packed |= (f(i) as u64) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        unsafe { *output.get_unchecked_mut(chunk) = packed }
    }
}

#[inline(never)]
pub fn collect_bool_slice(output: &mut [u64], input: &[bool]) {
    collect_bool(output, |i| input[i])
}

#[divan::bench(args=[1, 10, 40])]
fn arrow_pack_bool(bencher: Bencher, args: u64) {
    let mut rng = StdRng::seed_from_u64(0);

    let bools = (0..args * 1024).map(|_| rng.random_bool(0.5)).collect_vec();

    bencher
        .with_inputs(|| {
            (
                bools.clone(),
                (0..bools.len() / 64).map(|_| 0u64).collect_vec(),
            )
        })
        .bench_values(|(bools, mut output)| {
            collect_bool_slice(&mut output, &bools);
            output
        })
}

#[divan::bench(args=[1, 10, 40])]
fn arrow_pack_bool_sized(bencher: Bencher, args: u64) {
    let mut rng = StdRng::seed_from_u64(0);

    let bools = (0..args * 1024).map(|_| rng.random_bool(0.5)).collect_vec();

    bencher
        .with_inputs(|| {
            (
                bools.clone(),
                (0..bools.len() / 64).map(|_| 0).collect_vec(),
            )
        })
        .bench_values(|(bools, mut output)| {
            for i in 0..args as usize {
                collect_bool_sized(&mut output[i * 16..][..16], &bools[i * 1024..][..1024]);
            }
            output
        });

    {
        let output1 = {
            let mut output = (0..args as usize * 16).map(|_| 0u64).collect_vec();
            collect_bool_slice(&mut output, &bools);
            output
        };

        let output2 = {
            let mut output = (0..args as usize * 16).map(|_| 0u64).collect_vec();
            for i in 0..args as usize {
                collect_bool_sized(&mut output[i * 16..][..16], &bools[i * 1024..][..1024]);
            }
            output
        };

        assert_eq!(output1, output2);
    }
}

#[inline]
pub fn collect_bool2<F: FnMut(usize) -> bool>(output: &mut [u64], mut f: F) {
    let len = 1024;
    let chunks = len / 64;
    for chunk in 0..chunks {
        let mut packed = 0u64;
        for bit_idx in 0..64 {
            let i = bit_idx + chunk * 64;
            packed |= (f(i) as u64) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        unsafe { *output.get_unchecked_mut(chunk) = packed }
    }
}
#[inline(never)]
pub fn collect_bool_sized(output: &mut [u64], input: &[bool]) {
    collect_bool2(output, |i| input[i])
}

#[test]
mod tests {
    use itertools::Itertools;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{collect_bool_sized, collect_bool_slice};

    #[test]
    fn test123() {
        let args = 5;
        let mut rng = StdRng::seed_from_u64(0);

        let bools = (0..args * 1024).map(|_| rng.random_bool(0.5)).collect_vec();

        let output1 = {
            let output = (0..args as usize).map(|_| 0u64).collect_vec();
            for i in 0..args as usize {
                collect_bool_slice(&mut output[i * 16..][..16], &bools[i * 1024..][..1024]);
            }
            output
        };

        let output2 = {
            let output = (0..args as usize).map(|_| 0u64).collect_vec();
            for i in 0..args as usize {
                collect_bool_sized(&mut output[i * 16..][..16], &bools[i * 1024..][..1024]);
            }
            output
        };

        assert_eq!(output1, output2);
    }
}
