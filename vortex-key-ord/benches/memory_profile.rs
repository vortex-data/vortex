// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Output-array `nbytes` for each (input, kernel) combination. Static
//! measurement printed once at startup; the divan bench is a sentinel
//! so `cargo bench` runs the file.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use divan::Bencher;
use divan::counter::BytesCount;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_key_ord::stream_kernel::OvcKernel;

fn main() {
    print_report();
    divan::main();
}

const N: usize = 100_000;

fn make_primitive() -> PrimitiveArray {
    let buf: Vec<u64> = (0..N as u64).collect();
    PrimitiveArray::new(Buffer::<u64>::copy_from(&buf), Validity::NonNullable)
}

fn make_constant() -> ConstantArray {
    ConstantArray::new(42u64, N)
}

fn make_dict(dict_size: usize) -> DictArray {
    let values_buf: Vec<u64> = (0..dict_size as u64).collect();
    let codes_buf: Vec<u32> = (0..N).map(|i| (i % dict_size) as u32).collect();
    let values = PrimitiveArray::new(
        Buffer::<u64>::copy_from(&values_buf),
        Validity::NonNullable,
    )
    .into_array();
    let codes =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&codes_buf), Validity::NonNullable)
            .into_array();
    DictArray::new(codes, values)
}

fn print_report() {
    println!("\n==== MEMORY FOOTPRINT (output nbytes for N={N}) ====");

    let prim_out = <Primitive as OvcKernel>::ovc_encode(make_primitive().as_view(), 0);
    println!("Primitive in  -> Primitive out : {:>10} bytes", prim_out.nbytes());

    let cons_out = <Constant as OvcKernel>::ovc_encode(make_constant().as_view(), 0);
    println!("Constant  in  -> Constant  out : {:>10} bytes", cons_out.nbytes());

    for &ds in &[4usize, 64, 1024] {
        let d_out = <Dict as OvcKernel>::ovc_encode(make_dict(ds).as_view(), 0);
        println!("Dict(ds={ds:>4}) -> Dict     out : {:>10} bytes", d_out.nbytes());
    }

    let flat = PrimitiveArray::new(
        Buffer::<u64>::copy_from(&(0..N as u64).collect::<Vec<_>>()),
        Validity::NonNullable,
    )
    .into_array();
    println!("Naive flat baseline             : {:>10} bytes\n", flat.nbytes());
}

#[divan::bench(sample_count = 5)]
fn sentinel(bencher: Bencher) {
    let arr = make_constant();
    let bytes = <Constant as OvcKernel>::ovc_encode(arr.as_view(), 0).nbytes();
    bencher
        .counter(BytesCount::new(bytes))
        .bench_local(|| {
            black_box(<Constant as OvcKernel>::ovc_encode(arr.as_view(), 0))
        });
}
