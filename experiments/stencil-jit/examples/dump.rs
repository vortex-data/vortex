//! Dumps the stencil bytes for sanity-checking via `ndisasm -b 64`.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    use stencil_jit::{CmpOp, debug};

    let single = debug::stencil_bytes();
    println!(
        "single-block stencil = {} B, load @ {}..{}, op @ {}..{}",
        single.len(),
        debug::ffor_offset(),
        debug::ffor_offset() + debug::ffor_len(),
        debug::op_offset(),
        debug::op_offset() + debug::op_len(),
    );

    let bulk = debug::bulk_bytes();
    println!(
        "bulk stencil         = {} B, load_a @ {}, op_a @ {}, load_b @ {}, op_b @ {}",
        bulk.len(),
        debug::bulk_ffor_a_offset(),
        debug::bulk_op_a_offset(),
        debug::bulk_ffor_b_offset(),
        debug::bulk_op_b_offset(),
    );

    println!("\nbulk stencil bytes (eq, FFoR on, patched):");
    let mut b = bulk.to_vec();
    b[debug::bulk_ffor_a_offset()..debug::bulk_ffor_a_offset() + 5]
        .copy_from_slice(&[0xC5, 0xE5, 0xFC, 0x07, 0x90]); // vpaddb ymm0,ymm3,[rdi];nop
    b[debug::bulk_op_a_offset()..debug::bulk_op_a_offset() + 8]
        .copy_from_slice(debug::op_patch_bytes(CmpOp::Eq));
    b[debug::bulk_ffor_b_offset()..debug::bulk_ffor_b_offset() + 5]
        .copy_from_slice(&[0xC5, 0xE5, 0xFC, 0x47, 0x20]); // vpaddb ymm0,ymm3,[rdi+32]
    b[debug::bulk_op_b_offset()..debug::bulk_op_b_offset() + 8]
        .copy_from_slice(debug::op_patch_bytes(CmpOp::Eq));
    for (i, byte) in b.iter().enumerate() {
        if i % 16 == 0 {
            print!("\n  {i:04x}: ");
        }
        print!("{byte:02x} ");
    }
    println!();
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("stencil-jit prototype only supports x86_64 Linux");
}
