//! Dumps the AOT stencil bytes for every (ffor, op) configuration. Useful
//! for sanity-checking the encoding via `ndisasm -b 64`.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    use stencil_jit::{ChainConfig, CmpOp, debug};

    let bytes = debug::stencil_bytes();
    let f_off = debug::ffor_offset();
    let o_off = debug::op_offset();

    println!(
        "stencil = {} B, ffor @ {}..{}, op @ {}..{}",
        bytes.len(),
        f_off,
        f_off + 8,
        o_off,
        o_off + 8,
    );

    let configs = [
        (false, CmpOp::Eq),
        (true, CmpOp::Eq),
        (false, CmpOp::Lt),
        (true, CmpOp::Lt),
        (true, CmpOp::Ge),
    ];

    for (ffor, op) in configs {
        let mut full = bytes.to_vec();
        let ffor_src = if ffor {
            debug::ffor_add_patch()
        } else {
            debug::ffor_nop_patch()
        };
        full[f_off..f_off + 8].copy_from_slice(ffor_src);
        full[o_off..o_off + 8].copy_from_slice(debug::op_patch_bytes(op));
        let _ = ChainConfig { ffor, op }; // doc tie-in
        print!("ffor={:>5} op={:>3?}: ", ffor, op);
        for b in &full {
            print!("{b:02x} ");
        }
        println!();
    }
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("stencil-jit prototype only supports x86_64 Linux");
}
