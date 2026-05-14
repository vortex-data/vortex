//! Dumps the AOT stencil bytes and all six patched variants. Useful for
//! sanity-checking the encoding by piping through `ndisasm -b 64`.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    use stencil_jit::{CmpOp, debug};

    let bytes = debug::stencil_bytes();
    let off = debug::patch_offset();
    let len = debug::patch_len();

    println!(
        "stencil length = {} bytes, patch slot @ {}..{} ({} bytes)",
        bytes.len(),
        off,
        off + len,
        len,
    );

    for op in CmpOp::ALL {
        let mut full = bytes.to_vec();
        full[off..off + len].copy_from_slice(debug::op_patch(op));
        print!("{:>3?}: ", op);
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
