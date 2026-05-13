//! Dumps the AOT stencil bytes and the two patched variants. Useful for
//! sanity-checking the encoding by piping through `ndisasm -b 64`.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    use stencil_jit::debug;

    let bytes = debug::stencil_bytes();
    let off = debug::patch_offset();
    let len = debug::patch_len();

    let render = |label: &str, slot: &[u8]| {
        let mut full = bytes.to_vec();
        full[off..off + len].copy_from_slice(slot);
        print!("{label}: ");
        for b in &full {
            print!("{b:02x} ");
        }
        println!();
    };

    println!("stencil length = {} bytes, patch slot @ {}..{}", bytes.len(), off, off + len);
    render("eq ", debug::eq_patch());
    render("neq", debug::neq_patch());
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    eprintln!("stencil-jit prototype only supports x86_64 Linux");
}
