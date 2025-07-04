#[rustversion::nightly]
fn main() {
    println!("cargo:rustc-cfg=nightly");
}

#[rustversion::not(nightly)]
fn main() {}
