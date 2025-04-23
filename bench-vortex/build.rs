#[cfg(target_os = "linux")]
fn main() {
    custom_labels::build::emit_build_instructions();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("out dir = {}", std::env::var("OUT_DIR").unwrap());
}
