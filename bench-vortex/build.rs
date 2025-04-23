fn main() {
    #[cfg(target_os = "linux")]
    linux_build()
}

#[cfg(target_os = "linux")]
fn linux_build() {
    custom_labels::build::emit_build_instructions1();
}
