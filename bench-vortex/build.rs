fn main() {
    #[cfg(target_os = "linux")]
    custom_labels::build::emit_build_instructions();
}
