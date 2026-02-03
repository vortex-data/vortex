fn main() {
    #[cfg(not(target_os = "macos"))]
    custom_labels::build::emit_build_instructions();
}
