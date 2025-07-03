use xshell::Shell;

pub fn java_test_files() -> anyhow::Result<()> {
    let sh = Shell::new()?;
    xshell::cmd!(sh, "cargo run --manifest-path java/testfiles/Cargo.toml").run()?;
    Ok(())
}
