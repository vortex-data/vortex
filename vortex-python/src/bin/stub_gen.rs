use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    env_logger::init();
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let stub = vortex_python::stub_info()?;
    stub.generate()?;
    Ok(())
}
