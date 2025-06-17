#![allow(clippy::expect_used)]

use std::io::Write;
use std::path::Path;

use futures::executor::block_on;
use vortex::arrays::StructArray;
use vortex::buffer::ByteBufferMut;
use vortex::builders::{ArrayBuilder, DecimalBuilder, VarBinViewBuilder};
use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::file::VortexWriteOptions;
use vortex::validity::Validity;

/// Generate a test dataset with the following small set of rows:
///
/// ```markdown
/// =======================
/// Name | Salary  | State
/// =======================
/// Alice   1000    CA
/// Bob     2000    NY
/// Carol   3000    TX
/// Dan     4000    CA
/// Edward  5000    NY
/// Frida   6000    TX
/// George  7000    CA
/// Henry   8000    NY
/// Ida     9000    TX
/// John    10000   VA
/// =======================
/// ```
fn main() {
    let mut names = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), 10);
    names.append_value("Alice");
    names.append_value("Bob");
    names.append_value("Carol");
    names.append_value("Dan");
    names.append_value("Edward");
    names.append_value("Frida");
    names.append_value("George");
    names.append_value("Henry");
    names.append_value("Ida");
    names.append_value("John");
    let names = names.finish();

    let mut salary =
        DecimalBuilder::with_capacity::<i32>(10, DecimalDType::new(9, 2), Nullability::Nullable);
    for i in 1..=10 {
        salary.append_value(1_000 * i);
    }
    let salary = salary.finish();

    let mut states = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), 10);
    states.append_value("CA");
    states.append_value("NY");
    states.append_value("TX");
    states.append_value("CA");
    states.append_value("NY");
    states.append_value("TX");
    states.append_value("CA");
    states.append_value("NY");
    states.append_value("TX");
    states.append_value("VA");
    let states = states.finish();

    // Make the struct array
    let rows = StructArray::try_new(
        ["Name".into(), "Salary".into(), "State".into()].into(),
        vec![names, salary, states],
        10,
        Validity::NonNullable,
    )
    .expect("Could not create struct array");

    let writer = VortexWriteOptions::default();
    let written = block_on(writer.write(ByteBufferMut::empty(), rows.to_array_stream()))
        .expect("writing Vortex file")
        .freeze();

    // Save to file
    let minimal_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-jni/src/test/resources/minimal.vortex");
    let mut file = std::fs::File::create(&minimal_path).expect("opening Vortex file");
    file.write_all(written.as_ref())
        .expect("flushing Vortex file to disk");

    println!("Wrote Vortex file to {}", minimal_path.display());
}
