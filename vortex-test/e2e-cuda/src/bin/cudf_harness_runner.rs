// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::process::Command;
use std::process::ExitCode;

const PRIMITIVE_DTYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64",
];
const PRIMITIVE_DTYPE_ENV: &str = "VORTEX_CUDF_PRIMITIVE_DTYPE";

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    let [program, harness, library] = args.as_slice() else {
        eprintln!(
            "Usage: {} <cudf-test-harness> <library.so>",
            args.first().map_or("cudf_harness_runner", String::as_str)
        );
        return ExitCode::from(2);
    };

    for primitive_dtype in PRIMITIVE_DTYPES {
        eprintln!("running {program} with {PRIMITIVE_DTYPE_ENV}={primitive_dtype}");

        let status = Command::new("compute-sanitizer")
            .args(["--tool", "memcheck", "--error-exitcode", "1"])
            .arg(harness)
            .arg("check")
            .arg(library)
            .env(PRIMITIVE_DTYPE_ENV, primitive_dtype)
            .status();

        match status {
            Ok(status) if status.success() => {}
            Ok(status) => {
                eprintln!("cudf-test-harness failed with {status}");
                return ExitCode::from(1);
            }
            Err(err) => {
                eprintln!("failed to run cudf-test-harness: {err}");
                return ExitCode::from(1);
            }
        }
    }

    ExitCode::SUCCESS
}
