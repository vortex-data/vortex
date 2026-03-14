// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compat::fixtures::all_fixtures;

fn main() {
    let tmp = std::env::temp_dir().join("vortex-fixture-measure");
    std::fs::create_dir_all(&tmp).unwrap();

    println!(
        "{:<35} {:>8} {:>8} {:>12}",
        "FIXTURE", "CHUNKS", "ROWS", "BYTES"
    );
    println!("{}", "-".repeat(67));

    let mut total_rows = 0usize;
    let mut total_bytes = 0u64;

    for fixture in all_fixtures() {
        // Skip network-dependent fixtures
        if fixture.name().contains("tpch") || fixture.name().contains("clickbench") {
            println!("{:<35} {:>8}", fixture.name(), "SKIPPED");
            continue;
        }

        match fixture.build(&tmp) {
            Ok(chunks) => {
                let n_chunks = chunks.len();
                let rows: usize = chunks.iter().map(|c| c.len()).sum();
                let bytes: u64 = chunks.iter().map(|c| c.nbytes()).sum();
                total_rows += rows;
                total_bytes += bytes;
                println!(
                    "{:<35} {:>8} {:>8} {:>12}",
                    fixture.name(),
                    n_chunks,
                    rows,
                    bytes
                );
            }
            Err(e) => {
                println!("{:<35} ERROR: {}", fixture.name(), e);
            }
        }
    }
    println!("{}", "-".repeat(67));
    println!(
        "{:<35} {:>8} {:>8} {:>12}",
        "TOTAL", "", total_rows, total_bytes
    );
}
