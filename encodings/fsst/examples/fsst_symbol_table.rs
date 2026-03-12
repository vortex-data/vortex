// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads lines from stdin, trains an FSST symbol table, and prints it.
//!
//! Usage:
//!   cat urls.txt | cargo run -p vortex-fsst --example fsst_symbol_table
//!   duckdb -csv -noheader -c "SELECT URL FROM 'hits_0.parquet' LIMIT 100000" | cargo run ...

#![allow(clippy::expect_used)]

use std::io;
use std::io::BufRead;

use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

fn main() {
    let stdin = io::stdin();
    let lines: Vec<Option<Box<[u8]>>> = stdin
        .lock()
        .lines()
        .map(|l| {
            l.expect("failed to read line")
                .into_bytes()
                .into_boxed_slice()
        })
        .map(Some)
        .collect();

    let n = lines.len();
    eprintln!("Read {n} lines from stdin");

    let varbin = VarBinArray::from_iter(lines, DType::Utf8(Nullability::NonNullable));
    let compressor = fsst_train_compressor(&varbin);
    let fsst_array = fsst_compress(&varbin, &compressor);

    print!("{}", fsst_array.format_symbol_table());

    // Report duplicate symbols in the table.
    let symbols = compressor.symbol_table();
    let lengths = compressor.symbol_lengths();
    let total = symbols.len();
    let mut keys: Vec<(u64, u8)> = symbols
        .iter()
        .zip(lengths.iter())
        .map(|(sym, &len)| (sym.to_u64(), len))
        .collect();
    keys.sort();
    let unique_count = {
        keys.dedup();
        keys.len()
    };
    let duplicates = total - unique_count;
    eprintln!("Symbol table: {total} symbols, {duplicates} duplicates");
}
