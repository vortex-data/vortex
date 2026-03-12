// Quick script: read ClickBench parquet, FSST-compress the URL column,
// dump the symbol table, and show how LIKE patterns encode into the DFA.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::{DType, Nullability};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vortex-bench/data/clickbench_partitioned/parquet/hits_0.parquet".into());

    // --- 1. Read parquet, extract URL column ---
    let file = std::fs::File::open(&path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let schema = builder.schema().clone();

    // Find the URL column index
    let url_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("no URL column");
    println!("URL column index: {url_idx}");

    let reader = builder.build().expect("build reader");

    // Collect first batch of URLs
    let batch = reader.into_iter().next().expect("no batches").expect("batch error");
    let url_col = batch.column(url_idx);
    println!("Batch rows: {}, URL dtype: {:?}", batch.num_rows(), url_col.data_type());

    // Convert arrow StringArray to VarBinArray
    let urls: Vec<Option<&str>> = match url_col.data_type() {
        DataType::Utf8 => {
            let arr = url_col.as_string::<i32>();
            (0..arr.len()).map(|i| {
                if arr.is_null(i) { None } else { Some(arr.value(i)) }
            }).collect()
        }
        DataType::LargeUtf8 => {
            let arr = url_col.as_string::<i64>();
            (0..arr.len()).map(|i| {
                if arr.is_null(i) { None } else { Some(arr.value(i)) }
            }).collect()
        }
        DataType::Utf8View => {
            let arr = url_col.as_string_view();
            (0..arr.len()).map(|i| {
                if arr.is_null(i) { None } else { Some(arr.value(i)) }
            }).collect()
        }
        other => panic!("unexpected URL dtype: {other:?}"),
    };

    let n_urls = urls.len();
    let non_null = urls.iter().filter(|u| u.is_some()).count();
    println!("URLs: {n_urls} total, {non_null} non-null");

    // Show some sample URLs
    println!("\n=== Sample URLs ===");
    for (i, u) in urls.iter().enumerate().take(10) {
        if let Some(s) = u {
            let display = if s.len() > 100 { &s[..100] } else { s };
            println!("  [{i}] {display}");
        } else {
            println!("  [{i}] NULL");
        }
    }

    // --- 2. FSST compress ---
    let varbin = VarBinArray::from_iter(urls.iter().copied(), DType::Utf8(Nullability::Nullable));
    let compressor = vortex_fsst::fsst_train_compressor(&varbin);
    let fsst = vortex_fsst::fsst_compress(varbin, &compressor);

    let symbols = fsst.symbols();
    let symbol_lengths = fsst.symbol_lengths();

    println!("\n=== FSST Symbol Table ({} symbols) ===", symbols.len());
    println!("{:<6} {:<6} {:<20} {:<20}", "Code", "Len", "Hex", "ASCII");
    println!("{}", "-".repeat(60));

    for (code, (sym, &len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
        let bytes = sym.to_u64().to_le_bytes();
        let sym_bytes = &bytes[..len as usize];
        let hex: String = sym_bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
        let ascii: String = sym_bytes
            .iter()
            .map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' })
            .collect();
        println!("  {code:<4} {len:<6} {hex:<20} {ascii:<20}");
    }

    // --- 3. Show how patterns encode ---
    let patterns = [
        "google", "http", "://", ".com", "yandex", "mail", "search", "www.",
    ];
    let escape_code = fsst::ESCAPE_CODE;
    println!("\n=== Pattern Encoding (ESCAPE_CODE = 0x{escape_code:02x}) ===");

    for pattern in &patterns {
        print!("\nPattern \"{pattern}\":");
        // Compress the pattern string to see how it encodes
        let mut buf = vec![0u8; 2 * pattern.len() + 7];
        unsafe { compressor.compress_into(pattern.as_bytes(), &mut buf) };
        let codes = &buf[..];
        // Print the codes (stop at first zero if it looks like the output is shorter)
        let code_str: Vec<String> = codes.iter().map(|c| {
            if *c == escape_code {
                "ESC".to_string()
            } else {
                format!("0x{c:02x}")
            }
        }).collect();
        println!("  codes = [{}]", code_str.join(", "));

        // Annotate: walk codes and show what each one decodes to
        print!("  decoded: ");
        let mut pos = 0;
        while pos < codes.len() {
            let c = codes[pos];
            if c == escape_code {
                pos += 1;
                if pos < codes.len() {
                    let lit = codes[pos];
                    let ch = if lit.is_ascii_graphic() || lit == b' ' {
                        format!("{}", lit as char)
                    } else {
                        format!("\\x{lit:02x}")
                    };
                    print!("[ESC '{ch}'] ");
                }
            } else {
                let sym = symbols[c as usize];
                let len = symbol_lengths[c as usize] as usize;
                let bytes = sym.to_u64().to_le_bytes();
                let s: String = bytes[..len]
                    .iter()
                    .map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' })
                    .collect();
                print!("[{c}→\"{s}\"] ");
            }
            pos += 1;
        }
        println!();
    }

    // --- 4. Show a sample string's compressed codes ---
    println!("\n=== Sample Compressed Strings ===");
    let codes_varbin = fsst.codes();
    let offsets = codes_varbin.offsets().to_primitive();
    let all_bytes = codes_varbin.bytes();
    let all_bytes = all_bytes.as_slice();

    for i in 0..10.min(n_urls) {
        let start: usize = offsets.as_slice::<i32>()[i] as usize;
        let end: usize = offsets.as_slice::<i32>()[i + 1] as usize;
        let string_codes = &all_bytes[start..end];
        let original = urls[i].unwrap_or("NULL");
        let orig_len = original.len();
        let comp_len = string_codes.len();
        let ratio = if orig_len > 0 {
            comp_len as f64 / orig_len as f64
        } else {
            0.0
        };

        let display_orig = if original.len() > 60 { &original[..60] } else { original };
        println!(
            "  [{i}] {orig_len}B → {comp_len}B ({ratio:.2}x): \"{display_orig}...\""
        );

        // Show first 20 code bytes
        let show = &string_codes[..string_codes.len().min(20)];
        let hex: String = show
            .iter()
            .map(|b| {
                if *b == escape_code {
                    "ESC".to_string()
                } else {
                    format!("{b:02x}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("         codes: [{hex}{}]", if string_codes.len() > 20 { " ..." } else { "" });
    }

    // --- 5. Compression stats ---
    let total_orig: usize = urls.iter().filter_map(|u| u.map(|s| s.len())).sum();
    let total_comp: usize = {
        let off = offsets.as_slice::<i32>();
        off.last().copied().unwrap_or(0) as usize
    };
    println!("\n=== Compression Stats ===");
    println!("  Original:   {total_orig} bytes");
    println!("  Compressed: {total_comp} bytes");
    println!(
        "  Ratio:      {:.2}x",
        total_comp as f64 / total_orig as f64
    );
    println!(
        "  Savings:    {:.1}%",
        (1.0 - total_comp as f64 / total_orig as f64) * 100.0
    );
}
