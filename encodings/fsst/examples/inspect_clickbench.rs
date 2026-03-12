// Quick script: read ClickBench parquet, FSST-compress the URL column,
// dump the symbol table, and show how LIKE patterns encode into the DFA.

use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ToCanonical;
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

    let url_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("no URL column");
    println!("URL column index: {url_idx}");

    let reader = builder.build().expect("build reader");
    let batch = reader.into_iter().next().expect("no batches").expect("batch error");
    let url_col = batch.column(url_idx);
    println!("Batch rows: {}, URL dtype: {:?}", batch.num_rows(), url_col.data_type());

    let urls: Vec<Option<&str>> = match url_col.data_type() {
        DataType::Utf8 => {
            let arr = url_col.as_string::<i32>();
            (0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        }
        DataType::LargeUtf8 => {
            let arr = url_col.as_string::<i64>();
            (0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        }
        DataType::Utf8View => {
            let arr = url_col.as_string_view();
            (0..arr.len()).map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) }).collect()
        }
        other => panic!("unexpected URL dtype: {other:?}"),
    };

    let n_urls = urls.len();
    let non_null = urls.iter().filter(|u| u.is_some()).count();
    println!("URLs: {n_urls} total, {non_null} non-null");

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
    let fsst_arr = vortex_fsst::fsst_compress(varbin, &compressor);

    let symbols = fsst_arr.symbols();
    let symbol_lengths = fsst_arr.symbol_lengths();

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
        println!("\nPattern \"{pattern}\":");
        let mut buf = vec![0u8; 2 * pattern.len() + 7];
        unsafe { compressor.compress_into(pattern.as_bytes(), &mut buf) };

        // Walk codes and annotate what each one decodes to
        print!("  encoded: ");
        let mut pos = 0;
        while pos < buf.len() {
            let c = buf[pos];
            if c == escape_code {
                pos += 1;
                if pos < buf.len() {
                    let lit = buf[pos];
                    let ch = if lit.is_ascii_graphic() || lit == b' ' {
                        format!("{}", lit as char)
                    } else {
                        format!("\\x{lit:02x}")
                    };
                    print!("[ESC '{ch}'] ");
                }
            } else if (c as usize) < symbols.len() {
                let sym = symbols[c as usize];
                let len = symbol_lengths[c as usize] as usize;
                let bytes = sym.to_u64().to_le_bytes();
                let s: String = bytes[..len]
                    .iter()
                    .map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' })
                    .collect();
                print!("[0x{c:02x}→\"{s}\"] ");
            } else {
                print!("[0x{c:02x}?] ");
            }
            pos += 1;
        }
        println!();
    }

    // --- 4. Show sample compressed strings ---
    println!("\n=== Sample Compressed Strings ===");
    let codes_varbin = fsst_arr.codes();
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
            "  [{i}] {orig_len}B -> {comp_len}B ({ratio:.2}x): \"{display_orig}...\""
        );

        // Show first 30 code bytes with annotations
        let show_len = string_codes.len().min(30);
        let hex: String = string_codes[..show_len]
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
        println!("         codes: [{hex}{}]", if string_codes.len() > 30 { " ..." } else { "" });
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
    println!("  Ratio:      {:.2}x", total_comp as f64 / total_orig as f64);
    println!("  Savings:    {:.1}%", (1.0 - total_comp as f64 / total_orig as f64) * 100.0);
}