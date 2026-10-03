// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prints the shredding layout, compressed sizes and a correctness check for the LO2v2 labels.
//!
//! ```text
//! LO2_ROWS=1000000 cargo run --release -p vortex-shredded-map --example prom_labels_report
//! ```

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as _;
use arrow_array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_shredded_map::ops;

#[path = "../benches/common/mod.rs"]
mod common;

use common::DATA;
use common::SESSION;

fn mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn strings(array: &ArrayRef) -> Vec<Option<String>> {
    let array = array
        .clone()
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .unwrap();
    (0..array.len())
        .map(|i| {
            array
                .is_valid(i, &mut SESSION.create_execution_ctx())
                .unwrap()
                .then(|| String::from_utf8(array.bytes_at(i).to_vec()).unwrap())
        })
        .collect()
}

fn parquet_bytes(map: &arrow_array::MapArray) -> (Vec<u8>, f64) {
    let batch =
        RecordBatch::try_from_iter([("labels", Arc::new(map.clone()) as arrow_array::ArrayRef)])
            .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let start = Instant::now();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(buf.clone()))
        .unwrap()
        .with_batch_size(map.len())
        .build()
        .unwrap();
    let rows: usize = reader.map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(rows, map.len());
    (buf, start.elapsed().as_secs_f64())
}

fn main() {
    let data = LazyLock::force(&DATA);
    let mut ctx = SESSION.create_execution_ctx();
    println!(
        "rows: {}  series: {}  load: {:.1}s  shred: {:.2}s  encode: {:.2}s",
        data.rows, data.series, data.load_secs, data.shred_secs, data.encode_secs
    );

    println!("\nshredded columns (key, variant, compressed size):");
    for ((column, array), compressed) in data
        .shredded
        .data()
        .columns()
        .iter()
        .zip(data.shredded.columns().iter())
        .zip(data.shredded_compressed.columns().iter())
    {
        let valid = array.valid_count(&mut ctx).unwrap();
        println!(
            "  {:<55} {:<6} {:>6.1}% rows  {:>10} -> {:>10}",
            column.key,
            column
                .variant
                .map(|v| ["str", "int", "float", "bool"][v])
                .unwrap_or("full"),
            100.0 * valid as f64 / data.rows as f64,
            mib(array.nbytes()),
            mib(compressed.nbytes()),
        );
    }
    let residual_entries = common::compress(data.shredded.residual()).nbytes();
    println!(
        "  residual map: {} -> {}",
        mib(data.shredded.residual().nbytes()),
        mib(residual_entries)
    );

    let (parquet, parquet_read) = parquet_bytes(&data.arrow);
    println!("\nsizes:");
    println!("  arrow Map<Utf8, Utf8> (in memory)  {}", mib(data.arrow.get_array_memory_size() as u64));
    println!("  parquet zstd(3) Map<Utf8, Utf8>    {}  (read back in {:.3}s)", mib(parquet.len() as u64), parquet_read);
    println!("  vortex map (canonical)             {}", mib(data.map.nbytes()));
    println!("  vortex map + btrblocks             {}", mib(data.map_compressed.nbytes()));
    println!("  vortex shared map (canonical)      {}", mib(data.map_shared.nbytes()));
    println!("  vortex shared map + btrblocks      {}", mib(data.map_shared_compressed.nbytes()));
    println!(
        "  vortex shared map + compact        {}",
        mib(common::compress_with(&data.map_shared, true).nbytes())
    );
    println!("  shredded (canonical)               {}", mib(data.shredded.nbytes()));
    println!("  shredded + btrblocks               {}", mib(data.shredded_compressed.nbytes()));
    println!("  keyset (canonical)                 {}", mib(data.keyset.nbytes()));
    println!("  keyset + btrblocks                 {}", mib(data.keyset_compressed.nbytes()));
    println!(
        "  keyset + btrblocks compact         {}",
        mib(common::compress_children(&data.keyset, true).nbytes())
    );
    let plain_keyset = vortex_shredded_map::keyset::keyset_encode(
        &data.map,
        vortex_shredded_map::keyset::KeySetOptions { dedup_values: false },
        &mut ctx,
    )
    .unwrap()
    .into_array();
    println!(
        "  keyset, no value dedup + btrblocks {}",
        mib(common::compress_children(&plain_keyset, false).nbytes())
    );
    println!(
        "  map + btrblocks with map schemes   {}  ({}, {:.3}s)",
        mib(data.map_auto.nbytes()),
        data.map_auto.encoding_id(),
        data.map_auto_secs
    );
    println!(
        "  ... compact                        {}",
        mib(common::compress_auto(&data.map, true).nbytes())
    );
    println!("  encoded (canonical)                {}", mib(data.encoded.nbytes()));
    println!("  encoded + btrblocks                {}", mib(data.encoded_compressed.nbytes()));
    println!(
        "  encoded + btrblocks compact        {}",
        mib(common::compress_encoded(&data.encoded, true).nbytes())
    );
    println!(
        "  vortex map + btrblocks compact     {}",
        mib(common::compress_with(&data.map, true).nbytes())
    );
    println!(
        "  shredded + btrblocks compact       {}",
        mib(common::compress_shredded_with(&data.shredded, true).nbytes())
    );

    if std::env::var("REPORT_VERIFY").as_deref() == Ok("0") {
        return;
    }
    // Every label of every row must match between the canonical and shredded maps.
    let start = Instant::now();
    let keys = ops::map::distinct_label_names(&data.map, &mut ctx).unwrap();
    assert_eq!(
        keys,
        ops::shredded::distinct_label_names(&data.shredded_compressed, &mut ctx).unwrap()
    );
    let decoded: ArrayRef = ops::shredded::to_map(&data.shredded_compressed, &mut ctx)
        .unwrap()
        .into();
    let decoded_encoded: ArrayRef = ops::encoded::to_map(&data.encoded_compressed, &mut ctx)
        .unwrap()
        .into();
    for key in &keys {
        let t = Instant::now();
        let expected = strings(&ops::map::get_label_utf8(&data.map, key, &mut ctx).unwrap());
        let t1 = t.elapsed();
        let got = ops::shredded::get_label_utf8(&data.shredded_compressed, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "shredded label {key}");
        let t2 = t.elapsed();
        let got = ops::map::get_label_utf8(&decoded, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "decoded label {key}");
        let got = ops::encoded::get_label_utf8(&data.encoded_compressed, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "encoded label {key}");
        let got = ops::encoded::get_label_utf8(&data.keyset_compressed, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "keyset label {key}");
        let got = ops::encoded::get_label_utf8(&data.map_auto, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "map_auto label {key}");
        let got = ops::map::get_label_utf8(&decoded_encoded, key, &mut ctx).unwrap();
        assert_eq!(strings(&got), expected, "decoded encoded label {key}");
        if key == &keys[0] {
            eprintln!("map {t1:?} shredded {:?} decoded {:?}", t2 - t1, t.elapsed() - t2);
        }
    }
    println!(
        "\nverified {} labels on every row in {:.1}s",
        keys.len(),
        start.elapsed().as_secs_f64()
    );
}
