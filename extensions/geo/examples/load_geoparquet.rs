extern crate vortex_dtype;
extern crate vortex_geo;

use std::fs::File;

use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayRef, ToCanonical, TryIntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::arrow::ArrowTypeConversionRef;
use vortex_geo::POLYGON_ID;

#[used]
static TEST: fn() = || {
    // Dummy function to force linker to pick up geo plugin
    vortex_geo::arrow::registry_link()
};

#[allow(clippy::unwrap_used, clippy::expect_used)]
pub fn main() {
    println!(
        "number of conversions: {}",
        vortex_dtype::inventory::iter::<ArrowTypeConversionRef>().count()
    );
    let file = File::open("/Volumes/Code/Data/afghan.parquet").unwrap();
    let mut geo = geoarrow_geoparquet::GeoParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();

    let arrow_batch = geo.next().unwrap().unwrap();

    let vortex_batch: ArrayRef = arrow_batch
        .try_into_array()
        .expect("convert to vortex array");
    let geometry = vortex_batch
        .to_struct()
        .unwrap()
        .maybe_null_field_by_name("geometry")
        .unwrap();

    assert_eq!(geometry.dtype().as_extension().unwrap().id(), &*POLYGON_ID);

    println!("uncompressed:\n{}", geometry.tree_display());

    // Run through the compressor to see how it performs. It should just compress the individual storage
    // arrays.
    let compressed = BtrBlocksCompressor.compress(geometry.as_ref()).unwrap();
    println!("compressed:\n{}", compressed.tree_display());
}
