use std::fs::File;

use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayRef, ToCanonical, TryIntoArray};
use vortex_dtype as _;
use vortex_geo::POLYGON_ID;

pub fn main() {
    // Open the parquet file as GeoParquet.
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
}
