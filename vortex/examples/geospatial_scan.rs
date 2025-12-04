// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::{StreamExt, pin_mut};
use geo_types::{Coord, Geometry, Rect};
use geozero::wkb::WkbDialect;
use geozero::{GeozeroGeometry, wkb};
use vortex::VortexSessionDefault;
use vortex_array::Array;
use vortex_array::expr::session::ExprSessionExt;
use vortex_array::expr::st_contains::STContains;
use vortex_array::expr::{ExprVTable, VTableExt, col, lit};
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_session::VortexSession;

#[tokio::main]
pub async fn main() {
    let session = VortexSession::default();

    // Regsiter the ST_CONTAINS expression.
    session
        .expressions()
        .register(ExprVTable::new_static(&STContains));

    let mut target_wkb: Vec<u8> = vec![];
    let mut writer = wkb::WkbWriter::new(&mut target_wkb, WkbDialect::Wkb);

    // This should only yield 1 final building.
    let target = Geometry::Rect(Rect::new(
        Coord {
            x: -96.9582104,
            y: 20.1394955,
        },
        Coord {
            x: -96.9573294,
            y: 20.1400545,
        },
    ));
    target.process_geom(&mut writer).unwrap();
    let target_wkb = ByteBuffer::from(target_wkb);

    let st_contains_filter = STContains
        .try_new_expr((), [lit(target_wkb), col("geometry")])
        .expect("building new ST_Contains expression");

    println!("executing scan with row filter {st_contains_filter}");

    // Create the scan.
    let vxf = session.open_options()
        .open("/Users/aduffy/Downloads/BuildingsParquet/custom_download_20251204_095222.compact.vortex")
        .await.expect("open file");

    let stream = vxf
        .scan()
        .unwrap()
        .with_filter(st_contains_filter)
        .into_array_stream()
        .unwrap();

    pin_mut!(stream);

    let mut total_rows = 0;
    while let Some(next) = stream.next().await {
        let count = next.unwrap().len();
        println!("Rows matched: {:?}", count);
        total_rows += count;
    }

    println!("Total rows: {}", total_rows);
}
