// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::StreamExt;
use futures::pin_mut;
use geo_types::Coord;
use geo_types::Geometry;
use geo_types::Rect;
use geozero::GeozeroGeometry;
use geozero::wkb;
use geozero::wkb::WkbDialect;
use vortex::VortexSessionDefault;
use vortex_array::Array;
use vortex_array::expr::ExprVTable;
use vortex_array::expr::VTableExt;
use vortex_array::expr::col;
use vortex_array::expr::lit;
use vortex_array::expr::session::ExprSessionExt;
use vortex_array::expr::st_contains::STContains;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_layout::LayoutEncodingRef;
use vortex_layout::layouts::geo::GeoLayoutEncoding;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

#[tokio::main]
pub async fn main() {
    let session = VortexSession::default();

    // Regsiter the ST_CONTAINS expression.
    session
        .expressions()
        .register(ExprVTable::new_static(&STContains));

    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(GeoLayoutEncoding.as_ref()));

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
        .open("buildings_rtree.vortex")
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
