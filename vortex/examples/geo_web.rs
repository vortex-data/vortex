// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use axum::extract::Query;
use axum::extract::State;
use axum::response::Html;
use axum::routing::get;
use axum::Json;
use axum::Router;
use futures::pin_mut;
use futures::StreamExt;
use futures::TryStreamExt;
use geo::algorithm::centroid::Centroid;
use geo_types::Geometry;
use geo_types::Rect;
use geozero::geo_types::GeoWriter;
use geozero::wkb;
use geozero::wkb::WkbDialect;
use geozero::GeozeroGeometry;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::expr::col;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::session::ExprSessionExt;
use vortex_array::expr::st_contains::STContains;
use vortex_array::expr::ExprVTable;
use vortex_array::expr::VTableExt;
use vortex_array::Array;
use vortex_array::ToCanonical;
use vortex_buffer::ByteBuffer;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_layout::layouts::geo::GeoLayoutEncoding;
use vortex_layout::session::LayoutSessionExt;
use vortex_layout::LayoutEncodingRef;
use vortex_session::VortexSession;

#[derive(Deserialize)]
struct ViewportQuery {
    south: f64,
    west: f64,
    north: f64,
    east: f64,
    zoom: u8,
}

#[derive(Serialize)]
struct CountResponse {
    count: usize,
}

#[derive(Serialize)]
struct GeoJsonResponse {
    r#type: String,
    features: Vec<GeoJsonFeature>,
}

#[derive(Serialize)]
struct GeoJsonFeature {
    r#type: String,
    geometry: GeoJsonGeometry,
    properties: GeoJsonProperties,
}

#[derive(Serialize)]
struct GeoJsonGeometry {
    r#type: String,
    coordinates: [f64; 2],
}

#[derive(Serialize)]
struct GeoJsonProperties {
    name: String,
    description: String,
}

#[derive(Clone)]
struct AppState {
    /// File with geospatial data backing the state service.
    vxf: VortexFile,
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default();

    // Register our custom geospatial layouts and functions
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(GeoLayoutEncoding.as_ref()));
    session
        .expressions()
        .register(ExprVTable::new_static(&STContains));

    session
});

#[tokio::main]
pub async fn main() {
    // Open up the Vortex file.
    let vxf = SESSION
        .open_options()
        .open("buildings_rtree.vortex")
        .await
        .expect("Opening file failed");

    let state = Arc::new(AppState { vxf });

    let api_routes = Router::new()
        .route("/hello", get(hello_handler))
        .route("/pins", get(pins_handler))
        .route("/counts", get(count_handler))
        .with_state(state.clone());

    let app = Router::new()
        .route("/index.html", get(index_handler))
        .nest("/api", api_routes)
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001")
        .await
        .unwrap();

    println!("Server running on http://127.0.0.1:3001");
    println!("  - Static page: http://127.0.0.1:3001/index.html");
    println!("  - API endpoint: http://127.0.0.1:3001/api/hello");
    println!("  - API endpoint: http://127.0.0.1:3001/api/pins");

    axum::serve(listener, app).await.unwrap();
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

async fn hello_handler() -> &'static str {
    "Hello, World!"
}

// Query the backing file using the pins instead.
async fn pins_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ViewportQuery>,
) -> Json<GeoJsonResponse> {
    // For now, generate some sample pins within the viewport
    // In a real application, you'd query a database here
    let rect = Geometry::Rect(Rect::new(
        [params.west, params.south],
        [params.east, params.north],
    ));

    let mut target_wkb: Vec<u8> = vec![];
    let mut writer = wkb::WkbWriter::new(&mut target_wkb, WkbDialect::Wkb);
    rect.process_geom(&mut writer).unwrap();
    let target_wkb = ByteBuffer::from(target_wkb);

    let filter = STContains
        .try_new_expr((), [lit(target_wkb), col("geometry")])
        .expect("failed to build filter");

    // Perform the scan operation over the file.
    let stream = state
        .vxf
        .scan()
        .expect("creating scan")
        .with_filter(filter)
        .with_projection(pack(
            [
                ("geometry", col("geometry")),
                ("occupancy", col("occupancy")),
                ("last_update", col("last_update")),
            ],
            Nullability::NonNullable,
        ))
        .into_array_stream()
        .expect("into_array_stream");
    pin_mut!(stream);

    let result = stream
        .try_collect::<Vec<_>>()
        .await
        .expect("reading stream failed");

    let features = result
        .into_iter()
        .flat_map(|building| {
            // Parse the WKB and extract out the centroid of it.
            let building = building.to_struct();
            let geometry = building.field_by_name("geometry").unwrap();
            geometry
                .to_varbinview()
                .with_iterator(|wkbs| wkbs.filter_map(|wkb| Some(parse_wkb(wkb?))).collect_vec())
                .into_iter()
                .filter_map(|geom| geom.centroid())
                .map(|point| GeoJsonFeature {
                    r#type: "Feature".to_string(),
                    geometry: GeoJsonGeometry {
                        r#type: "Point".to_string(),
                        coordinates: [point.x(), point.y()],
                    },
                    properties: GeoJsonProperties {
                        name: "a point".to_string(),
                        description: "a description".to_string(),
                    },
                })
        })
        .collect_vec();

    Json(GeoJsonResponse {
        r#type: "FeatureCollection".to_string(),
        features,
    })
}

async fn count_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ViewportQuery>,
) -> Json<CountResponse> {
    // For now, generate some sample pins within the viewport
    // In a real application, you'd query a database here
    let rect = Geometry::Rect(Rect::new(
        [params.west, params.south],
        [params.east, params.north],
    ));

    let mut target_wkb: Vec<u8> = vec![];
    let mut writer = wkb::WkbWriter::new(&mut target_wkb, WkbDialect::Wkb);
    rect.process_geom(&mut writer).unwrap();
    let target_wkb = ByteBuffer::from(target_wkb);

    let filter = STContains
        .try_new_expr((), [lit(target_wkb), col("geometry")])
        .expect("failed to build filter");

    // Perform the scan operation over the file.
    let stream = state
        .vxf
        .scan()
        .expect("creating scan")
        .with_filter(filter)
        .with_projection(pack(
            [("occupancy", col("occupancy"))],
            Nullability::NonNullable,
        ))
        .into_array_stream()
        .expect("into_array_stream");

    pin_mut!(stream);

    let counts = stream
        .map(|chunk| VortexResult::Ok(chunk?.len()))
        .try_collect::<Vec<_>>()
        .await
        .expect("counting stream failed");
    let count = counts.into_iter().sum::<usize>();

    Json(CountResponse { count })
}

fn parse_wkb(wkb: &[u8]) -> Geometry {
    let mut writer = GeoWriter::new();
    wkb::Wkb(wkb)
        .process_geom(&mut writer)
        .expect("wkb parsing left");
    writer.take_geometry().expect("wkb should yield geometry")
}
