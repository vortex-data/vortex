// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use axum::{
    response::Html,
    routing::get,
    Router,
};

#[tokio::main]
pub async fn main() {
    let api_routes = Router::new().route("/hello", get(hello_handler));

    let app = Router::new()
        .route("/index.html", get(index_handler))
        .nest("/api", api_routes);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001")
        .await
        .unwrap();

    println!("Server running on http://127.0.0.1:3001");
    println!("  - Static page: http://127.0.0.1:3001/index.html");
    println!("  - API endpoint: http://127.0.0.1:3001/api/hello");

    axum::serve(listener, app).await.unwrap();
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

async fn hello_handler() -> &'static str {
    "Hello, World!"
}