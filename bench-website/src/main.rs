#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use std::net::SocketAddr;

    use axum::Router;
    use bench_website::App;
    use bench_website::db::DbPool;
    use leptos::config::LeptosOptions;
    use leptos::prelude::*;
    use leptos_axum::LeptosRoutes;
    use leptos_axum::generate_route_list;
    use tokio::net::TcpListener;

    tracing_subscriber::fmt::init();

    // Initialize database with mock data
    let db = DbPool::new_with_mock_data().expect("Failed to initialize database");

    // Build Leptos configuration manually (works from any directory)
    let addr: SocketAddr = "127.0.0.1:3000".parse().unwrap();
    let leptos_options = LeptosOptions::builder()
        .output_name("bench-website")
        .site_root("target/site")
        .site_pkg_dir("pkg")
        .site_addr(addr)
        .reload_port(3001)
        .build();
    let routes = generate_route_list(App);

    // Build router with Leptos routes
    let app = Router::new()
        .leptos_routes_with_context(
            &leptos_options,
            routes,
            {
                let db = db.clone();
                move || provide_context(db.clone())
            },
            {
                let leptos_options = leptos_options.clone();
                move || shell(leptos_options.clone())
            },
        )
        .with_state(leptos_options);

    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Listening on http://{}", addr);
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[cfg(feature = "ssr")]
fn shell(options: leptos::config::LeptosOptions) -> impl leptos::prelude::IntoView {
    use bench_website::App;
    use leptos::prelude::*;
    use leptos_meta::MetaTags;

    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone()/>
                <HydrationScripts options/>
                <MetaTags/>
                <style>{INLINE_CSS}</style>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

/// Minimal Tailwind-style utility classes for Phase 1 prototype.
/// Phase 2 will use proper Tailwind CSS with PostCSS.
#[cfg(feature = "ssr")]
const INLINE_CSS: &str = r#"
/* === Reset & Base === */
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; color: #1f2937; }
a { text-decoration: none; color: inherit; }
a:hover { opacity: 0.8; }

/* === Layout === */
.min-h-screen { min-height: 100vh; }
.container { width: 100%; max-width: 1280px; margin: 0 auto; }
.mx-auto { margin-left: auto; margin-right: auto; }
.flex { display: flex; }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
.justify-center { justify-content: center; }

/* === Spacing === */
.gap-2 { gap: 0.5rem; }
.gap-6 { gap: 1.5rem; }
.p-4 { padding: 1rem; }
.px-4 { padding-left: 1rem; padding-right: 1rem; }
.py-8 { padding-top: 2rem; padding-bottom: 2rem; }
.mb-4 { margin-bottom: 1rem; }
.mb-6 { margin-bottom: 1.5rem; }
.mb-8 { margin-bottom: 2rem; }
.mt-4 { margin-top: 1rem; }

/* === Sizing === */
.w-4 { width: 1rem; }
.w-full { width: 100%; }
.h-3 { height: 0.75rem; }
.h-96 { height: 24rem; }

/* === Typography === */
.text-sm { font-size: 0.875rem; }
.text-lg { font-size: 1.125rem; }
.text-xl { font-size: 1.25rem; }
.text-3xl { font-size: 1.875rem; }
.font-bold { font-weight: 700; }
.font-semibold { font-weight: 600; }

/* === Colors === */
.bg-white { background-color: #ffffff; }
.bg-gray-100 { background-color: #f3f4f6; }
.bg-gray-200 { background-color: #e5e7eb; }
.bg-gray-900 { background-color: #111827; }
.bg-red-50 { background-color: #fef2f2; }
.text-white { color: #ffffff; }
.text-gray-300 { color: #d1d5db; }
.text-gray-700 { color: #374151; }
.text-gray-800 { color: #1f2937; }
.text-gray-900 { color: #111827; }
.text-red-600 { color: #dc2626; }

/* === Borders & Shadows === */
.rounded-sm { border-radius: 0.125rem; }
.rounded-lg { border-radius: 0.5rem; }
.shadow-md { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
.shadow-lg { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }

/* === Transitions & Animations === */
.transition-colors { transition: color 0.15s ease-in-out; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

/* === Chart-specific === */
.chart-container { max-width: 100%; overflow-x: auto; }
.chart-svg { display: block; max-width: 100%; height: auto; }
"#;

#[cfg(not(feature = "ssr"))]
fn main() {
    // Client-side only - hydration is handled by lib.rs
}
