use leptos::prelude::*;
use leptos_meta::Meta;
use leptos_meta::Title;
use leptos_meta::provide_meta_context;
use leptos_router::components::Route;
use leptos_router::components::Router;
use leptos_router::components::Routes;
use leptos_router::path;

use crate::components::HomePage;

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Title text="Vortex Benchmarks"/>
        <Meta name="description" content="Performance benchmarks for Vortex columnar file format"/>

        <Router>
            <main class="min-h-screen bg-gray-100">
                <nav class="bg-gray-900 text-white p-4 shadow-lg">
                    <div class="container mx-auto flex items-center justify-between">
                        <a href="/" class="text-xl font-bold hover:text-gray-300 transition-colors">
                            "Vortex Benchmarks"
                        </a>
                        <a
                            href="https://github.com/spiraldb/vortex"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="text-gray-300 hover:text-white transition-colors"
                        >
                            "GitHub"
                        </a>
                    </div>
                </nav>
                <Routes fallback=|| view! {
                    <div class="container mx-auto py-8 px-4">
                        <h1 class="text-2xl font-bold text-gray-800">"Page not found"</h1>
                    </div>
                }>
                    <Route path=path!("/") view=HomePage />
                </Routes>
            </main>
        </Router>
    }
}
