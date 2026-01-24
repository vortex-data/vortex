use leptos::prelude::*;

use crate::components::Chart;
use crate::db::ChartData;

#[component]
pub fn HomePage() -> impl IntoView {
    // Server-side resource to fetch chart data
    let chart_data = Resource::new(|| (), |_| get_chart_data());

    view! {
        <div class="container mx-auto py-8 px-4">
            <h1 class="text-3xl font-bold text-gray-900 mb-8">
                "Benchmark Results"
            </h1>

            <Suspense fallback=move || view! {
                <div class="animate-pulse bg-gray-200 rounded-lg h-96 w-full"></div>
            }>
                {move || Suspend::new(async move {
                    match chart_data.await {
                        Ok(data) => view! { <Chart data=data /> }.into_any(),
                        Err(e) => view! {
                            <div class="text-red-600 p-4 bg-red-50 rounded-lg">
                                {format!("Error loading data: {}", e)}
                            </div>
                        }.into_any(),
                    }
                })}
            </Suspense>
        </div>
    }
}

/// Server function to fetch chart data from DuckDB.
#[server(GetChartData)]
async fn get_chart_data() -> Result<ChartData, ServerFnError> {
    use crate::db::DbPool;
    use crate::db::get_random_access_data;

    let db = expect_context::<DbPool>();
    get_random_access_data(&db, 50).map_err(|e| ServerFnError::new(e.to_string()))
}
