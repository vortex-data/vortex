// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use crate::engines::df::execute_query;

pub async fn run_tpch_query(ctx: &SessionContext, query: &str) -> (usize, Arc<dyn ExecutionPlan>) {
    let (record_batches, metrics) = execute_query(ctx, query).await.unwrap();
    (record_batches.iter().map(|r| r.num_rows()).sum(), metrics)
}
