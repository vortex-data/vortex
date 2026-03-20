// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP-RD compression benchmark.
//!
//! Everything — table schema, column generators, and SQL queries — is defined
//! in [`alp_compress_benchmark`] so it's easy to tweak in one place.

use std::sync::Arc;

use anyhow::Result;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_schema::DataType;
use rand::RngExt;

use crate::synthetic::SyntheticBenchmark;

/// Default number of rows per scale-factor unit.
const ROWS_PER_SF: usize = 2_000_000;

/// Build the ALP-RD compression benchmark.
///
/// All tables, columns, and queries are defined here.
pub fn alp_compress_benchmark(scale_factor: usize) -> Result<SyntheticBenchmark> {
    let n_rows = scale_factor * ROWS_PER_SF;

    SyntheticBenchmark::builder("alp-compress", n_rows)
        // ── table: alp_floats ──────────────────────────────────────────
        .table("alp_floats", |t| {
            t.column("id", DataType::Int64, |start, len, _rng| {
                Arc::new(Int64Array::from_iter_values(
                    (start as i64)..((start + len) as i64),
                ))
            });

            // Shared exponent around 98.6, full mantissa variation.
            t.column("sensor_reading", DataType::Float64, |start, len, rng| {
                Arc::new(Float64Array::from_iter_values((0..len).map(|j| {
                    let i = start + j;
                    let noise = ((i * 7 + 13) % 1000) as f64 / 10_000.0;
                    98.6 + noise + rng.random::<f64>() * 1e-10
                })))
            });

            // Slowly drifting doubles around 1000.
            t.column("price", DataType::Float64, |start, len, rng| {
                Arc::new(Float64Array::from_iter_values((0..len).map(|j| {
                    let i = start + j;
                    1000.0 + (i as f64) * 0.001 + rng.random::<f64>() * 1e-10
                })))
            });

            // Oscillating real doubles.
            t.column("measurement", DataType::Float64, |start, len, rng| {
                Arc::new(Float64Array::from_iter_values((0..len).map(|j| {
                    let i = start + j;
                    (i as f64 * 0.001).sin() * 100.0 + rng.random::<f64>() * 1e-10
                })))
            });

            // Cycling through a few base values with sub-millidegree noise.
            t.column("temperature", DataType::Float64, |start, len, rng| {
                let base_temps = [20.0, 21.5, 22.3, 23.1];
                Arc::new(Float64Array::from_iter_values((0..len).map(|j| {
                    let i = start + j;
                    let base = base_temps[i % base_temps.len()];
                    base + ((i * 3 % 997) as f64) * 1e-6 + rng.random::<f64>() * 1e-12
                })))
            });

            // Clustered around 300.0 with gaussian-like noise (Box-Muller).
            t.column("velocity", DataType::Float64, |_start, len, rng| {
                Arc::new(Float64Array::from_iter_values((0..len).map(|_| {
                    let u1: f64 = rng.random::<f64>().max(1e-15);
                    let u2: f64 = rng.random::<f64>();
                    let normal: f64 = (-2.0_f64 * u1.ln()).sqrt()
                        * (2.0_f64 * std::f64::consts::PI * u2).cos();
                    300.0 + normal * 2.0
                })))
            });

            // Low-cardinality grouping key.
            t.column("label", DataType::Int64, |start, len, _rng| {
                Arc::new(Int64Array::from_iter_values((0..len).map(|j| {
                    let i = start + j;
                    (i % 100) as i64
                })))
            });
        })
        // ── queries ────────────────────────────────────────────────────
        .queries(&[
            // Q0: Full scan, sum all float columns (decompression throughput).
            "SELECT SUM(sensor_reading), SUM(price), SUM(measurement), \
                    SUM(temperature), SUM(velocity) FROM alp_floats",
            // Q1: Filtered scan on id range.
            "SELECT SUM(sensor_reading), AVG(price) FROM alp_floats \
                    WHERE id BETWEEN 100000 AND 200000",
            // Q2: Group-by aggregation over low-cardinality key.
            "SELECT label, AVG(sensor_reading), AVG(price), AVG(temperature) \
                    FROM alp_floats GROUP BY label",
            // Q3: Filter on a float column value range.
            "SELECT COUNT(*), AVG(velocity) FROM alp_floats \
                    WHERE temperature > 22.0 AND temperature < 23.0",
            // Q4: Multi-column projection with filter.
            "SELECT id, sensor_reading, price FROM alp_floats \
                    WHERE velocity > 299.0 AND velocity < 301.0",
            // Q5: ORDER BY on a float column with LIMIT (top-k).
            "SELECT id, measurement FROM alp_floats \
                    ORDER BY measurement DESC LIMIT 100",
            // Q6: Heavy aggregation with multiple agg functions.
            "SELECT label, MIN(sensor_reading), MAX(sensor_reading), \
                    AVG(price), SUM(measurement), COUNT(*) \
                    FROM alp_floats GROUP BY label ORDER BY label",
            // Q7: Arithmetic expression on compressed columns.
            "SELECT SUM(sensor_reading * price + measurement) FROM alp_floats",
        ])
        .build()
}
