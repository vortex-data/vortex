// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Statistical and Population Genetics Benchmark
//!
//! A custom benchmark for statistical and population genetics workloads using the gnomAD v3.1.2
//! release of the jointly called One Thousand Genomes (1kG) and Human Genome Diversity Project
//! (HGDP) dataset (1kG+HGDP). Only a prefix of Chromosome 21 is used for benchmarking.
//!
//! Data source: <https://gnomad.broadinstitute.org/>
//!
//! Queries test common genomics analytics patterns:
//!
//! - Allele frequency calculations.
//!
//! - Hardy-Weinberg equilibrium statistics calculation.
//!
//! - Variant filtering based on frequency.
//!
//! - Variant filtering based on locus interval.
//!
//! - Random access to a specific variant by contig and position.

mod builder;
mod download_vcf;
mod schema;
mod statpopgen_benchmark;
mod vcf_conversion;

pub use statpopgen_benchmark::*;
