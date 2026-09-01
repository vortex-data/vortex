// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-exactness tests against the reference C generator.
//!
//! Each digest below is the SHA-256 of the `.tbl` output of `eyalroz/ssb-dbgen` at
//! `ae1e254aa4d603d8ef1f44078e5abed011634b23`, built with `cmake -DCMAKE_BUILD_TYPE=Release` and
//! run as `dbgen -s 1 -f` under `TZ=UTC`, hashed with `shasum -a 256`. Reproduce with:
//!
//! ```text
//! git clone https://github.com/eyalroz/ssb-dbgen && cd ssb-dbgen
//! git checkout ae1e254aa4d603d8ef1f44078e5abed011634b23
//! cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target dbgen
//! mkdir -p out && TZ=UTC DSS_CONFIG=$PWD/build DSS_PATH=$PWD/out ./build/dbgen -s 1 -f
//! shasum -a 256 out/*.tbl
//! ```
//!
//! The dimensions are hashed whole. `lineorder` is hashed over its first
//! [`LINEORDER_PREFIX_ROWS`] rows to keep the test cheap; all five tables were compared in full at
//! SF 1 and SF 10 when this landed, and [`dump_tbl`] regenerates the `.tbl` files to repeat that
//! at any scale factor.

use std::fmt::Display;
use std::fmt::Write as _;
use std::num::NonZeroUsize;

use sha2::Digest;
use sha2::Sha256;

use crate::ssb::ssbgen::CustomerGenerator;
use crate::ssb::ssbgen::DwdateGenerator;
use crate::ssb::ssbgen::LineorderGenerator;
use crate::ssb::ssbgen::PartGenerator;
use crate::ssb::ssbgen::SupplierGenerator;
use crate::ssb::ssbgen::arrow::SsbTable;
use crate::ssb::ssbgen::max_supported_scale_factor;
use crate::ssb::ssbgen::part_key_max;
use crate::ssb::ssbgen::part_row_count;
use crate::ssb::ssbgen::validate_scale_factor;

const PART_SF1: &str = "7c1b8e710b9677d75f34939b965cb512cbe4d313d1316301d25bf6fc5965f54c";
const SUPPLIER_SF1: &str = "d09faf5a22389e895eb27eb3989934715c8124a0048fd66b06c75a833b420e2e";
const CUSTOMER_SF1: &str = "1948243884d70a882945cf4f65df79f46ee7e5b8d2daa304bfaa3d20925b5326";
const DWDATE_SF1: &str = "a389677a574fc53906f17b3bfb0be93dd4ebe9c12295d45c01eb1c58a1152aa7";
const LINEORDER_SF1_PREFIX: &str =
    "7b5d609d724952832845ba7f424285fce128329ee1989b4abc2235ac3d046366";

/// Rows of `lineorder` covered by [`LINEORDER_SF1_PREFIX`].
const LINEORDER_PREFIX_ROWS: usize = 100_000;

/// Same reference generator at `-s 10`. The dimensions are cheap enough to hash whole even at this
/// scale, which covers the scale-dependent stream arithmetic that SF 1 cannot: `part`'s cardinality
/// formula, and `customer`'s carry-over past a ten-times-larger `supplier`.
const PART_SF10: &str = "306be63ec61dfc509a26000776069a25cf5bebc506af8151c8ac7ef49251942d";
const SUPPLIER_SF10: &str = "ef4a51fb9a006187f8171796afb37e0fdb9adc975892282398969b955c027683";
const CUSTOMER_SF10: &str = "25fa596b57468730e69a6869e4ced945a35be552f5955ac07621be10acbafaea";

/// The tail of `lineorder` at SF 1, rows 5,900,001 onwards. The prefix digest cannot catch a
/// divergence in late per-order stream state; this can.
const LINEORDER_SF1_TAIL: &str = "b6598ced6a31a900775d08a24dd0ae4d0d94af7e7c48ec4af8866c07dd45dc80";
/// Rows to skip before hashing for [`LINEORDER_SF1_TAIL`].
const LINEORDER_TAIL_SKIP: usize = 5_900_000;

/// SHA-256 over the `.tbl` rendering of `rows`, newline terminated as the reference writes it.
fn digest(rows: impl Iterator<Item = impl Display>) -> String {
    let mut hasher = Sha256::new();
    let mut line = String::new();
    for row in rows {
        line.clear();
        writeln!(line, "{row}").expect("writing to a String cannot fail");
        hasher.update(line.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").expect("writing to a String cannot fail");
            out
        })
}

#[test]
fn part_matches_reference() {
    assert_eq!(digest(PartGenerator::new(1.0).iter()), PART_SF1);
}

#[test]
fn supplier_matches_reference() {
    assert_eq!(digest(SupplierGenerator::new(1.0).iter()), SUPPLIER_SF1);
}

#[test]
fn customer_matches_reference() {
    assert_eq!(digest(CustomerGenerator::new(1.0).iter()), CUSTOMER_SF1);
}

#[test]
fn dwdate_matches_reference() {
    assert_eq!(digest(DwdateGenerator::new().iter()), DWDATE_SF1);
}

#[test]
fn lineorder_matches_reference() {
    let rows = LineorderGenerator::new(1.0)
        .iter()
        .take(LINEORDER_PREFIX_ROWS);
    assert_eq!(digest(rows), LINEORDER_SF1_PREFIX);
}

#[test]
fn scale_factor_cardinalities() {
    // The row count uses a base-2 log while `lo_partkey`'s range uses a natural log, so from
    // SF 3 up the fact table cannot reference every part. Reproduced from the reference; see the
    // module docs.
    assert_eq!(part_row_count(1.0), 200_000);
    assert_eq!(part_key_max(1.0), 200_000);
    assert_eq!(part_row_count(10.0), 800_000);
    assert_eq!(part_key_max(10.0), 600_000);

    assert_eq!(SupplierGenerator::new(10.0).row_count(), 20_000);
    assert_eq!(CustomerGenerator::new(10.0).row_count(), 300_000);
    assert_eq!(LineorderGenerator::new(10.0).order_count(), 15_000_000);
    assert_eq!(DwdateGenerator::new().row_count(), 2557);
}

#[test]
fn schemas_cover_every_column() {
    // Guards the Arrow builders against drifting from the schemas they claim to fill.
    for table in SsbTable::ALL {
        let schema = table.schema();
        let batch = table
            .batches(1.0, NonZeroUsize::new(8).expect("8 is nonzero"))
            .next()
            .unwrap_or_else(|| panic!("{} generated no rows", table.name()));
        assert_eq!(batch.schema(), schema, "{}", table.name());
        assert_eq!(batch.num_rows(), 8, "{}", table.name());
    }
}

#[test]
fn supplier_matches_reference_at_sf10() {
    assert_eq!(digest(SupplierGenerator::new(10.0).iter()), SUPPLIER_SF10);
}

#[test]
fn customer_matches_reference_at_sf10() {
    // Also pins the cross-table carry-over: customer's city and phone streams must start where
    // supplier's 20,000 rows left off, not at their base seeds.
    assert_eq!(digest(CustomerGenerator::new(10.0).iter()), CUSTOMER_SF10);
}

#[test]
#[ignore = "slow (800k rows of 92-draw permutations); run with --ignored"]
fn part_matches_reference_at_sf10() {
    assert_eq!(digest(PartGenerator::new(10.0).iter()), PART_SF10);
}

#[test]
#[ignore = "slow (generates 6M rows to reach the tail); run with --ignored"]
fn lineorder_tail_matches_reference() {
    let rows = LineorderGenerator::new(1.0)
        .iter()
        .skip(LINEORDER_TAIL_SKIP);
    assert_eq!(digest(rows), LINEORDER_SF1_TAIL);
}

#[test]
fn unsupported_scale_factors_are_rejected() {
    // Each of these used to generate a directory of Parquet and exit successfully. At SF 0.01 the
    // part dimension came out empty while lineorder drew keys from an inverted range, producing
    // negative part keys that join to nothing.
    for bad in [0.01, 0.5, 0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            validate_scale_factor(bad).is_err(),
            "scale factor {bad} should be rejected",
        );
    }

    // And the upper bound, where the sparse lo_orderkey outgrows the schema's INTEGER.
    let max = max_supported_scale_factor();
    assert!(
        validate_scale_factor(max as f64).is_ok(),
        "SF {max} should be supported"
    );
    assert!(
        validate_scale_factor((max + 1) as f64).is_err(),
        "SF {} should be rejected",
        max + 1,
    );
    // Pinned so a change to the key arithmetic or the schema width shows up here.
    assert_eq!(max, 357);
}

#[test]
fn supported_scale_factors_are_accepted() {
    for sf in [1.0, 1.5, 10.0, 100.0, 300.0] {
        assert!(
            validate_scale_factor(sf).is_ok(),
            "scale factor {sf} should be supported"
        );
    }
}

/// Write every table's `.tbl` rendering to `$SSB_TBL_OUT` at `$SSB_TBL_SF` (default SF 1), for
/// diffing against the reference generator's output.
///
/// ```text
/// SSB_TBL_OUT=/tmp/ssb SSB_TBL_SF=1 cargo test -p vortex-bench --lib \
///     ssb::ssbgen::tests::dump_tbl -- --ignored
/// ```
#[test]
#[ignore = "developer tool: writes .tbl files for offline comparison"]
fn dump_tbl() -> anyhow::Result<()> {
    use std::fs;
    use std::io::BufWriter;
    use std::io::Write as _;
    use std::path::PathBuf;

    let out = PathBuf::from(std::env::var("SSB_TBL_OUT")?);
    let scale_factor: f64 = std::env::var("SSB_TBL_SF")
        .unwrap_or_else(|_| "1".to_string())
        .parse()?;
    fs::create_dir_all(&out)?;

    fn write(path: PathBuf, rows: impl Iterator<Item = impl Display>) -> anyhow::Result<()> {
        let mut file = BufWriter::new(fs::File::create(path)?);
        for row in rows {
            writeln!(file, "{row}")?;
        }
        Ok(file.flush()?)
    }

    write(
        out.join("part.tbl"),
        PartGenerator::new(scale_factor).iter(),
    )?;
    write(
        out.join("supplier.tbl"),
        SupplierGenerator::new(scale_factor).iter(),
    )?;
    write(
        out.join("customer.tbl"),
        CustomerGenerator::new(scale_factor).iter(),
    )?;
    write(out.join("date.tbl"), DwdateGenerator::new().iter())?;
    write(
        out.join("lineorder.tbl"),
        LineorderGenerator::new(scale_factor).iter(),
    )
}
