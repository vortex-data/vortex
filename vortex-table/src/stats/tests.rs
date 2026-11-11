// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_error::VortexResult;
use vortex_file::Footer;

use super::TableStats;
use crate::footer::parse_footer;
use crate::tests::SESSION;
use crate::tests::table;

async fn footers() -> VortexResult<Vec<Footer>> {
    let (_, manifest) = table(true).await?;
    manifest
        .entries()
        .iter()
        .map(|entry| {
            let region = entry.footer.clone().expect("inlined footer");
            parse_footer(region, entry.byte_size, &SESSION)
        })
        .collect()
}

/// File `f` holds ids `f * 1000 .. (f + 1) * 1000`, so these filters keep the listed files. Names
/// compare as strings: file 0 spans `"row-0" ..= "row-999"`, which contains `"row-3999"`.
#[rstest]
#[case::range(gt_eq(col("id"), lit(2500u64)), &[false, false, true, true])]
#[case::point(eq(col("id"), lit(1234u64)), &[false, true, false, false])]
#[case::none(gt(col("id"), lit(10_000u64)), &[false, false, false, false])]
#[case::all(lt(col("id"), lit(10_000u64)), &[true, true, true, true])]
#[case::conjunction(
    and(gt_eq(col("id"), lit(1000u64)), lt(col("id"), lit(2000u64))),
    &[false, true, false, false],
)]
#[case::disjunction(
    or(lt(col("id"), lit(500u64)), gt_eq(col("id"), lit(3500u64))),
    &[true, false, false, true],
)]
#[case::string(eq(col("name"), lit("row-3999")), &[true, false, false, true])]
#[case::row_count(is_null(col("id")), &[false, false, false, false])]
#[tokio::test]
async fn prune_matches_each_footer(
    #[case] filter: Expression,
    #[case] kept: &[bool],
) -> VortexResult<()> {
    let footers = footers().await?;
    let stats = TableStats::try_new(&footers, footers[0].dtype(), &SESSION)?;
    // The stats a written manifest reads back, without inlined footers to fall back on.
    let (_, manifest) = table(false).await?;
    let read = manifest.stats().expect("the manifest records stats");
    for pruned in [stats.prune(&filter)?, read.prune(&filter)?] {
        for (i, footer) in footers.iter().enumerate() {
            assert_eq!(
                pruned.value(i),
                footer.can_prune(&filter, &SESSION)?,
                "file {i} under {filter}"
            );
            assert_eq!(!pruned.value(i), kept[i], "file {i} under {filter}");
        }
    }
    Ok(())
}
