// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::buffer::ByteBufferMut;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

use crate::*;

fn test_array() -> anyhow::Result<ArrayRef> {
    let strings: Vec<&str> = ["alpha", "beta", "gamma"]
        .into_iter()
        .cycle()
        .take(10_000)
        .collect();
    Ok(StructArray::from_fields(&[
        ("ints", PrimitiveArray::from_iter(0i64..10_000).into_array()),
        ("strs", VarBinArray::from(strings).into_array()),
    ])?
    .into_array())
}

async fn write_test_file(session: &VortexSession, path: &Path) -> anyhow::Result<()> {
    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buf, test_array()?.to_array_stream())
        .await?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, buf.as_slice())?;
    Ok(())
}

async fn check(checks: &str, excludes: &[&str]) -> anyhow::Result<Vec<SectionOutcome>> {
    let session = VortexSession::default();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test.vortex");
    write_test_file(&session, &path).await?;

    let excludes = excludes
        .iter()
        .map(|pattern| Glob::new(pattern))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let units = load_units(&session, &path, DumpOptions::default(), &excludes).await?;
    run_checks(&units, &parse_check_file(checks)?.sections)
}

#[test]
fn glob_matches_whole_path_with_literal_brackets() -> anyhow::Result<()> {
    let glob = Glob::new("lineitem.l_comment/data[*]")?;
    assert!(glob.matches("lineitem.l_comment/data[0]"));
    assert!(glob.matches("lineitem.l_comment/data[12]"));
    assert!(!glob.matches("lineitem.l_comment/zones"));
    assert!(!glob.matches("xlineitem.l_comment/data[0]"));
    assert!(Glob::new("*/zones")?.matches("a.b/zones"));
    assert!(Glob::new("ints?zones")?.matches("ints/zones"));
    Ok(())
}

#[test]
fn check_file_splits_into_sections() -> anyhow::Result<()> {
    let check = parse_check_file(
        "files: tpch/data/lineitem\\.vortex\n\
         check: a\n\
         # Comments may mention `units: <glob>` and check: directives freely.\n\
         not: b\n\
         \n\
         units: */zones\n\
         check: c\n",
    )?;
    assert_eq!(check.files.len(), 1);
    assert!(check.applies_to("tpch/data/lineitem.vortex"));
    assert!(!check.applies_to("tpch/data/lineitem.vortex.bak"));
    assert!(!check.applies_to("x/tpch/data/lineitem.vortex"));

    let sections = &check.sections;
    assert_eq!(sections.len(), 2);
    assert!(sections[0].selector.is_none());
    let directives: Vec<_> = sections[0]
        .directives
        .iter()
        .map(|directive| (directive.line, directive.text.as_str()))
        .collect();
    assert_eq!(directives, [(2, "check: a"), (4, "not: b")]);
    assert_eq!(sections[1].line, Some(6));
    assert_eq!(sections[1].directives[0].text, "check: c");
    assert!(
        sections[1]
            .selector
            .as_ref()
            .is_some_and(|glob| glob.matches("x/zones"))
    );
    Ok(())
}

#[rstest]
#[case::files_after_units("units: a*\ncheck: b\nfiles: c\\.vortex\n", "line 3")]
#[case::misspelled_directive("chek: root:\n", "line 1")]
#[case::singular_units("unit: a*\ncheck: b\n", "line 1")]
#[case::units_without_space("units:a*\ncheck: b\n", "line 1")]
#[case::empty_pattern("check:\nnot: b\n", "line 1")]
#[case::directive_without_space("check:root:\n", "put a space")]
#[case::empty_units_section("check: a\nunits: b*\n", "line 2")]
#[case::no_directives("files: a\\.vortex\n# only a comment\n", "no directives")]
#[case::directive_mid_line("see check: a\n", "line 1")]
fn malformed_check_files_are_rejected(#[case] text: &str, #[case] message: &str) {
    let err = parse_check_file(text).expect_err("check file must be rejected");
    assert!(
        format!("{err:#}").contains(message),
        "expected {message:?} in {err:#}"
    );
}

#[tokio::test]
async fn failures_name_the_directive_that_missed() -> anyhow::Result<()> {
    let outcomes = check(
        "units: ints*\ncheck: root:\nnextln: vortex.no_such_encoding\n",
        &[],
    )
    .await?;
    let failure = &outcomes[0].failures[0];
    let lines: Vec<_> = failure.directives.iter().map(|d| d.line).collect();
    assert_eq!(lines, [3]);
    assert!(
        failure
            .context
            .iter()
            .any(|line| line.starts_with("Missed #1"))
    );
    Ok(())
}

#[tokio::test]
async fn failures_name_a_trailing_not_that_matched() -> anyhow::Result<()> {
    let outcomes = check("units: ints*\ncheck: unit:\nnot: unit:\nnot: root:\n", &[]).await?;
    let failure = &outcomes[0].failures[0];
    let lines: Vec<_> = failure.directives.iter().map(|d| d.line).collect();
    assert_eq!(lines, [3, 4]);
    assert!(
        failure
            .context
            .iter()
            .any(|line| line.starts_with("> root:"))
    );
    Ok(())
}

#[tokio::test]
async fn whole_file_checks_match_partially_in_order() -> anyhow::Result<()> {
    let outcomes = check(
        "check: unit: ints\ncheck: root: vortex.\ncheck: unit: strs\n",
        &[],
    )
    .await?;
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].passed(), "{outcomes:?}");

    let outcomes = check("check: unit: strs\ncheck: unit: ints\n", &[]).await?;
    assert!(!outcomes[0].passed());
    Ok(())
}

#[tokio::test]
async fn unit_sections_check_every_matching_unit() -> anyhow::Result<()> {
    let outcomes = check(
        "units: strs*\ncheck: root:\n\nunits: ints*\ncheck: vortex.no_such_encoding\n",
        &[],
    )
    .await?;
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes[0].passed(), "{outcomes:?}");
    assert!(outcomes[0].checked > 0);
    assert_eq!(outcomes[1].failures.len(), outcomes[1].checked);
    assert!(outcomes[1].failures.iter().all(|f| f.unit.is_some()));
    Ok(())
}

#[tokio::test]
async fn sections_matching_no_units_fail() -> anyhow::Result<()> {
    let outcomes = check("units: no_such_column*\ncheck: root:\n", &[]).await?;
    assert_eq!(outcomes[0].checked, 0);
    assert!(!outcomes[0].passed());
    Ok(())
}

#[tokio::test]
async fn excluded_units_are_not_checked() -> anyhow::Result<()> {
    let outcomes = check("not: unit: ints\n", &["ints*"]).await?;
    assert!(outcomes[0].passed(), "{outcomes:?}");
    Ok(())
}

#[tokio::test]
async fn check_tree_checks_the_files_the_regexes_match() -> anyhow::Result<()> {
    let session = VortexSession::default();
    let root = tempfile::tempdir()?;
    write_test_file(&session, &root.path().join("tpch/data/a.vortex")).await?;
    write_test_file(&session, &root.path().join("tpch/data/b.vortex")).await?;
    write_test_file(&session, &root.path().join("other/a.vortex")).await?;

    let check = parse_check_file("files: tpch/data/.*\\.vortex\nunits: ints*\ncheck: root:\n")?;
    let outcomes = check_tree(&session, root.path(), &check, DumpOptions::default(), &[]).await?;
    let checked: Vec<_> = outcomes
        .iter()
        .map(|outcome| outcome.path.strip_prefix(root.path()))
        .collect::<Result<_, _>>()?;
    assert_eq!(
        checked,
        [
            Path::new("tpch/data/a.vortex"),
            Path::new("tpch/data/b.vortex")
        ]
    );
    assert!(outcomes.iter().all(FileOutcome::passed));
    Ok(())
}

#[tokio::test]
async fn check_tree_requires_a_matching_file() -> anyhow::Result<()> {
    let session = VortexSession::default();
    let root = tempfile::tempdir()?;
    write_test_file(&session, &root.path().join("a.vortex")).await?;

    for text in [
        "units: *\ncheck: root:\n",
        "files: b\\.vortex\ncheck: root:\n",
    ] {
        let check = parse_check_file(text)?;
        assert!(
            check_tree(&session, root.path(), &check, DumpOptions::default(), &[])
                .await
                .is_err()
        );
    }
    Ok(())
}
