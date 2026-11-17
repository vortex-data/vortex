// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check files: parsing, running their directives, and reporting the outcomes.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context as _;
use anyhow::bail;
use filecheck::CheckerBuilder;
use filecheck::NO_VARIABLES;
use regex::Regex;

use crate::units::Unit;

/// How many lines of the match trace to show on each side of the failing directive.
const CONTEXT_LINES: usize = 4;

/// How many failing units to report per section.
const MAX_REPORTED_FAILURES: usize = 5;

/// The directives the filecheck crate understands, at the start of a line.
const DIRECTIVE_RX: &str = r"^\s*(check|sameln|nextln|unordered|not|regex):(.*)$";

/// A filecheck `Missed #N: ...` or `Matched #N not: ...` line of a match trace.
const FAILED_MATCH_RX: &str = r"^(?:Missed #(\d+):|Matched #(\d+) not:)";

/// A path glob where `*` matches any run of characters and `?` matches one character.
#[derive(Debug, Clone)]
pub struct Glob {
    pattern: String,
    regex: Regex,
}

impl Glob {
    /// Compile a glob pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the translated regular expression does not compile.
    pub fn new(pattern: &str) -> anyhow::Result<Self> {
        let mut rx = String::from("^");
        for c in pattern.chars() {
            match c {
                '*' => rx.push_str(".*"),
                '?' => rx.push('.'),
                c => rx.push_str(&regex::escape(c.encode_utf8(&mut [0; 4]))),
            }
        }
        rx.push('$');
        Ok(Self {
            pattern: pattern.to_string(),
            regex: Regex::new(&rx)?,
        })
    }

    /// The glob as written.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Whether `path` matches the whole glob.
    pub fn matches(&self, path: &str) -> bool {
        self.regex.is_match(path)
    }
}

/// How a failing section is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Severity {
    /// Report the failure but do not fail the run.
    Warn,
    /// Fail the run.
    #[default]
    Error,
}

impl Severity {
    fn parse(value: &str, lineno: usize) -> anyhow::Result<Self> {
        match value {
            "warn" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            _ => bail!("line {lineno}: `severity:` must be `warn` or `error`, found `{value}`"),
        }
    }
}

/// A parsed check file.
#[derive(Debug)]
pub struct CheckFile {
    /// The `files: <regex>` patterns naming the Vortex files the check file applies to, each
    /// anchored to match a whole path.
    pub files: Vec<Regex>,
    /// The directive sections, starting with the whole-file section.
    pub sections: Vec<CheckSection>,
}

impl CheckFile {
    /// Whether `path` matches one of the `files:` patterns.
    pub fn applies_to(&self, path: &str) -> bool {
        self.files.iter().any(|files| files.is_match(path))
    }
}

/// One filecheck directive of a check file.
#[derive(Debug, Clone)]
pub struct Directive {
    /// The 1-based line of the directive in the check file.
    pub line: usize,
    /// The directive as written, such as `check: root: vortex.fsst`.
    pub text: String,
}

impl Directive {
    fn is_not(&self) -> bool {
        self.text.trim_start().starts_with("not:")
    }

    fn is_positive(&self) -> bool {
        !self.is_not() && !self.text.trim_start().starts_with("regex:")
    }
}

/// A group of directives from a check file.
#[derive(Debug)]
pub struct CheckSection {
    /// Units to check one by one, or `None` to check the dump of the whole file once.
    pub selector: Option<Glob>,
    /// The 1-based line of the `units:` directive, or `None` for the leading section.
    pub line: Option<usize>,
    /// The filecheck directives of the section, in order.
    pub directives: Vec<Directive>,
    /// How the section is reported if it fails.
    pub severity: Severity,
}

impl CheckSection {
    fn label(&self) -> String {
        match (&self.selector, self.line) {
            (Some(glob), Some(line)) => format!("line {line}: units: {}", glob.pattern),
            _ => "whole file".to_string(),
        }
    }
}

/// One failed check.
#[derive(Debug)]
pub struct Failure {
    /// The unit that failed, or `None` for a whole-file section.
    pub unit: Option<String>,
    /// The directives that failed: the one that missed or matched, or every `not:` directive
    /// that may have matched after the last positive match.
    pub directives: Vec<Directive>,
    /// The lines of the match trace around the failure.
    pub context: Vec<String>,
    /// The full filecheck match trace.
    pub explanation: String,
}

/// The outcome of checking one section.
#[derive(Debug)]
pub struct SectionOutcome {
    /// A human-readable description of the section.
    pub label: String,
    /// How many texts were checked: matching units, or 1 for the whole file.
    pub checked: usize,
    /// The checks that failed.
    pub failures: Vec<Failure>,
    /// How the section is reported if it fails.
    pub severity: Severity,
}

impl SectionOutcome {
    /// Whether every check in the section passed.
    ///
    /// A `units:` section that matched no unit fails, so a typo in a glob cannot pass silently.
    pub fn passed(&self) -> bool {
        self.checked > 0 && self.failures.is_empty()
    }

    /// Whether the section failed with [`Severity::Error`], so the run fails.
    pub fn is_error(&self) -> bool {
        !self.passed() && self.severity == Severity::Error
    }

    /// Whether the section failed with [`Severity::Warn`].
    pub fn is_warning(&self) -> bool {
        !self.passed() && self.severity == Severity::Warn
    }

    /// Report a failure of this section as a warning, as when checks run in warn-only mode.
    pub fn downgrade_to_warning(&mut self) {
        self.severity = Severity::Warn;
    }

    fn status(&self) -> &'static str {
        if self.passed() {
            "PASS"
        } else if self.is_warning() {
            "WARN"
        } else {
            "FAIL"
        }
    }
}

/// Read and parse the check file at `path`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not parse.
pub fn read_check_file(path: &Path) -> anyhow::Result<CheckFile> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading check file {}", path.display()))?;
    parse_check_file(&text).with_context(|| format!("parsing check file {}", path.display()))
}

/// Parse a check file.
///
/// Every line must be blank, a `#` comment, a `files: <regex>` line, a `units: <glob>` line,
/// a `severity: warn|error` line, or a filecheck directive, each at the start of the line.
///
/// # Errors
///
/// Returns an error, naming the line, for any other line, a directive with an empty pattern,
/// a malformed regex or glob, a `files:` line after the first `units:` line, a second
/// `severity:` line for the same section or file, a `units:` section without directives, or a
/// check file without any directive.
pub fn parse_check_file(text: &str) -> anyhow::Result<CheckFile> {
    let files_rx = Regex::new(r"^\s*files:(.*)$")?;
    let units_rx = Regex::new(r"^\s*units:(.*)$")?;
    let severity_rx = Regex::new(r"^\s*severity:(.*)$")?;
    let directive_rx = Regex::new(DIRECTIVE_RX)?;
    let mut files = Vec::new();
    let mut sections = vec![CheckSection {
        selector: None,
        line: None,
        directives: Vec::new(),
        severity: Severity::default(),
    }];
    // The file's default severity, set before the first `units:` line, and each section's own.
    let mut default_severity = None;
    let mut section_severities = vec![None];
    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(caps) = files_rx.captures(line) {
            let pattern = non_empty(&caps[1], lineno, "files:")?;
            if sections.len() > 1 {
                bail!("line {lineno}: `files:` must come before the first `units:` line");
            }
            files.push(
                Regex::new(&format!("^(?:{pattern})$"))
                    .with_context(|| format!("line {lineno}: invalid `files:` regex"))?,
            );
        } else if let Some(caps) = units_rx.captures(line) {
            let glob = non_empty(&caps[1], lineno, "units:")?;
            sections.push(CheckSection {
                selector: Some(Glob::new(glob)?),
                line: Some(lineno),
                directives: Vec::new(),
                severity: Severity::default(),
            });
            section_severities.push(None);
        } else if let Some(caps) = severity_rx.captures(line) {
            let severity = Severity::parse(non_empty(&caps[1], lineno, "severity:")?, lineno)?;
            let slot = if sections.len() == 1 {
                &mut default_severity
            } else {
                section_severities
                    .last_mut()
                    .context("every section has a severity slot")?
            };
            if slot.replace(severity).is_some() {
                bail!("line {lineno}: `severity:` is already set for this section");
            }
        } else if let Some(caps) = directive_rx.captures(line) {
            non_empty(&caps[2], lineno, &format!("{}:", &caps[1]))?;
            // Fail on a malformed pattern now, with its line, rather than when checking.
            CheckerBuilder::new()
                .directive(line)
                .with_context(|| format!("line {lineno}: invalid directive"))?;
            if let Some(section) = sections.last_mut() {
                section.directives.push(Directive {
                    line: lineno,
                    text: trimmed.to_string(),
                });
            }
        } else {
            bail!(
                "line {lineno}: expected `check:`, `nextln:`, `sameln:`, `not:`, `unordered:`, \
                 `regex:`, `files:`, `units:`, `severity:` or a `#` comment, found `{trimmed}`"
            );
        }
    }

    for (section, severity) in sections.iter_mut().zip(section_severities) {
        if let Some(line) = section.line
            && section.directives.is_empty()
        {
            bail!("line {line}: `units:` section has no directives");
        }
        section.severity = severity.or(default_severity).unwrap_or_default();
    }
    if sections.iter().all(|section| section.directives.is_empty()) {
        bail!("the check file has no directives");
    }
    Ok(CheckFile { files, sections })
}

/// The pattern after a directive's colon, which must be separated from it by whitespace: the
/// filecheck crate ignores `check:foo`.
fn non_empty<'a>(rest: &'a str, lineno: usize, directive: &str) -> anyhow::Result<&'a str> {
    if rest.starts_with(|c: char| !c.is_whitespace()) {
        bail!("line {lineno}: put a space after `{directive}`");
    }
    let rest = rest.trim();
    if rest.is_empty() {
        bail!("line {lineno}: `{directive}` needs a pattern");
    }
    Ok(rest)
}

/// Check `units` against each section, skipping sections without directives.
///
/// # Errors
///
/// Returns an error if a directive is malformed.
pub fn run_checks(
    units: &[Unit],
    sections: &[CheckSection],
) -> anyhow::Result<Vec<SectionOutcome>> {
    let failed_match_rx = Regex::new(FAILED_MATCH_RX)?;
    let mut outcomes = Vec::new();
    for section in sections {
        if section.directives.is_empty() {
            continue;
        }
        let mut builder = CheckerBuilder::new();
        for directive in &section.directives {
            builder
                .directive(&directive.text)
                .with_context(|| format!("line {}: invalid directive", directive.line))?;
        }
        let checker = builder.finish();

        let whole_file;
        let targets: Vec<(Option<&str>, &str)> = match &section.selector {
            None => {
                whole_file = units
                    .iter()
                    .map(|unit| unit.text.as_str())
                    .collect::<String>();
                vec![(None, whole_file.as_str())]
            }
            Some(glob) => units
                .iter()
                .filter(|unit| glob.matches(&unit.path))
                .map(|unit| (Some(unit.path.as_str()), unit.text.as_str()))
                .collect(),
        };

        let mut failures = Vec::new();
        for (unit, text) in &targets {
            let (passed, explanation) = checker.explain(text, NO_VARIABLES)?;
            if !passed {
                failures.push(describe_failure(
                    *unit,
                    text,
                    &section.directives,
                    &failed_match_rx,
                    explanation,
                ));
            }
        }
        outcomes.push(SectionOutcome {
            label: section.label(),
            checked: targets.len(),
            failures,
            severity: section.severity,
        });
    }
    Ok(outcomes)
}

/// Find the failing directive in a filecheck match trace, and the trace lines around it.
fn describe_failure(
    unit: Option<&str>,
    text: &str,
    directives: &[Directive],
    failed_match_rx: &Regex,
    explanation: String,
) -> Failure {
    let lines: Vec<&str> = explanation.lines().collect();
    let failed = lines.iter().enumerate().rev().find_map(|(idx, line)| {
        let caps = failed_match_rx.captures(line)?;
        let directive: usize = caps.get(1).or_else(|| caps.get(2))?.as_str().parse().ok()?;
        Some((idx, directive))
    });

    let (directives, context) = match failed {
        Some((idx, directive)) => {
            let start = idx.saturating_sub(CONTEXT_LINES);
            let end = (idx + CONTEXT_LINES + 1).min(lines.len());
            (
                directives.get(directive).cloned().into_iter().collect(),
                lines[start..end]
                    .iter()
                    .map(|line| line.to_string())
                    .collect(),
            )
        }
        // filecheck reports nothing when a `not:` after the last positive match matches, so
        // point at the `not:` directives after it, and at the text a literal one matches.
        None => {
            let last_positive = directives.iter().rposition(Directive::is_positive);
            let trailing_nots: Vec<Directive> = directives
                .iter()
                .skip(last_positive.map_or(0, |idx| idx + 1))
                .filter(|directive| directive.is_not())
                .cloned()
                .collect();
            let context = trailing_nots
                .iter()
                .filter_map(|directive| {
                    let pattern = directive.text.trim_start().strip_prefix("not:")?.trim();
                    text.lines().find(|line| line.contains(pattern))
                })
                .map(|line| format!("> {line}"))
                .collect();
            (trailing_nots, context)
        }
    };

    Failure {
        unit: unit.map(str::to_string),
        directives,
        context,
        explanation,
    }
}

/// Describe check outcomes: one `PASS`, `WARN` or `FAIL` line per section, then for up to five
/// failing units the failing directive and the match trace around it, or the full trace if
/// `explain`, and finally a count of passed, warned and failed sections.
pub fn format_report(outcomes: &[SectionOutcome], explain: bool) -> String {
    let mut out = String::new();
    // Writing to a `String` cannot fail.
    write_report(&mut out, outcomes, explain).unwrap_or_default();
    out
}

fn write_report(out: &mut String, outcomes: &[SectionOutcome], explain: bool) -> std::fmt::Result {
    for outcome in outcomes {
        let status = outcome.status();
        if outcome.passed() {
            writeln!(
                out,
                "{status} {}: {} checked",
                outcome.label, outcome.checked
            )?;
            continue;
        }
        if outcome.checked == 0 {
            writeln!(out, "{status} {}: matched no units", outcome.label)?;
            continue;
        }
        writeln!(
            out,
            "{status} {}: {}/{} failed",
            outcome.label,
            outcome.failures.len(),
            outcome.checked
        )?;
        for failure in outcome.failures.iter().take(MAX_REPORTED_FAILURES) {
            if let Some(unit) = &failure.unit {
                writeln!(out, "  unit {unit}:")?;
            }
            match failure.directives.as_slice() {
                [] => {}
                [directive] => writeln!(
                    out,
                    "    failed: line {}: {}",
                    directive.line, directive.text
                )?,
                directives => {
                    writeln!(out, "    one of these matched:")?;
                    for directive in directives {
                        writeln!(out, "      line {}: {}", directive.line, directive.text)?;
                    }
                }
            }
            let trace: Vec<&str> = if explain || failure.context.is_empty() {
                failure.explanation.lines().collect()
            } else {
                failure.context.iter().map(String::as_str).collect()
            };
            for line in trace {
                writeln!(out, "    {line}")?;
            }
        }
        if outcome.failures.len() > MAX_REPORTED_FAILURES {
            writeln!(
                out,
                "  ... and {} more failing units",
                outcome.failures.len() - MAX_REPORTED_FAILURES
            )?;
        }
    }
    let warned = outcomes
        .iter()
        .filter(|outcome| outcome.is_warning())
        .count();
    let failed = outcomes.iter().filter(|outcome| outcome.is_error()).count();
    writeln!(
        out,
        "{} sections: {} passed, {warned} warned, {failed} failed",
        outcomes.len(),
        outcomes.len() - warned - failed,
    )?;
    Ok(())
}
