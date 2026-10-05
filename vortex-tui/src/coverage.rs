// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registration coverage diagnostics. This reports candidates, not runtime applicability.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use vortex::aggregate_fn::session::AggregateFnSession;
use vortex::array::optimizer::kernels::KernelSession;
use vortex::array::session::ArraySession;
use vortex::layout::session::LayoutSession;
use vortex::scalar_fn::session::ScalarFnSession;
use vortex::session::SessionExt;
use vortex::session::VortexSession;

/// Arguments for registration coverage diagnostics.
#[derive(Debug, clap::Args)]
pub struct CoverageArgs {
    /// JSON manifest of expected registrations, grouped by encoding.
    #[arg(long)]
    expectations: Option<PathBuf>,
    /// Fail after reporting missing expectations (warnings are the default).
    #[arg(long, requires = "expectations")]
    strict: bool,
    /// Emit a machine-readable report.
    #[arg(long, conflicts_with = "baseline")]
    json: bool,
    /// Emit the current registrations as an editable expectations manifest.
    #[arg(long, conflicts_with_all = ["expectations", "strict"])]
    baseline: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Compute,
    SessionReduce,
    StaticReduce,
    StaticParentReduce,
    Aggregate,
    GroupedAggregate,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Compute => "compute",
            Self::SessionReduce => "session_reduce",
            Self::StaticReduce => "static_reduce",
            Self::StaticParentReduce => "static_parent_reduce",
            Self::Aggregate => "aggregate",
            Self::GroupedAggregate => "grouped_aggregate",
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Requirement {
    kind: Kind,
    operation: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct Expectations {
    require_all_encodings: bool,
    layouts: BTreeSet<String>,
    scalar_functions: BTreeSet<String>,
    aggregate_functions: BTreeSet<String>,
    // An empty list still requires the encoding to be present. "*" holds encoding-agnostic kernels.
    encodings: BTreeMap<String, BTreeSet<Requirement>>,
}

#[derive(Debug, Serialize)]
struct Registration {
    encoding: String,
    #[serde(flatten)]
    requirement: Requirement,
    candidates: usize,
}

#[derive(Debug, Serialize)]
struct OpaqueReductions {
    encoding: String,
    self_reduce: bool,
    parent_reduce: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    encodings: BTreeSet<String>,
    layouts: BTreeSet<String>,
    scalar_functions: BTreeSet<String>,
    aggregate_functions: BTreeSet<String>,
    registrations: Vec<Registration>,
    opaque_reductions: Vec<OpaqueReductions>,
    warnings: Vec<String>,
}

impl Report {
    fn collect(session: &VortexSession) -> Self {
        // Diagnostics must not install default registries and hide missing initialization.
        let arrays = session.get_opt::<ArraySession>();
        let layouts = session.get_opt::<LayoutSession>();
        let scalar_fns = session.get_opt::<ScalarFnSession>();
        let aggregate_fns = session.get_opt::<AggregateFnSession>();
        let kernels = session.get_opt::<KernelSession>();
        let mut report = Self {
            schema_version: 1,
            encodings: BTreeSet::new(),
            layouts: layouts
                .as_ref()
                .map(|layouts| {
                    layouts.registry().read(|registry| {
                        registry.keys().map(ToString::to_string).collect()
                    })
                })
                .unwrap_or_default(),
            scalar_functions: scalar_fns
                .as_ref()
                .map(|fns| {
                    fns.registry().read(|registry| {
                        registry.keys().map(ToString::to_string).collect()
                    })
                })
                .unwrap_or_default(),
            aggregate_functions: aggregate_fns
                .iter()
                .flat_map(|fns| fns.registered_ids())
                .map(|id| id.to_string())
                .collect(),
            registrations: Vec::new(),
            opaque_reductions: Vec::new(),
            warnings: Vec::new(),
        };
        for plugin in arrays.iter().flat_map(|arrays| arrays.plugins()) {
            let encoding = plugin.id().to_string();
            report.encodings.insert(encoding.clone());
            let rules = plugin.reduction_rules();
            if rules.reduce.is_none() || rules.parent_reduce.is_none() {
                report.opaque_reductions.push(OpaqueReductions {
                    encoding: encoding.clone(),
                    self_reduce: rules.reduce.is_none(),
                    parent_reduce: rules.parent_reduce.is_none(),
                });
            }
            for (kind, descriptions) in [
                (Kind::StaticReduce, rules.reduce),
                (Kind::StaticParentReduce, rules.parent_reduce),
            ] {
                for description in descriptions.into_iter().flatten() {
                    report.add(&encoding, kind.clone(), description, 1);
                }
            }
        }
        for entry in kernels.iter().flat_map(|kernels| kernels.registrations()) {
            for (kind, count) in [
                (Kind::Compute, entry.execute),
                (Kind::SessionReduce, entry.reduce),
            ] {
                if count > 0 {
                    report.add(entry.child.as_str(), kind, entry.parent.to_string(), count);
                }
            }
        }
        for entry in aggregate_fns.iter().flat_map(|fns| fns.kernel_registrations()) {
            report.add(
                entry.encoding.as_ref().map_or("*", |id| id.as_str()),
                if entry.grouped {
                    Kind::GroupedAggregate
                } else {
                    Kind::Aggregate
                },
                entry
                    .function
                    .map_or_else(|| "*".to_string(), |id| id.to_string()),
                1,
            );
        }
        report.registrations.sort_by(|left, right| {
            (&left.encoding, &left.requirement).cmp(&(&right.encoding, &right.requirement))
        });
        report
    }

    fn add(&mut self, encoding: &str, kind: Kind, operation: String, candidates: usize) {
        self.registrations.push(Registration {
            encoding: encoding.to_string(),
            requirement: Requirement { kind, operation },
            candidates,
        });
    }

    fn baseline(&self) -> Expectations {
        let mut encodings: BTreeMap<String, BTreeSet<Requirement>> = self
            .encodings
            .iter()
            .map(|encoding| (encoding.clone(), BTreeSet::new()))
            .collect();
        for entry in &self.registrations {
            encodings
                .entry(entry.encoding.clone())
                .or_default()
                .insert(entry.requirement.clone());
        }
        Expectations {
            require_all_encodings: true,
            layouts: self.layouts.clone(),
            scalar_functions: self.scalar_functions.clone(),
            aggregate_functions: self.aggregate_functions.clone(),
            encodings,
        }
    }

    fn validate(&mut self, expectations: &Expectations) {
        let actual = self.baseline();
        if expectations.require_all_encodings {
            for encoding in &self.encodings {
                if !expectations.encodings.contains_key(encoding) {
                    self.warnings.push(format!(
                        "No expectations declared for encoding: {encoding}"
                    ));
                }
            }
        }
        for (kind, expected, available) in [
            ("layout", &expectations.layouts, &actual.layouts),
            (
                "scalar function",
                &expectations.scalar_functions,
                &actual.scalar_functions,
            ),
            (
                "aggregate function",
                &expectations.aggregate_functions,
                &actual.aggregate_functions,
            ),
        ] {
            for missing in expected.difference(available) {
                self.warnings.push(format!("Missing {kind}: {missing}"));
            }
        }
        for (encoding, required) in &expectations.encodings {
            let Some(available) = actual.encodings.get(encoding) else {
                self.warnings
                    .push(format!("Missing encoding or kernel owner: {encoding}"));
                continue;
            };
            for missing in required.difference(available) {
                let static_rule = matches!(
                    missing.kind,
                    Kind::StaticReduce | Kind::StaticParentReduce
                );
                let opaque = static_rule && !self.encodings.contains(encoding)
                    || self.opaque_reductions.iter().any(|hooks| {
                        hooks.encoding == *encoding
                            && match missing.kind {
                                Kind::StaticReduce => hooks.self_reduce,
                                Kind::StaticParentReduce => hooks.parent_reduce,
                                _ => false,
                            }
                    });
                if opaque {
                    self.warnings.push(format!(
                        "Cannot verify opaque reduction hook: {encoding} {} {}",
                        missing.kind, missing.operation,
                    ));
                    continue;
                }
                self.warnings.push(format!(
                    "Missing registration: {encoding} {} {}",
                    missing.kind, missing.operation,
                ));
            }
        }
    }

    fn write_text(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "Registration coverage (candidates may decline an input)")?;
        writeln!(
            out,
            "{} encodings, {} layouts, {} scalar functions, {} aggregate functions, {} registration entries",
            self.encodings.len(),
            self.layouts.len(),
            self.scalar_functions.len(),
            self.aggregate_functions.len(),
            self.registrations.len(),
        )?;
        for (kind, ids) in [
            ("Encodings", &self.encodings),
            ("Layouts", &self.layouts),
            ("Scalar functions", &self.scalar_functions),
            ("Aggregate functions", &self.aggregate_functions),
        ] {
            writeln!(
                out,
                "{kind}: {}",
                ids.iter().cloned().collect::<Vec<_>>().join(", "),
            )?;
        }
        for entry in &self.registrations {
            writeln!(
                out,
                "{}\t{}\t{}\t{}",
                entry.encoding,
                entry.requirement.kind,
                entry.requirement.operation,
                entry.candidates,
            )?;
        }
        for opaque in &self.opaque_reductions {
            writeln!(
                out,
                "{}: opaque reduction hooks (self={}, parent={}); no exposed table does not imply no support",
                opaque.encoding, opaque.self_reduce, opaque.parent_reduce,
            )?;
        }
        for warning in &self.warnings {
            writeln!(out, "WARNING: {warning}")?;
        }
        Ok(())
    }
}

/// Report session registrations and optionally validate an expectations manifest.
///
/// # Errors
///
/// Returns an error for invalid manifests, output failures, or unmet expectations in strict mode.
pub fn exec_coverage(session: &VortexSession, args: CoverageArgs) -> anyhow::Result<()> {
    let expectations = args
        .expectations
        .as_ref()
        .map(|path| {
            let file = File::open(path)
                .with_context(|| format!("opening expectations {}", path.display()))?;
            serde_json::from_reader::<_, Expectations>(file)
                .with_context(|| format!("reading expectations {}", path.display()))
        })
        .transpose()?;
    let mut report = Report::collect(session);
    if let Some(expectations) = expectations {
        report.validate(&expectations);
    }
    let mut out = io::stdout().lock();
    if args.baseline {
        serde_json::to_writer_pretty(&mut out, &report.baseline())?;
        writeln!(out)?;
    } else if args.json {
        serde_json::to_writer_pretty(&mut out, &report)?;
        writeln!(out)?;
    } else {
        report.write_text(&mut out)?;
    }
    out.flush()?;
    anyhow::ensure!(
        !args.strict || report.warnings.is_empty(),
        "{} unmet coverage expectations",
        report.warnings.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::optimizer::kernels::ArrayKernelsExt;
    use vortex::array::optimizer::kernels::KernelSession;
    use vortex::error::VortexResult;
    use vortex::session::SessionExt;
    use vortex::session::VortexSession;
    use vortex::session::registry::CachedId;

    use super::Expectations;
    use super::Kind;
    use super::Report;
    use super::Requirement;

    #[test]
    fn test_empty_session_is_not_initialized_by_diagnostics() {
        let session = VortexSession::empty();
        let report = Report::collect(&session);
        assert!(report.encodings.is_empty());
        assert!(report.layouts.is_empty());
        assert!(report.scalar_functions.is_empty());
        assert!(report.aggregate_functions.is_empty());
        assert!(report.registrations.is_empty());
        assert!(session.get_opt::<KernelSession>().is_none());
    }

    #[test]
    fn test_baseline_roundtrip() -> anyhow::Result<()> {
        let mut report = Report::collect(&VortexSession::default());
        let baseline = serde_json::to_string(&report.baseline())?;
        report.validate(&serde_json::from_str(&baseline)?);
        assert!(report.warnings.is_empty());
        assert!(report.registrations.iter().any(|entry| {
            entry.encoding == "vortex.dict" && entry.requirement.kind == Kind::StaticParentReduce
        }));
        Ok(())
    }

    #[test]
    fn test_missing_encoding_policy_is_reported() {
        let mut report = Report::collect(&VortexSession::default());
        let mut expectations = report.baseline();
        expectations.encodings.remove("vortex.dict");
        report.validate(&expectations);
        assert_eq!(
            report.warnings,
            ["No expectations declared for encoding: vortex.dict"],
        );
    }

    #[test]
    fn test_missing_compute_registration_is_reported() {
        let session = VortexSession::default();
        let expectations = Report::collect(&session).baseline();
        session.register(KernelSession::empty());
        let mut report = Report::collect(&session);
        report.validate(&expectations);
        assert!(report.warnings.iter().any(|warning| {
            warning == "Missing registration: vortex.bool compute vortex.binary"
        }));
    }

    #[test]
    fn test_missing_expectations_are_reported() -> anyhow::Result<()> {
        let mut report = Report::collect(&VortexSession::default());
        let expectations: Expectations = serde_json::from_str(
            r#"{
            "layouts": ["missing.layout"],
            "encodings": {
                "missing.encoding": [],
                "vortex.dict": [{"kind": "compute", "operation": "missing.function"}]
            }
        }"#,
        )?;
        report.validate(&expectations);
        assert_eq!(report.warnings.len(), 3);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("missing.function"))
        );
        Ok(())
    }

    #[test]
    fn test_wildcard_aggregate_is_not_exact_registration() {
        let mut report = Report::collect(&VortexSession::default());
        assert!(report.registrations.iter().any(|entry| {
            entry.encoding == "vortex.chunked"
                && entry.requirement.kind == Kind::Aggregate
                && entry.requirement.operation == "*"
        }));
        let mut expectations = Expectations::default();
        expectations.encodings.insert(
            "vortex.chunked".to_string(),
            BTreeSet::from([
                Requirement {
                    kind: Kind::Aggregate,
                    operation: "*".to_string(),
                },
                Requirement {
                    kind: Kind::Aggregate,
                    operation: "missing.aggregate".to_string(),
                },
            ]),
        );
        report.validate(&expectations);
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].contains("missing.aggregate"));
    }

    fn decline_reduce(
        _child: &ArrayRef,
        _parent: &ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }

    #[test]
    fn test_custom_session_registration_is_discovered() {
        let session = VortexSession::default();
        static PARENT: CachedId = CachedId::new("custom.parent");
        static CHILD: CachedId = CachedId::new("custom.child");
        session
            .kernels()
            .register_reduce_parent(*PARENT, *CHILD, &[decline_reduce, decline_reduce]);
        let report = Report::collect(&session);
        let entry = report
            .registrations
            .iter()
            .find(|entry| entry.encoding == "custom.child")
            .expect("custom kernel is in the report even without a serde plugin");
        assert_eq!(entry.requirement.kind, Kind::SessionReduce);
        assert_eq!(entry.requirement.operation, "custom.parent");
        assert_eq!(entry.candidates, 2);
    }

    #[test]
    fn test_manifest_rejects_unknown_fields_and_kinds() {
        assert!(serde_json::from_str::<Expectations>(r#"{"encoding": {}}"#).is_err());
        assert!(
            serde_json::from_str::<Expectations>(r#"{
                "encodings": {"vortex.dict": [{"kind": "unsupported", "operation": "vortex.cast"}]}
            }"#)
            .is_err()
        );
    }
}
