// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Group + chart-link discovery from the five fact tables.
//!
//! `collect_groups` is the entry point: it scans every fact table to find
//! distinct group dimensions, materialises [`super::dto::Group`] records,
//! attaches their summaries, and applies the canonical [`super::dto::GROUP_ORDER`].

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;

use super::descriptions::group_description;
use super::dto::ChartLink;
use super::dto::Group;
use super::dto::group_sort_key;
use super::summary::collect_group_summary;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// Collect every group + chart link derivable from the data. Used by both
/// `GET /api/groups` and the HTML landing page.
pub(crate) fn collect_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut groups: Vec<Group> = Vec::new();

    let qm_groups = collect_query_groups(conn).context("collect_query_groups")?;
    groups.extend(qm_groups);

    if let Some(g) = collect_compression_time_group(conn)? {
        groups.push(g);
    }
    if let Some(g) = collect_compression_size_group(conn)? {
        groups.push(g);
    }
    if let Some(g) = collect_random_access_group(conn)? {
        groups.push(g);
    }
    let vsr_groups = collect_vector_search_groups(conn)?;
    groups.extend(vsr_groups);

    for group in &mut groups {
        let key = GroupKey::from_slug(&group.slug)
            .with_context(|| format!("invalid generated group slug: {}", group.slug))?;
        group.summary = collect_group_summary(conn, &key, &group.charts)?;
        group.description = group_description(&group.name);
    }

    // Apply canonical ordering. `sort_by_key` is stable, so groups whose
    // names map to the same key (the `GROUP_ORDER.len()` bucket — i.e. not in
    // the canonical list) keep the order the discovery passes produced.
    groups.sort_by(|a, b| group_sort_key(&a.name).cmp(&group_sort_key(&b.name)));

    Ok(groups)
}

fn collect_query_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant, scale_factor, storage, query_idx
          FROM query_measurements
         GROUP BY dataset, dataset_variant, scale_factor, storage, query_idx
         ORDER BY dataset, dataset_variant NULLS FIRST,
                  scale_factor NULLS FIRST, storage, query_idx
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i32>(4)?,
        ))
    })?;

    let mut groups: Vec<Group> = Vec::new();
    let mut current: Option<(String, Option<String>, Option<String>, String)> = None;
    for row in rows {
        let (dataset, dataset_variant, scale_factor, storage, query_idx) = row?;
        let key = (
            dataset.clone(),
            dataset_variant.clone(),
            scale_factor.clone(),
            storage.clone(),
        );
        let need_new_group = current.as_ref() != Some(&key);
        if need_new_group {
            let group_slug = GroupKey::QueryGroup {
                dataset: dataset.clone(),
                dataset_variant: dataset_variant.clone(),
                scale_factor: scale_factor.clone(),
                storage: storage.clone(),
            }
            .to_slug();
            groups.push(Group {
                name: group_name_query(&dataset, &dataset_variant, &scale_factor, &storage),
                slug: group_slug,
                charts: Vec::new(),
                summary: None,
                description: None,
            });
            current = Some(key);
        }
        let slug = ChartKey::QueryMeasurement {
            dataset,
            dataset_variant,
            scale_factor,
            storage,
            query_idx,
        }
        .to_slug();
        groups
            .last_mut()
            .expect("just pushed")
            .charts
            .push(ChartLink {
                name: format!("Q{query_idx}"),
                slug,
            });
    }
    Ok(groups)
}

/// Render a query group name in the same shape v2 used (per the hard-coded
/// list in `origin/ct/vfvb:benchmarks-website/index.html`):
///
/// - `tpch` + storage + scale_factor → `TPC-H (NVMe) (SF=1)`
/// - `tpcds` + storage + scale_factor → `TPC-DS (NVMe) (SF=1)`
/// - `clickbench` → `Clickbench`
/// - anything else → fall back to the legacy `dataset[/variant] sf=N [storage]`
///   shape so unknown datasets still get a deterministic name.
///
/// Variant disambiguation: for tpch/tpcds, if `dataset_variant` is set we
/// append ` / variant`, since v2's list flattened variants but v3 ingests
/// them. Without this, two ingestion variants would collide.
fn group_name_query(
    dataset: &str,
    dataset_variant: &Option<String>,
    scale_factor: &Option<String>,
    storage: &str,
) -> String {
    let storage_label = match storage {
        "nvme" => Some("NVMe"),
        "s3" => Some("S3"),
        _ => None,
    };
    let base = match (dataset, storage_label, scale_factor.as_deref()) {
        ("tpch", Some(s), Some(sf)) => Some(format!("TPC-H ({s}) (SF={sf})")),
        ("tpcds", Some(s), Some(sf)) => Some(format!("TPC-DS ({s}) (SF={sf})")),
        ("clickbench", ..) => Some("Clickbench".to_string()),
        _ => None,
    };
    if let Some(mut name) = base {
        if let Some(v) = dataset_variant {
            name.push_str(" / ");
            name.push_str(v);
        }
        return name;
    }
    // Legacy fallback for unknown datasets — keeps the page rendering rather
    // than silently dropping data.
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    if let Some(sf) = scale_factor {
        name.push_str(" sf=");
        name.push_str(sf);
    }
    name.push_str(" [");
    name.push_str(storage);
    name.push(']');
    name
}

fn collect_compression_time_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant
          FROM compression_times
         GROUP BY dataset, dataset_variant
         ORDER BY dataset, dataset_variant NULLS FIRST
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| {
            let dataset: String = row.get(0)?;
            let dataset_variant: Option<String> = row.get(1)?;
            Ok((dataset, dataset_variant))
        })?
        .map(|r| {
            r.map(|(dataset, dataset_variant)| {
                let key = ChartKey::CompressionTime {
                    dataset: dataset.clone(),
                    dataset_variant: dataset_variant.clone(),
                };
                let mut name = dataset;
                if let Some(v) = &dataset_variant {
                    name.push('/');
                    name.push_str(v);
                }
                ChartLink {
                    name,
                    slug: key.to_slug(),
                }
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Compression".into(),
            slug: GroupKey::CompressionTimeGroup.to_slug(),
            charts,
            summary: None,
            description: None,
        }))
    }
}

fn collect_compression_size_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant
          FROM compression_sizes
         GROUP BY dataset, dataset_variant
         ORDER BY dataset, dataset_variant NULLS FIRST
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| {
            let dataset: String = row.get(0)?;
            let dataset_variant: Option<String> = row.get(1)?;
            Ok((dataset, dataset_variant))
        })?
        .map(|r| {
            r.map(|(dataset, dataset_variant)| {
                let key = ChartKey::CompressionSize {
                    dataset: dataset.clone(),
                    dataset_variant: dataset_variant.clone(),
                };
                let mut name = dataset;
                if let Some(v) = &dataset_variant {
                    name.push('/');
                    name.push_str(v);
                }
                ChartLink {
                    name,
                    slug: key.to_slug(),
                }
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Compression Size".into(),
            slug: GroupKey::CompressionSizeGroup.to_slug(),
            charts,
            summary: None,
            description: None,
        }))
    }
}

fn collect_random_access_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT DISTINCT dataset
          FROM random_access_times
         ORDER BY dataset
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .map(|r| {
            r.map(|dataset| ChartLink {
                name: dataset.clone(),
                slug: ChartKey::RandomAccess { dataset }.to_slug(),
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Random Access".into(),
            slug: GroupKey::RandomAccessGroup.to_slug(),
            charts,
            summary: None,
            description: None,
        }))
    }
}

fn collect_vector_search_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, layout, threshold
          FROM vector_search_runs
         GROUP BY dataset, layout, threshold
         ORDER BY dataset, layout, threshold
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;

    let mut groups: Vec<Group> = Vec::new();
    let mut current: Option<(String, String)> = None;
    for row in rows {
        let (dataset, layout, threshold) = row?;
        let key = (dataset.clone(), layout.clone());
        if current.as_ref() != Some(&key) {
            let group_slug = GroupKey::VectorSearchGroup {
                dataset: dataset.clone(),
                layout: layout.clone(),
            }
            .to_slug();
            groups.push(Group {
                name: format!("{dataset} / {layout}"),
                slug: group_slug,
                charts: Vec::new(),
                summary: None,
                description: None,
            });
            current = Some(key);
        }
        let slug = ChartKey::VectorSearch {
            dataset,
            layout,
            threshold,
        }
        .to_slug();
        groups
            .last_mut()
            .expect("just pushed")
            .charts
            .push(ChartLink {
                name: format!("threshold={threshold}"),
                slug,
            });
    }
    Ok(groups)
}
