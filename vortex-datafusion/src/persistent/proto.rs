// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization of Vortex scans in DataFusion physical plans.
//!
//! DataFusion's protobuf schema has no node for Vortex scans, so a [`VortexSource`] encodes
//! itself through [`FileSource::try_to_proto`] as an extension node, and
//! [`VortexPhysicalExtensionCodec`] decodes it. Distributed engines that ship physical plans
//! between processes, such as Ballista or datafusion-distributed, pass the codec wherever they
//! decode plans.
//!
//! The node carries everything that shapes the scan's output: DataFusion's shared file scan
//! configuration (files, schema, projection, ordering, statistics and limit), the pushed
//! predicates, the read order chosen by sort pushdown, and the table options. Per-process state
//! is rebuilt on decode instead: the codec's [`VortexSession`], the task's file metadata cache,
//! fresh metrics, the default reader factory and the default expression convertor. A source with
//! a custom [`ExpressionConvertor`] or [`VortexReaderFactory`] decodes with the defaults.
//!
//! [`FileSource::try_to_proto`]: datafusion_datasource::file::FileSource::try_to_proto
//! [`ExpressionConvertor`]: crate::convert::ExpressionConvertor
//! [`VortexReaderFactory`]: crate::reader::VortexReaderFactory

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::config::ExtensionOptions;
use datafusion_common::internal_datafusion_err;
use datafusion_common::not_impl_err;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr_common::sort_expr::sort_exprs_try_to_proto;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::proto::ExecutionPlanEncodeCtx;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_proto::physical_plan::PhysicalPlanDecodeContext;
use datafusion_proto::physical_plan::PhysicalProtoConverterExtension;
use datafusion_proto::physical_plan::from_proto::parse_physical_expr_with_converter;
use datafusion_proto::physical_plan::from_proto::parse_physical_sort_exprs;
use datafusion_proto::physical_plan::from_proto::parse_protobuf_file_scan_config;
use datafusion_proto_models::protobuf;
use datafusion_proto_models::protobuf::physical_plan_node::PhysicalPlanType;
use prost::Message;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;

use crate::VortexSource;
use crate::VortexTableOptions;
use crate::persistent::sort::ReadOrder;

/// Prefixes the payload of every Vortex scan extension node, so that a codec composed with others
/// can tell its nodes apart.
const VORTEX_SCAN_TAG: &[u8] = b"vortex.scan.v1:";

/// The extension node payload for a Vortex scan.
#[derive(Clone, PartialEq, Message)]
struct VortexScanNode {
    #[prost(message, optional, tag = "1")]
    base_conf: Option<protobuf::FileScanExecConf>,
    /// The predicate used to prune files, see `VortexSource::full_predicate`.
    #[prost(message, optional, tag = "2")]
    full_predicate: Option<protobuf::PhysicalExprNode>,
    /// The predicate evaluated by the scan, see `VortexSource::vortex_predicate`.
    #[prost(message, optional, tag = "3")]
    vortex_predicate: Option<protobuf::PhysicalExprNode>,
    #[prost(bool, tag = "4")]
    ordered: bool,
    /// The sort order chosen by an inexact sort pushdown, empty without one.
    #[prost(message, repeated, tag = "5")]
    read_order: Vec<protobuf::PhysicalSortExprNode>,
    #[prost(bool, tag = "6")]
    reverse_splits: bool,
    /// The table options that are set, by name.
    #[prost(btree_map = "string, string", tag = "7")]
    options: BTreeMap<String, String>,
}

impl VortexSource {
    /// Encodes this source and its scan configuration as a DataFusion extension plan node.
    pub(crate) fn scan_to_proto(
        &self,
        base: &FileScanConfig,
        ctx: &ExecutionPlanEncodeCtx<'_>,
    ) -> DFResult<protobuf::PhysicalPlanNode> {
        let node = VortexScanNode {
            base_conf: Some(base.try_to_proto(ctx)?),
            full_predicate: self
                .full_predicate
                .as_ref()
                .map(|expr| ctx.encode_expr(expr))
                .transpose()?,
            vortex_predicate: self
                .vortex_predicate
                .as_ref()
                .map(|expr| ctx.encode_expr(expr))
                .transpose()?,
            ordered: self.ordered,
            read_order: self
                .read_order
                .as_ref()
                .map(|order| sort_exprs_try_to_proto(order.sort_order.iter(), &ctx.expr_ctx()))
                .transpose()?
                .unwrap_or_default(),
            reverse_splits: self
                .read_order
                .as_ref()
                .is_some_and(|order| order.reverse_splits),
            options: self
                .options()
                .entries()
                .into_iter()
                .filter_map(|entry| Some((entry.key, entry.value?)))
                .collect(),
        };
        let mut buf = VORTEX_SCAN_TAG.to_vec();
        node.encode(&mut buf)
            .map_err(|e| internal_datafusion_err!("Failed to encode Vortex scan: {e}"))?;
        Ok(protobuf::PhysicalPlanNode {
            physical_plan_type: Some(PhysicalPlanType::Extension(
                protobuf::PhysicalExtensionNode {
                    node: buf,
                    inputs: vec![],
                },
            )),
        })
    }
}

/// Decodes the Vortex scans in DataFusion physical plans serialized with `datafusion-proto`.
///
/// Pass it wherever plans are decoded, for example to
/// `datafusion_proto::bytes::physical_plan_from_bytes_with_extension_codec`. Vortex scans encode
/// themselves, so the codec is not needed to encode plans. To decode other extension nodes too,
/// compose it with their codecs; expressions nested in a Vortex scan are decoded with this codec,
/// so functions they reference are resolved by name.
pub struct VortexPhysicalExtensionCodec {
    session: VortexSession,
}

impl VortexPhysicalExtensionCodec {
    /// Creates a codec that decodes scans reading through `session`.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }
}

impl Default for VortexPhysicalExtensionCodec {
    fn default() -> Self {
        Self::new(VortexSession::default())
    }
}

impl Debug for VortexPhysicalExtensionCodec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexPhysicalExtensionCodec")
            .finish_non_exhaustive()
    }
}

impl PhysicalExtensionCodec for VortexPhysicalExtensionCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let Some(payload) = buf.strip_prefix(VORTEX_SCAN_TAG) else {
            return not_impl_err!("VortexPhysicalExtensionCodec only decodes Vortex scans");
        };
        if !inputs.is_empty() {
            return Err(internal_datafusion_err!(
                "A Vortex scan has no inputs, got {}",
                inputs.len()
            ));
        }
        let node = VortexScanNode::decode(payload)
            .map_err(|e| internal_datafusion_err!("Failed to decode Vortex scan: {e}"))?;
        let base_conf = node
            .base_conf
            .as_ref()
            .ok_or_else(|| internal_datafusion_err!("Vortex scan is missing its base config"))?;

        let decode_ctx = PhysicalPlanDecodeContext::new(ctx, self);
        let table_schema = FileScanConfig::parse_table_schema_from_proto(base_conf)?;
        let schema = Arc::clone(table_schema.table_schema());
        let decode_expr = |expr: &protobuf::PhysicalExprNode| {
            parse_physical_expr_with_converter(expr, &schema, &decode_ctx, proto_converter)
        };

        let mut options = VortexTableOptions::default();
        for (key, value) in &node.options {
            options.set(key, value)?;
        }
        let mut source = VortexSource::new(table_schema, self.session.clone())
            .with_options(options)
            .with_file_metadata_cache(ctx.runtime_env().cache_manager.get_file_metadata_cache());
        source.full_predicate = node.full_predicate.as_ref().map(decode_expr).transpose()?;
        source.vortex_predicate = node
            .vortex_predicate
            .as_ref()
            .map(decode_expr)
            .transpose()?;
        source.ordered = node.ordered;
        source.read_order = LexOrdering::new(parse_physical_sort_exprs(
            &node.read_order,
            &decode_ctx,
            &schema,
            proto_converter,
        )?)
        .map(|sort_order| ReadOrder {
            sort_order,
            reverse_splits: node.reverse_splits,
        });

        let config = parse_protobuf_file_scan_config(
            base_conf,
            &decode_ctx,
            proto_converter,
            Arc::new(source),
        )?;
        Ok(DataSourceExec::from_data_source(config))
    }

    fn try_encode(
        &self,
        _node: Arc<dyn ExecutionPlan>,
        _buf: &mut Vec<u8>,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> DFResult<()> {
        not_impl_err!("Vortex scans encode themselves through `FileSource::try_to_proto`")
    }
}
