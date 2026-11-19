// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental DataFusion functions over `vortex.protobuf` columns, without compatibility
//! guarantees.
//!
//! DataFusion sees a protobuf column as its storage struct. Its Arrow field names the extension
//! type `vortex.protobuf` and holds the message schema in its extension metadata, so each function
//! reads the schema from the field of its message argument:
//!
//! - `protobuf_enum_name(msg, 'path.to.enum_field')` returns the name of the value of a singular
//!   enum field as `Utf8`. It is null where the field or a message on the path is null, and where
//!   the field holds a number that the enum does not define.
//! - `protobuf_oneof_case(msg, 'path.to.oneof')` returns the name of the member of a oneof that
//!   is set as `Utf8`, or null where no member is set or a message on the path is null.
//! - `protobuf_encode(msg)` returns the protobuf wire format of each message as `Binary`, or null
//!   for a null message.
//!
//! A path is a string literal of dotted field names, as in
//! [`protobuf_field`](vortex_protobuf::protobuf_field), and planning a call fails if the path
//! does not name a field of the right kind. Each call is the Vortex expression that
//! [`vortex_protobuf`] builds for it. The [`DefaultExpressionConverter`] converts a call to that
//! expression, so a Vortex scan evaluates it and reads only the message fields that it uses. When
//! DataFusion evaluates a call itself, such as over another data source, the function imports the
//! messages from Arrow and evaluates the same expression, so the results are the same.
//!
//! Register the functions with [`register_functions`]:
//!
//! ```
//! use datafusion::prelude::SessionContext;
//!
//! let mut ctx = SessionContext::new();
//! vortex_datafusion::protobuf::register_functions(&mut ctx)?;
//! # Ok::<(), datafusion_common::DataFusionError>(())
//! ```
//!
//! [`DefaultExpressionConverter`]: crate::convert::DefaultExpressionConverter

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_err;
use datafusion_common::internal_err;
use datafusion_common::plan_datafusion_err;
use datafusion_common::plan_err;
use datafusion_expr::ColumnarValue;
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::ScalarFunctionArgs;
use datafusion_expr::ScalarUDF;
use datafusion_expr::ScalarUDFImpl;
use datafusion_expr::Signature;
use datafusion_expr::Volatility;
use datafusion_expr::registry::FunctionRegistry;
use itertools::Itertools;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexError;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_protobuf::Protobuf;
use vortex_protobuf::ProtobufMetadata;
use vortex_protobuf::protobuf_encode;
use vortex_protobuf::protobuf_enum_name;
use vortex_protobuf::protobuf_oneof_case;

/// The session that imports messages from Arrow and evaluates the functions in DataFusion.
///
/// The storage of a protobuf column holds only built-in dtypes and the dtypes that
/// [`vortex_protobuf::initialize`] registers, so this session imports every protobuf column.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default();
    vortex_protobuf::initialize(&session);
    session
});

/// Returns the protobuf functions.
pub fn functions() -> Vec<Arc<ScalarUDF>> {
    [Operation::EnumName, Operation::OneofCase, Operation::Encode]
        .into_iter()
        .map(|operation| Arc::new(ScalarUDF::from(ProtobufFunction::new(operation))))
        .collect()
}

/// Registers the protobuf functions with `registry`, such as a
/// [`SessionContext`](https://docs.rs/datafusion/latest/datafusion/prelude/struct.SessionContext.html),
/// replacing functions of the same names.
///
/// # Errors
///
/// Returns an error if the registry rejects a function.
pub fn register_functions(registry: &mut dyn FunctionRegistry) -> DFResult<()> {
    for function in functions() {
        registry.register_udf(function)?;
    }
    Ok(())
}

/// What a protobuf function computes from messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Operation {
    /// `protobuf_enum_name(msg, path)`, with [`protobuf_enum_name`].
    EnumName,
    /// `protobuf_oneof_case(msg, path)`, with [`protobuf_oneof_case`].
    OneofCase,
    /// `protobuf_encode(msg)`, with [`protobuf_encode`].
    Encode,
}

/// A DataFusion function over protobuf messages, which evaluates a Vortex expression.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct ProtobufFunction {
    /// What the function computes.
    operation: Operation,
    /// The arguments: the messages, then the literal path if the operation has one.
    signature: Signature,
}

impl ProtobufFunction {
    /// Creates the function that computes `operation`.
    fn new(operation: Operation) -> Self {
        let arity = match operation {
            Operation::EnumName | Operation::OneofCase => 2,
            Operation::Encode => 1,
        };
        Self {
            operation,
            signature: Signature::any(arity, Volatility::Immutable),
        }
    }

    /// Returns the Vortex expression that a call evaluates over `message`, whose dtype has
    /// `metadata`.
    ///
    /// `literals` holds the value of each argument after the messages, or `None` for an argument
    /// that is not a literal.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is not a string literal, or if it does not name a field of
    /// the kind that the function reads.
    pub(crate) fn expression(
        &self,
        metadata: &ProtobufMetadata,
        literals: &[Option<&ScalarValue>],
        message: Expression,
    ) -> DFResult<Expression> {
        let expr = match self.operation {
            Operation::EnumName => protobuf_enum_name(metadata, self.path(literals)?, message),
            Operation::OneofCase => protobuf_oneof_case(metadata, self.path(literals)?, message),
            Operation::Encode => Ok(protobuf_encode(message)),
        };
        expr.map_err(|err| plan_datafusion_err!("{}: {err}", self.name()))
    }

    /// Returns the path argument from the literal arguments after the messages.
    fn path<'a>(&self, literals: &[Option<&'a ScalarValue>]) -> DFResult<&'a str> {
        match literals {
            [Some(path)] => path
                .try_as_str()
                .flatten()
                .ok_or_else(|| plan_datafusion_err!("{} requires a string path", self.name())),
            _ => plan_err!("{} requires a string literal path", self.name()),
        }
    }

    /// Returns the protobuf metadata of the messages that `field` describes.
    fn message_metadata(&self, field: &Field) -> DFResult<ProtobufMetadata> {
        let dtype = SESSION.arrow().from_arrow_field(field).map_err(external)?;
        dtype
            .as_extension_opt()
            .and_then(|ext_dtype| ext_dtype.metadata_opt::<Protobuf>())
            .cloned()
            .ok_or_else(|| {
                plan_datafusion_err!(
                    "{} requires {} messages, got {}",
                    self.name(),
                    Protobuf::ID,
                    field.data_type()
                )
            })
    }
}

impl ScalarUDFImpl for ProtobufFunction {
    fn name(&self) -> &str {
        match self.operation {
            Operation::EnumName => "protobuf_enum_name",
            Operation::OneofCase => "protobuf_oneof_case",
            Operation::Encode => "protobuf_encode",
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        internal_err!(
            "{} computes its return field from its arguments",
            self.name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        let (Some(message), Some((_, literals))) =
            (args.arg_fields.first(), args.scalar_arguments.split_first())
        else {
            return plan_err!("{} requires messages", self.name());
        };
        // Building the expression checks the path.
        self.expression(&self.message_metadata(message)?, literals, root())?;
        let field = match self.operation {
            Operation::EnumName | Operation::OneofCase => {
                Field::new(self.name(), DataType::Utf8, true)
            }
            Operation::Encode => Field::new(self.name(), DataType::Binary, message.is_nullable()),
        };
        Ok(Arc::new(field))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let (Some((messages, literals)), Some(field)) =
            (args.args.split_first(), args.arg_fields.first())
        else {
            return exec_err!("{} requires messages", self.name());
        };
        let literals = literals
            .iter()
            .map(|literal| match literal {
                ColumnarValue::Scalar(value) => Some(value),
                ColumnarValue::Array(_) => None,
            })
            .collect_vec();
        let expr = self.expression(&self.message_metadata(field)?, &literals, root())?;

        let (messages, scalar) = match messages {
            ColumnarValue::Array(array) => (Arc::clone(array), false),
            ColumnarValue::Scalar(value) => (value.to_array()?, true),
        };
        let mut ctx = SESSION.create_execution_ctx();
        let result = SESSION
            .arrow()
            .from_arrow_array(messages, field)
            .and_then(|messages| messages.apply(&expr))
            .and_then(|result| result.execute_arrow(Some(&args.return_field), &mut ctx))
            .map_err(external)?;
        Ok(if scalar {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&result, 0)?)
        } else {
            ColumnarValue::Array(result)
        })
    }
}

/// Wraps a Vortex error for DataFusion.
fn external(err: VortexError) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}
