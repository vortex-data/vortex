/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import static io.trino.spi.function.table.ReturnTypeSpecification.GenericTable.GENERIC_TABLE;

import com.google.common.base.Preconditions;
import dev.vortex.api.DType;
import dev.vortex.api.File;
import dev.vortex.api.Files;
import io.airlift.slice.Slice;
import io.trino.spi.connector.ConnectorAccessControl;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.*;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Table function for reading Vortex files.
 *
 * <p>Usage in Trino SQL:
 * <pre>
 * SELECT * FROM TABLE(vortex.system.read_vortex(uri => 'file:///path/to/file.vortex'))
 * </pre>
 */
public final class VortexTableFunction extends AbstractConnectorTableFunction {

    private static final String FUNCTION_NAME = "read_vortex";
    private static final String SCHEMA_NAME = "system";

    public VortexTableFunction() {
        super(
                SCHEMA_NAME,
                FUNCTION_NAME,
                List.of(
                        TableArgumentSpecification.builder()
                                        .rowSemantics()
                        ScalarArgumentSpecification.builder()
                        .name("URI")
                        .type(VarcharType.VARCHAR)
                        .build()),
                GENERIC_TABLE);
    }

    @Override
    public TableFunctionAnalysis analyze(
            ConnectorSession session,
            ConnectorTransactionHandle transaction,
            Map<String, Argument> arguments,
            ConnectorAccessControl accessControl) {
        // Extract the URI argument
        ScalarArgument uriArg = (ScalarArgument) arguments.get("URI");
        String uri = ((Slice) Preconditions.checkNotNull(uriArg.getValue())).toStringUtf8();

        // Open the file to read its schema
        File vxf = Files.open(uri);
        DType dtype = vxf.getDType();

        if (dtype.getVariant() != DType.Variant.STRUCT) {
            throw new IllegalArgumentException("Vortex file must have a struct schema at the top level");
        }

        List<String> fieldNames = dtype.getFieldNames();
        List<DType> fieldTypes = dtype.getFieldTypes();

        // Build the descriptor with column information
        List<Type> trinoTypes = new ArrayList<>();
        for (int i = 0; i < fieldNames.size(); i++) {
            trinoTypes.add(VortexTypeConverter.toTrinoType(fieldTypes.get(i)));
        }

        Descriptor descriptor = Descriptor.descriptor(fieldNames,
                trinoTypes);

        return TableFunctionAnalysis.builder()
                .returnedType(descriptor)
                .handle(new VortexTableFunctionHandle(vxf))
                .build();
    }
}
