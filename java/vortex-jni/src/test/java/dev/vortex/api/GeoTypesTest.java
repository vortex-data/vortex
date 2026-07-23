// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Round-trips a Vortex geo extension column ({@code vortex.geo.wkb}) through the JNI boundary.
 * Geo columns cross the boundary as Arrow fields tagged with the GeoArrow extension name
 * ({@code geoarrow.wkb}) and JSON metadata carrying the CRS.
 */
public final class GeoTypesTest {
    private static final String EXTENSION_NAME_KEY = "ARROW:extension:name";
    private static final String EXTENSION_METADATA_KEY = "ARROW:extension:metadata";
    private static final String GEOARROW_WKB = "geoarrow.wkb";
    private static final String CRS_METADATA = "{\"crs\":\"OGC:CRS84\"}";

    @TempDir
    static Path tempDir;

    static String writePath;

    private static final List<byte[]> WKB_POINTS = new ArrayList<>();

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @BeforeAll
    static void setup() throws IOException {
        writePath = tempDir.resolve("geo.vortex").toAbsolutePath().toUri().toString();

        WKB_POINTS.add(wkbPoint(1.0, 2.0));
        WKB_POINTS.add(wkbPoint(-111.7610, 34.8697));
        WKB_POINTS.add(null);
        WKB_POINTS.add(wkbPoint(0.0, 0.0));

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = new Schema(List.of(wkbField("geom")));
        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.create(session, writePath, schema, new HashMap<>(), allocator);
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            VarBinaryVector geomVec = (VarBinaryVector) root.getVector("geom");
            geomVec.allocateNew(WKB_POINTS.size());
            for (int i = 0; i < WKB_POINTS.size(); i++) {
                byte[] wkb = WKB_POINTS.get(i);
                if (wkb == null) {
                    geomVec.setNull(i);
                } else {
                    geomVec.setSafe(i, wkb);
                }
            }
            root.setRowCount(WKB_POINTS.size());

            try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);
                writer.writeBatch(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
            }
        }
    }

    @Test
    public void testSchemaCarriesGeoArrowExtension() {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);

        Schema schema = ds.arrowSchema(allocator);
        Field geom = schema.findField("geom");
        assertEquals(GEOARROW_WKB, geom.getMetadata().get(EXTENSION_NAME_KEY));
        assertEquals(CRS_METADATA, geom.getMetadata().get(EXTENSION_METADATA_KEY));
        assertTrue(geom.isNullable());
    }

    @Test
    public void testScanSchemaCarriesGeoArrowExtension() {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);

        Schema schema = ds.scan(ScanOptions.of()).arrowSchema(allocator);
        Field geom = schema.findField("geom");
        assertEquals(GEOARROW_WKB, geom.getMetadata().get(EXTENSION_NAME_KEY));
        assertEquals(CRS_METADATA, geom.getMetadata().get(EXTENSION_METADATA_KEY));
    }

    @Test
    public void testWkbValuesRoundTrip() throws Exception {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);

        List<byte[]> values = new ArrayList<>();
        Scan scan = ds.scan(ScanOptions.of());
        while (scan.hasNext()) {
            Partition partition = scan.next();
            try (ArrowReader arrowReader = partition.scanArrow(allocator)) {
                while (arrowReader.loadNextBatch()) {
                    VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
                    var geomVec = root.getVector("geom");
                    for (int i = 0; i < root.getRowCount(); i++) {
                        values.add(geomVec.isNull(i) ? null : (byte[]) geomVec.getObject(i));
                    }
                }
            }
        }

        assertEquals(WKB_POINTS.size(), values.size());
        for (int i = 0; i < WKB_POINTS.size(); i++) {
            byte[] expected = WKB_POINTS.get(i);
            if (expected == null) {
                assertNull(values.get(i));
            } else {
                assertArrayEquals(expected, values.get(i));
            }
        }
    }

    private static Field wkbField(String name) {
        Map<String, String> metadata =
                Map.of(EXTENSION_NAME_KEY, GEOARROW_WKB, EXTENSION_METADATA_KEY, CRS_METADATA);
        return new Field(name, new FieldType(true, ArrowType.Binary.INSTANCE, null, metadata), null);
    }

    /** Little-endian WKB encoding of {@code POINT(x y)}. */
    private static byte[] wkbPoint(double x, double y) {
        ByteBuffer buffer = ByteBuffer.allocate(21).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put((byte) 1); // little-endian marker
        buffer.putInt(1); // geometry type: point
        buffer.putDouble(x);
        buffer.putDouble(y);
        return buffer.array();
    }
}
