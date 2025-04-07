/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.spark;

import com.google.common.collect.ImmutableMap;
import dev.vortex.jni.NativeLogging;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

final class VortexScanTest {
    private static final Path TPCH_VORTEX =
            Paths.get("/Volumes/Code/vortex/bench-vortex/data/tpch/1/vortex_compressed");
    private static final Path TPCH_PARQUET = Paths.get("/Volumes/Code/vortex/bench-vortex/data/tpch/1/parquet");

    static {
        NativeLogging.initLogging(NativeLogging.ERROR);
    }

    private void registerTables(SparkSession spark, Path root, String format) {
        var tables = ImmutableMap.of(
                "lineitem", "lineitem." + format,
                "part", "part." + format,
                "supplier", "supplier." + format,
                "partsupp", "partsupp." + format,
                "customer", "customer." + format,
                "orders", "orders." + format,
                "nation", "nation." + format,
                "region", "region." + format);

        for (var entry : tables.entrySet()) {
            var tableName = entry.getKey();
            var fileName = entry.getValue();
            var filePath = root.resolve(fileName).toAbsolutePath().toString();
            var table = spark.read().format(format).load(filePath);
            table.createOrReplaceTempView(tableName);
        }
    }

    @Test
    public void testSparkRead() {
        SparkSession spark =
                SparkSession.builder().appName("test").master("local").getOrCreate();

        registerTables(spark, TPCH_VORTEX, "vortex");

        spark.sql("select l_shipdate, count(*) as orders from lineitem group by l_shipdate ORDER BY orders")
                .show(20);

        // Execute the TPC-H queries
        var q1 = "select\n" + "    l_returnflag,\n"
                + "    l_linestatus,\n"
                + "    sum(l_quantity) as sum_qty,\n"
                + "    sum(l_extendedprice) as sum_base_price,\n"
                + "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,\n"
                + "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,\n"
                + "    avg(l_quantity) as avg_qty,\n"
                + "    avg(l_extendedprice) as avg_price,\n"
                + "    avg(l_discount) as avg_disc,\n"
                + "    count(*) as count_order\n"
                + "from\n"
                + "    lineitem\n"
                + "where\n"
                + "        l_shipdate <= date '1998-09-02'\n"
                + "group by\n"
                + "    l_returnflag,\n"
                + "    l_linestatus\n"
                + "order by\n"
                + "    l_returnflag,\n"
                + "    l_linestatus\n";

        var plan = spark.sql(q1);

        long start = System.nanoTime();
        var results = plan.collectAsList();
        long duration = System.nanoTime() - start;
        // plan.queryExecution().debug().codegen();
        System.out.println("Q1 (" + duration + " nanos) results: " + results);
    }

    @Test
    public void testParquetRead() {
        SparkSession spark =
                SparkSession.builder().appName("test").master("local").getOrCreate();

        registerTables(spark, TPCH_PARQUET, "parquet");

        spark.sql("select l_shipdate, count(*) as orders from lineitem group by l_shipdate ORDER BY orders")
                .show(20);

        // Execute the TPC-H queries
        var q1 = "select\n" + "    l_returnflag,\n"
                + "    l_linestatus,\n"
                + "    sum(l_quantity) as sum_qty,\n"
                + "    sum(l_extendedprice) as sum_base_price,\n"
                + "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,\n"
                + "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,\n"
                + "    avg(l_quantity) as avg_qty,\n"
                + "    avg(l_extendedprice) as avg_price,\n"
                + "    avg(l_discount) as avg_disc,\n"
                + "    count(*) as count_order\n"
                + "from\n"
                + "    lineitem\n"
                + "where\n"
                + "        l_shipdate <= date '1998-09-02'\n"
                + "group by\n"
                + "    l_returnflag,\n"
                + "    l_linestatus\n"
                + "order by\n"
                + "    l_returnflag,\n"
                + "    l_linestatus\n";

        var plan = spark.sql(q1);

        long start = System.nanoTime();
        var results = plan.collectAsList();
        long duration = System.nanoTime() - start;
        System.out.println("Q1 (" + duration + " nanos) results: " + results);
    }
}
