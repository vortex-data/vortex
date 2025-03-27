package dev.vortex;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.api.File;
import dev.vortex.api.Files;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.net.URI;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(value = Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Open
public class BenchFile {
    static final URI FILE = URI.create("s3a://vortex-iceberg-dev/warehouse/db/trips/data/202409-citibike-tripdata_2.vortex");
    static final String AWS_ACCESS_KEY = System.getenv("AWS_ACCESS_KEY");
    static final String AWS_SECRET_KEY = System.getenv("AWS_SECRET_KEY");
    static final Map<String, String> PROPS = Map.of(
            "aws_access_key_id", AWS_ACCESS_KEY,
            "aws_secret_access_key", AWS_SECRET_KEY
    );

    File opened;


    @Benchmark
    public void open() {
        opened = Files.open(FILE, PROPS);
    }

    @TearDown(Level.Invocation)
    public void tearDown() {
        if (opened != null) {
            opened.close();
        }
    }
}
