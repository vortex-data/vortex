// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.google.common.collect.ImmutableMap;
import java.io.Serializable;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * The Vortex format options of a read or write, resolved case-insensitively.
 *
 * <p>Spark lower-cases the keys of the {@link CaseInsensitiveStringMap} it hands to a table, so an option a user
 * spelled {@code vortex.workerThreads} arrives as {@code vortex.workerthreads}. Holding the options here means every
 * call site gets that matching for free, and each option has one place that names it, documents its default and
 * validates it — following the shape of Spark's own {@code ParquetOptions}.
 *
 * <p>Instances cross Spark's serialization boundary to the executors, so the case-insensitive view is {@code transient}
 * and rebuilt on demand from the raw map.
 */
public final class VortexOptions implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Number of native worker threads used to decode a scan. */
    public static final String WORKER_THREADS = "vortex.workerThreads";

    /** Rows buffered before a batch is written out. */
    public static final String WRITE_BATCH_SIZE = "vortex.write.batch.size";

    /** Legacy spelling of {@link #WRITE_BATCH_SIZE}, still honoured. */
    public static final String LEGACY_WRITE_BATCH_SIZE = "batch.size";

    /** Class name of a {@link VortexSessionProvider} supplying the native session. */
    public static final String SESSION_PROVIDER = "vortex.session.provider";

    /** Default number of native worker threads. */
    public static final int DEFAULT_WORKER_THREADS = 4;

    /** Default number of rows buffered before a write batch is flushed. */
    public static final int DEFAULT_WRITE_BATCH_SIZE = 2048;

    /** Smallest accepted {@link #WRITE_BATCH_SIZE}. */
    public static final int MIN_WRITE_BATCH_SIZE = 1;

    /** Largest accepted {@link #WRITE_BATCH_SIZE}. */
    public static final int MAX_WRITE_BATCH_SIZE = 65536;

    private final ImmutableMap<String, String> options;

    private transient CaseInsensitiveStringMap cached;

    private VortexOptions(Map<String, String> options) {
        this.options = ImmutableMap.copyOf(options);
    }

    /** Wraps the supplied options; the map is copied, so later changes to it are not observed. */
    public static VortexOptions of(Map<String, String> options) {
        return new VortexOptions(Objects.requireNonNull(options, "options"));
    }

    /** Empty options, for reads and writes that configure nothing. */
    public static VortexOptions empty() {
        return new VortexOptions(ImmutableMap.of());
    }

    /**
     * Returns these options with {@code overrides} applied on top, matching keys case-insensitively so that an override
     * replaces the option it means to rather than sitting beside it under a different spelling.
     */
    public VortexOptions withOverrides(Map<String, String> overrides) {
        if (overrides.isEmpty()) {
            return this;
        }
        Set<String> overridden = new HashSet<>();
        overrides.keySet().forEach(key -> overridden.add(fold(key)));
        Map<String, String> merged = new LinkedHashMap<>();
        options.forEach((key, value) -> {
            if (!overridden.contains(fold(key))) {
                merged.put(key, value);
            }
        });
        merged.putAll(overrides);
        return new VortexOptions(merged);
    }

    /**
     * Number of native worker threads to decode with, {@value #DEFAULT_WORKER_THREADS} if unset.
     *
     * @throws IllegalArgumentException if the value is not a non-negative integer
     */
    public int workerThreads() {
        int threads = intOption(WORKER_THREADS, DEFAULT_WORKER_THREADS);
        if (threads < 0) {
            throw new IllegalArgumentException(
                    String.format("%s must be a non-negative integer, got %d", WORKER_THREADS, threads));
        }
        return threads;
    }

    /**
     * Rows to buffer before writing a batch, {@value #DEFAULT_WRITE_BATCH_SIZE} if unset. A value outside
     * [{@value #MIN_WRITE_BATCH_SIZE}, {@value #MAX_WRITE_BATCH_SIZE}] falls back to the default; use
     * {@link #rejectedWriteBatchSize()} to report which value was ignored.
     *
     * @throws IllegalArgumentException if the value is not an integer
     */
    public int writeBatchSize() {
        int configured = configuredWriteBatchSize();
        if (configured < MIN_WRITE_BATCH_SIZE || configured > MAX_WRITE_BATCH_SIZE) {
            return DEFAULT_WRITE_BATCH_SIZE;
        }
        return configured;
    }

    /**
     * The out-of-range batch size {@link #writeBatchSize()} had to ignore, together with the key it was set under, or
     * empty when the configured value was usable.
     */
    public Optional<RejectedOption> rejectedWriteBatchSize() {
        int configured = configuredWriteBatchSize();
        if (configured >= MIN_WRITE_BATCH_SIZE && configured <= MAX_WRITE_BATCH_SIZE) {
            return Optional.empty();
        }
        String key = caseInsensitive().get(WRITE_BATCH_SIZE) == null ? LEGACY_WRITE_BATCH_SIZE : WRITE_BATCH_SIZE;
        return Optional.of(new RejectedOption(key, configured));
    }

    /** An option value that was parsed but fell outside its accepted range. */
    public record RejectedOption(String key, int value) {}

    private int configuredWriteBatchSize() {
        Integer current = optionalIntOption(WRITE_BATCH_SIZE);
        if (current != null) {
            return current;
        }
        Integer legacy = optionalIntOption(LEGACY_WRITE_BATCH_SIZE);
        return legacy != null ? legacy : DEFAULT_WRITE_BATCH_SIZE;
    }

    /** Class name of the session provider to use, empty to use the default session. */
    public Optional<String> sessionProvider() {
        String provider = caseInsensitive().get(SESSION_PROVIDER);
        return provider == null || provider.isEmpty() ? Optional.empty() : Optional.of(provider);
    }

    /** The raw options, as the native bindings expect them. */
    public Map<String, String> asMap() {
        return options;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof VortexOptions && options.equals(((VortexOptions) other).options);
    }

    @Override
    public int hashCode() {
        return options.hashCode();
    }

    @Override
    public String toString() {
        return options.toString();
    }

    private int intOption(String key, int defaultValue) {
        Integer value = optionalIntOption(key);
        return value != null ? value : defaultValue;
    }

    private Integer optionalIntOption(String key) {
        String value = caseInsensitive().get(key);
        if (value == null) {
            return null;
        }
        try {
            return Integer.valueOf(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(String.format("%s must be an integer, got \"%s\"", key, value), e);
        }
    }

    private CaseInsensitiveStringMap caseInsensitive() {
        CaseInsensitiveStringMap local = cached;
        if (local == null) {
            local = new CaseInsensitiveStringMap(options);
            cached = local;
        }
        return local;
    }

    /** Folds a key the same way {@link CaseInsensitiveStringMap} does, so the two agree on what collides. */
    private static String fold(String key) {
        return key.toLowerCase(Locale.ROOT);
    }
}
