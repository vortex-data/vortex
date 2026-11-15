// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.google.common.collect.ImmutableMap;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * The format options of one Spark read or write, resolved without regard to key case.
 *
 * <p>Spark treats data source options as case-insensitive, but the connector is reached through maps that disagree: a
 * V2 scan carries the keys as the user typed them, while the V1 file format lower-cases them first. Every
 * {@code vortex.*} lookup goes through here so both paths answer the same.
 *
 * <p>The keys as given are kept too. Hadoop configuration keys are case-sensitive, so {@link #asCaseSensitiveMap()} is
 * what feeds a Hadoop {@code Configuration}.
 *
 * <p>Built on the driver and shipped to executors inside the reader and writer factories.
 */
public final class VortexOptions implements Serializable {
    private static final long serialVersionUID = 1L;

    private final ImmutableMap<String, String> original;
    private final ImmutableMap<String, String> byLowerCasedKey;

    /** Captures {@code options} as given. A null map is read as no options at all. */
    public static VortexOptions of(Map<String, String> options) {
        ImmutableMap<String, String> original = options == null ? ImmutableMap.of() : ImmutableMap.copyOf(options);

        // A map arriving from Spark may already hold keys that differ only in case. Later entries win, as they
        // do in Spark's own CaseInsensitiveStringMap, and `original` still carries both.
        Map<String, String> lowerCased = new HashMap<>(original.size());
        original.forEach((key, value) -> lowerCased.put(key.toLowerCase(Locale.ROOT), value));
        return new VortexOptions(original, ImmutableMap.copyOf(lowerCased));
    }

    /** No options at all, for callers with no Spark read or write to draw them from. */
    public static VortexOptions empty() {
        return of(ImmutableMap.of());
    }

    private VortexOptions(ImmutableMap<String, String> original, ImmutableMap<String, String> byLowerCasedKey) {
        this.original = original;
        this.byLowerCasedKey = byLowerCasedKey;
    }

    /** The options with the keys as given, for the case-sensitive world of Hadoop configuration. */
    public Map<String, String> asCaseSensitiveMap() {
        return original;
    }

    /** The value set for {@code key} under any casing. */
    public Optional<String> get(String key) {
        Objects.requireNonNull(key, "key");
        return Optional.ofNullable(byLowerCasedKey.get(key.toLowerCase(Locale.ROOT)));
    }

    /** The value set for {@code key} under any casing, or {@code fallback} when it is not set. */
    public String get(String key, String fallback) {
        return get(key).orElse(fallback);
    }

    /**
     * The value set for {@code key} as an integer, or {@code fallback} when it is not set.
     *
     * @throws IllegalArgumentException if the value is not an integer
     */
    public int getInt(String key, int fallback) {
        Optional<String> value = get(key);
        if (value.isEmpty()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.get().trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "%s must be an integer, got '%s'", key, value.get()), e);
        }
    }

    /**
     * The value set for {@code key} as a boolean, or {@code fallback} when it is not set.
     *
     * @throws IllegalArgumentException if the value is neither {@code true} nor {@code false}
     */
    public boolean getBoolean(String key, boolean fallback) {
        Optional<String> value = get(key);
        if (value.isEmpty()) {
            return fallback;
        }
        String trimmed = value.get().trim();
        if (trimmed.equalsIgnoreCase("true")) {
            return true;
        }
        if (trimmed.equalsIgnoreCase("false")) {
            return false;
        }
        throw new IllegalArgumentException(
                String.format(Locale.ROOT, "%s must be true or false, got '%s'", key, value.get()));
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof VortexOptions options && original.equals(options.original);
    }

    @Override
    public int hashCode() {
        return original.hashCode();
    }

    @Override
    public String toString() {
        return original.toString();
    }
}
