package dev.vortex.api;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Objects;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import javax.annotation.ParametersAreNonnullByDefault;
import javax.annotation.concurrent.Immutable;
import javax.annotation.concurrent.NotThreadSafe;
import org.immutables.value.Generated;

/**
 * Immutable implementation of {@link ScanOptions}.
 * <p>
 * Use the builder to create immutable instances:
 * {@code ImmutableScanOptions.builder()}.
 */
@Generated(from = "ScanOptions", generator = "Immutables")
@SuppressWarnings({"all"})
@ParametersAreNonnullByDefault
@javax.annotation.processing.Generated("org.immutables.processor.ProxyProcessor")
@Immutable
@CheckReturnValue
public final class ImmutableScanOptions implements ScanOptions {

  private ImmutableScanOptions(ImmutableScanOptions.Builder builder) {
  }

  /**
   * This instance is equal to all instances of {@code ImmutableScanOptions} that have equal attribute values.
   * @return {@code true} if {@code this} is equal to {@code another} instance
   */
  @Override
  public boolean equals(@Nullable Object another) {
    if (this == another) return true;
    return another instanceof ImmutableScanOptions
        && equalTo(0, (ImmutableScanOptions) another);
  }

  @SuppressWarnings("MethodCanBeStatic")
  private boolean equalTo(int synthetic, ImmutableScanOptions another) {
    return true;
  }

  /**
   * Returns a constant hash code value.
   * @return hashCode value
   */
  @Override
  public int hashCode() {
    return -855334843;
  }

  /**
   * Prints the immutable value {@code ScanOptions}.
   * @return A string representation of the value
   */
  @Override
  public String toString() {
    return "ScanOptions{}";
  }

  /**
   * Creates an immutable copy of a {@link ScanOptions} value.
   * Uses accessors to get values to initialize the new immutable instance.
   * If an instance is already immutable, it is returned as is.
   * @param instance The instance to copy
   * @return A copied immutable ScanOptions instance
   */
  public static ImmutableScanOptions copyOf(ScanOptions instance) {
    if (instance instanceof ImmutableScanOptions) {
      return (ImmutableScanOptions) instance;
    }
    return ImmutableScanOptions.builder()
        .from(instance)
        .build();
  }

  /**
   * Creates a builder for {@link ImmutableScanOptions ImmutableScanOptions}.
   * <pre>
   * ImmutableScanOptions.builder()
   *    .build();
   * </pre>
   * @return A new ImmutableScanOptions builder
   */
  public static ImmutableScanOptions.Builder builder() {
    return new ImmutableScanOptions.Builder();
  }

  /**
   * Builds instances of type {@link ImmutableScanOptions ImmutableScanOptions}.
   * Initialize attributes and then invoke the {@link #build()} method to create an
   * immutable instance.
   * <p><em>{@code Builder} is not thread-safe and generally should not be stored in a field or collection,
   * but instead used immediately to create instances.</em>
   */
  @Generated(from = "ScanOptions", generator = "Immutables")
  @NotThreadSafe
  public static final class Builder {

    private Builder() {
    }

    /**
     * Fill a builder with attribute values from the provided {@code ScanOptions} instance.
     * Regular attribute values will be replaced with those from the given instance.
     * Absent optional values will not replace present values.
     * @param instance The instance from which to copy values
     * @return {@code this} builder for use in a chained invocation
     */
    @CanIgnoreReturnValue 
    public final Builder from(ScanOptions instance) {
      Objects.requireNonNull(instance, "instance");
      return this;
    }

    /**
     * Builds a new {@link ImmutableScanOptions ImmutableScanOptions}.
     * @return An immutable instance of ScanOptions
     * @throws java.lang.IllegalStateException if any required attributes are missing
     */
    public ImmutableScanOptions build() {
      return new ImmutableScanOptions(this);
    }
  }
}
