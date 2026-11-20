// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexSparkSessionTest {
    private static final AtomicInteger providerInitializations = new AtomicInteger();
    private static final AtomicInteger providerConstructions = new AtomicInteger();

    @ParameterizedTest
    @ValueSource(strings = {"many", "-1"})
    void everyWorkerThreadCountIsCheckedAfterThePoolIsSized(String value) {
        VortexSparkSession.get(VortexOptions.empty());

        VortexOptions options = VortexOptions.of(Map.of(VortexSparkSession.WORKER_THREADS_OPTION, value));
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> VortexSparkSession.get(options));
        assertTrue(failure.getMessage().contains(VortexSparkSession.WORKER_THREADS_OPTION), failure::getMessage);
    }

    @Test
    void validatingAProviderNeitherInitializesNorConstructsIt() {
        VortexOptions options =
                VortexOptions.of(Map.of(VortexSparkSession.PROVIDER_OPTION, CountingProvider.class.getName()));

        VortexSparkSession.validate(options);
        assertEquals(0, providerInitializations.get());
        assertEquals(0, providerConstructions.get());

        VortexSparkSession.get(options);
        assertEquals(1, providerInitializations.get());
        assertEquals(1, providerConstructions.get());
    }

    @ParameterizedTest
    @ValueSource(classes = {String.class, AbstractProvider.class, PrivateConstructorProvider.class})
    void anUnusableProviderClassIsRejected(Class<?> provider) {
        VortexOptions options = VortexOptions.of(Map.of(VortexSparkSession.PROVIDER_OPTION, provider.getName()));

        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> VortexSparkSession.validate(options));
        assertTrue(failure.getMessage().contains(provider.getName()), failure::getMessage);
    }

    @Test
    void aProviderTheDriverCannotLoadIsLeftToTheExecutors() {
        // A class added with SparkContext.addJar reaches the executors' classpath but not the driver's.
        VortexSparkSession.validate(
                VortexOptions.of(Map.of(VortexSparkSession.PROVIDER_OPTION, "dev.vortex.spark.NoSuchProvider")));
    }

    @Test
    void aProviderTheSessionCanConstructIsAccepted() {
        VortexOptions options =
                VortexOptions.of(Map.of(VortexSparkSession.PROVIDER_OPTION, PackagePrivateProvider.class.getName()));

        VortexSparkSession.validate(options);
        VortexSparkSession.get(options);
    }

    public static final class CountingProvider implements VortexSessionProvider {
        static {
            providerInitializations.incrementAndGet();
        }

        public CountingProvider() {
            providerConstructions.incrementAndGet();
        }

        @Override
        public Session get() {
            return VortexSparkSession.get();
        }
    }

    static final class PackagePrivateProvider implements VortexSessionProvider {
        @Override
        public Session get() {
            return VortexSparkSession.get();
        }
    }

    public abstract static class AbstractProvider implements VortexSessionProvider {}

    public static final class PrivateConstructorProvider implements VortexSessionProvider {
        private PrivateConstructorProvider() {}

        @Override
        public Session get() {
            return VortexSparkSession.get();
        }
    }
}
