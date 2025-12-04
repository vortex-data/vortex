// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration object for the benchmark chart. Modify these values to
 * customize the chart for different benchmarks.
 */
const CHART_CONFIG = {
    // Data sources.
    wasmModulePath: "./pkg/vortex_wasm.js",

    // GitHub repository for constructing commit URLs.
    githubRepo: "https://github.com/spiraldb/vortex",

    // Which group and chart to display (for now, hardcoded to random-access).
    targetGroup: "random-access",
    targetChart: "random-access",

    // Series configuration.
    seriesNames: ["vortex-nvme", "parquet-nvme", "lance-nvme"],
    seriesColors: {
        "vortex-nvme": "#101010",
        "parquet-nvme": "#5DADE2",
        "lance-nvme": "#ef7f1d",
    },

    // Chart display settings.
    defaultVisibleCommits: 50,

    // Zoom and scroll configuration.
    minWindowSize: 10,
    zoomSpeed: 0.1,

    // Y-axis label.
    yAxisLabel: "Time (ms)",
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Updates the status display with a message and type.
 *
 * @param {string} message - The status message to display.
 * @param {string} type - The status type: "loading", "success", or "error".
 */
function setStatus(message, type = "loading") {
    const status = document.getElementById("status");
    const text = document.getElementById("status-text");
    status.className = `status ${type}`;
    text.textContent = message;

    // Hide spinner for success/error.
    const spinner = status.querySelector(".spinner");
    if (spinner) {
        spinner.style.display = type === "loading" ? "inline-block" : "none";
    }
}

/**
 * Formats time in human-readable format with appropriate units.
 *
 * @param {number} ms - Time in milliseconds.
 * @returns {string} Formatted time string.
 */
function formatTime(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

// ============================================================================
// Data Loading Functions
// ============================================================================

/**
 * Loads and initializes the WASM module.
 *
 * @returns {Promise<Object>} The initialized WASM module.
 */
async function loadWasmModule() {
    setStatus("Loading WASM module...");
    const wasm = await import(CHART_CONFIG.wasmModulePath);
    await wasm.default();
    console.log("WASM loaded:", wasm.get_version());
    return wasm;
}

/**
 * Loads all benchmark data from WASM (benchmarks + commits).
 *
 * @param {Object} wasm - The WASM module.
 * @returns {Promise<Object>} Object with benchmarks and commits.
 */
async function loadBenchmarkData(wasm) {
    setStatus("Loading benchmark data from Vortex files via WASM...");
    const data = await wasm.load_benchmark_data();
    console.log(`Loaded benchmark data with ${data.commits.length} commits`);
    return data;
}

// ============================================================================
// Data Processing Functions
// ============================================================================

/**
 * Processes benchmark data into chart-ready format.
 *
 * @param {Object} data - The data object from WASM with benchmarks and commits.
 * @returns {Object} Object containing seriesData and chartCommits.
 */
function processChartData(data) {
    const { benchmarks, commits } = data;

    // Get the target group and chart.
    const group = benchmarks[CHART_CONFIG.targetGroup];
    if (!group) {
        throw new Error(`Group '${CHART_CONFIG.targetGroup}' not found in benchmark data`);
    }

    const chart = group.charts[CHART_CONFIG.targetChart];
    if (!chart) {
        throw new Error(`Chart '${CHART_CONFIG.targetChart}' not found in group '${CHART_CONFIG.targetGroup}'`);
    }

    const alignedSeries = chart.aligned_series;

    // Convert commits to chart-friendly format with URLs and short IDs.
    const processedCommits = commits.map((commit, index) => ({
        ...commit,
        // Use commit_id for display (first 7 chars).
        id: commit.commit_id,
        // Construct GitHub URL.
        url: `${CHART_CONFIG.githubRepo}/commit/${commit.commit_id}`,
        // Keep original index for reference.
        sortedIndex: index,
    }));

    // Convert series data from nanoseconds to milliseconds.
    const seriesData = new Map();
    for (const name of CHART_CONFIG.seriesNames) {
        const rawData = alignedSeries[name];
        if (rawData) {
            // Convert nanoseconds to milliseconds, preserving nulls.
            const msData = rawData.map(v => v !== null ? { value: v / 1_000_000 } : null);
            seriesData.set(name, msData);
        } else {
            // Series not found, fill with nulls.
            seriesData.set(name, new Array(commits.length).fill(null));
        }
    }

    // Find the range of data (first and last non-null indices).
    let firstDataIndex = commits.length;
    let lastDataIndex = 0;
    for (const data of seriesData.values()) {
        for (let i = 0; i < data.length; i++) {
            if (data[i] !== null) {
                firstDataIndex = Math.min(firstDataIndex, i);
                lastDataIndex = Math.max(lastDataIndex, i);
            }
        }
    }

    // Slice to show only the range with data.
    const startIndex = Math.max(0, firstDataIndex);
    const endIndex = lastDataIndex + 1;
    const chartCommits = processedCommits.slice(startIndex, endIndex);
    const slicedSeriesData = new Map();

    for (const [name, data] of seriesData.entries()) {
        slicedSeriesData.set(name, data.slice(startIndex, endIndex));
    }

    console.log(`Processed ${commits.length} commits, showing range ${startIndex}-${endIndex}`);

    return { seriesData: slicedSeriesData, chartCommits };
}

/**
 * Calculates summary statistics for the latest data point.
 *
 * @param {Map} seriesData - Map of series names to data arrays.
 * @returns {Object} Object containing results array and fastestTime.
 */
function calculateSummary(seriesData) {
    const latestResults = new Map();

    for (const [seriesName, data] of seriesData.entries()) {
        // Find the most recent non-null value.
        for (let i = data.length - 1; i >= 0; i--) {
            if (data[i] !== null) {
                latestResults.set(seriesName, data[i].value);
                break;
            }
        }
    }

    const fastestTime = Math.min(...latestResults.values());
    const sortedResults = Array.from(latestResults.entries())
        .sort((a, b) => a[1] - b[1]);

    return { results: sortedResults, fastestTime };
}

// ============================================================================
// Rendering Functions
// ============================================================================

/**
 * Renders the summary table showing latest benchmark results.
 *
 * @param {Object} summary - Summary data containing results and fastestTime.
 */
function renderSummary({ results, fastestTime }) {
    const summaryList = document.getElementById("summary-list");
    summaryList.innerHTML = "";

    results.forEach(([name, time], index) => {
        const ratio = time / fastestTime;
        const item = document.createElement("div");
        item.className = "score-item";
        item.innerHTML = `
            <span class="score-rank">#${index + 1}</span>
            <span class="score-series" style="color: ${CHART_CONFIG.seriesColors[name]}">${name}</span>
            <div class="score-metrics">
                <span class="score-runtime">${formatTime(time)}</span>
                <span class="score-ratio">${ratio.toFixed(2)}x</span>
            </div>
        `;
        summaryList.appendChild(item);
    });

    // Show the benchmark section.
    document.getElementById("random-access-benchmark").classList.remove("hidden");
}

/**
 * Sets up collapsible benchmark sections.
 */
function setupCollapsibleBenchmarks() {
    document.querySelectorAll('.benchmark-header').forEach(header => {
        header.addEventListener('click', () => {
            const benchmarkSet = header.closest('.benchmark-set');
            benchmarkSet.classList.toggle('collapsed');
        });
    });
}

/**
 * Creates Chart.js datasets from series data.
 *
 * @param {Map} seriesData - Map of series names to data arrays.
 * @returns {Array} Array of Chart.js dataset objects.
 */
function createDatasets(seriesData) {
    return CHART_CONFIG.seriesNames.map(name => {
        const data = seriesData.get(name);
        return {
            label: name,
            data: data.map(d => d?.value ?? null),
            borderColor: CHART_CONFIG.seriesColors[name],
            backgroundColor: CHART_CONFIG.seriesColors[name],
            borderWidth: 1.5,
            borderJoinStyle: 'round',
            pointRadius: 2,
            tension: 0,
            spanGaps: true,
        };
    });
}

// ============================================================================
// Chart Configuration Functions
// ============================================================================

/**
 * Creates the vertical line plugin for Chart.js.
 *
 * @returns {Object} Chart.js plugin object.
 */
function createVerticalLinePlugin() {
    return {
        id: 'verticalLine',
        afterDatasetsDraw(chart) {
            if (chart.tooltip?._active?.length) {
                const activePoint = chart.tooltip._active[0];
                const ctx = chart.ctx;
                const x = activePoint.element.x;
                const topY = chart.scales.y.top;
                const bottomY = chart.scales.y.bottom;

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(x, topY);
                ctx.lineTo(x, bottomY);
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(89, 113, 253, 0.5)';
                ctx.stroke();
                ctx.restore();
            }
        }
    };
}

/**
 * Handles click events on chart data points.
 *
 * @param {Array} elements - Array of clicked chart elements.
 * @param {Array} chartCommits - Array of commit objects.
 */
function handleChartClick(elements, chartCommits) {
    if (elements.length > 0) {
        const index = elements[0].index;
        const commit = chartCommits[index];
        if (commit?.url) {
            window.open(commit.url, "_blank");
        }
    }
}

/**
 * Creates tooltip configuration for Chart.js.
 *
 * @param {Array} chartCommits - Array of commit objects.
 * @returns {Object} Chart.js tooltip configuration.
 */
function createTooltipConfig(chartCommits) {
    return {
        enabled: false,
        external: (context) => renderExternalTooltip(context),
        callbacks: {
            footer: (tooltipItems) => getTooltipFooter(tooltipItems, chartCommits)
        }
    };
}

/**
 * Gets tooltip footer content with commit details.
 *
 * @param {Array} tooltipItems - Array of tooltip items.
 * @param {Array} chartCommits - Array of commit objects.
 * @returns {Array} Array of footer lines.
 */
function getTooltipFooter(tooltipItems, chartCommits) {
    if (tooltipItems.length === 0) return [];
    const commit = chartCommits[tooltipItems[0].dataIndex];
    if (!commit) return [];

    // Handle timestamp - it's Unix seconds from Rust.
    const date = new Date(commit.timestamp * 1000).toLocaleDateString();

    return [
        commit.message.split("\n")[0].slice(0, 60),
        `${commit.author.name} - ${date}`
    ];
}

/**
 * Renders the external tooltip element.
 *
 * @param {Object} context - Chart.js tooltip context.
 */
function renderExternalTooltip(context) {
    const tooltipEl = document.getElementById('chartjs-tooltip');
    const tooltipModel = context.tooltip;

    // Hide if no tooltip.
    if (tooltipModel.opacity === 0) {
        tooltipEl.classList.remove('active');
        return;
    }

    // Set tooltip content.
    if (tooltipModel.body) {
        tooltipEl.innerHTML = buildTooltipHTML(tooltipModel);
        positionTooltip(tooltipEl, context, tooltipModel);
    }
}

/**
 * Builds HTML content for the tooltip.
 *
 * @param {Object} tooltipModel - Chart.js tooltip model.
 * @returns {string} HTML string for tooltip content.
 */
function buildTooltipHTML(tooltipModel) {
    const titleLines = tooltipModel.title || [];
    const footerLines = tooltipModel.footer || [];

    let html = '<div class="chartjs-tooltip-body">';

    // Add title (commit ID).
    titleLines.forEach(title => {
        html += `<div style="font-weight: bold; margin-bottom: 4px;">${title}</div>`;
    });

    // Add body (series values) - sorted high to low.
    const sortedItems = tooltipModel.dataPoints.sort((a, b) => b.parsed.y - a.parsed.y);
    sortedItems.forEach((item) => {
        const color = item.dataset.borderColor;
        const value = item.formattedValue;
        const label = item.dataset.label;
        html += `
            <div class="chartjs-tooltip-item">
                <div class="chartjs-tooltip-color" style="background-color: ${color}"></div>
                <span>${label}: ${value}ms</span>
            </div>
        `;
    });

    // Add footer (commit details).
    if (footerLines.length > 0) {
        html += '<div class="chartjs-tooltip-footer">';
        footerLines.forEach(footer => {
            html += `<div>${footer}</div>`;
        });
        html += '</div>';
    }

    html += '</div>';
    return html;
}

/**
 * Positions the tooltip below the chart.
 *
 * @param {HTMLElement} tooltipEl - The tooltip element.
 * @param {Object} context - Chart.js context.
 * @param {Object} tooltipModel - Chart.js tooltip model.
 */
function positionTooltip(tooltipEl, context, tooltipModel) {
    const canvas = context.chart.canvas;
    const canvasRect = canvas.getBoundingClientRect();

    tooltipEl.classList.add('active');
    tooltipEl.style.left = canvasRect.left + window.pageXOffset + tooltipModel.caretX + 'px';
    tooltipEl.style.top = canvasRect.bottom + window.pageYOffset + 10 + 'px';
    tooltipEl.style.transform = 'translateX(-50%)';
}

/**
 * Creates chart options configuration.
 *
 * @param {Array} chartCommits - Array of commit objects.
 * @returns {Object} Chart.js options configuration.
 */
function createChartOptions(chartCommits) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: { left: 0, right: 0, top: 0, bottom: 0 }
        },
        interaction: {
            intersect: false,
            mode: "index",
        },
        scales: {
            x: {
                title: { display: false },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45,
                    autoSkipPadding: 10,
                },
                min: Math.max(0, chartCommits.length - CHART_CONFIG.defaultVisibleCommits),
            },
            y: {
                title: {
                    display: true,
                    text: CHART_CONFIG.yAxisLabel,
                },
                beginAtZero: true,
            },
        },
        plugins: {
            verticalLine: {},
            legend: { position: "top" },
            tooltip: createTooltipConfig(chartCommits),
        },
        onClick: (event, elements) => handleChartClick(elements, chartCommits),
    };
}

/**
 * Creates the Chart.js instance.
 *
 * @param {Array} chartCommits - Array of commit objects.
 * @param {Map} seriesData - Map of series names to data arrays.
 * @returns {Object} The Chart.js instance.
 */
function createChart(chartCommits, seriesData) {
    const ctx = document.getElementById("chart").getContext("2d");
    const datasets = createDatasets(seriesData);

    // Register vertical line plugin.
    Chart.register(createVerticalLinePlugin());

    const chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartCommits.map(c => c.id.slice(0, 7)),
            datasets: datasets,
        },
        options: createChartOptions(chartCommits),
    });

    return chartInstance;
}

// ============================================================================
// Timeline Control Functions
// ============================================================================

/**
 * Creates chart context for timeline state management.
 *
 * @param {number} totalCommits - Total number of commits.
 * @returns {Object} Chart context object.
 */
function createChartContext(totalCommits) {
    return {
        totalCommits,
        minWindowSize: CHART_CONFIG.minWindowSize,
        maxWindowSize: totalCommits,
        defaultWindowSize: CHART_CONFIG.defaultVisibleCommits,
        currentWindowSize: CHART_CONFIG.defaultVisibleCommits,
        currentPosition: totalCommits
    };
}

/**
 * Updates scrollbar dimensions to match current window size.
 *
 * @param {Object} elements - DOM elements for timeline controls.
 * @param {Object} chartContext - Chart context state.
 */
function updateScrollbarDimensions(elements, chartContext) {
    const containerWidth = elements.scrollbarContainer.clientWidth;
    const ratio = chartContext.totalCommits / chartContext.currentWindowSize;
    const contentWidth = Math.max(containerWidth * ratio, containerWidth * 1.01);
    elements.scrollbarContent.style.width = `${contentWidth}px`;
}

/**
 * Updates chart view and UI to reflect current state.
 *
 * @param {Object} elements - DOM elements for timeline controls.
 * @param {Object} chartContext - Chart context state.
 * @param {Object} chartInstance - The Chart.js instance.
 * @param {boolean} updateScrollbar - Whether to update scrollbar position.
 */
function updateChartView(elements, chartContext, chartInstance, updateScrollbar) {
    const windowSize = chartContext.currentWindowSize;
    const position = chartContext.currentPosition;

    const endIndex = Math.min(position, chartContext.totalCommits);
    const startIndex = Math.max(0, endIndex - windowSize);

    // Update chart x-axis scale.
    chartInstance.options.scales.x.min = startIndex;
    chartInstance.options.scales.x.max = endIndex - 1;
    chartInstance.update('none');

    // Update UI labels.
    elements.controlInfoText.textContent =
        `Showing commits ${startIndex + 1}-${endIndex} of ${chartContext.totalCommits} (${windowSize} visible)`;

    // Update scrollbar.
    if (updateScrollbar) {
        updateScrollbarDimensions(elements, chartContext);
        const scrollPercentage = (endIndex - windowSize) / (chartContext.totalCommits - windowSize);
        elements.scrollbarContainer.scrollLeft = scrollPercentage *
            (elements.scrollbarContent.clientWidth - elements.scrollbarContainer.clientWidth);
    }
}

/**
 * Sets up scrollbar event handler.
 *
 * @param {Object} elements - DOM elements for timeline controls.
 * @param {Object} chartContext - Chart context state.
 * @param {Object} chartInstance - The Chart.js instance.
 */
function setupScrollbarHandler(elements, chartContext, chartInstance) {
    elements.scrollbarContainer.addEventListener("scroll", () => {
        const scrollLeft = elements.scrollbarContainer.scrollLeft;
        const maxScroll = elements.scrollbarContent.clientWidth - elements.scrollbarContainer.clientWidth;
        const scrollPercentage = maxScroll > 0 ? scrollLeft / maxScroll : 0;

        const windowSize = chartContext.currentWindowSize;
        const newPosition = Math.round(windowSize + scrollPercentage * (chartContext.totalCommits - windowSize));
        chartContext.currentPosition = Math.min(chartContext.totalCommits, Math.max(windowSize, newPosition));

        updateChartView(elements, chartContext, chartInstance, false);
    });
}

/**
 * Sets up zoom button click handlers.
 *
 * @param {Object} elements - DOM elements for timeline controls.
 * @param {Object} chartContext - Chart context state.
 * @param {Object} chartInstance - The Chart.js instance.
 */
function setupZoomButtons(elements, chartContext, chartInstance) {
    const zoom = (step, direction) => {
        const currentWindowSize = chartContext.currentWindowSize;

        // Snap to next multiple of step in the given direction.
        let newWindowSize;
        if (direction > 0) {
            // Zoom out: go to next higher multiple.
            newWindowSize = Math.ceil((currentWindowSize + 1) / step) * step;
        } else {
            // Zoom in: go to next lower multiple.
            newWindowSize = Math.floor((currentWindowSize - 1) / step) * step;
        }
        newWindowSize = Math.max(chartContext.minWindowSize, Math.min(chartContext.maxWindowSize, newWindowSize));

        // Keep centered while zooming.
        const currentStart = chartContext.currentPosition - currentWindowSize;
        const currentCenter = currentStart + currentWindowSize / 2;
        chartContext.currentWindowSize = newWindowSize;
        chartContext.currentPosition = Math.min(
            chartContext.totalCommits,
            Math.max(newWindowSize, Math.round(currentCenter + newWindowSize / 2))
        );

        updateChartView(elements, chartContext, chartInstance, true);
    };

    // Small zoom: 25 commits, large zoom: 250 commits.
    elements.zoomInSmallBtn?.addEventListener("click", () => zoom(25, -1));
    elements.zoomInLargeBtn?.addEventListener("click", () => zoom(250, -1));
    elements.zoomOutSmallBtn?.addEventListener("click", () => zoom(25, 1));
    elements.zoomOutLargeBtn?.addEventListener("click", () => zoom(250, 1));
}

/**
 * Sets up mouse wheel pan handler.
 *
 * @param {Object} elements - DOM elements for timeline controls.
 * @param {Object} chartContext - Chart context state.
 * @param {Object} chartInstance - The Chart.js instance.
 */
function setupWheelPanHandler(elements, chartContext, chartInstance) {
    elements.chartCanvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        const delta = Math.sign(e.deltaY);
        const panAmount = Math.max(1, Math.round(chartContext.currentWindowSize * 0.1));

        chartContext.currentPosition = Math.min(
            chartContext.totalCommits,
            Math.max(chartContext.currentWindowSize, chartContext.currentPosition + delta * panAmount)
        );

        updateChartView(elements, chartContext, chartInstance, true);
    });
}

/**
 * Initializes timeline controls for chart navigation.
 *
 * @param {Object} chartInstance - The Chart.js instance.
 * @param {Array} chartCommits - Array of commit objects.
 */
function initializeTimelineControls(chartInstance, chartCommits) {
    const elements = {
        scrollbarContainer: document.querySelector(".timeline-scrollbar-container"),
        scrollbarContent: document.getElementById("timeline-scrollbar-content"),
        controlInfoText: document.getElementById("control-info-text"),
        chartCanvas: document.getElementById("chart"),
        zoomInSmallBtn: document.getElementById("zoom-in-small"),
        zoomInLargeBtn: document.getElementById("zoom-in-large"),
        zoomOutSmallBtn: document.getElementById("zoom-out-small"),
        zoomOutLargeBtn: document.getElementById("zoom-out-large"),
    };

    const chartContext = createChartContext(chartCommits.length);

    // Handle edge cases.
    if (chartContext.totalCommits === 0) {
        elements.scrollbarContainer.style.display = "none";
        elements.controlInfoText.textContent = "No data available";
        return;
    }

    if (chartContext.totalCommits === 1) {
        elements.scrollbarContainer.style.display = "none";
        chartContext.maxWindowSize = 1;
    }

    // Setup event handlers.
    setupScrollbarHandler(elements, chartContext, chartInstance);
    setupZoomButtons(elements, chartContext, chartInstance);
    setupWheelPanHandler(elements, chartContext, chartInstance);

    // Initial render.
    updateChartView(elements, chartContext, chartInstance, true);
}

// ============================================================================
// Main Orchestration
// ============================================================================

/**
 * Main function that orchestrates the entire benchmark chart initialization.
 */
async function main() {
    try {
        // Load data from WASM (benchmarks + commits in one call).
        const wasm = await loadWasmModule();
        const data = await loadBenchmarkData(wasm);

        // Process data for the target group/chart.
        const { seriesData, chartCommits } = processChartData(data);
        const summary = calculateSummary(seriesData);

        // Render UI.
        renderSummary(summary);
        const chartInstance = createChart(chartCommits, seriesData);
        initializeTimelineControls(chartInstance, chartCommits);

        // Set up collapsible behavior.
        setupCollapsibleBenchmarks();

        setStatus(`Loaded ${data.commits.length} commits, showing ${chartCommits.length} with data`, "success");

    } catch (error) {
        console.error("Error:", error);
        setStatus(`Error: ${error.message}`, "error");
    }
}

main();
