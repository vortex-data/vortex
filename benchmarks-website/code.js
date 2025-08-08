"use strict";

// Import modules
import { CONFIG, SERIES_COLOR_MAP, VORTEX_COLORS, FALLBACK_PALETTE, BENCHMARK_DESCRIPTIONS, CATEGORY_TAGS, SCALE_FACTOR_DESCRIPTIONS, QUERY_NAME_MAP, ENGINE_LABELS, BENCHMARK_GROUPS } from './config.js';
import { utils } from './utils.js';
import { dataProcessor } from './data-processor.js';
import { chartManager } from './chart-manager.js';
import { scoring } from './scoring.js';
import { zoomSync } from './zoom-sync.js';
import { workerManager } from './worker-manager.js';

// Fun rendering messages
const RENDERING_MESSAGES = [
  "Materializing charts from the data dimension...",
  "Rendering graphs faster than light travels...",
  "Drawing charts with the precision of Leonardo da Vinci...",
  "Manifesting benchmarks from the quantum realm...",
  "Generating plots like Bob Ross paints happy trees...",
  "Building charts with the power of the One Ring...",
  "Crafting visualizations like Tony Stark builds suits...",
  "Assembling graphs with Voltron-like precision...",
];

function getRandomRenderingMessage() {
  return RENDERING_MESSAGES[Math.floor(Math.random() * RENDERING_MESSAGES.length)];
}

// Main module
window.initAndRender = (function () {
  // State management
  const state = {
    currentView: "grid",
    expandedSections: new Set(), // Start with all sections collapsed
    activeCategory: "all",
    activeTag: "all",
    activeEngines: new Set(["all"]), // Changed to Set for multiple selections
    searchTerm: "",
    charts: [],
    chartInstances: new Map(),
    pendingZoomUpdates: new Map(),
    lastWindowWidth: window.innerWidth,
    isResizing: false,
  };

  // DOM element cache
  const domElements = {};
  let chartObserver = null;






  // UI module
  const ui = {
    getTpchDescription(categoryName) {
      const scaleFactorMatch = categoryName.match(/SF=(\d+)/);
      const scaleFactor = scaleFactorMatch ? scaleFactorMatch[1] : null;
      const scaleFactorInfo =
        SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

      if (categoryName.includes("NVMe")) {
        return `TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance at ${scaleFactorInfo}`;
      } else if (categoryName.includes("S3")) {
        return `TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads at ${scaleFactorInfo}`;
      }
      return "";
    },

    getDescription(categoryName) {
      if (categoryName.startsWith("TPC-H")) {
        return this.getTpchDescription(categoryName);
      }
      const baseCategory = categoryName.split(" (")[0];
      return BENCHMARK_DESCRIPTIONS[baseCategory] || "";
    },

    benchmarkGroupHasData(benchSet) {
      if (!benchSet || benchSet.size === 0) return false;

      // Check if any query in the benchmark set has data
      for (const [queryName, queryData] of benchSet.entries()) {
        if (!queryData.series || queryData.series.size === 0) continue;

        // Check if any series has any non-null data
        for (const [seriesName, seriesData] of queryData.series.entries()) {
          for (let i = 0; i < seriesData.length; i++) {
            if (seriesData[i] && seriesData[i].value !== null && seriesData[i].value !== undefined) {
              return true;
            }
          }
        }
      }

      return false;
    },

    createBenchmarkSection(name, benchSet, groupFilterSettings = {}) {
      const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } =
        groupFilterSettings;

      const section = document.createElement("div");
      section.className = "benchmark-set";
      section.setAttribute("data-category", name);

      // Check if this benchmark group has any data
      const hasData = this.benchmarkGroupHasData(benchSet);
      if (!hasData) {
        section.classList.add("no-data");
      }

      // Create wrapper for sticky header to maintain space
      const stickyWrapper = document.createElement("div");
      stickyWrapper.className = "sticky-header-wrapper";

      // Create sticky header container
      const stickyContainer = document.createElement("div");
      stickyContainer.className = "sticky-header-container";

      // Add header
      const header = this.createSectionHeader(name, benchSet, keptCharts);
      stickyContainer.appendChild(header);

      // Add controls
      const controls = this.createSectionControls(name);
      if (controls) {
        stickyContainer.appendChild(controls);
      }

      stickyWrapper.appendChild(stickyContainer);
      section.appendChild(stickyWrapper);

      // Add scoring summary for query benchmarks (after sticky container)
      if (scoring.isQueryBenchmark(name) && benchSet) {
        const scores = scoring.calculateClickBenchScore(benchSet);
        const scoreSummary = scoring.formatScoresSummary(scores);
        if (scoreSummary) {
          section.appendChild(scoreSummary);
        }
      }

      // Add summary for Random Access benchmarks
      if (scoring.isRandomAccessBenchmark(name) && benchSet) {
        const metrics = scoring.calculateRandomAccessMetrics(benchSet);
        const metricsSummary = scoring.formatRandomAccessSummary(metrics);
        if (metricsSummary) {
          section.appendChild(metricsSummary);
        }
      }

      // Add summary for Compression benchmarks
      if (scoring.isCompressionBenchmark(name) && benchSet) {
        const metrics = scoring.calculateCompressionMetrics(benchSet);
        const metricsSummary = scoring.formatCompressionSummary(metrics);
        if (metricsSummary) {
          section.appendChild(metricsSummary);
        }
      }

      // Add summary for Compression Size benchmarks
      if (scoring.isCompressionSizeBenchmark(name) && benchSet) {
        const metrics = scoring.calculateCompressionSizeMetrics(benchSet);
        const metricsSummary = scoring.formatCompressionSizeSummary(metrics);
        if (metricsSummary) {
          section.appendChild(metricsSummary);
        }
      }

      // Add charts container
      const chartsContainer = document.createElement("div");
      chartsContainer.className = "benchmark-graphs";

      // Add single-chart class if there's only one chart
      const chartCount = keptCharts ? keptCharts.length : benchSet?.size || 0;
      if (chartCount === 1) {
        chartsContainer.classList.add("single-chart");
      }

      section.appendChild(chartsContainer);

      // Collapse by default
      section.classList.add("collapsed");

      return { section, chartsContainer };
    },

    createSectionHeader(name, benchSet, keptCharts) {
      const h1id = name.replace(/\s+/g, "_");

      const header = document.createElement("div");
      header.className = "benchmark-header";
      
      // Check if the parent section has the no-data class
      const section = document.querySelector(`[data-category="${name}"]`);
      const hasNoData = section && section.classList.contains("no-data");
      
      if (!hasNoData) {
        header.onclick = (e) => {
          // Don't toggle if clicking on info icon
          if (!e.target.closest(".info-icon")) {
            this.toggleSection(name);
          }
        };
      }

      const titleWrapper = document.createElement("div");
      titleWrapper.className = "title-wrapper";

      const title = document.createElement("h1");
      title.id = h1id;
      title.className = "benchmark-title";
      title.innerHTML = `<span class="collapse-icon">▼</span> ${name}`;

      // Create a secondary container for link, info, and charts count
      const secondaryInfo = document.createElement("div");
      secondaryInfo.className = "benchmark-secondary-info";

      const linkBtn = document.createElement("button");
      linkBtn.className = "group-link-btn";
      linkBtn.setAttribute("aria-label", "Copy link to this section");
      linkBtn.innerHTML = "🔗";
      linkBtn.onclick = (e) => {
        e.stopPropagation();
        this.linkToGroup(name);
      };
      secondaryInfo.appendChild(linkBtn);

      // Add info icon with tooltip
      const description = this.getDescription(name);
      if (description) {
        const infoIcon = document.createElement("div");
        infoIcon.className = "info-icon";
        infoIcon.innerHTML = "ⓘ";
        infoIcon.setAttribute("data-tooltip", description);
        secondaryInfo.appendChild(infoIcon);
      }

      const meta = document.createElement("div");
      meta.className = "benchmark-meta";
      const chartCount = keptCharts ? keptCharts.length : benchSet?.size || 0;
      
      // Check if the parent section has the no-data class
      const sectionHasNoData = section && section.classList.contains("no-data");
      if (sectionHasNoData) {
        meta.textContent = "No data available";
      } else {
        meta.textContent = `${chartCount} charts`;
      }
      secondaryInfo.appendChild(meta);

      titleWrapper.appendChild(title);
      titleWrapper.appendChild(secondaryInfo);

      header.appendChild(titleWrapper);

      return header;
    },

    createSectionControls(name) {
      const tags = CATEGORY_TAGS[name] || [];
      const isQueryGroup = tags.some((tag) => tag.includes("Queries"));

      const container = document.createElement("div");
      container.className = "engine-filter-container";

      if (isQueryGroup) {
        const label = document.createElement("span");
        label.className = "engine-filter-label";
        label.textContent = "Show: ";
        container.appendChild(label);

        Object.entries(ENGINE_LABELS).forEach(([engine, label]) => {
          const btn = document.createElement("button");
          btn.className =
            "engine-filter-btn" +
            (state.activeEngines.has(engine) ? " active" : "");
          btn.textContent = label;
          btn.setAttribute("data-engine", engine);
          btn.setAttribute("data-category", name);
          btn.onclick = () => this.filterEngine(name, engine);
          container.appendChild(btn);
        });

        const separator = document.createElement("span");
        separator.className = "filter-separator";
        separator.textContent = "|";
        container.appendChild(separator);
      }

      const resetBtn = document.createElement("button");
      resetBtn.className = "reset-zoom-btn";
      resetBtn.textContent = "Reset X-Axis";
      resetBtn.setAttribute("data-category", name);
      resetBtn.onclick = () => window.zoomSync.resetZoomForCategory(name, state, utils, CONFIG);
      container.appendChild(resetBtn);

      return container;
    },

    toggleSection(name) {
      const section = document.querySelector(`[data-category="${name}"]`);
      if (!section) return;
      
      // Don't toggle if section has no data
      if (section.classList.contains("no-data")) return;

      if (state.expandedSections.has(name)) {
        state.expandedSections.delete(name);
        section.classList.add("collapsed");
      } else {
        state.expandedSections.add(name);
        section.classList.remove("collapsed");
      }
    },

    linkToGroup(name) {
      urlManager.updateParams({ group: name });

      const targetSection = document.querySelector(`[data-category="${name}"]`);

      navigator.clipboard.writeText(window.location.href).then(() => {
        if (targetSection) {
          const linkBtn = targetSection.querySelector(".group-link-btn");
          if (linkBtn) {
            const originalText = linkBtn.innerHTML;
            linkBtn.innerHTML = "✓";
            linkBtn.classList.add("copied");
            setTimeout(() => {
              linkBtn.innerHTML = originalText;
              linkBtn.classList.remove("copied");
            }, CONFIG.LINK_FEEDBACK_DURATION);
          }
        }
      });
    },

    filterEngine(categoryName, engine) {
      // Handle "all" button specially
      if (engine === "all") {
        state.activeEngines.clear();
        state.activeEngines.add("all");
      } else {
        // Remove "all" if selecting specific engine
        if (state.activeEngines.has("all")) {
          state.activeEngines.clear();
        }

        // Toggle the selected engine
        if (state.activeEngines.has(engine)) {
          state.activeEngines.delete(engine);
          // If no engines selected, revert to "all"
          if (state.activeEngines.size === 0) {
            state.activeEngines.add("all");
          }
        } else {
          state.activeEngines.add(engine);
        }
      }

      // Update URL with comma-separated engines
      const engineParam = state.activeEngines.has("all")
        ? "all"
        : Array.from(state.activeEngines).join(",");
      urlManager.updateParams({ engine: engineParam });

      // Update all engine filter buttons
      document
        .querySelectorAll(".engine-filter-container")
        .forEach((container) => {
          container.querySelectorAll(".engine-filter-btn").forEach((btn) => {
            const btnEngine = btn.getAttribute("data-engine");
            btn.classList.toggle("active", state.activeEngines.has(btnEngine));
          });
        });

      // Apply filter to charts
      this.applyEngineFilter();
    },

    applyEngineFilter() {
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        const tags = CATEGORY_TAGS[category] || [];
        const isQueryGroup = tags.some((tag) => tag.includes("Queries"));

        if (isQueryGroup) {
          const containers = section.querySelectorAll(".chart-container");
          containers.forEach((container, index) => {
            const chartKey = `${category}-${index}`;
            const chartData = state.chartInstances.get(chartKey);

            if (chartData?.chart) {
              this.updateChartVisibility(chartData.chart);
            }
          });
        }
      });
    },

    updateChartVisibility(chart) {
      const updates = [];

      chart.data.datasets.forEach((dataset, index) => {
        const label = dataset.label.toLowerCase();

        // Check if dataset should be visible based on selected engines
        const shouldShow =
          state.activeEngines.has("all") ||
          Array.from(state.activeEngines).some((engine) =>
            label.includes(engine)
          );

        if (chart.isDatasetVisible(index) !== shouldShow) {
          updates.push({ index, visible: shouldShow });
        }
      });

      if (updates.length > 0) {
        updates.forEach(({ index, visible }) => {
          chart.setDatasetVisibility(index, visible);
        });
        chart.update("none");
      }
    },

    setView(view) {
      state.currentView = view;
      document.querySelectorAll(".benchmark-graphs").forEach((graphs) => {
        graphs.classList.toggle("list-view", view === "list");
      });

      document.querySelectorAll(".view-btn").forEach((btn) => {
        btn.classList.remove("active");
      });
      document.getElementById(`${view}-view`).classList.add("active");
    },
  };

  // URL management module
  const urlManager = {
    getParams() {
      const params = new URLSearchParams(window.location.search);
      return {
        tag: params.get("tag") || "all",
        engine: params.get("engine") || "all",
        expanded: params.get("expanded") || "false",
        group: params.get("group") || null,
      };
    },

    updateParams(updates) {
      const params = new URLSearchParams(window.location.search);

      Object.entries(updates).forEach(([key, value]) => {
        if (
          value &&
          value !== "all" &&
          !(key === "expanded" && value === "false")
        ) {
          params.set(key, value);
        } else {
          params.delete(key);
        }
      });

      const newURL =
        window.location.pathname +
        (params.toString() ? "?" + params.toString() : "");
      window.history.replaceState({}, "", newURL);
    },

    initializeFromParams() {
      const params = this.getParams();

      state.activeTag = params.tag;

      // Handle comma-separated engines
      if (params.engine && params.engine !== "all") {
        const engines = params.engine.split(",");
        state.activeEngines.clear();
        engines.forEach((engine) => state.activeEngines.add(engine.trim()));
      }

      const categoryFilter = domElements.categoryFilter;
      if (categoryFilter) {
        categoryFilter.value = params.tag;
        filterManager.filterByTag(params.tag);
      }

      if (params.engine !== "all") {
        ui.applyEngineFilter();
      }

      if (params.group) {
        setTimeout(() => {
          navigationManager.focusOnGroup(params.group);
        }, CONFIG.URL_INIT_DELAY);
      } else if (params.expanded === "true") {
        navigationManager.expandAll();
      }
    },
  };

  // Filter management module
  const filterManager = {
    filterByTag(tag) {
      state.activeTag = tag;
      urlManager.updateParams({ tag });

      // Filter sections
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        const tags = CATEGORY_TAGS[category] || [];
        section.style.display =
          tag === "all" || tags.includes(tag) ? "block" : "none";
      });

      // Filter navigation
      document.querySelectorAll(".toc-list li").forEach((navItem) => {
        const link = navItem.querySelector("a");
        if (link) {
          const targetId = link.getAttribute("href").substring(1);
          const targetSection = document.getElementById(targetId);

          if (targetSection?.closest(".benchmark-set")) {
            const category = targetSection
              .closest(".benchmark-set")
              .getAttribute("data-category");
            const tags = CATEGORY_TAGS[category] || [];
            navItem.style.display =
              tag === "all" || tags.includes(tag) ? "block" : "none";
          }
        }
      });

      // Update clear filter button
      const clearBtn = domElements.clearFilter;
      if (clearBtn) {
        clearBtn.style.display = tag === "all" ? "none" : "block";
        clearBtn.textContent = tag === "all" ? "" : `Clear Filter: ${tag}`;
      }
    },

    filterBySearch(term) {
      state.searchTerm = term.toLowerCase();

      document.querySelectorAll(".chart-container").forEach((chart) => {
        const benchmarkName = chart
          .getAttribute("data-benchmark")
          .toLowerCase();
        const chartName = chart.getAttribute("data-chart").toLowerCase();
        const matches =
          benchmarkName.includes(state.searchTerm) ||
          chartName.includes(state.searchTerm);
        chart.style.display = matches ? "block" : "none";
      });

      // Then, hide sections that have no visible charts
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const visibleCharts = section.querySelectorAll(
          ".chart-container[style='display: block;'], .chart-container:not([style]), .chart-container[style='']"
        );
        section.style.display = visibleCharts.length > 0 ? "block" : "none";
      });
    },
  };

  // Navigation management module
  const navigationManager = {
    expandAll() {
      const sections = document.querySelectorAll(".benchmark-set");
      const updates = [];

      sections.forEach((section) => {
        // Skip sections with no data
        if (section.classList.contains("no-data")) return;
        
        const category = section.getAttribute("data-category");
        state.expandedSections.add(category);
        if (section.classList.contains("collapsed")) {
          updates.push(() => section.classList.remove("collapsed"));
        }
      });

      utils.batchDOMUpdates(updates);
      urlManager.updateParams({ expanded: "true" });
    },

    collapseAll() {
      const sections = document.querySelectorAll(".benchmark-set");
      const updates = [];

      sections.forEach((section) => {
        const category = section.getAttribute("data-category");
        state.expandedSections.delete(category);
        if (!section.classList.contains("collapsed")) {
          updates.push(() => section.classList.add("collapsed"));
        }
      });

      utils.batchDOMUpdates(updates);
      urlManager.updateParams({ expanded: "false" });
    },

    focusOnGroup(groupName) {
      // Find target section
      const targetSection = document.querySelector(
        `[data-category="${groupName}"]`
      );
      if (targetSection) {
        // Just scroll to the section without expanding it
        // The user can click to expand if they want to see the charts

        // Close sidebar after navigation on mobile
        if (utils.isMobile()) {
          domElements.sidebar.classList.remove("active");
        }

        // Scroll to section
        const targetId = targetSection.querySelector(".benchmark-title").id;
        const targetElement = document.getElementById(targetId);
        const headerHeight =
          document.querySelector(".sticky-header").offsetHeight;
        const elementPosition =
          targetElement.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition =
          elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth",
        });

        this.updateActiveNavItem(targetId);
      }
    },

    updateActiveNavItem(id) {
      document.querySelectorAll(".toc-list a").forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
      });
    },

    handleScroll() {
      const scrollY = window.scrollY;
      domElements.backToTop.classList.toggle(
        "visible",
        scrollY > CONFIG.BACK_TO_TOP_THRESHOLD
      );

      // Update active nav item
      const sections = document.querySelectorAll(".benchmark-set");
      let current = "";

      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= CONFIG.SCROLL_ACTIVE_THRESHOLD) {
          current = section.querySelector(".benchmark-title").id;
        }
      });

      if (current) {
        this.updateActiveNavItem(current);
      }
    },
  };

  // Initialization module
  const initializer = {
    async loadData() {
      const [dataResponse, commitsResponse] = await Promise.all([
        this.fetchGzippedData(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
        ),
        fetch(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json"
        ).then((r) => r.text()),
      ]);

      // Return raw text data for worker processing
      return { 
        benchmarkData: dataResponse, 
        commitsData: commitsResponse 
      };
    },

    async fetchGzippedData(url) {
      const response = await fetch(url);
      const decompressedStream = response.body.pipeThrough(
        new DecompressionStream("gzip")
      );
      const reader = decompressedStream.getReader();
      const decoder = new TextDecoder();
      let result = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
      }

      result += decoder.decode();
      return result;
    },

    parseJsonl(jsonl) {
      return jsonl
        .split("\n")
        .filter((line) => line.trim().length !== 0)
        .map((line) => JSON.parse(line));
    },

    updateLoadingProgress(progress, message) {
      const main = domElements.main || document.getElementById("main");
      const loadingIndicator = main.querySelector('.loading-indicator');
      
      if (loadingIndicator) {
        const progressText = loadingIndicator.querySelector('p');
        if (progressText) {
          progressText.textContent = `${message} (${Math.round(progress)}%)`;
        }
      }
    },

    initializeControls() {
      // Cache DOM elements
      const elementIds = [
        "menu-toggle",
        "sidebar",
        "sidebar-close",
        "sidebar-overlay",
        "expand-all",
        "collapse-all",
        "grid-view",
        "list-view",
        "category-filter",
        "clear-filter",
        "search-filter",
        "back-to-top",
        "modal-close",
        "chart-modal",
        "main",
        "toc",
      ];

      elementIds.forEach((id) => {
        const camelCaseId = id.replace(/-(.)/g, (match, char) =>
          char.toUpperCase()
        );
        domElements[camelCaseId] = document.getElementById(id);
      });

      // Initialize sidebar state on desktop
      if (window.innerWidth >= 1200) {
        const sidebarPref = localStorage.getItem("sidebarCollapsed");
        if (domElements.sidebar) {
          // Set default to collapsed (true) on first visit  
          if (sidebarPref === null) {
            localStorage.setItem("sidebarCollapsed", "true");
          }
          
          // Apply saved preference
          if (sidebarPref === "false") {
            // User previously opened sidebar via toggle
            domElements.sidebar.classList.remove("collapsed");
            domElements.sidebar.classList.add("open");
          } else {
            // Default collapsed or user previously closed it via toggle
            domElements.sidebar.classList.add("collapsed");
            domElements.sidebar.classList.remove("open");
          }
        }
      }

      // Initialize chart observer for lazy loading
      if ("IntersectionObserver" in window) {
        chartObserver = new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              if (entry.isIntersecting) {
                const container = entry.target;
                if (!container.hasAttribute("data-chart-loaded")) {
                  container.setAttribute("data-chart-loaded", "true");
                  const chartData = container.chartData;
                  if (chartData) {
                    chartManager.createChartInstance(chartData);
                  }
                }
              }
            });
          },
          {
            rootMargin: CONFIG.CHART_OBSERVER_MARGIN,
          }
        );

        // Initialize sticky header observer
        const stickyObserver = new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              const stickyContainer = entry.target.querySelector(
                ".sticky-header-container"
              );
              if (stickyContainer) {
                if (entry.intersectionRatio < 1) {
                  stickyContainer.classList.add("is-stuck");
                } else {
                  stickyContainer.classList.remove("is-stuck");
                }
              }
            });
          },
          {
            threshold: [1],
            rootMargin: "-72px 0px 0px 0px", // Adjust based on header height
          }
        );

        // Observe all sticky header wrappers after DOM is ready
        setTimeout(() => {
          document.querySelectorAll(".sticky-header-wrapper").forEach((wrapper) => {
            stickyObserver.observe(wrapper);
          });
        }, 100);
      }

      // Make chartObserver globally available for modules
      window.chartObserver = chartObserver;
      
      // Initialize zoom sync
      zoomSync.init(state, utils);

      // Set up event listeners
      this.setupEventListeners();
    },

    setupEventListeners() {
      // Sidebar toggle (for both mobile and desktop)
      domElements.menuToggle.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          // On desktop, toggle collapsed state
          domElements.sidebar.classList.toggle("collapsed");
          domElements.sidebar.classList.toggle("open");

          // Save preference to localStorage
          const isCollapsed =
            domElements.sidebar.classList.contains("collapsed");
          localStorage.setItem("sidebarCollapsed", isCollapsed.toString());
        } else {
          // On mobile/tablet, toggle active state
          domElements.sidebar.classList.toggle("active");
        }
      });

      domElements.sidebarClose.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          domElements.sidebar.classList.add("collapsed");
          domElements.sidebar.classList.remove("open");
          localStorage.setItem("sidebarCollapsed", "true");
        } else {
          domElements.sidebar.classList.remove("active");
        }
      });

      // Sidebar overlay click (mobile and desktop) - only closes, doesn't toggle
      domElements.sidebarOverlay.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          // On desktop, close sidebar (don't toggle, just close)
          domElements.sidebar.classList.add("collapsed");
          domElements.sidebar.classList.remove("open");
          localStorage.setItem("sidebarCollapsed", "true");
        } else {
          // On mobile/tablet, remove active state
          domElements.sidebar.classList.remove("active");
        }
      });

      // Expand/Collapse
      domElements.expandAll.addEventListener("click", () =>
        navigationManager.expandAll()
      );
      domElements.collapseAll.addEventListener("click", () =>
        navigationManager.collapseAll()
      );

      // View controls
      domElements.gridView.addEventListener("click", () => ui.setView("grid"));
      domElements.listView.addEventListener("click", () => ui.setView("list"));

      // Filters
      domElements.categoryFilter.addEventListener("change", (e) => {
        filterManager.filterByTag(e.target.value);
      });

      domElements.clearFilter.addEventListener("click", () => {
        domElements.categoryFilter.value = "all";
        filterManager.filterByTag("all");
        urlManager.updateParams({ tag: "all" });
      });

      const debouncedSearch = utils.debounce(
        (term) => filterManager.filterBySearch(term),
        CONFIG.SEARCH_DEBOUNCE
      );
      domElements.searchFilter.addEventListener("input", (e) => {
        debouncedSearch(e.target.value);
      });

      // Scroll handling
      const throttledScroll = utils.throttle(
        () => navigationManager.handleScroll(),
        CONFIG.THROTTLE_SCROLL
      );
      window.addEventListener("scroll", throttledScroll);

      domElements.backToTop.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });

      // Modal
      domElements.modalClose.addEventListener("click", () =>
        chartManager.closeModal()
      );
      domElements.chartModal.addEventListener("click", (e) => {
        if (e.target.id === "chart-modal") {
          chartManager.closeModal();
        }
      });

      // Outside click for sidebar on mobile only
      document.addEventListener("click", (e) => {
        if (
          utils.isMobile() &&
          !domElements.sidebar.contains(e.target) &&
          !domElements.menuToggle.contains(e.target) &&
          domElements.sidebar.classList.contains("active")
        ) {
          domElements.sidebar.classList.remove("active");
        }
      });

      // Window resize handler
      const debouncedResize = utils.debounce(() => {
        chartManager.updateChartsForResize();

        const isDesktop = window.innerWidth >= 1200;
        const wasDesktop = state.lastWindowWidth >= 1200;

        // Handle sidebar state when crossing desktop/mobile threshold
        if (wasDesktop && !isDesktop) {
          // Moving from desktop to mobile
          domElements.sidebar.classList.remove("collapsed");
          domElements.sidebar.classList.remove("active");
        } else if (!wasDesktop && isDesktop) {
          // Moving from mobile to desktop
          domElements.sidebar.classList.remove("active");
          // Restore saved collapsed state
          const sidebarPref = localStorage.getItem("sidebarCollapsed");
          // Set default to collapsed (true) on first visit  
          if (sidebarPref === null) {
            localStorage.setItem("sidebarCollapsed", "true");
          }
          
          // Apply saved preference
          if (sidebarPref === "false") {
            domElements.sidebar.classList.remove("collapsed");
            domElements.sidebar.classList.add("open");
          } else {
            domElements.sidebar.classList.add("collapsed");
            domElements.sidebar.classList.remove("open");
          }
        }

        // Update last window width
        state.lastWindowWidth = window.innerWidth;
      }, CONFIG.RESIZE_DEBOUNCE);

      window.addEventListener("resize", debouncedResize);
    },
  };

  // Async function to render all benchmark sets with batching
  async function renderBenchmarkSetsAsync(grouped, main, toc, keptGroups) {
    const batchSize = 1; // Render 1 benchmark set at a time for maximum responsiveness
    let currentBatch = 0;
    let totalSets = 0;
    
    // Determine what we're rendering
    let renderQueue = [];
    if (keptGroups === undefined) {
      renderQueue = grouped.map(({ name, dataSet }) => ({ name, dataSet, groupFilterSettings: {} }));
    } else {
      const dataSetsMap = new Map(
        grouped.map(({ name, dataSet }) => [name, dataSet])
      );
      renderQueue = keptGroups.map(([name, groupFilterSettings]) => ({
        name,
        dataSet: dataSetsMap.get(name),
        groupFilterSettings
      }));
    }
    
    totalSets = renderQueue.length;
    
    // Process in batches
    for (let i = 0; i < renderQueue.length; i += batchSize) {
      const batch = renderQueue.slice(i, i + batchSize);
      
      // Render this batch
      for (const { name, dataSet, groupFilterSettings } of batch) {
        // Add placeholder while rendering
        const placeholder = document.createElement('div');
        placeholder.className = 'benchmark-placeholder';
        placeholder.innerHTML = `
          <div class="benchmark-header">
            <div class="title-wrapper">
              <h1 class="benchmark-title">
                <span class="collapse-icon">▼</span> ${name}
              </h1>
              <div class="benchmark-secondary-info">
                <div class="benchmark-meta">Loading...</div>
              </div>
            </div>
          </div>
        `;
        main.appendChild(placeholder);
        
        await renderBenchmarkSet(name, dataSet, main, toc, groupFilterSettings);
        
        // Remove placeholder after rendering
        if (placeholder.parentNode) {
          placeholder.remove();
        }
        
        currentBatch++;
      }
      
      // Update progress and yield control after each batch
      const progress = (currentBatch / totalSets) * 100;
      const progressElement = document.getElementById('rendering-progress');
      if (progressElement) {
        progressElement.textContent = `${getRandomRenderingMessage()} ${currentBatch}/${totalSets} (${Math.round(progress)}%)`;
      }
      
      if (i + batchSize < renderQueue.length) {
        // Yield control to prevent UI freezing
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }

  // Async render benchmark set function
  async function renderBenchmarkSet(
    name,
    benchSet,
    main,
    toc,
    groupFilterSettings = {}
  ) {
    const { section, chartsContainer } = ui.createBenchmarkSection(
      name,
      benchSet,
      groupFilterSettings
    );
    
    // Add fade-in animation effect
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    section.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
    
    main.appendChild(section);
    
    // Trigger fade-in animation
    requestAnimationFrame(() => {
      section.style.opacity = '1';
      section.style.transform = 'translateY(0)';
    });

    // Create TOC entry
    const tocLi = document.createElement("li");
    const tocLink = document.createElement("a");
    const h1id = name.replace(/\s+/g, "_");
    tocLink.href = "#" + h1id;
    tocLink.innerHTML = name;
    tocLink.onclick = (e) => {
      e.preventDefault();

      // Auto-expand the section if it's collapsed (but not if it has no data)
      const targetSection = document.querySelector(`[data-category="${name}"]`);
      if (targetSection && targetSection.classList.contains("collapsed") && !targetSection.classList.contains("no-data")) {
        state.expandedSections.add(name);
        targetSection.classList.remove("collapsed");
      }

      // Close sidebar after navigation on mobile
      if (utils.isMobile()) {
        domElements.sidebar.classList.remove("active");
      }

      const targetElement = document.getElementById(h1id);
      const headerHeight =
        document.querySelector(".sticky-header").offsetHeight;
      const elementPosition =
        targetElement.getBoundingClientRect().top + window.pageYOffset;
      const offsetPosition =
        elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });

      navigationManager.updateActiveNavItem(h1id);
    };
    tocLi.appendChild(tocLink);
    toc.appendChild(tocLi);

    // Render charts
    let chartIndex = 0;
    const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } =
      groupFilterSettings;

    // Render charts with async yielding to prevent UI blocking
    await renderChartsAsync(
      benchSet,
      keptCharts,
      chartsContainer,
      name,
      hiddenDatasets,
      removedDatasets,
      renamedDatasets,
      chartIndex
    );
    
    // Update zoom sync cache for this category
    window.zoomSync.updateCacheForCategory(name);
  }

  // Async function to render charts with yielding
  async function renderChartsAsync(
    benchSet,
    keptCharts,
    chartsContainer,
    name,
    hiddenDatasets,
    removedDatasets,
    renamedDatasets,
    startIndex
  ) {
    let chartIndex = startIndex;
    const chartsToRender = [];
    
    // Collect all charts to render
    if (keptCharts === undefined) {
      if (benchSet !== undefined) {
        for (const [benchName, benches] of benchSet.entries()) {
          chartsToRender.push({ benchName, benches });
        }
      }
    } else if (keptCharts) {
      for (const benchName of keptCharts) {
        const benches = benchSet.get(benchName);
        if (benches) {
          chartsToRender.push({ benchName, benches });
        }
      }
    } else {
      // This is the case when keptCharts is not defined at all (not undefined, just missing)
      if (benchSet !== undefined) {
        for (const [benchName, benches] of benchSet.entries()) {
          chartsToRender.push({ benchName, benches });
        }
      }
    }
    
    // Render charts with yielding between each one
    for (let i = 0; i < chartsToRender.length; i++) {
      const { benchName, benches } = chartsToRender[i];
      
      state.charts.push(
        chartManager.renderChart(
          chartsContainer,
          name,
          benchName,
          benches,
          hiddenDatasets,
          removedDatasets,
          renamedDatasets,
          chartIndex++
        )
      );
      
      // Yield control after each chart to prevent blocking
      // Only yield if there are more charts to render
      if (i < chartsToRender.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }

  // Main initialization
  return async function initAndRender(keptGroups) {
    try {
      // Make necessary objects globally available for modules BEFORE rendering charts
      window.state = state;
      window.domElements = domElements;
      window.zoomSync = zoomSync;
      window.utils = utils;
      
      // Initialize workers
      workerManager.init();
      
      const { benchmarkData, commitsData } = await initializer.loadData();
      
      // Process data using worker or fallback
      const grouped = await workerManager.processData(
        benchmarkData,
        commitsData,
        keptGroups,
        initializer.updateLoadingProgress
      );

      const main = domElements.main || document.getElementById("main");
      const toc = domElements.toc || document.getElementById("toc");

      // Clear loading indicator and show rendering progress
      main.innerHTML = `
        <div class="loading-indicator">
          <div class="loading-spinner"></div>
          <p id="rendering-progress">${getRandomRenderingMessage()} Preparing to render charts...</p>
        </div>
      `;

      // Render all charts with batching to prevent UI freezing
      await renderBenchmarkSetsAsync(grouped, main, toc, keptGroups);
      
      // Remove rendering progress indicator
      const loadingIndicator = main.querySelector('.loading-indicator');
      if (loadingIndicator) {
        loadingIndicator.remove();
      }

      initializer.initializeControls();
      urlManager.initializeFromParams();
      
      // Clean up workers after initialization (give more time for any pending operations)
      setTimeout(() => {
        workerManager.terminate();
      }, 5000);
      
      // Ensure workers and zoom sync are cleaned up on page unload
      window.addEventListener('beforeunload', () => {
        workerManager.terminate();
        window.zoomSync.cleanup();
      });
    } catch (error) {
      console.error("Failed to load benchmark data:", error);
      const main = domElements.main || document.getElementById("main");
      main.innerHTML = `
        <div class="loading-indicator">
          <p style="color: red;">Failed to load benchmark data. Please try refreshing the page.</p>
          <p>${error.message}</p>
        </div>
      `;
    }
  };
})();
