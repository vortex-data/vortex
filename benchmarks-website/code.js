"use strict";
window.initAndRender = (function () {
  // State management
  const state = {
    currentView: 'grid',
    expandedSections: new Set(),
    activeCategory: 'all',
    activeTag: 'all',
    activeEngine: 'all',
    searchTerm: '',
    charts: [],
    chartInstances: new Map(),
    benchmarkDescriptions: {
      'Random Access': 'Measures random access performance across different data structures',
      'Compression': 'Compression and decompression time benchmarks for various encodings',
      'Compression Size': 'Size comparison of compressed data using different algorithms',
      'TPC-H (NVMe)': 'TPC-H benchmark queries on local NVMe storage',
      'TPC-H (S3)': 'TPC-H benchmark queries on S3 storage',
      'Clickbench': 'ClickHouse benchmark queries for analytical workloads'
    },
    categoryTags: {
      'Random Access': ['Read/Write'],
      'Compression': ['Read/Write'],
      'Compression Size': ['Read/Write'],
      'Clickbench': ['Queries (NVMe)'],
      'TPC-H (NVMe) (SF=1)': ['Queries (NVMe)', 'TPC-H (SF=1)'],
      'TPC-H (S3) (SF=1)': ['Queries (S3)', 'TPC-H (SF=1)'],
      'TPC-H (NVMe) (SF=10)': ['Queries (NVMe)', 'TPC-H (SF=10)'],
      'TPC-H (S3) (SF=10)': ['Queries (S3)', 'TPC-H (SF=10)'],
      'TPC-H (NVMe) (SF=100)': ['Queries (NVMe)', 'TPC-H (SF=100)'],
      'TPC-H (S3) (SF=100)': ['Queries (S3)', 'TPC-H (SF=100)'],
      'TPC-H (NVMe) (SF=1000)': ['Queries (NVMe)', 'TPC-H (SF=1000)'],
      'TPC-H (S3) (SF=1000)': ['Queries (S3)', 'TPC-H (SF=1000)']
    }
  };

  function stringToColor(str) {
    // Spiral brand colors
    const SPIRAL_COLORS = {
      primary: "#5971FD",    // Spiral Blue
      accent: "#CEE562",     // Spiral Green  
      pink: "#EEB3E1",       // Spiral Pink
      black: "#101010",      // Spiral Black
      gray: "#666666",       // Secondary gray
    };
    
    // Specific mappings using brand colors
    const MAP = {
      "datafusion:arrow": SPIRAL_COLORS.gray,
      "datafusion:parquet": "#FF8C42",  // Orange complement
      "datafusion:vortex": SPIRAL_COLORS.primary,

      "duckdb:parquet": "#B8336A",  // Pink variant
      "duckdb:vortex": SPIRAL_COLORS.accent,
      "duckdb:duckdb": "#726DA8",   // Purple complement
    };

    if (MAP[str]) {
      return MAP[str];
    }

    // Fallback palette for unmapped series
    const fallbackPalette = [
      SPIRAL_COLORS.primary,
      SPIRAL_COLORS.accent,
      SPIRAL_COLORS.pink,
      "#FF8C42",  // Orange
      "#B8336A",  // Deep pink
      "#726DA8",  // Purple
      "#2D936C",  // Teal
      "#E9B44C",  // Gold
    ];
    
    // Use hash to consistently pick from palette
    let hash = new Hashes.MD5().hex(str);
    const index = parseInt(hash.slice(0, 2), 16) % fallbackPalette.length;
    return fallbackPalette[index];
  }

  function downloadAndGroupData(data, commit_metadata, seriesRenameFn) {
    // It's desirable for all our graphs to line up in terms of X-axis.
    // As such, we collect all unique {commit,entry} first, and then assign
    // data points to them for each graph. Commits are sorted by date.
    const commits = [];
    Object.values(commit_metadata)
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
      .forEach((commit, commitSortedIndex) => {
        commit.sortedIndex = commitSortedIndex;
        commits.push(commit);
      });

    // Prepare data points for charts
    let groups = {
      "Random Access": new Map(),
      Compression: new Map(),
      "Compression Size": new Map(),
      Clickbench: new Map(),
      "TPC-H (NVMe) (SF=1)": new Map(),
      "TPC-H (S3) (SF=1)": new Map(),
      "TPC-H (NVMe) (SF=10)": new Map(),
      "TPC-H (S3) (SF=10)": new Map(),
      "TPC-H (NVMe) (SF=100)": new Map(),
      "TPC-H (S3) (SF=100)": new Map(),
      "TPC-H (NVMe) (SF=1000)": new Map(),
      "TPC-H (S3) (SF=1000)": new Map(),
    };

    let uncategorizable_names = new Set();
    let missing_commits = new Set();

    for (let benchmark_result of data) {
      let commit_id = benchmark_result.commit_id;
      benchmark_result["commit"] = commit_metadata[commit_id];
      if (!benchmark_result["commit"]) {
        missing_commits.add(commit_id);
        benchmark_result["commit"] = commit_metadata[commit_id] = {
          author: { email: "daniel.zidan.king@gmail.com", name: "Dan King" },
          committer: { email: "noreply@github.com", name: "GitHub" },
          id: commit_id,
          message: "!! This commit is missing from commits.json !!",
          timestamp: "1970-01-01T00:00:00Z",
          tree_id: null,
          url: "https://github.com/spiraldb/vortex/commit/" + commit_id,
        };
      }

      let { name, unit, value, commit } = benchmark_result;
      let storage = benchmark_result.storage;
      let dataset = benchmark_result.dataset;
      let group = undefined;
      let group_id = undefined;

      if (dataset !== undefined) {
        if (dataset.tpch !== undefined) {
          let scale_factor = dataset.tpch.scale_factor;
          let nvme = storage === undefined || storage === "nvme";
          if (Number(scale_factor) === 1) {
            group_id = nvme ? "TPC-H (NVMe) (SF=1)" : "TPC-H (S3) (SF=1)";
          } else if (Number(scale_factor) === 10) {
            group_id = nvme ? "TPC-H (NVMe) (SF=10)" : "TPC-H (S3) (SF=10)";
          } else if (Number(scale_factor) === 100) {
            group_id = nvme ? "TPC-H (NVMe) (SF=100)" : "TPC-H (S3) (SF=100)";
          } else if (Number(scale_factor) === 1000) {
            group_id = nvme ? "TPC-H (NVMe) (SF=1000)" : "TPC-H (S3) (SF=1000)";
          } else {
            console.warn("no scale factor found in benchmark");
          }
        } else if (dataset.clickbench !== undefined) {
          group_id = "Clickbench";
        } else {
          console.warn("unknown dataset please implement");
        }
      } else if (name.startsWith("random-access/")) {
        group_id = "Random Access";
      } else if (name.includes("compress time/")) {
        group_id = "Compression";
      } else if (name.startsWith("vortex size/")) {
        if (unit === null || unit === undefined) {
          unit = "bytes"; // Unit information was missing before the commit that adds this comment.
        }
        group_id = "Compression Size";
      } else if (
        name.startsWith("vortex:raw size/") ||
        name.startsWith("vortex:parquet-zstd size/")
      ) {
        if (unit === null || unit === undefined) {
          unit = "ratio"; // The unit becomes the y-axis label.
        }
        group_id = "Compression Size";
      } else if (name.startsWith("tpch_q")) {
        if (storage === undefined || storage === "nvme") {
          group_id = "TPC-H (NVMe) (SF=1)";
        } else {
          group_id = "TPC-H (S3) (SF=1)";
        }
      } else if (name.startsWith("clickbench")) {
        group_id = "Clickbench";
      } else {
        uncategorizable_names.add(name);
        continue;
      }
      group = groups[group_id];

      if (group === undefined) {
        console.warn("cannot find group element in group");
        console.log(group_id)
        continue;
      }

      // Normalize name and units
      let [q, seriesName] = name.split("/");
      if (seriesName.endsWith(" throughput")) {
        seriesName = seriesName.slice(
          0,
          seriesName.length - " throughput".length
        );
        q = q.replace("time", "throughput");
      } else if (seriesName.endsWith("throughput")) {
        seriesName = seriesName.slice(
          0,
          seriesName.length - "throughput".length
        );
        q = q.replace("time", "throughput");
      }

      // Rename old series names to new ones,
      // e.g. vortex-file-compressed -> datafusion:vortex
      // also new series DataFusion:vortex-file-compressed -> datafusion:vortex.
      const renamer = seriesRenameFn?.find((n, v) => n[0] === group_id);
      if (
        renamer !== undefined &&
        renamer[1] !== undefined &&
        renamer[1]["renamedDatasets"] !== undefined
      ) {
        const renameDict = renamer[1]["renamedDatasets"];
        seriesName =
          seriesName in renameDict ? renameDict[seriesName] : seriesName;
      }

      let prettyQ = q
        .replace("_", " ")
        .toUpperCase()
        .replace("VORTEX:RAW SIZE", "VORTEX COMPRESSION RATIO")
        .replace("VORTEX:PARQUET-ZSTD SIZE", "VORTEX:PARQUET-ZSTD SIZE RATIO");
      if (prettyQ.includes("PARQUET-UNC")) {
        return;
      }

      const is_nanos = unit === "ns/iter" || unit === "ns";
      const is_bytes = unit === "bytes";
      const is_throughput = unit === "bytes/ns";

      let sort_position =
        q.slice(0, 4) === "tpch"
          ? parseInt(prettyQ.split(" ")[1].substring(1), 10)
          : 0;



      let arr = group.get(prettyQ);
      if (arr === undefined) {
        group.set(prettyQ, {
          sort_position,
          commits,
          unit: is_nanos
            ? "ms/iter"
            : is_bytes
            ? "MiB"
            : is_throughput
            ? "MiB/s"
            : unit,
          series: new Map(),
        });
        arr = group.get(prettyQ);
      }

      let series = arr.series.get(seriesName);
      if (series === undefined) {
        arr.series.set(seriesName, new Array(commits.length).fill(null));
        series = arr.series.get(seriesName);
      }

      series[commit.sortedIndex] = {
        range: "this was the range",
        value: is_nanos
          ? value / 1_000_000
          : is_bytes
          ? value / 1_048_576
          : is_throughput
          ? (value * 1_000_000_000) / 1_048_576
          : value,
      };
    }

    function sortByPositionThenName(a, b) {
      let position_compare = a[1].sort_position - b[1].sort_position;
      if (position_compare !== 0) {
        return position_compare;
      }
      return a[0].localeCompare(b[0]);
    }

    Object.entries(groups).forEach((pair) => {
      let [name, charts] = pair;
      groups[name] = new Map(
        [...charts.entries()].sort(sortByPositionThenName)
      );
    });

    console.warn(
      "these commits were missing from commits.json so the commit message is missing and the datetime is set to 1970-01-01T00:00:00Z",
      missing_commits
    );
    console.warn(
      "could not categorizes benchmarks with these names, they will not be shown: ",
      uncategorizable_names
    );

    return Object.keys(groups).map((name) => ({
      name,
      dataSet: groups[name],
    }));
  }

  function createChartContainer(name, benchName, index) {
    const container = document.createElement('div');
    container.className = 'chart-container fade-in';
    container.setAttribute('data-benchmark', name);
    container.setAttribute('data-chart', benchName);
    
    const header = document.createElement('div');
    header.className = 'chart-header';
    
    const title = document.createElement('h3');
    title.className = 'chart-title';
    title.textContent = benchName;
    
    const actions = document.createElement('div');
    actions.className = 'chart-actions';
    
    const fullscreenBtn = document.createElement('button');
    fullscreenBtn.className = 'chart-action-btn';
    fullscreenBtn.textContent = 'Fullscreen';
    fullscreenBtn.onclick = () => openChartModal(name, benchName, index);
    
    actions.appendChild(fullscreenBtn);
    header.appendChild(title);
    header.appendChild(actions);
    container.appendChild(header);
    
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${name}-${index}`;
    container.appendChild(canvas);
    
    return { container, canvas };
  }

  function renderChart(
    parent,
    name,
    benchName,
    dataset,
    hiddenDatasets,
    removedDatasets,
    renamedDatasets,
    index
  ) {
    const { container, canvas } = createChartContainer(name, benchName, index);
    parent.appendChild(container);

    const data = {
      labels: dataset.commits.map((commit) => commit.id.slice(0, 7)),
      datasets: Array.from(dataset.series)
        .filter(([name, benches]) => {
          return removedDatasets === undefined || !removedDatasets.has(name);
        })
        .map(([name, benches]) => {
          const renamedName =
            renamedDatasets === undefined
              ? name
              : renamedDatasets[name] || name;
          const color = stringToColor(renamedName);
          return {
            label: renamedName,
            data: benches.map((b) => (b ? b.value : null)),
            borderColor: color,
            backgroundColor: color + "60", // Add alpha for #rrggbbaa
            hidden: hiddenDatasets !== undefined && hiddenDatasets.has(name),
          };
        }),
    };
    
    const y_axis_scale = {
      title: {
        display: true,
        text: dataset.commits.length > 0 ? dataset.unit : "",
      },
      suggestedMin: 0,
    };

    if (
      benchName.includes("COMPRESS") &&
      benchName.includes("THROUGHPUT") &&
      dataset.unit === "MiB/s"
    ) {
      y_axis_scale.suggestedMax = 1024;
      y_axis_scale.max = 1024;
    }

    if (
      benchName.includes("DECOMPRESS") &&
      benchName.includes("THROUGHPUT") &&
      dataset.unit === "MiB/s"
    ) {
      y_axis_scale.suggestedMax = 8192;
      y_axis_scale.max = 8192;
    }

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      spanGaps: true,
      pointStyle: "crossRot",
      elements: {
        line: {
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: benchName,
            padding: { bottom: 50 },
          },
          // By default, show the last 50 commits
          min: Math.max(0, dataset.commits.length - 50),
        },
        y: y_axis_scale,
      },
      plugins: {
        zoom: {
          zoom: {
            wheel: { enabled: true },
            mode: "x",
            drag: { enabled: true },
          },
        },
        legend: {
          display: true,
          onClick: function (e, legendItem) {
            const index = legendItem.datasetIndex;
            const chart = this.chart;
            const dataset = chart.data.datasets[index];
            dataset.hidden = !dataset.hidden;
            chart.update();
          },
        },
      },
    };

    const chart = new Chart(canvas, {
      type: "line",
      data: data,
      options: options,
    });

    const chartKey = `${name}-${index}`;
    state.chartInstances.set(chartKey, { chart, data, options });
    
    return chart;
  }

  function renderBenchSet(name, benchSet, main, toc, groupFilterSettings) {
    const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } =
      groupFilterSettings === undefined
        ? {
            keptCharts: undefined,
            hiddenDatasets: undefined,
            removedDatasets: undefined,
            renamedDatasets: undefined,
          }
        : groupFilterSettings;
    
    // Create collapsible section
    const setElem = document.createElement("div");
    setElem.className = "benchmark-set";
    setElem.setAttribute('data-category', name);
    main.appendChild(setElem);

    const h1id = name.replace(/\s+/g, "_");
    
    // Create header with collapse functionality
    const headerElem = document.createElement('div');
    headerElem.className = 'benchmark-header';
    headerElem.onclick = () => toggleSection(name);
    
    const titleWrapper = document.createElement('div');
    titleWrapper.className = 'title-wrapper';
    
    const nameElem = document.createElement("h1");
    nameElem.id = h1id;
    nameElem.className = "benchmark-title";
    nameElem.innerHTML = `<span class="collapse-icon">▼</span> ${name}`;
    
    const linkBtn = document.createElement('button');
    linkBtn.className = 'group-link-btn';
    linkBtn.setAttribute('aria-label', 'Copy link to this section');
    linkBtn.innerHTML = '🔗';
    linkBtn.onclick = (e) => {
      e.stopPropagation(); // Prevent triggering collapse/expand
      linkToGroup(name);
    };
    
    titleWrapper.appendChild(nameElem);
    titleWrapper.appendChild(linkBtn);
    
    const metaElem = document.createElement('div');
    metaElem.className = 'benchmark-meta';
    const chartCount = keptCharts ? keptCharts.length : (benchSet ? benchSet.size : 0);
    metaElem.textContent = `${chartCount} charts`;
    
    headerElem.appendChild(titleWrapper);
    headerElem.appendChild(metaElem);
    setElem.appendChild(headerElem);
    
    // Add description if available
    const baseCategory = name.split(' (')[0];
    if (state.benchmarkDescriptions[baseCategory]) {
      const descElem = document.createElement('div');
      descElem.className = 'benchmark-description';
      descElem.textContent = state.benchmarkDescriptions[baseCategory];
      setElem.appendChild(descElem);
    }
    
    // Add engine filters for query groups
    const tags = state.categoryTags[name] || [];
    const isQueryGroup = tags.some(tag => tag.includes('Queries'));
    if (isQueryGroup) {
      const filterContainer = document.createElement('div');
      filterContainer.className = 'engine-filter-container';
      
      const filterLabel = document.createElement('span');
      filterLabel.className = 'engine-filter-label';
      filterLabel.textContent = 'Show: ';
      filterContainer.appendChild(filterLabel);
      
      const engines = ['all', 'duckdb', 'datafusion', 'vortex', 'parquet'];
      const engineLabels = {
        'all': 'All',
        'duckdb': 'DuckDB',
        'datafusion': 'DataFusion',
        'vortex': 'Vortex',
        'parquet': 'Parquet'
      };
      
      engines.forEach(engine => {
        const btn = document.createElement('button');
        btn.className = 'engine-filter-btn' + (engine === state.activeEngine ? ' active' : '');
        btn.textContent = engineLabels[engine];
        btn.setAttribute('data-engine', engine);
        btn.setAttribute('data-category', name);
        btn.onclick = () => filterEngineForCategory(name, engine);
        filterContainer.appendChild(btn);
      });
      
      setElem.appendChild(filterContainer);
    }

    // Create TOC entry
    const tocLi = document.createElement("li");
    const tocLink = document.createElement("a");
    tocLink.href = "#" + h1id;
    tocLink.innerHTML = name;
    tocLink.onclick = (e) => {
      e.preventDefault();
      const targetElement = document.getElementById(h1id);
      const headerHeight = document.querySelector('.sticky-header').offsetHeight;
      const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
      const offsetPosition = elementPosition - headerHeight - 20; // 20px extra padding
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
      
      updateActiveNavItem(h1id);
    };
    tocLi.appendChild(tocLink);
    toc.appendChild(tocLi);

    // Don't add categories to dropdown anymore - we use tags instead

    const graphsElem = document.createElement("div");
    graphsElem.className = "benchmark-graphs";
    setElem.appendChild(graphsElem);

    let chartIndex = 0;
    if (keptCharts === undefined) {
      if (benchSet !== undefined) {
        for (const [benchName, benches] of benchSet.entries()) {
          state.charts.push(
            renderChart(
              graphsElem,
              name,
              benchName,
              benches,
              hiddenDatasets,
              removedDatasets,
              renamedDatasets,
              chartIndex++
            )
          );
        }
      }
    } else {
      for (const benchName of keptCharts) {
        const benches = benchSet.get(benchName);
        if (benches) {
          state.charts.push(
            renderChart(
              graphsElem,
              name,
              benchName,
              benches,
              hiddenDatasets,
              removedDatasets,
              renamedDatasets,
              chartIndex++
            )
          );
        }
      }
    }
    
    // Expand by default
    state.expandedSections.add(name);
  }

  function renderAllCharts(dataSets, keptGroups) {
    const main = document.getElementById("main");
    const toc = document.getElementById("toc");
    
    // Clear loading indicator
    main.innerHTML = '';
    
    if (keptGroups === undefined) {
      for (const { name, dataSet } of dataSets) {
        renderBenchSet(name, dataSet, main, toc, undefined);
      }
    } else {
      const dataSetsMap = new Map(
        dataSets.map(({ name, dataSet }) => [name, dataSet])
      );
      for (const [name, groupFilterSettings] of keptGroups) {
        const dataSet = dataSetsMap.get(name);
        renderBenchSet(name, dataSet, main, toc, groupFilterSettings);
      }
    }
    
    // Initialize UI controls
    initializeControls();
    
    // Apply URL parameters after controls are initialized
    initializeFromURL();
  }

  // UI Control Functions
  function toggleSection(name) {
    const section = document.querySelector(`[data-category="${name}"]`);
    if (!section) return;
    
    if (state.expandedSections.has(name)) {
      state.expandedSections.delete(name);
      section.classList.add('collapsed');
    } else {
      state.expandedSections.add(name);
      section.classList.remove('collapsed');
    }
  }

  function expandAll() {
    document.querySelectorAll('.benchmark-set').forEach(section => {
      const category = section.getAttribute('data-category');
      state.expandedSections.add(category);
      section.classList.remove('collapsed');
    });
    updateURLParams({ expanded: 'true' });
  }

  function collapseAll() {
    document.querySelectorAll('.benchmark-set').forEach(section => {
      const category = section.getAttribute('data-category');
      state.expandedSections.delete(category);
      section.classList.add('collapsed');
    });
    updateURLParams({ expanded: 'false' });
  }

  function setView(view) {
    state.currentView = view;
    document.querySelectorAll('.benchmark-graphs').forEach(graphs => {
      if (view === 'list') {
        graphs.classList.add('list-view');
      } else {
        graphs.classList.remove('list-view');
      }
    });
    
    // Update active button
    document.querySelectorAll('.view-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    document.getElementById(`${view}-view`).classList.add('active');
  }

  function filterByTag(tag) {
    state.activeTag = tag;
    
    // Update URL
    updateURLParams({ tag });
    
    // Filter both the main content and navigation items
    document.querySelectorAll('.benchmark-set').forEach(section => {
      const sectionCategory = section.getAttribute('data-category');
      const tags = state.categoryTags[sectionCategory] || [];
      
      if (tag === 'all' || tags.includes(tag)) {
        section.style.display = 'block';
      } else {
        section.style.display = 'none';
      }
    });
    
    // Filter navigation items
    document.querySelectorAll('.toc-list li').forEach(navItem => {
      const link = navItem.querySelector('a');
      if (link) {
        const href = link.getAttribute('href');
        const targetId = href.substring(1); // Remove #
        const targetSection = document.getElementById(targetId);
        
        if (targetSection && targetSection.closest('.benchmark-set')) {
          const sectionCategory = targetSection.closest('.benchmark-set').getAttribute('data-category');
          const tags = state.categoryTags[sectionCategory] || [];
          
          if (tag === 'all' || tags.includes(tag)) {
            navItem.style.display = 'block';
          } else {
            navItem.style.display = 'none';
          }
        }
      }
    });
    
    // Show/hide and update clear filter button
    const clearFilterBtn = document.getElementById('clear-filter');
    if (clearFilterBtn) {
      if (tag === 'all') {
        clearFilterBtn.style.display = 'none';
      } else {
        clearFilterBtn.style.display = 'block';
        clearFilterBtn.textContent = `Clear Filter: ${tag}`;
      }
    }
  }

  function filterBySearch(term) {
    state.searchTerm = term.toLowerCase();
    document.querySelectorAll('.chart-container').forEach(chart => {
      const benchmarkName = chart.getAttribute('data-benchmark').toLowerCase();
      const chartName = chart.getAttribute('data-chart').toLowerCase();
      if (benchmarkName.includes(state.searchTerm) || chartName.includes(state.searchTerm)) {
        chart.style.display = 'block';
      } else {
        chart.style.display = 'none';
      }
    });
  }

  function updateActiveNavItem(id) {
    document.querySelectorAll('.toc-list a').forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${id}`) {
        link.classList.add('active');
      }
    });
  }
  
  function filterEngineForCategory(categoryName, engine) {
    // Update global state
    state.activeEngine = engine;
    
    // Update URL
    updateURLParams({ engine });
    
    // Update all categories with engine filters
    document.querySelectorAll('.engine-filter-container').forEach(container => {
      container.querySelectorAll('.engine-filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-engine') === engine);
      });
    });
    
    // Apply filter to all query categories
    document.querySelectorAll('.benchmark-set').forEach(categorySection => {
      const category = categorySection.getAttribute('data-category');
      const tags = state.categoryTags[category] || [];
      const isQueryGroup = tags.some(tag => tag.includes('Queries'));
      
      if (isQueryGroup) {
        const chartContainers = categorySection.querySelectorAll('.chart-container');
        chartContainers.forEach((container, index) => {
          const chartKey = `${category}-${index}`;
          const chartData = state.chartInstances.get(chartKey);
          
          if (chartData && chartData.chart) {
            const chart = chartData.chart;
            
            // Toggle dataset visibility based on engine filter
            chart.data.datasets.forEach((dataset, datasetIndex) => {
              const label = dataset.label.toLowerCase();
              
              if (engine === 'all') {
                // Show all datasets
                chart.setDatasetVisibility(datasetIndex, true);
              } else {
                // Show only datasets that match the engine
                const shouldShow = label.includes(engine);
                chart.setDatasetVisibility(datasetIndex, shouldShow);
              }
            });
            
            chart.update('none'); // Update without animation for better performance
          }
        });
      }
    });
  }

  function openChartModal(benchmarkName, chartName, index) {
    const modal = document.getElementById('chart-modal');
    const modalCanvas = document.getElementById('modal-chart');
    
    // Get original chart data
    const chartKey = `${benchmarkName}-${index}`;
    const originalChart = state.chartInstances.get(chartKey);
    if (!originalChart) return;
    
    // Clone the chart configuration
    const modalChart = new Chart(modalCanvas, {
      type: 'line',
      data: JSON.parse(JSON.stringify(originalChart.data)),
      options: {
        ...originalChart.options,
        maintainAspectRatio: false,
        responsive: true,
      }
    });
    
    modal.classList.add('active');
    
    // Store modal chart instance for cleanup
    modal.modalChart = modalChart;
  }

  function closeChartModal() {
    const modal = document.getElementById('chart-modal');
    if (modal.modalChart) {
      modal.modalChart.destroy();
      modal.modalChart = null;
    }
    modal.classList.remove('active');
  }

  // URL parameter handling
  function getURLParams() {
    const params = new URLSearchParams(window.location.search);
    return {
      tag: params.get('tag') || 'all',
      engine: params.get('engine') || 'all',
      expanded: params.get('expanded') || 'true',
      group: params.get('group') || null
    };
  }
  
  function updateURLParams(updates) {
    const params = new URLSearchParams(window.location.search);
    
    Object.entries(updates).forEach(([key, value]) => {
      if (value && value !== 'all' && !(key === 'expanded' && value === 'true')) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    });
    
    const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
    window.history.replaceState({}, '', newURL);
  }
  
  function linkToGroup(groupName) {
    // Update URL with group parameter
    updateURLParams({ group: groupName });
    
    // Find the target section for the copy feedback
    const targetSection = document.querySelector(`[data-category="${groupName}"]`);
    
    // Copy URL to clipboard
    navigator.clipboard.writeText(window.location.href).then(() => {
      // Show temporary feedback
      if (targetSection) {
        const linkBtn = targetSection.querySelector('.group-link-btn');
        if (linkBtn) {
          const originalText = linkBtn.innerHTML;
          linkBtn.innerHTML = '✓';
          linkBtn.classList.add('copied');
          setTimeout(() => {
            linkBtn.innerHTML = originalText;
            linkBtn.classList.remove('copied');
          }, 2000);
        }
      }
    });
  }
  
  function focusOnGroup(groupName) {
    // Collapse all sections first
    document.querySelectorAll('.benchmark-set').forEach(section => {
      const category = section.getAttribute('data-category');
      state.expandedSections.delete(category);
      section.classList.add('collapsed');
    });
    
    // Expand only the selected group
    const targetSection = document.querySelector(`[data-category="${groupName}"]`);
    if (targetSection) {
      state.expandedSections.add(groupName);
      targetSection.classList.remove('collapsed');
      
      // Scroll to the section with offset for sticky header
      const targetId = targetSection.querySelector('.benchmark-title').id;
      const targetElement = document.getElementById(targetId);
      const headerHeight = document.querySelector('.sticky-header').offsetHeight;
      const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
      const offsetPosition = elementPosition - headerHeight - 20; // 20px extra padding
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
      
      // Update active nav item
      updateActiveNavItem(targetId);
    }
  }

  function initializeFromURL() {
    const urlParams = getURLParams();
    
    // Set initial state from URL
    state.activeTag = urlParams.tag;
    state.activeEngine = urlParams.engine;
    
    // Apply tag filter
    const categoryFilter = document.getElementById('category-filter');
    if (categoryFilter) {
      categoryFilter.value = urlParams.tag;
      filterByTag(urlParams.tag);
    }
    
    // Apply engine filter
    if (urlParams.engine !== 'all') {
      filterEngineForCategory(null, urlParams.engine);
    }
    
    // Apply expand/collapse state or handle specific group
    if (urlParams.group) {
      // If a specific group is linked, collapse all and expand only that group
      setTimeout(() => {
        focusOnGroup(urlParams.group);
      }, 100); // Small delay to ensure DOM is ready
    } else if (urlParams.expanded === 'false') {
      collapseAll();
    }
  }

  function initializeControls() {
    // Mobile menu toggle
    document.getElementById('menu-toggle').addEventListener('click', () => {
      document.getElementById('sidebar').classList.toggle('active');
    });
    
    document.getElementById('sidebar-close').addEventListener('click', () => {
      document.getElementById('sidebar').classList.remove('active');
    });
    
    // Expand/Collapse controls
    document.getElementById('expand-all').addEventListener('click', expandAll);
    document.getElementById('collapse-all').addEventListener('click', collapseAll);
    
    // View controls
    document.getElementById('grid-view').addEventListener('click', () => setView('grid'));
    document.getElementById('list-view').addEventListener('click', () => setView('list'));
    
    // Tag filter
    document.getElementById('category-filter').addEventListener('change', (e) => {
      filterByTag(e.target.value);
    });
    
    // Clear filter button
    document.getElementById('clear-filter').addEventListener('click', () => {
      document.getElementById('category-filter').value = 'all';
      filterByTag('all');
      updateURLParams({ tag: 'all' });
    });
    
    // Search filter
    let searchTimeout;
    document.getElementById('search-filter').addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => filterBySearch(e.target.value), 300);
    });
    
    // Back to top button
    const backToTop = document.getElementById('back-to-top');
    window.addEventListener('scroll', () => {
      if (window.scrollY > 200) {
        backToTop.classList.add('visible');
      } else {
        backToTop.classList.remove('visible');
      }
      
      // Update active nav item based on scroll position
      const sections = document.querySelectorAll('.benchmark-set');
      let current = '';
      sections.forEach(section => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= 100) {
          current = section.querySelector('.benchmark-title').id;
        }
      });
      if (current) {
        updateActiveNavItem(current);
      }
    });
    
    backToTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    // Modal controls
    document.getElementById('modal-close').addEventListener('click', closeChartModal);
    document.getElementById('chart-modal').addEventListener('click', (e) => {
      if (e.target.id === 'chart-modal') {
        closeChartModal();
      }
    });
    
    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
      const sidebar = document.getElementById('sidebar');
      const menuToggle = document.getElementById('menu-toggle');
      if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        sidebar.classList.remove('active');
      }
    });
  }

  function parse_jsonl(jsonl) {
    return jsonl
      .split("\n")
      .filter((line) => line.trim().length !== 0)
      .map((line) => JSON.parse(line));
  }

  async function fetchAndDecompressGzip(url) {
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

    result += decoder.decode(); // Flush any remaining bytes
    return result;
  }

  // Main initialization function
  return async function initAndRender(keptGroups) {
    try {
      const [dataResponse, commitsResponse] = await Promise.all([
        fetchAndDecompressGzip(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
        ),
        fetch(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json"
        ).then((r) => r.text()),
      ]);

      const data = parse_jsonl(dataResponse);
      const commitsArray = parse_jsonl(commitsResponse);
      
      // Convert commits array to object keyed by commit id
      const commits = {};
      commitsArray.forEach(commit => {
        commits[commit.id] = commit;
      });

      const grouped = downloadAndGroupData(data, commits, keptGroups);
      renderAllCharts(grouped, keptGroups);
    } catch (error) {
      console.error("Failed to load benchmark data:", error);
      document.getElementById("main").innerHTML = `
        <div class="loading-indicator">
          <p style="color: red;">Failed to load benchmark data. Please try refreshing the page.</p>
          <p>${error.message}</p>
        </div>
      `;
    }
  };
})();