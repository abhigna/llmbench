// src/llm_comparison/static/js/main.js

// Global store for fetched data
let allModelsData = {};
let allBenchmarksInfo = {};
let availableBenchmarks = [];

// Global variable to track current sorting state
let currentSort = {
    column: 'weightedScore', // Default sort column
    direction: 'desc'        // Default sort direction
};

// --- Data Loading ---

async function fetchJson(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} for ${url}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Failed to fetch ${url}:`, error);
        return null; // Return null to indicate failure
    }
}

// Fetches all necessary data
async function loadAllData() {
    console.log("Loading all data...");
    const models = await fetchJson('models.json');
    const benchmarksList = await fetchJson('benchmarks.json');

    if (!models || !benchmarksList) {
        showError("Failed to load core model or benchmark list.");
        return;
    }

    availableBenchmarks = benchmarksList; // Store globally

    // Fetch info and latest scores for each benchmark concurrently
    const benchmarkDataPromises = benchmarksList.map(async (bench) => {
        const info = await fetchJson(`benchmarks/${bench.id}/info.json`);
        // Use last_updated date from benchmarks.json to get the correct file
        const scores = await fetchJson(`benchmarks/${bench.id}/${bench.last_updated}.json`);

        if (info && scores) {
            // Store benchmark info globally, keyed by id
            allBenchmarksInfo[bench.id] = { ...info, id: bench.id, last_updated_date: scores.date };
            return { id: bench.id, scores: scores.scores, date: scores.date };
        } else {
            console.warn(`Failed to load data for benchmark: ${bench.id}`);
            allBenchmarksInfo[bench.id] = { name: bench.name, id: bench.id, error: true }; // Mark error
            return null; // Indicate failure for this benchmark
        }
    });

    const benchmarkScoresData = (await Promise.all(benchmarkDataPromises)).filter(Boolean); // Filter out nulls

    // Consolidate data: Map models to their scores across benchmarks
    allModelsData = {};
    for (const modelId in models) {
        allModelsData[modelId] = {
            ...models[modelId], // Copy model base info
            id: modelId,
            scores: {} // Initialize scores object { benchmark_id: { score: value, date: date } }
        };
    }

    // Populate scores for each model
    benchmarkScoresData.forEach(benchmarkResult => {
        const benchId = benchmarkResult.id;
        const benchInfo = allBenchmarksInfo[benchId];
        const primaryMetric = benchInfo.display?.primary_metric;
        const primaryDimension = benchInfo.display?.primary_dimension || 'overall'; // Default dimension

        if (!primaryMetric) {
            console.warn(`Primary metric not defined for benchmark: ${benchId}`);
            return; // Skip if we don't know what score to use
        }

        benchmarkResult.scores.forEach(scoreData => {
            const modelId = scoreData.model_id;
            if (allModelsData[modelId]) {
                const dimensionData = scoreData.dimensions?.[primaryDimension] || {};
                const scoreValue = dimensionData[primaryMetric];

                if (scoreValue !== undefined && scoreValue !== null) {
                    allModelsData[modelId].scores[benchId] = {
                        score: scoreValue,
                        date: benchmarkResult.date // Date the benchmark data file was generated
                    };
                } else {
                     // Store N/A but still include the date if the model was listed but had no score
                     allModelsData[modelId].scores[benchId] = {
                        score: 'N/A',
                        date: benchmarkResult.date
                    };
                }
            } else {
                console.warn(`Model ${modelId} found in benchmark ${benchId} but not in models.json`);
            }
        });
    });

    console.log("Data loaded and consolidated:", allModelsData);
    console.log("Benchmark Info:", allBenchmarksInfo);
    
    updateTable(); // Initial table render
    updateLastUpdatedTimes(); // Update footer/info dates
}


// --- Table Rendering ---

function updateTableHeader(visibleBenchmarks) {
    const headerRow = document.querySelector('#llm-table thead tr');
    if (!headerRow) return;

    // Static columns with sort indicators
    headerRow.innerHTML = `
      <th data-sort="name" class="sortable">Model <span class="sort-indicator"></span></th>
      <th data-sort="weightedScore" class="sortable">Weighted Score <span class="sort-indicator"></span></th>
    `;

    // Dynamic benchmark columns
    visibleBenchmarks.forEach(benchId => {
        const benchInfo = allBenchmarksInfo[benchId];
        const th = document.createElement('th');
        th.textContent = benchInfo?.name || benchId; // Use name, fallback to ID
        th.title = benchInfo?.description || '';     // Add description as tooltip
        th.className = 'sortable';
        th.dataset.sort = `benchmark-${benchId}`;
        th.innerHTML = `${benchInfo?.name || benchId} <span class="sort-indicator"></span>`;
        headerRow.appendChild(th);
    });

    // Add sort event listeners to all sortable headers
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.sort;
            // If clicking the same column, toggle direction, otherwise set to desc
            const direction = (column === currentSort.column) 
                ? (currentSort.direction === 'asc' ? 'desc' : 'asc') 
                : 'desc';
            
            sortTable(column, direction);
        });
    });

    // Update sort indicators based on current sort
    updateSortIndicators();
}

// Update the sort indicators in the table header
function updateSortIndicators() {
    // Clear all indicators first
    document.querySelectorAll('.sort-indicator').forEach(indicator => {
        indicator.textContent = '';
    });
    
    // Set the indicator for the current sort column
    const currentHeader = document.querySelector(`th[data-sort="${currentSort.column}"]`);
    if (currentHeader) {
        const indicator = currentHeader.querySelector('.sort-indicator');
        indicator.textContent = currentSort.direction === 'asc' ? '▲' : '▼';
    }
}

// Sort the table based on column and direction
function sortTable(column, direction) {
    console.log(`Sorting by ${column} in ${direction} order`);
    
    // Update current sort state
    currentSort.column = column;
    currentSort.direction = direction;
    
    // Get all model data as an array
    const modelsArray = Object.values(allModelsData);
    
    // Filter models based on current visible benchmarks
    const visibleBenchmarks = getVisibleBenchmarks();
    const modelsToDisplay = modelsArray.filter(modelData => {
        return visibleBenchmarks.some(benchId => modelData.scores.hasOwnProperty(benchId));
    });
    
    // Sort the filtered models
    modelsToDisplay.sort((a, b) => {
        let valueA, valueB;
        
        if (column === 'name') {
            // Sort by model name
            valueA = a.name?.toLowerCase() || '';
            valueB = b.name?.toLowerCase() || '';
        } 
        else if (column === 'weightedScore') {
            // Sort by weighted score
            valueA = calculateWeightedScore(a, visibleBenchmarks);
            valueB = calculateWeightedScore(b, visibleBenchmarks);
            
            // Handle 'N/A' values
            if (valueA === 'N/A') valueA = -Infinity;
            if (valueB === 'N/A') valueB = -Infinity;
        }
        else if (column.startsWith('benchmark-')) {
            // Sort by specific benchmark score
            const benchId = column.replace('benchmark-', '');
            valueA = a.scores[benchId]?.score;
            valueB = b.scores[benchId]?.score;
            
            // Handle 'N/A' or undefined values
            if (valueA === 'N/A' || valueA === undefined) valueA = -Infinity;
            if (valueB === 'N/A' || valueB === undefined) valueB = -Infinity;
        }
        
        // Apply sort direction
        const modifier = direction === 'asc' ? 1 : -1;
        
        // Handle string comparison
        if (typeof valueA === 'string' && typeof valueB === 'string') {
            return modifier * valueA.localeCompare(valueB);
        }
        
        // Handle numeric comparison
        return modifier * (valueA - valueB);
    });
    
    // Update the table with sorted data
    renderTableRows(modelsToDisplay, visibleBenchmarks);
    
    // Update the sort indicators
    updateSortIndicators();
}

// Helper function to get currently visible benchmarks based on filters
function getVisibleBenchmarks() {
    const selectedTypes = Array.from(document.querySelectorAll('.type-filter:checked'))
        .map(checkbox => checkbox.value);
    const selectAllTypes = selectedTypes.includes('all');

    return Object.keys(allBenchmarksInfo).filter(benchId => {
        const benchInfo = allBenchmarksInfo[benchId];
        return !benchInfo.error && (selectAllTypes || selectedTypes.includes(benchInfo.type));
    });
}


function renderTableRows(modelsToDisplay, visibleBenchmarks) {
    const tableBody = document.getElementById('llm-table-body');
    if (!tableBody) return;
    tableBody.innerHTML = ''; // Clear existing rows

    if (modelsToDisplay.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="99">No models match the current filters.</td></tr>'; // Use colspan 99 as a large enough number
        return;
    }

    modelsToDisplay.forEach(modelData => {
        const row = document.createElement('tr');
        row.dataset.modelId = modelData.id; // Store model ID for potential future use

        // 1. Model Name
        const nameCell = document.createElement('td');
        nameCell.textContent = modelData.name;
        nameCell.classList.add('model-name');
        row.appendChild(nameCell);

        // 2. Weighted Score
        const weightedScore = calculateWeightedScore(modelData, visibleBenchmarks);
        const weightedScoreCell = document.createElement('td');
        weightedScoreCell.textContent = formatScore(weightedScore);
        weightedScoreCell.classList.add('benchmark-score');
        weightedScoreCell.title = 'Simple average of primary scores from visible benchmarks. Not normalized.';
        row.appendChild(weightedScoreCell);

        // 3. Individual Benchmark Scores
        visibleBenchmarks.forEach(benchId => {
            const scoreData = modelData.scores[benchId];
            const scoreCell = document.createElement('td');
            scoreCell.classList.add('benchmark-score');

            if (scoreData && scoreData.score !== 'N/A') {
                scoreCell.innerHTML = `${formatScore(scoreData.score)} <sup class="score-date" title="Benchmark data from ${scoreData.date}">${formatDate(scoreData.date)}</sup>`;
            } else if (scoreData && scoreData.date) {
                 // Model listed in benchmark data but score was null/undefined
                 scoreCell.innerHTML = `N/A <sup class="score-date" title="Benchmark data from ${scoreData.date}">${formatDate(scoreData.date)}</sup>`;
            }
            else {
                scoreCell.textContent = 'N/A';
            }
            row.appendChild(scoreCell);
        });

        tableBody.appendChild(row);
    });
}

// --- Calculations and Formatting ---


// Calculates min and max scores for each benchmark across all models
function calculateBenchmarkRanges() {
    const ranges = {};
    
    // Initialize ranges for each benchmark
    Object.keys(allBenchmarksInfo).forEach(benchId => {
        ranges[benchId] = {
            min: Infinity,
            max: -Infinity
        };
    });
    
    // Find min and max for each benchmark
    Object.values(allModelsData).forEach(modelData => {
        Object.entries(modelData.scores).forEach(([benchId, scoreData]) => {
            if (typeof scoreData.score === 'number') {
                ranges[benchId].min = Math.min(ranges[benchId].min, scoreData.score);
                ranges[benchId].max = Math.max(ranges[benchId].max, scoreData.score);
            }
        });
    });
    
    // Handle case where no scores were found (min still Infinity)
    Object.keys(ranges).forEach(benchId => {
        if (ranges[benchId].min === Infinity) {
            ranges[benchId].min = 0;
            ranges[benchId].max = 1; // Avoid division by zero
        }
        // Handle case where min equals max (would cause division by zero)
        if (ranges[benchId].min === ranges[benchId].max) {
            // If there's only one value, slightly adjust max to avoid division by zero
            ranges[benchId].max += 0.00001;
        }
    });
    
    return ranges;
}

function calculateWeightedScore(modelData, visibleBenchmarks) {
    // Calculate benchmark ranges once
    const benchmarkRanges = calculateBenchmarkRanges();
    
    let totalNormalizedScore = 0;
    let count = 0;

    visibleBenchmarks.forEach(benchId => {
        const scoreData = modelData.scores[benchId];
        // Ensure score exists and is a number for averaging
        if (scoreData && typeof scoreData.score === 'number') {
            const range = benchmarkRanges[benchId];
            
            // Normalize score: (score - min) / (max - min)
            // This scales all scores to range [0, 1]
            const normalizedScore = (scoreData.score - range.min) / (range.max - range.min);
            
            totalNormalizedScore += normalizedScore;
            count++;
        }
    });

    return count > 0 ? totalNormalizedScore / count : 'N/A';
}

// Format score (simple number formatting for now)
function formatScore(score) {
    if (typeof score === 'number') {
        // For normalized weighted scores (always between 0 and 1)
        if (score >= 0 && score <= 1 && !Number.isInteger(score)) {
            return score.toFixed(2);
        }
        // Basic check: if it looks like a percentage or small decimal, use 2 decimal places
        else if (Math.abs(score) <= 1.0 && !Number.isInteger(score) || (Math.abs(score) > 1 && Math.abs(score) <= 100)) {
            return score.toFixed(2);
        }
        else if (Math.abs(score) > 1 && Math.abs(score) < 10) {
            return score.toFixed(2);
        }
        else if (Math.abs(score) > 1000) {
            return score.toLocaleString(undefined, { maximumFractionDigits: 0}); // Large numbers like ELO
        }
        return score.toFixed(1); // Default to 1 decimal place for other numbers
    }
    return score; // Return as is if not a number (e.g., 'N/A')
}


// Format date for superscript display (e.g., 'MM/DD')
function formatDate(dateString) {
    if (!dateString) return '';
    try {
        const date = new Date(dateString + 'T00:00:00Z'); // Assume UTC date if no time provided
        const month = (date.getUTCMonth() + 1).toString().padStart(2, '0');
        const day = date.getUTCDate().toString().padStart(2, '0');
        return `${month}/${day}`;
    } catch (e) {
        console.error("Error formatting date:", dateString, e);
        return ''; // Return empty string on error
    }
}

// --- Filtering ---

// Update the applyFilters function to use sorting
function applyFilters() {
    // 1. Get visible benchmarks based on type filters
    const visibleBenchmarks = getVisibleBenchmarks();

    // 2. Filter models: Show a model if it has a score in *at least one* of the visible benchmarks
    const modelsToDisplay = Object.values(allModelsData).filter(modelData => {
        return visibleBenchmarks.some(benchId => modelData.scores.hasOwnProperty(benchId));
    });

    console.log("Applying filters. Visible benchmarks:", visibleBenchmarks.length, "Models to display:", modelsToDisplay.length);

    // 3. Update the table header based on visible benchmarks
    updateTableHeader(visibleBenchmarks);

    // 4. Sort and render the table rows
    sortTable(currentSort.column, currentSort.direction);

    // 5. Update model counter
    updateModelCounter(modelsToDisplay.length, Object.keys(allModelsData).length);
}

// Add CSS for sortable columns to your page
function addSortingStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .sortable {
            cursor: pointer;
            user-select: none;
        }
        .sortable:hover {
            background-color: #f0f0f0;
        }
        .sort-indicator {
            display: inline-block;
            margin-left: 5px;
            font-size: 0.8em;
        }
    `;
    document.head.appendChild(style);
}


function updateTable() {
    applyFilters();
}

// --- UI Updates ---

function updateModelCounter(visibleCount, totalCount) {
    const counterElem = document.getElementById('model-counter');
    if (counterElem) {
        counterElem.textContent = `Showing ${visibleCount} of ${totalCount} models`;
    }
}

function updateLastUpdatedTimes() {
     const lastUpdatedElem = document.getElementById('last-updated');
     if(lastUpdatedElem) {
         const dates = Object.values(allBenchmarksInfo)
            .map(info => info.last_updated_date)
            .filter(Boolean); // Filter out undefined/null dates

         if (dates.length > 0) {
            // Find the most recent date among all benchmarks
            const mostRecentDate = dates.reduce((latest, current) => {
                return new Date(current) > new Date(latest) ? current : latest;
            });
             lastUpdatedElem.textContent = `Data Sources Updated: Up to ${mostRecentDate}`;
         } else {
            lastUpdatedElem.textContent = `Data Sources Updated: N/A`;
         }
     }
}

function showError(message) {
    const tableBody = document.getElementById('llm-table-body');
    if (tableBody) {
        tableBody.innerHTML = `<tr><td colspan="99" style="color: red; text-align: center;">Error: ${message}</td></tr>`;
    }
     const counterElem = document.getElementById('model-counter');
     if (counterElem) counterElem.textContent = "Error loading data";
}


// --- Event Listeners ---

function setupEventListeners() {
    // Type filters
    document.querySelectorAll('.type-filter').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const isAllCheckbox = checkbox.value === 'all';
            const allCheckbox = document.querySelector('.type-filter[value="all"]');
            const otherCheckboxes = Array.from(document.querySelectorAll('.type-filter:not([value="all"])'));

            if (isAllCheckbox && checkbox.checked) {
                // Check all others
                otherCheckboxes.forEach(cb => cb.checked = true);
            } else if (isAllCheckbox && !checkbox.checked) {
                // Uncheck all others if 'All' is manually unchecked
                 otherCheckboxes.forEach(cb => cb.checked = false);
            } else if (!isAllCheckbox && !checkbox.checked) {
                // If an individual type is unchecked, uncheck 'All'
                allCheckbox.checked = false;
            } else if (!isAllCheckbox && checkbox.checked) {
                 // If an individual type is checked, check 'All' if all others are also checked
                 const allOthersChecked = otherCheckboxes.every(cb => cb.checked);
                 if (allOthersChecked) {
                    allCheckbox.checked = true;
                 }
            }
            applyFilters(); // Re-apply filters whenever a checkbox changes
        });
    });
}


// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded. Initializing...");
    addSortingStyles();      // Add sorting styles
    loadAllData();       // Load data first
    setupEventListeners(); // Then setup listeners
});