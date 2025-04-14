// src/llm_comparison/static/js/main.js

// Function to fetch and display data from JSON files
async function loadModelData(benchmark = 'lmsys') {
    try {
      // Fetch model data, benchmark info, and scores
      const modelsResponse = await fetch('models.json');
      const models = await modelsResponse.json();
      
      const benchmarksResponse = await fetch('benchmarks.json');
      const benchmarks = await benchmarksResponse.json();
      
      // Find the selected benchmark
      const selectedBenchmark = benchmarks.find(b => b.id === benchmark);
      if (!selectedBenchmark) {
        throw new Error(`Benchmark '${benchmark}' not found`);
      }
      
      // Get the benchmark data
      const benchmarkResponse = await fetch(`benchmarks/${benchmark}/${selectedBenchmark.last_updated}.json`);
      const benchmarkData = await benchmarkResponse.json();
      
      // Get the benchmark info
      const benchmarkInfoResponse = await fetch(`benchmarks/${benchmark}/info.json`);
      const benchmarkInfo = await benchmarkInfoResponse.json();
      
      // Clear existing table
      const tableBody = document.getElementById('llm-table-body');
      tableBody.innerHTML = '';
      
      // Update the table header based on benchmark info
      updateTableHeader(benchmarkInfo);
      
      // Populate table with model data + scores
      benchmarkData.scores.forEach(scoreData => {
        const modelId = scoreData.model_id;
        const modelInfo = models[modelId];
        
        if (!modelInfo) {
          console.warn(`Model info not found for ${modelId}`);
          return;
        }
        
        const row = document.createElement('tr');
        row.dataset.modelType = modelInfo.organization;
        
        // Get primary dimension and metric from benchmark info
        const primaryDimension = benchmarkInfo.display?.primary_dimension || 'overall';
        const primaryMetric = benchmarkInfo.display?.primary_metric || Object.keys(benchmarkInfo.metrics)[0].id;
        
        // Get score for primary dimension and metric
        const dimensionData = scoreData.dimensions?.[primaryDimension] || {};
        const score = dimensionData[primaryMetric];
        
        // Build table row
        row.innerHTML = `
          <td class="model-name">${modelInfo.name}</td>
          <td>${modelInfo.organization}</td>
          <td>${modelInfo.release_date || 'N/A'}</td>
          <td>${modelInfo.context_window ? modelInfo.context_window.toLocaleString() : 'N/A'}</td>
          <td class="benchmark-score">${score !== undefined ? formatScore(score) : 'N/A'}</td>
          <td>${scoreData.rank || 'N/A'}</td>
        `;
        
        tableBody.appendChild(row);
      });
      
      // Apply filters
      applyFilters();
      
      // Update benchmark title and description
      const benchmarkTitle = document.getElementById('benchmark-title');
      if (benchmarkTitle) {
        benchmarkTitle.textContent = selectedBenchmark.name;
      }
      
      const benchmarkDesc = document.getElementById('benchmark-description');
      if (benchmarkDesc) {
        benchmarkDesc.textContent = selectedBenchmark.description;
      }
      
      // Update last updated text
      const lastUpdated = new Date(benchmarkData.date);
      const lastUpdatedElem = document.getElementById('last-updated');
      if (lastUpdatedElem) {
        lastUpdatedElem.textContent = `Last updated: ${lastUpdated.toLocaleString()}`;
      }
      
    } catch (error) {
      console.error('Error loading data:', error);
      const tableBody = document.getElementById('llm-table-body');
      tableBody.innerHTML = `<tr><td colspan="6">Error loading data: ${error.message}</td></tr>`;
    }
  }
  
  // Update table header based on benchmark info
  function updateTableHeader(benchmarkInfo) {
    const headerRow = document.querySelector('#llm-table thead tr');
    if (!headerRow) return;
    
    // Get primary metric name
    const primaryMetricId = benchmarkInfo.display?.primary_metric || Object.keys(benchmarkInfo.metrics)[0];
    const primaryMetric = benchmarkInfo.metrics.find(m => m.id === primaryMetricId);
    const metricName = primaryMetric ? primaryMetric.name : 'Score';
    
    // Set standard headers
    headerRow.innerHTML = `
      <th>Model</th>
      <th>Organization</th>
      <th>Release Date</th>
      <th>Context Window</th>
      <th>${metricName}</th>
      <th>Rank</th>
    `;
  }
  
  // Format score based on its value
  function formatScore(score) {
    if (typeof score === 'number') {
      return score.toFixed(2);
    }
    return score;
  }
  
  // Function to apply filters
  function applyFilters() {
    const minYear = document.getElementById('min-year').value || '2000';
    const selectedOrgs = Array.from(document.querySelectorAll('.org-filter:checked'))
      .map(checkbox => checkbox.value);
      
    const allSelected = selectedOrgs.includes('all');
    
    // Filter the table rows
    const rows = document.querySelectorAll('#llm-table-body tr');
    rows.forEach(row => {
      const org = row.dataset.modelType;
      const releaseCell = row.querySelector('td:nth-child(3)');
      const releaseYear = releaseCell.textContent.split('-')[0] || '2000';
      
      const orgMatch = allSelected || selectedOrgs.some(type => 
        org === type && type !== 'all');
      const yearMatch = releaseYear >= minYear;
      
      row.style.display = (orgMatch && yearMatch) ? '' : 'none';
    });
    
    // Update count of displayed models
    const visibleRows = document.querySelectorAll('#llm-table-body tr:not([style*="display: none"])').length;
    const totalRows = rows.length;
    const counterElem = document.getElementById('model-counter');
    if (counterElem) {
      counterElem.textContent = `Showing ${visibleRows} of ${totalRows} models`;
    }
  }
  
  // Set up event listeners once the DOM is loaded
  document.addEventListener('DOMContentLoaded', () => {
    // Fetch available benchmarks and populate the selector
    fetch('benchmarks.json')
      .then(response => response.json())
      .then(benchmarks => {
        const selector = document.getElementById('benchmark-selector');
        if (selector) {
          selector.innerHTML = '';
          benchmarks.forEach(benchmark => {
            const option = document.createElement('option');
            option.value = benchmark.id;
            option.textContent = benchmark.name;
            selector.appendChild(option);
          });
          
          // Initial data load with the first benchmark
          if (benchmarks.length > 0) {
            loadModelData(benchmarks[0].id);
          }
        }
      })
      .catch(error => console.error('Error loading benchmarks:', error));
    
    // Handle organization filters
    document.querySelectorAll('.org-filter').forEach(checkbox => {
      checkbox.addEventListener('change', () => {
        // Handle the "All" checkbox
        if (checkbox.value === 'all' && checkbox.checked) {
          document.querySelectorAll('.org-filter').forEach(cb => {
            if (cb.value !== 'all') cb.checked = true;
          });
        } else if (checkbox.value === 'all' && !checkbox.checked) {
          document.querySelectorAll('.org-filter').forEach(cb => {
            if (cb.value !== 'all') cb.checked = false;
          });
        }
        
        // Update "All" checkbox state based on other checkboxes
        const allCheckbox = document.querySelector('.org-filter[value="all"]');
        const otherCheckboxes = Array.from(document.querySelectorAll('.org-filter:not([value="all"])'));
        const allOthersChecked = otherCheckboxes.every(cb => cb.checked);
        const noneChecked = otherCheckboxes.every(cb => !cb.checked);
        
        if (allOthersChecked) {
          allCheckbox.checked = true;
        } else if (noneChecked) {
          allCheckbox.checked = false;
        }
        
        applyFilters();
      });
    });
    
    // Handle year filter
    document.getElementById('min-year').addEventListener('input', applyFilters);
    
    // Handle benchmark selector
    document.getElementById('benchmark-selector').addEventListener('change', (e) => {
      loadModelData(e.target.value);
    });
  });