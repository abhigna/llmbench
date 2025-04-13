// src/llm_comparison/static/js/main.js

// Function to fetch and display data from JSON files
async function loadModelData(benchmark = 'lmsys') {
    try {
        // Fetch model data and benchmark scores
        const modelsResponse = await fetch(`models.json`);
        const models = await modelsResponse.json();
        
        const benchmarkResponse = await fetch(`${benchmark}.json`);
        const benchmarkData = await benchmarkResponse.json();
        
        // Clear existing table
        const tableBody = document.getElementById('llm-table-body');
        tableBody.innerHTML = '';
        
        // Populate table with model data + scores
        models.forEach(model => {
            // Get the model's score for the selected benchmark
            const score = benchmarkData[model.id] || 'N/A';
            
            const row = document.createElement('tr');
            row.dataset.modelType = model.developer;
            
            row.innerHTML = `
                <td class="model-name">${model.name}</td>
                <td>${model.pricePerMillion ? '$' + model.pricePerMillion.toFixed(2) : 'N/A'}</td>
                <td>${model.priceInput ? '$' + model.priceInput.toFixed(4) : 'N/A'}</td>
                <td>${model.priceOutput ? '$' + model.priceOutput.toFixed(4) : 'N/A'}</td>
                <td>${model.contextWindow.toLocaleString()}</td>
                <td>${model.releaseDate}</td>
                <td>${model.developer}</td>
                <td class="benchmark-score">${score !== 'N/A' ? score.toFixed(2) : 'N/A'}</td>
                <td><a href="${model.apiLink}" class="affiliate-link" target="_blank">API Docs</a></td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Apply filters
        applyFilters();
        
        // Update last updated text if available
        if (benchmarkData._last_updated) {
            const lastUpdated = new Date(benchmarkData._last_updated);
            const lastUpdatedElem = document.getElementById('last-updated');
            if (lastUpdatedElem) {
                lastUpdatedElem.textContent = `Last updated: ${lastUpdated.toLocaleString()}`;
            }
        }
        
    } catch (error) {
        console.error('Error loading data:', error);
        const tableBody = document.getElementById('llm-table-body');
        tableBody.innerHTML = `<tr><td colspan="9">Error loading data: ${error.message}</td></tr>`;
    }
}

// Function to apply filters
function applyFilters() {
    const minPrice = parseFloat(document.getElementById('min-price').value) || 0;
    const maxPrice = parseFloat(document.getElementById('max-price').value) || Infinity;
    
    const selectedModelTypes = Array.from(document.querySelectorAll('.model-filter:checked'))
        .map(checkbox => checkbox.value);
        
    const allSelected = selectedModelTypes.includes('all');
    
    // Filter the table rows
    const rows = document.querySelectorAll('#llm-table-body tr');
    rows.forEach(row => {
        const modelType = row.dataset.modelType;
        const priceCell = row.querySelector('td:nth-child(2)');
        const price = parseFloat(priceCell.textContent.replace('$', '')) || 0;
        
        const modelTypeMatch = allSelected || selectedModelTypes.some(type => 
            modelType.includes(type) && type !== 'all');
        const priceMatch = price >= minPrice && price <= maxPrice;
        
        row.style.display = (modelTypeMatch && priceMatch) ? '' : 'none';
    });
    
    // Update count of displayed models
    const visibleRows = document.querySelectorAll('#llm-table-body tr[style=""]').length;
    const totalRows = rows.length;
    const counterElem = document.getElementById('model-counter');
    if (counterElem) {
        counterElem.textContent = `Showing ${visibleRows} of ${totalRows} models`;
    }
}

// Set up event listeners once the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Handle filter checkboxes
    document.querySelectorAll('.model-filter').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            // Handle the "All" checkbox
            if (checkbox.value === 'all' && checkbox.checked) {
                document.querySelectorAll('.model-filter').forEach(cb => {
                    if (cb.value !== 'all') cb.checked = true;
                });
            } else if (checkbox.value === 'all' && !checkbox.checked) {
                document.querySelectorAll('.model-filter').forEach(cb => {
                    if (cb.value !== 'all') cb.checked = false;
                });
            }
            
            // Update "All" checkbox state based on other checkboxes
            const allCheckbox = document.querySelector('.model-filter[value="all"]');
            const otherCheckboxes = Array.from(document.querySelectorAll('.model-filter:not([value="all"])'));
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
    
    // Handle price filters
    document.getElementById('min-price').addEventListener('input', applyFilters);
    document.getElementById('max-price').addEventListener('input', applyFilters);
    
    // Handle benchmark selector
    document.getElementById('benchmark-selector').addEventListener('change', (e) => {
        loadModelData(e.target.value);
    });
    
    // Initial data load
    loadModelData();
});