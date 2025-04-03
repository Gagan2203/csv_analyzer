// Global variables
let dataInfo = null;
let numericColumns = [];
let categoricalColumns = [];
let geminiConfigured = false;

// DOM elements
const uploadForm = document.getElementById('upload-form');
const fileUpload = document.getElementById('file-upload');
const saveApiKey = document.getElementById('save-api-key');
const geminiApiKey = document.getElementById('gemini-api-key');
const downloadCsv = document.getElementById('download-csv');
const downloadJson = document.getElementById('download-json');
const welcomeMessage = document.getElementById('welcome-message');
const dataAnalysis = document.getElementById('data-analysis');
const errorAlert = document.getElementById('error-alert');
const successAlert = document.getElementById('success-alert');
const loading = document.querySelector('.loading');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // File upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        uploadFile();
    });
    
    // Save API key
    saveApiKey.addEventListener('click', function() {
        saveGeminiApiKey();
    });
    
    // Download buttons
    downloadCsv.addEventListener('click', function() {
        downloadData('csv');
    });
    
    downloadJson.addEventListener('click', function() {
        downloadData('json');
    });
    
    // Analysis type change
    document.getElementById('analysis-type').addEventListener('change', function() {
        const customQuestionContainer = document.getElementById('custom-question-container');
        if (this.value === 'Custom Analysis') {
            customQuestionContainer.style.display = 'block';
        } else {
            customQuestionContainer.style.display = 'none';
        }
    });
    
    // Visualization buttons
    document.getElementById('generate-distribution').addEventListener('click', generateDistributionPlots);
    document.getElementById('generate-correlation').addEventListener('click', generateCorrelationHeatmap);
    document.getElementById('generate-categorical').addEventListener('click', generateCategoricalPlot);
    document.getElementById('generate-scatter').addEventListener('click', generateScatterPlot);
    
    // Statistics buttons
    document.getElementById('generate-descriptive').addEventListener('click', generateDescriptiveStatistics);
    document.getElementById('generate-group-by').addEventListener('click', generateGroupByStatistics);
    document.getElementById('generate-missing-values').addEventListener('click', generateMissingValuesAnalysis);
    document.getElementById('generate-outliers').addEventListener('click', generateOutliersAnalysis);
    
    // ML buttons
    document.getElementById('generate-pca').addEventListener('click', generatePcaAnalysis);
    document.getElementById('generate-clustering').addEventListener('click', generateClusteringAnalysis);
    document.getElementById('generate-regression').addEventListener('click', generateRegressionAnalysis);
    document.getElementById('generate-classification').addEventListener('click', generateClassificationAnalysis);
    
    // Gemini button
    document.getElementById('run-ai-analysis').addEventListener('click', runAiAnalysis);
});

// Helper functions
function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    errorAlert.textContent = message;
    errorAlert.style.display = 'block';
    setTimeout(() => {
        errorAlert.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    successAlert.textContent = message;
    successAlert.style.display = 'block';
    setTimeout(() => {
        successAlert.style.display = 'none';
    }, 5000);
}

// API functions
async function uploadFile() {
    if (!fileUpload.files[0]) {
        showError('Please select a file to upload');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileUpload.files[0]);
    
    showLoading();
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showSuccess(result.message);
            await loadDataInfo();
            welcomeMessage.style.display = 'none';
            dataAnalysis.style.display = 'block';
            downloadCsv.disabled = false;
            downloadJson.disabled = false;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error uploading file: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function loadDataInfo() {
    showLoading();
    
    try {
        const response = await fetch('/api/data_info');
        const result = await response.json();
        
        if (result.status === 'success') {
            dataInfo = result.data_info;
            numericColumns = result.numeric_columns;
            categoricalColumns = result.categorical_columns;
            
            // Update data info table
            const dataInfoTable = document.getElementById('data-info-table');
            dataInfoTable.innerHTML = `
                <tr><th>Filename</th><td>${dataInfo.filename}</td></tr>
                <tr><th>Rows</th><td>${dataInfo.rows}</td></tr>
                <tr><th>Columns</th><td>${dataInfo.columns}</td></tr>
                <tr><th>Memory Usage</th><td>${dataInfo.memory_usage}</td></tr>
                <tr><th>Missing Values</th><td>${dataInfo.missing_values}</td></tr>
            `;
            
            // Update column info table
            const columnInfoTable = document.getElementById('column-info-table').querySelector('tbody');
            columnInfoTable.innerHTML = '';
            
            result.column_info.forEach(col => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${col.Column}</td>
                    <td>${col['Data Type']}</td>
                    <td>${col['Unique Values']}</td>
                    <td>${col['Missing Values']}</td>
                    <td>${col['Missing (%)']}</td>
                `;
                columnInfoTable.appendChild(row);
            });
            
            // Update data preview table
            const dataPreviewTable = document.getElementById('data-preview-table');
            dataPreviewTable.innerHTML = '';
            
            // Add header row
            const headerRow = document.createElement('thead');
            headerRow.innerHTML = '<tr>' + result.preview.columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
            dataPreviewTable.appendChild(headerRow);
            
            // Add data rows
            const tbody = document.createElement('tbody');
            result.preview.data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = row.map(cell => `<td>${cell !== null ? cell : 'null'}</td>`).join('');
                tbody.appendChild(tr);
            });
            dataPreviewTable.appendChild(tbody);
            
            // Update select options for visualizations
            updateSelectOptions();
            
            // Create data types chart
            createDataTypesChart(result.column_info);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error loading data info: ' + error.message);
    } finally {
        hideLoading();
    }
}

function updateSelectOptions() {
    // Update numeric column selects
    const numericSelects = [
        document.getElementById('distribution-column'),
        document.getElementById('scatter-x-column'),
        document.getElementById('scatter-y-column'),
        document.getElementById('agg-column'),
        document.getElementById('regression-target'),
        document.getElementById('pca-columns'),
        document.getElementById('clustering-columns'),
        document.getElementById('regression-features'),
        document.getElementById('classification-features')
    ];
    
    numericSelects.forEach(select => {
        if (select) {
            select.innerHTML = '';
            numericColumns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                select.appendChild(option);
            });
        }
    });
    
    // Update categorical column selects
    const categoricalSelects = [
        document.getElementById('categorical-column'),
        document.getElementById('group-column'),
        document.getElementById('classification-target')
    ];
    
    categoricalSelects.forEach(select => {
        if (select) {
            select.innerHTML = '';
            categoricalColumns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                select.appendChild(option);
            });
        }
    });
    
    // Special case for scatter-y-column (exclude first column)
    const scatterYSelect = document.getElementById('scatter-y-column');
    if (scatterYSelect && numericColumns.length > 1) {
        scatterYSelect.selectedIndex = 1;
    }
}

function createDataTypesChart(columnInfo) {
    const dataTypeCounts = {};
    
    columnInfo.forEach(col => {
        const type = col['Data Type'];
        dataTypeCounts[type] = (dataTypeCounts[type] || 0) + 1;
    });
    
    const types = Object.keys(dataTypeCounts);
    const counts = Object.values(dataTypeCounts);
    
    const ctx = document.createElement('canvas');
    document.getElementById('data-types-chart').innerHTML = '';
    document.getElementById('data-types-chart').appendChild(ctx);
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: types,
            datasets: [{
                data: counts,
                backgroundColor: [
                    '#4CAF50',
                    '#2196F3',
                    '#FFC107',
                    '#F44336',
                    '#9C27B0',
                    '#FF9800'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display: true,
                    text: 'Column Data Types'
                }
            }
        }
    });
}

function saveGeminiApiKey() {
    const apiKey = geminiApiKey.value.trim();
    
    if (!apiKey) {
        showError('Please enter a Gemini API key');
        return;
    }
    
    localStorage.setItem('geminiApiKey', apiKey);
    geminiConfigured = true;
    
    document.getElementById('gemini-not-configured').style.display = 'none';
    document.getElementById('gemini-configured').style.display = 'block';
    
    showSuccess('Gemini API key saved successfully');
}

function downloadData(format) {
    window.location.href = `/api/download/${format}`;
}

// Visualization functions
async function generateDistributionPlots() {
    const column = document.getElementById('distribution-column').value;
    
    if (!column) {
        showError('Please select a column');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/visualization/distribution`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ column })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const histogramContainer = document.getElementById('histogram-container');
            const boxplotContainer = document.getElementById('boxplot-container');
            
            histogramContainer.innerHTML = `<img src="${result.result.histogram}" alt="Histogram">`;
            boxplotContainer.innerHTML = `<img src="${result.result.boxplot}" alt="Box Plot">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating distribution plots: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateCorrelationHeatmap() {
    showLoading();
    
    try {
        const response = await fetch(`/api/visualization/correlation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const correlationContainer = document.getElementById('correlation-container');
            correlationContainer.innerHTML = `<img src="${result.result}" alt="Correlation Heatmap">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating correlation heatmap: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateCategoricalPlot() {
    const column = document.getElementById('categorical-column').value;
    
    if (!column) {
        showError('Please select a column');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/visualization/categorical`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ column })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const categoricalContainer = document.getElementById('categorical-container');
            categoricalContainer.innerHTML = `<img src="${result.result}" alt="Categorical Plot">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating categorical plot: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateScatterPlot() {
    const xColumn = document.getElementById('scatter-x-column').value;
    const yColumn = document.getElementById('scatter-y-column').value;
    
    if (!xColumn || !yColumn) {
        showError('Please select both X and Y columns');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/visualization/scatter`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ x_column: xColumn, y_column: yColumn })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const scatterContainer = document.getElementById('scatter-container');
            scatterContainer.innerHTML = `<img src="${result.result}" alt="Scatter Plot">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating scatter plot: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Statistics functions
async function generateDescriptiveStatistics() {
    showLoading();
    
    try {
        const response = await fetch(`/api/statistics/descriptive`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const descriptiveTable = document.getElementById('descriptive-table');
            
            // Create table header
            let headerHtml = '<thead><tr><th>Statistic</th>';
            const firstRow = result.result[0];
            const columns = Object.keys(firstRow).filter(key => key !== 'index');
            
            columns.forEach(col => {
                headerHtml += `<th>${col}</th>`;
            });
            
            headerHtml += '</tr></thead>';
            
            // Create table body
            let bodyHtml = '<tbody>';
            const metrics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
            
            metrics.forEach(metric => {
                bodyHtml += `<tr><th>${metric}</th>`;
                
                result.result.forEach(row => {
                    bodyHtml += `<td>${row[metric] !== undefined ? row[metric].toFixed(4) : 'N/A'}</td>`;
                });
                
                bodyHtml += '</tr>';
            });
            
            bodyHtml += '</tbody>';
            
            descriptiveTable.innerHTML = headerHtml + bodyHtml;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating descriptive statistics: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateGroupByStatistics() {
    const groupColumn = document.getElementById('group-column').value;
    const aggColumn = document.getElementById('agg-column').value;
    const aggFunc = document.getElementById('agg-function').value;
    
    if (!groupColumn || !aggColumn) {
        showError('Please select both group and aggregate columns');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/statistics/group_by`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                group_column: groupColumn, 
                agg_column: aggColumn,
                agg_func: aggFunc
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Update table
            const groupByTable = document.getElementById('group-by-table');
            
            let html = `
                <thead>
                    <tr>
                        <th>${groupColumn}</th>
                        <th>${aggFunc} of ${aggColumn}</th>
                    </tr>
                </thead>
                <tbody>
            `;
            
            result.result.forEach(row => {
                html += `
                    <tr>
                        <td>${row.group}</td>
                        <td>${typeof row.value === 'number' ? row.value.toFixed(4) : row.value}</td>
                    </tr>
                `;
            });
            
            html += '</tbody>';
            groupByTable.innerHTML = html;
            
            // Request visualization
            const vizResponse = await fetch(`/api/visualization/group_by`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    group_column: groupColumn, 
                    agg_column: aggColumn,
                    agg_func: aggFunc
                })
            });
            
            const vizResult = await vizResponse.json();
            
            if (vizResult.status === 'success') {
                const groupByContainer = document.getElementById('group-by-container');
                groupByContainer.innerHTML = `<img src="${vizResult.result.plot}" alt="Group By Plot">`;
            }
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error generating group by statistics: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateMissingValuesAnalysis() {
    showLoading();
    
    try {
        const response = await fetch(`/api/statistics/missing_values`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const missingValuesTable = document.getElementById('missing-values-table').querySelector('tbody');
            missingValuesTable.innerHTML = '';
            
            result.result.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.column}</td>
                    <td>${row.missing_values}</td>
                    <td>${row.missing_percent.toFixed(2)}%</td>
                `;
                missingValuesTable.appendChild(tr);
            });
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error analyzing missing values: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateOutliersAnalysis() {
    const method = document.getElementById('outlier-method').value;
    
    showLoading();
    
    try {
        const response = await fetch(`/api/statistics/outliers`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ method })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const outliersTable = document.getElementById('outliers-table').querySelector('tbody');
            outliersTable.innerHTML = '';
            
            result.result.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.column}</td>
                    <td>${row.outlier_count}</td>
                    <td>${row.outlier_percent.toFixed(2)}%</td>
                    <td>${row.min_outlier !== null ? row.min_outlier.toFixed(4) : 'N/A'}</td>
                    <td>${row.max_outlier !== null ? row.max_outlier.toFixed(4) : 'N/A'}</td>
                `;
                outliersTable.appendChild(tr);
            });
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error analyzing outliers: ' + error.message);
    } finally {
        hideLoading();
    }
}

// ML functions
async function generatePcaAnalysis() {
    const columnsSelect = document.getElementById('pca-columns');
    const columns = Array.from(columnsSelect.selectedOptions).map(option => option.value);
    const nComponents = parseInt(document.getElementById('pca-components').value);
    
    if (columns.length < 2) {
        showError('Please select at least 2 columns for PCA');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/ml/pca`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                columns,
                n_components: nComponents
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Show results container
            document.getElementById('pca-results').style.display = 'block';
            
            // Update visualizations
            document.getElementById('pca-variance-container').innerHTML = 
                `<img src="${result.result.plots.variance}" alt="PCA Variance">`;
            
            document.getElementById('pca-scatter-container').innerHTML = 
                `<img src="${result.result.plots.scatter}" alt="PCA Scatter">`;
            
            document.getElementById('pca-importance-container').innerHTML = 
                `<img src="${result.result.plots.importance}" alt="PCA Feature Importance">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error performing PCA analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateClusteringAnalysis() {
    const columnsSelect = document.getElementById('clustering-columns');
    const columns = Array.from(columnsSelect.selectedOptions).map(option => option.value);
    const k = parseInt(document.getElementById('clustering-k').value);
    
    if (columns.length < 2) {
        showError('Please select at least 2 columns for clustering');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/ml/clustering`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                columns,
                n_clusters: k
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Show results container
            document.getElementById('clustering-results').style.display = 'block';
            
            // Update cluster centers table
            const centersTable = document.getElementById('cluster-centers-table');
            
            let html = '<thead><tr><th>Cluster</th>';
            columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            for (let i = 0; i < result.result.n_clusters; i++) {
                html += `<tr><th>${i}</th>`;
                for (let j = 0; j < columns.length; j++) {
                    html += `<td>${result.result.cluster_centers[i][j].toFixed(4)}</td>`;
                }
                html += '</tr>';
            }
            
            html += '</tbody>';
            centersTable.innerHTML = html;
            
            // Update visualizations
            document.getElementById('clustering-scatter-container').innerHTML = 
                `<img src="${result.result.plots.scatter}" alt="Clustering Scatter">`;
            
            document.getElementById('clustering-distribution-container').innerHTML = 
                `<img src="${result.result.plots.distribution}" alt="Clustering Distribution">`;
            
            // Update download link
            document.getElementById('download-clustered-data').href = result.result.clustered_data_path;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error performing clustering analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateRegressionAnalysis() {
    const targetColumn = document.getElementById('regression-target').value;
    const featuresSelect = document.getElementById('regression-features');
    const featureColumns = Array.from(featuresSelect.selectedOptions).map(option => option.value);
    const testSize = parseInt(document.getElementById('regression-test-size').value) / 100;
    
    if (!targetColumn) {
        showError('Please select a target column');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/ml/regression`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                target_column: targetColumn,
                feature_columns: featureColumns.length > 0 ? featureColumns : null,
                test_size: testSize
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Show results container
            document.getElementById('regression-results').style.display = 'block';
            
            // Update metrics table
            const metricsTable = document.getElementById('regression-metrics-table');
            
            metricsTable.innerHTML = `
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Mean Squared Error (MSE)</td>
                        <td>${result.result.metrics.mse.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>Root Mean Squared Error (RMSE)</td>
                        <td>${result.result.metrics.rmse.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>R² Score</td>
                        <td>${result.result.metrics.r2.toFixed(4)}</td>
                    </tr>
                </tbody>
            `;
            
            // Update visualizations
            document.getElementById('regression-predictions-container').innerHTML = 
                `<img src="${result.result.plots.predictions}" alt="Actual vs Predicted">`;
            
            document.getElementById('regression-residuals-container').innerHTML = 
                `<img src="${result.result.plots.residuals}" alt="Residuals Plot">`;
            
            document.getElementById('regression-importance-container').innerHTML = 
                `<img src="${result.result.plots.importance}" alt="Feature Importance">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error performing regression analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateClassificationAnalysis() {
    const targetColumn = document.getElementById('classification-target').value;
    const featuresSelect = document.getElementById('classification-features');
    const featureColumns = Array.from(featuresSelect.selectedOptions).map(option => option.value);
    const testSize = parseInt(document.getElementById('classification-test-size').value) / 100;
    
    if (!targetColumn) {
        showError('Please select a target column');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/ml/classification`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                target_column: targetColumn,
                feature_columns: featureColumns.length > 0 ? featureColumns : null,
                test_size: testSize
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Show results container
            document.getElementById('classification-results').style.display = 'block';
            
            // Update accuracy
            document.getElementById('classification-accuracy').innerHTML = 
                `<div class="alert alert-info">Accuracy: ${(result.result.metrics.accuracy * 100).toFixed(2)}%</div>`;
            
            // Update visualizations
            document.getElementById('classification-confusion-matrix-container').innerHTML = 
                `<img src="${result.result.plots.confusion_matrix}" alt="Confusion Matrix">`;
            
            document.getElementById('classification-report-container').innerHTML = 
                `<img src="${result.result.plots.class_report}" alt="Classification Report">`;
            
            document.getElementById('classification-importance-container').innerHTML = 
                `<img src="${result.result.plots.importance}" alt="Feature Importance">`;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error performing classification analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Gemini AI functions
async function runAiAnalysis() {
    const apiKey = geminiApiKey.value.trim();
    const analysisType = document.getElementById('analysis-type').value;
    const customQuestion = document.getElementById('custom-question').value;
    
    if (!apiKey) {
        showError('Please enter a Gemini API key');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/gemini`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                api_key: apiKey,
                analysis_type: analysisType,
                custom_question: analysisType === 'Custom Analysis' ? customQuestion : null
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Show results container
            document.getElementById('ai-analysis-results').style.display = 'block';
            
            // Update content
            document.getElementById('ai-analysis-content').innerHTML = 
                result.result.response.replace(/\n/g, '<br>');
            
            // Update download link
            document.getElementById('download-ai-analysis').href = result.result.file_paths.text;
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Error performing AI analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Check if Gemini API key is saved in localStorage
const savedApiKey = localStorage.getItem('geminiApiKey');
if (savedApiKey) {
    geminiApiKey.value = savedApiKey;
    geminiConfigured = true;
    document.getElementById('gemini-not-configured').style.display = 'none';
    document.getElementById('gemini-configured').style.display = 'block';
}
