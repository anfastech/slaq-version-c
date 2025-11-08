// Analysis Charts with Chart.js

function initAnalysisChart(data) {
    const ctx = document.getElementById('metricsChart');
    
    if (!ctx) {
        console.error('Canvas element not found');
        return;
    }
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mismatch %', 'Confidence', 'CTC Loss', 'Stutter Freq', 'Total Events'],
            datasets: [{
                label: 'Analysis Metrics',
                data: [
                    data.mismatchPercentage,
                    data.confidenceScore * 100, // Scale to percentage
                    data.ctcLoss * 100, // Scale for visibility
                    data.stutterFrequency,
                    data.totalEvents
                ],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',   // Red for mismatch
                    'rgba(59, 130, 246, 0.7)',  // Blue for confidence
                    'rgba(168, 85, 247, 0.7)',  // Purple for CTC loss
                    'rgba(249, 115, 22, 0.7)',  // Orange for frequency
                    'rgba(0, 144, 80, 0.7)'     // Brand green for events
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(59, 130, 246, 1)',
                    'rgba(168, 85, 247, 1)',
                    'rgba(249, 115, 22, 1)',
                    'rgba(0, 144, 80, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            let value = context.parsed.y;
                            
                            // Format based on metric type
                            if (label === 'Mismatch %') {
                                return `Mismatch: ${value.toFixed(1)}%`;
                            } else if (label === 'Confidence') {
                                return `Confidence: ${(value / 100).toFixed(2)}`;
                            } else if (label === 'CTC Loss') {
                                return `CTC Loss: ${(value / 100).toFixed(4)}`;
                            } else if (label === 'Stutter Freq') {
                                return `Frequency: ${value.toFixed(1)}/min`;
                            } else if (label === 'Total Events') {
                                return `Events: ${value}`;
                            }
                            
                            return `${label}: ${value}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Value'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Metrics'
                    }
                }
            }
        }
    });
}

// Progress Chart for Dashboard (trend over time)
function initProgressChart(trendData) {
    const ctx = document.getElementById('progressChart');
    
    if (!ctx) {
        return;
    }
    
    // Extract dates and mismatch percentages
    const labels = trendData.map(item => {
        const date = new Date(item.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const mismatchData = trendData.map(item => item.mismatch_percentage);
    const frequencyData = trendData.map(item => item.stutter_frequency);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Mismatch %',
                    data: mismatchData,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Stutter Frequency',
                    data: frequencyData,
                    borderColor: 'rgba(0, 144, 80, 1)',
                    backgroundColor: 'rgba(0, 144, 80, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Progress Over Time'
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Mismatch %'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Frequency (per min)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Severity Distribution Pie Chart
function initSeverityChart(distribution) {
    const ctx = document.getElementById('severityChart');
    
    if (!ctx) {
        return;
    }
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['None', 'Mild', 'Moderate', 'Severe'],
            datasets: [{
                data: [
                    distribution.none || 0,
                    distribution.mild || 0,
                    distribution.moderate || 0,
                    distribution.severe || 0
                ],
                backgroundColor: [
                    'rgba(34, 197, 94, 0.7)',   // Green
                    'rgba(250, 204, 21, 0.7)',  // Yellow
                    'rgba(249, 115, 22, 0.7)',  // Orange
                    'rgba(239, 68, 68, 0.7)'    // Red
                ],
                borderColor: [
                    'rgba(34, 197, 94, 1)',
                    'rgba(250, 204, 21, 1)',
                    'rgba(249, 115, 22, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Severity Distribution'
                }
            }
        }
    });
}

// Event Types Bar Chart
function initEventTypesChart(eventTypes) {
    const ctx = document.getElementById('eventTypesChart');
    
    if (!ctx) {
        return;
    }
    
    const labels = Object.keys(eventTypes);
    const data = Object.values(eventTypes);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                label: 'Count',
                data: data,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.7)',  // Blue
                    'rgba(168, 85, 247, 0.7)',  // Purple
                    'rgba(239, 68, 68, 0.7)',   // Red
                    'rgba(34, 197, 94, 0.7)'    // Green
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(168, 85, 247, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(34, 197, 94, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Stutter Event Types'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}
