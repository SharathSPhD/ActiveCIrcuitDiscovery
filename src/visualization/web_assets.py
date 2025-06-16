# CSS and JavaScript assets for web visualization

CSS_STYLES = """
/* ActiveCircuitDiscovery Web Visualizer Styles */

.main-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f8f9fa;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header-title {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
}

.header-subtitle {
    margin: 5px 0;
    font-size: 16px;
    opacity: 0.9;
}

.header-info {
    margin: 5px 0 0 0;
    font-size: 14px;
    opacity: 0.8;
}

.controls-panel {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
}

.control-group {
    min-width: 200px;
}

.control-group label {
    font-weight: 500;
    color: #333;
    margin-bottom: 5px;
    display: block;
}

.button-group {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.control-button {
    background: #667eea;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.3s;
}

.control-button:hover {
    background: #5a6fd8;
}

.graph-container {
    background: white;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.info-container {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
}

.info-panel {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.info-panel h3 {
    margin-top: 0;
    color: #333;
    border-bottom: 2px solid #667eea;
    padding-bottom: 10px;
}

/* Responsive design */
@media (max-width: 1200px) {
    .info-container {
        grid-template-columns: 1fr 1fr;
    }
}

@media (max-width: 768px) {
    .info-container {
        grid-template-columns: 1fr;
    }
    
    .controls-panel {
        flex-direction: column;
        align-items: stretch;
    }
    
    .control-group {
        min-width: auto;
    }
}

/* Plotly customizations */
.js-plotly-plot .plotly .main-svg {
    border-radius: 5px;
}

/* Animation for loading states */
.loading {
    opacity: 0.7;
    transition: opacity 0.3s;
}

/* Node selection highlighting */
.selected-node {
    stroke: #ff6b6b !important;
    stroke-width: 3px !important;
}

/* Annotation styles */
.annotation-item {
    background: #f1f3f4;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    border-left: 4px solid #667eea;
}

.annotation-timestamp {
    font-size: 12px;
    color: #666;
    margin-top: 5px;
}
"""

def get_external_stylesheets():
    """Get external stylesheet URLs."""
    return [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap'
    ]

def inject_custom_css(app):
    """Inject custom CSS into Dash app."""
    app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {CSS_STYLES}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''