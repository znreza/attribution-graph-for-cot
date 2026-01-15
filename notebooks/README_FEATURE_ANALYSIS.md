# Feature Concept Analysis Tool

This tool analyzes features from attribution graphs by fetching detailed information from Neuronpedia and verifying whether the automated concept descriptions match the actual feature behavior in your specific prompt.

## Features

- **Automatic API Fetching**: Retrieves feature data from Neuronpedia API
- **Smart Caching**: Caches API responses to avoid repeated requests
- **Activation Analysis**: Examines top activating examples for each feature
- **Relevance Detection**: Analyzes whether features are relevant to your prompt
- **Comprehensive Reports**: Generates detailed text and JSON reports

## Installation

No additional dependencies beyond standard Python libraries:
```bash
pip install requests
```

## Usage

### Basic Usage

```bash
python analyze_graph_features.py graphs/dallas-austin-subgraph-top1-E.json
```

This will:
1. Load the graph file
2. Extract all cross-layer transcoder features
3. Fetch concept descriptions and top activations from Neuronpedia
4. Analyze relevance to each token in your prompt
5. Generate a comprehensive report

### Output Files

The script generates two files:
- `<graph_name>_analysis.txt` - Human-readable detailed report
- `<graph_name>_analysis.json` - Machine-readable data

### Advanced Options

```bash
# Specify custom output file
python analyze_graph_features.py graphs/my_graph.json --output my_report.txt

# Use different model
python analyze_graph_features.py graphs/my_graph.json --model gemma-2-9b

# Specify cache file location
python analyze_graph_features.py graphs/my_graph.json --cache .cache/features.pkl
```

### Command-Line Arguments

- `graph_file` (required): Path to your graph JSON file
- `--model`: Model ID for Neuronpedia (default: `gemma-2-2b`)
- `--output`: Custom output file path
- `--cache`: Cache file for API responses (default: `feature_cache.pkl`)

## Understanding the Output

### Report Sections

#### 1. Summary Statistics
Shows how many features were analyzed and how many seem relevant vs suspicious.

#### 2. Detailed Analysis by Token Position
For each token in your prompt:
- Lists all features that activate at that position
- Shows the Neuronpedia concept description
- Displays top activating examples from the SAE dataset
- Provides relevance analysis

### Status Indicators

- **✅ RELEVANT**: Feature appears to be correctly related to the token/context
- **❓ SUSPICIOUS**: Feature's concept doesn't obviously match the token

### Relevance Analysis

The tool checks:
1. **Direct matches**: Does the concept mention the token?
2. **Keyword matches**: Are related keywords present?
3. **Activation patterns**: Do top activations contain similar text?
4. **Positional context**: Is this a predictive feature at the end of the prompt?
5. **Polysemanticity**: Could this be a polysemantic feature?

## Example Output

```
================================================================================
Position 2: Token = ' capital'
================================================================================

✅ RELEVANT
Node ID: 16_7171_2
Layer 16, Feature 7171
Activation: 45.000 | Influence: 0.800

Neuronpedia Concept:
  capital

Top Activating Examples:
  1. ...**capital**▁of...
  2. ...**Capital**▁Markets...
  3. ...**capital**▁city...

Analysis:
  Concept mentions 'capital'

Verify at: https://www.neuronpedia.org/gemma-2-2b/16-gemmascope-transcoder-16k/7171
```

## Understanding Polysemantic Features

Many features are **polysemantic** - they activate on multiple related (or sometimes unrelated) concepts. For example:

- A "code" feature might activate on: code syntax + capitalized words + structured text
- A "legal" feature might activate on: legal text + jurisdiction + place names

The automated Neuronpedia descriptions are based on the most common activations and may not capture all aspects of what a feature does.

## Tips for Verification

1. **Check high-activation features first**: Features with activation > 10.0 are more likely to be primary concepts
2. **Visit Neuronpedia links**: Manually inspect the top activations for suspicious features
3. **Consider context**: Features at the last token position are often predictive
4. **Look for patterns**: Multiple "suspicious" features might share a common pattern

## Caching

The script caches API responses to `feature_cache.pkl` by default. This:
- Speeds up repeated analysis
- Reduces API load
- Allows offline re-analysis

To clear the cache, simply delete the cache file.

## Programmatic Usage

You can also import and use the analyzer in your own scripts:

```python
from analyze_graph_features import FeatureAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = FeatureAnalyzer(model_id="gemma-2-2b")

# Analyze a graph
features = analyzer.analyze_graph(Path("graphs/my_graph.json"))

# Generate report
analyzer.generate_report(features, Path("my_report.txt"))

# Access individual features
for feat in features:
    if feat.seems_relevant:
        print(f"L{feat.layer} F{feat.feat_idx}: {feat.concept}")
```

## Troubleshooting

### API Rate Limiting
The script includes a 0.5-second delay between requests. If you hit rate limits, the requests will fail but won't crash the script.

### Missing Features
Some features might not be available on Neuronpedia. These will show a warning and be skipped.

### Large Graphs
For graphs with many features (>50), the analysis may take several minutes due to API requests.

## Example: Analyzing the Dallas-Austin Graph

```bash
cd /path/to/notebooks
python analyze_graph_features.py graphs/dallas-austin-subgraph-top1-E.json
```

Expected output:
- `dallas-austin-subgraph-top1-E_analysis.txt` - Detailed report
- `dallas-austin-subgraph-top1-E_analysis.json` - JSON data
- Console summary of relevant vs suspicious features

The report will show that most geography-related features (capital, Dallas, place names) are correctly identified, while some "code" features are polysemantic and activate on capitalized words.
