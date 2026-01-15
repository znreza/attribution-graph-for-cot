#!/usr/bin/env python3
"""
Feature Concept Analyzer for Attribution Graphs

This script analyzes features from attribution graphs by:
1. Extracting all features from a graph JSON file
2. Fetching detailed information from Neuronpedia API
3. Analyzing top activations to understand true feature behavior
4. Generating a comprehensive report with verification

Usage:
    python analyze_graph_features.py <graph_file.json>

Example:
    python analyze_graph_features.py graphs/dallas-austin-subgraph-top1-E.json
"""

import json
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle


@dataclass
class FeatureInfo:
    """Information about a feature from the graph and Neuronpedia"""
    node_id: str
    layer: int
    feat_idx: int
    position: int
    token: str
    activation: float
    influence: float

    # From Neuronpedia
    concept: str
    top_activations: List[str]
    max_activation_value: float

    # Analysis
    seems_relevant: bool
    relevance_reason: str


class FeatureAnalyzer:
    """Analyzes features from attribution graphs using Neuronpedia API"""

    def __init__(self, model_id: str = "gemma-2-2b", cache_file: str = "feature_cache.pkl"):
        self.model_id = model_id
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cached feature data to avoid repeated API calls"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save feature cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def parse_node_id(self, node_id: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Parse node_id to extract layer, feature index, and position"""
        parts = node_id.split('_')
        if len(parts) == 3:
            try:
                return int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                pass
        return None, None, None

    def fetch_feature_data(self, layer: int, feat_idx: int) -> Optional[Dict]:
        """Fetch feature data from Neuronpedia API with caching"""
        cache_key = f"{self.model_id}_{layer}_{feat_idx}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Fetch from API
        sae_id = f"{layer}-gemmascope-transcoder-16k"
        url = f"https://www.neuronpedia.org/api/feature/{self.model_id}/{sae_id}/{feat_idx}"

        try:
            time.sleep(0.5)  # Rate limiting
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = data
                self._save_cache()
                return data
            else:
                print(f"  ⚠ Error {response.status_code} for L{layer} F{feat_idx}")
                return None

        except Exception as e:
            print(f"  ⚠ Request failed for L{layer} F{feat_idx}: {e}")
            return None

    def extract_top_activations(self, feature_data: Dict, n: int = 5) -> Tuple[List[str], float]:
        """Extract top activation examples from feature data"""
        activations = feature_data.get('activations', [])
        examples = []
        max_val = 0

        for act in activations[:n]:
            tokens = act.get('tokens', [])
            max_idx = act.get('maxActivation', 0)
            max_val = max(max_val, act.get('values', [0])[max_idx] if act.get('values') else 0)

            # Get context around max activation
            start = max(0, max_idx - 3)
            end = min(len(tokens), max_idx + 4)
            context_tokens = tokens[start:end]

            # Highlight max activation token
            text_parts = []
            for j, tok in enumerate(context_tokens):
                if start + j == max_idx:
                    text_parts.append(f"**{tok}**")
                else:
                    text_parts.append(tok)

            examples.append(''.join(text_parts))

        return examples, max_val

    def get_concept_description(self, feature_data: Dict) -> str:
        """Extract concept description from feature data"""
        explanations = feature_data.get('explanations', [])
        if explanations:
            return explanations[0].get('description', 'No description')
        return "No description available"

    def analyze_relevance(self, token: str, concept: str, top_activations: List[str],
                         position: int, total_positions: int) -> Tuple[bool, str]:
        """
        Analyze if the feature seems relevant to the token/context.
        Returns (is_relevant, reason)
        """
        token_clean = token.lower().strip()
        concept_clean = concept.lower()

        # Direct token match in concept
        if token_clean and token_clean in concept_clean:
            return True, f"Concept mentions '{token}'"

        # Check for keyword matches
        keyword_matches = {
            'the': ['article', 'the', 'determiner'],
            'capital': ['capital', 'capitalized', 'uppercase', 'letter'],
            'dallas': ['place', 'location', 'city', 'proper noun', 'geographic'],
            'is': ['is', 'verb', 'auxiliary', 'copula'],
            'containing': ['contain', 'attach', 'belong', 'include'],
            'state': ['state', 'jurisdiction', 'government', 'region', 'sport'],
            'of': ['of', 'possessive', 'relation'],
        }

        for key, keywords in keyword_matches.items():
            if key in token_clean:
                if any(kw in concept_clean for kw in keywords):
                    return True, f"Concept related to '{token}': {concept}"

        # Check if activations contain similar patterns
        activations_text = ' '.join(top_activations).lower()
        if token_clean and token_clean in activations_text:
            return True, f"Top activations contain '{token}'"

        # Check for polysemantic place name features
        # Look for capitalized proper nouns in activations
        place_indicators = ['scottsdale', 'phoenix', 'austin', 'boston', 'chicago',
                          'android', 'java', 'python']  # Including code that looks like places
        if any(indicator in activations_text for indicator in place_indicators):
            if token_clean in ['capital', 'dallas', 'state', 'of']:
                return True, f"Polysemantic: activations show place names/capitalized terms relevant to geography"

        # Last position features might be predictive
        if position == total_positions - 1:
            predictive_keywords = ['place', 'location', 'city', 'proper noun', 'geographic',
                                  'name', 'state', 'capital', 'legal', 'court', 'county',
                                  'born', 'pioneer', 'settlement']
            if any(kw in concept_clean for kw in predictive_keywords):
                return True, "Predictive feature for location/descriptive output"

        # Polysemantic capitalization features
        if token.strip() and token.strip()[0].isupper():
            if any(kw in concept_clean for kw in ['capital', 'proper', 'name', 'code']):
                return True, f"Polysemantic: capitalized words (token '{token}' is capitalized)"

        # Pattern detection (e.g., "capital of")
        if any(kw in concept_clean for kw in ['capital', 'place', 'location']):
            return True, "Possible contextual pattern detection"

        # Check for "of" with capital/place concepts
        if token_clean == 'of':
            if any(kw in concept_clean for kw in ['capital', 'place', 'government', 'institute']):
                return True, f"Pattern detector for 'capital/place of' constructions"

        return False, "No clear relevance"

    def analyze_graph(self, graph_path: Path) -> List[FeatureInfo]:
        """Analyze all features in a graph file"""
        print(f"Loading graph from {graph_path}...")

        with open(graph_path, 'r') as f:
            graph_data = json.load(f)

        metadata = graph_data.get('metadata', {})
        prompt_tokens = metadata.get('prompt_tokens', [])
        nodes = graph_data.get('nodes', [])

        print(f"Prompt: {metadata.get('prompt', 'N/A')}")
        print(f"Tokens: {prompt_tokens}")
        print(f"Total nodes: {len(nodes)}")
        print(f"\nAnalyzing features...\n")

        features = []
        transcoder_nodes = [n for n in nodes if n.get('feature_type') == 'cross layer transcoder']

        for i, node in enumerate(transcoder_nodes, 1):
            node_id = node['node_id']
            layer, feat_idx, position = self.parse_node_id(node_id)

            if layer is None:
                continue

            token = prompt_tokens[position] if position < len(prompt_tokens) else "???"
            activation = node.get('activation', 0)
            influence = node.get('influence', 0)

            print(f"[{i}/{len(transcoder_nodes)}] Fetching L{layer} F{feat_idx} (pos {position}: '{token}')...")

            # Fetch from Neuronpedia
            feature_data = self.fetch_feature_data(layer, feat_idx)

            if feature_data:
                concept = self.get_concept_description(feature_data)
                top_acts, max_val = self.extract_top_activations(feature_data)

                is_relevant, reason = self.analyze_relevance(
                    token, concept, top_acts, position, len(prompt_tokens)
                )

                feature_info = FeatureInfo(
                    node_id=node_id,
                    layer=layer,
                    feat_idx=feat_idx,
                    position=position,
                    token=token,
                    activation=activation,
                    influence=influence,
                    concept=concept,
                    top_activations=top_acts,
                    max_activation_value=max_val,
                    seems_relevant=is_relevant,
                    relevance_reason=reason
                )

                features.append(feature_info)
            else:
                print(f"  ⚠ Failed to fetch data")

        return features

    def generate_report(self, features: List[FeatureInfo], output_file: Path):
        """Generate a comprehensive analysis report"""

        # Group by position
        by_position = {}
        for feat in features:
            if feat.position not in by_position:
                by_position[feat.position] = []
            by_position[feat.position].append(feat)

        # Generate report
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("FEATURE CONCEPT ANALYSIS REPORT")
        report_lines.append("=" * 100)
        report_lines.append("")

        # Summary statistics
        total = len(features)
        relevant = sum(1 for f in features if f.seems_relevant)
        suspicious = total - relevant

        report_lines.append("SUMMARY")
        report_lines.append("-" * 100)
        report_lines.append(f"Total features analyzed: {total}")
        report_lines.append(f"Seems relevant: {relevant} ({relevant/total*100:.1f}%)")
        report_lines.append(f"Suspicious: {suspicious} ({suspicious/total*100:.1f}%)")
        report_lines.append("")

        # Detailed analysis by position
        report_lines.append("=" * 100)
        report_lines.append("DETAILED ANALYSIS BY TOKEN POSITION")
        report_lines.append("=" * 100)
        report_lines.append("")

        for pos in sorted(by_position.keys()):
            feats = sorted(by_position[pos], key=lambda x: x.layer)
            token = feats[0].token if feats else "???"

            report_lines.append("=" * 100)
            report_lines.append(f"Position {pos}: Token = '{token}'")
            report_lines.append("=" * 100)
            report_lines.append("")

            for feat in feats:
                status = "✅ RELEVANT" if feat.seems_relevant else "❓ SUSPICIOUS"

                report_lines.append(f"{status}")
                report_lines.append(f"Node ID: {feat.node_id}")
                report_lines.append(f"Layer {feat.layer}, Feature {feat.feat_idx}")
                report_lines.append(f"Activation: {feat.activation:.3f} | Influence: {feat.influence:.3f}")
                report_lines.append(f"")
                report_lines.append(f"Neuronpedia Concept:")
                report_lines.append(f"  {feat.concept}")
                report_lines.append(f"")
                report_lines.append(f"Top Activating Examples:")
                for i, example in enumerate(feat.top_activations, 1):
                    report_lines.append(f"  {i}. ...{example}...")
                report_lines.append(f"")
                report_lines.append(f"Analysis:")
                report_lines.append(f"  {feat.relevance_reason}")
                report_lines.append(f"")
                report_lines.append(f"Verify at: https://www.neuronpedia.org/{self.model_id}/{feat.layer}-gemmascope-transcoder-16k/{feat.feat_idx}")
                report_lines.append("")
                report_lines.append("-" * 100)
                report_lines.append("")

        # Write report
        report_text = "\n".join(report_lines)

        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"\n{'='*100}")
        print(f"Report saved to: {output_file}")
        print(f"{'='*100}")

        # Also save as JSON for programmatic access
        json_output = output_file.with_suffix('.json')
        with open(json_output, 'w') as f:
            json.dump([asdict(feat) for feat in features], f, indent=2)

        print(f"JSON data saved to: {json_output}")

        return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Analyze feature concepts from attribution graphs using Neuronpedia'
    )
    parser.add_argument(
        'graph_file',
        type=str,
        help='Path to graph JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemma-2-2b',
        help='Model ID (default: gemma-2-2b)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output report file (default: <graph_name>_analysis.txt)'
    )
    parser.add_argument(
        '--cache',
        type=str,
        default='feature_cache.pkl',
        help='Cache file for API responses (default: feature_cache.pkl)'
    )

    args = parser.parse_args()

    # Setup paths
    graph_path = Path(args.graph_file)
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = graph_path.parent / f"{graph_path.stem}_analysis.txt"

    # Run analysis
    analyzer = FeatureAnalyzer(model_id=args.model, cache_file=args.cache)
    features = analyzer.analyze_graph(graph_path)

    if not features:
        print("No features found to analyze!")
        return 1

    # Generate report
    analyzer.generate_report(features, output_path)

    # Print summary to console
    print(f"\nAnalyzed {len(features)} features")
    print(f"Relevant: {sum(1 for f in features if f.seems_relevant)}")
    print(f"Suspicious: {sum(1 for f in features if not f.seems_relevant)}")

    return 0


if __name__ == '__main__':
    exit(main())
