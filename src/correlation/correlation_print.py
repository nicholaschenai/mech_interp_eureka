"""
Functions to print correlation results from correlation analyzer in a readable format.
"""
from typing import List, Dict, Tuple

from .correlation_analyzer import CorrelationAnalyzer


def pretty_print_correlated_neurons(
    analyzer: CorrelationAnalyzer,
    layer_name: str,
    feature_keys: List[str],
    threshold: float = 0.7,
    top_n: int = 10,
    show_negative: bool = True
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Pretty print neurons that correlate with specific features.

    Args:
        analyzer: CorrelationAnalyzer instance
        layer_name: Name of the layer to analyze
        feature_keys: List of features to analyze
        threshold: Correlation threshold
        top_n: Maximum number of neurons to show per feature
        show_negative: Whether to distinguish between positive and negative correlations
    """
    if not feature_keys:
        print("No features specified.")
        return

    # Get the total number of neurons in the layer
    total_neurons = analyzer.dataset.get_activation_matrix(layer_name).shape[1]

    # Gather correlations for each feature
    feature_correlations = {}
    neuron_counts = {}
    unique_neurons = set()

    for feature in feature_keys:
        correlated_neurons = analyzer.find_correlated_neurons(
            layer_name, feature, threshold
        )
        feature_correlations[feature] = correlated_neurons
        neuron_counts[feature] = len(correlated_neurons)
        unique_neurons.update([n[0] for n in correlated_neurons])

    if not feature_correlations:
        print("No correlated neurons found.")
        return

    print(f"\n{'=' * 70}")
    print(f"FEATURE-CORRELATED NEURONS SUMMARY (LAYER: {layer_name}, TOTAL NEURONS: {total_neurons})")
    print(f"{'=' * 70}")

    # Print correlations by feature
    for feature, correlations in feature_correlations.items():
        # Skip if no correlations found
        if not correlations:
            continue

        # Format feature name
        feature_name = feature.replace('_', ' ').title()

        # Calculate percentage of total neurons
        percentage = (len(correlations) / total_neurons) * 100

        # Print header for feature
        print(f"\n{feature_name.upper()}: {len(correlations)} neurons ({percentage:.1f}% of layer) with |correlation| >= {threshold}")
        print(f"{'-' * 70}")

        # Limit to top_n
        correlations_to_display = correlations[:top_n]

        if not correlations_to_display:
            print("  No neurons met the criteria.")
            continue

        # Format as a table
        print(f"  {'Neuron ID':<10} {'Correlation':<12} {'Bar'}")
        print(f"  {'-'*9:<10} {'-'*11:<12} {'-'*30}")

        # Print each neuron with a visual bar representing correlation
        for idx, corr in correlations_to_display:
            # Scale bar appropriately
            bar_length = min(30, int(abs(corr) * 30))

            # Use different symbols for positive and negative correlations
            if show_negative and corr < 0:
                bar = '▒' * bar_length  # Negative correlation
                corr_str = f"{corr:<12.3f}"
            else:
                bar = '█' * bar_length  # Positive correlation
                corr_str = f"{corr:<12.3f}"

            print(f"  {idx:<10} {corr_str} {bar}")

        # Note if there are more correlations than displayed
        if len(correlations) > top_n:
            print(f"  ... and {len(correlations) - top_n} more neurons")

    # Summary section
    print(f"\n{'=' * 70}")
    print(f"CORRELATION SUMMARY:")

    # Sort features by neuron count (descending)
    sorted_features = sorted(neuron_counts.keys(), key=lambda f: neuron_counts[f], reverse=True)

    for feature in sorted_features:
        percentage = (neuron_counts[feature] / total_neurons) * 100
        print(f"  {feature.replace('_', ' ').title()}: {neuron_counts[feature]} neurons ({percentage:.1f}% of layer)")

    # Calculate unique neuron percentage
    unique_percentage = (len(unique_neurons) / total_neurons) * 100
    print(f"\nTotal unique neurons correlated with at least one feature: {len(unique_neurons)} ({unique_percentage:.1f}% of layer)")
    print(f"{'=' * 70}\n")
    return feature_correlations


def pretty_print_phase_neurons(phase_neurons, top_n=10, min_selectivity=None):
    """
    Pretty print phase-selective neurons in a readable format.

    Args:
        phase_neurons: Dictionary from identify_all_phase_neurons
        top_n: Maximum number of neurons to show per phase
        min_selectivity: Optional minimum selectivity to display
    """
    if not phase_neurons:
        print("No phase-selective neurons found.")
        return

    total_neurons = phase_neurons['total_neurons']
    phase_counts = {phase: len(neurons) for phase, neurons in phase_neurons.items() if phase != 'total_neurons'}

    print(f"\n{'=' * 60}")
    print(f"PHASE-SELECTIVE NEURONS SUMMARY")
    print(f"{'=' * 60}")

    for phase, neurons in phase_neurons.items():
        if phase == 'total_neurons':
            continue
        # Filter by minimum selectivity if specified
        if min_selectivity is not None:
            neurons = {idx: score for idx, score in neurons.items() if score >= min_selectivity}

        # Get total count before limiting display
        total_count = len(neurons)

        # Limit to top_n neurons
        neurons_to_display = dict(list(neurons.items())[:top_n])

        print(f"\n{phase.upper()} PHASE: {total_count} neurons")
        print(f"{'-' * 40}")

        if not neurons_to_display:
            print("  No neurons met the criteria.")
            continue

        # Format as a table
        print(f"  {'Neuron ID':<10} {'Selectivity':<12} {'Bar'}")
        print(f"  {'-'*9:<10} {'-'*11:<12} {'-'*30}")

        # Print each neuron with a visual bar representing selectivity
        for idx, score in neurons_to_display.items():
            bar_length = min(30, int(score * 6))  # Scale bar appropriately
            bar = '█' * bar_length
            print(f"  {idx:<10} {score:<12.3f} {bar}")

        if total_count > top_n:
            print(f"  ... and {total_count - top_n} more neurons")

    print(f"\n{'=' * 60}")
    print(f"TOTAL NEURONS BY PHASE:")
    for phase, count in phase_counts.items():
        percentage = count / total_neurons * 100
        print(f"  {phase.upper()}: {count} neurons ({percentage:.1f}% of layer)")

    print(f"{'=' * 60}\n")
