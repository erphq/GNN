"""
Core analysis functions for process mining with fixed warnings.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

from processmine.process_mining.conformance import ConformanceChecker

logger = logging.getLogger(__name__)

def analyze_bottlenecks(df, freq_threshold=5, percentile_threshold=90.0):
    """
    Identify bottlenecks in the process
    
    Args:
        df: Process data dataframe
        freq_threshold: Minimum frequency threshold for transitions
        percentile_threshold: Percentile threshold for bottlenecks
        
    Returns:
        Tuple of (bottleneck_stats, significant_bottlenecks)
    """
    try:
        # Create transition pairs (current activity -> next activity)
        transitions = df.copy()
        
        # Make sure we have consistent timezone handling
        # Convert both timestamp columns to naive datetime (remove timezone info)
        if pd.api.types.is_datetime64_dtype(transitions["timestamp"]):
            transitions["timestamp"] = transitions["timestamp"].dt.tz_localize(None)
        
        if "next_timestamp" in transitions.columns and pd.api.types.is_datetime64_dtype(transitions["next_timestamp"]):
            transitions["next_timestamp"] = transitions["next_timestamp"].dt.tz_localize(None)
        elif "timestamp" in transitions.columns:
            # Create next_timestamp by shifting timestamp
            transitions["next_timestamp"] = transitions.groupby("case_id")["timestamp"].shift(-1)
            
        # Calculate wait time in seconds between activities
        transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
        
        # Convert to hours
        transitions["wait_hours"] = transitions["wait_sec"] / 3600.0
        
        # Filter out negative wait times (shouldn't happen in a proper event log)
        transitions = transitions[transitions["wait_hours"] >= 0]
        
        # Group by transition
        transition_stats = transitions.groupby(["task_id", "next_task"]).agg({
            "wait_hours": ["count", "mean", "median", "std", "min", "max"],
            "case_id": "nunique"
        })
        
        # Flatten multi-level columns
        transition_stats.columns = ["_".join(col).strip() for col in transition_stats.columns.values]
        
        # Reset index to make grouping variables into columns
        transition_stats = transition_stats.reset_index()
        
        # Rename to match expected column names for visualizations
        transition_stats = transition_stats.rename(columns={
            "next_task": "next_task_id",
            "wait_hours_count": "freq",
            "wait_hours_mean": "mean_hours",
            "wait_hours_median": "median_hours",
            "wait_hours_std": "std_hours",
            "wait_hours_min": "min_hours",
            "wait_hours_max": "max_hours",
            "case_id_nunique": "unique_cases"
        })
        
        # Add count column (alias for freq) for compatibility
        transition_stats["count"] = transition_stats["freq"]
        
        # Filter transitions with at least freq_threshold occurrences
        transition_stats = transition_stats[transition_stats["freq"] >= freq_threshold]
        
        # Identify bottlenecks based on percentile threshold
        percentile_value = transition_stats["mean_hours"].quantile(percentile_threshold / 100.0)
        bottlenecks = transition_stats[transition_stats["mean_hours"] > percentile_value].copy()
        
        # Sort by mean wait time (descending)
        bottlenecks = bottlenecks.sort_values("mean_hours", ascending=False)
        
        return transition_stats, bottlenecks
    
    except Exception as e:
        import traceback
        logger.error(f"Error in analyze_bottlenecks: {e}")
        traceback.print_exc()
        # Return empty dataframes in case of error
        empty_stats = pd.DataFrame(columns=["task_id", "next_task_id", "freq", "count", "mean_hours", "median_hours", 
                                         "std_hours", "min_hours", "max_hours", "unique_cases"])
        return empty_stats, empty_stats.copy()
    
def analyze_cycle_times(df):
    """
    Analyze cycle times for cases
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (case_stats, long_cases, p95)
    """
    try:
        # Ensure we have consistent timezone handling
        if pd.api.types.is_datetime64_dtype(df["timestamp"]):
            timestamp_col = df["timestamp"].dt.tz_localize(None)
        else:
            timestamp_col = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        
        # Calculate case statistics - use strings instead of functions
        case_stats = df.groupby("case_id", observed=False).agg({
            timestamp_col.name: ["min", "max"],  # Using strings instead of functions
            "task_name": "nunique",
            "resource_id": "nunique",
            "task_id": "count"
        })
        
        # Flatten multi-level columns
        case_stats.columns = ["_".join(col).strip() for col in case_stats.columns.values]
        
        # Rename columns
        case_stats = case_stats.rename(columns={
            f"{timestamp_col.name}_min": "start_time",
            f"{timestamp_col.name}_max": "end_time",
            "task_name_nunique": "unique_activities",
            "resource_id_nunique": "unique_resources",
            "task_id_count": "num_events"
        })
        
        # Calculate duration
        case_stats["duration"] = case_stats["end_time"] - case_stats["start_time"]
        case_stats["duration_h"] = case_stats["duration"] / pd.Timedelta(hours=1)
        
        # Calculate percentiles
        p95 = case_stats["duration_h"].quantile(0.95)
        
        # Identify long-running cases (above 95th percentile)
        long_cases = case_stats[case_stats["duration_h"] > p95].copy()
        
        return case_stats, long_cases, p95
    
    except Exception as e:
        logger.error(f"Error in analyze_cycle_times: {e}")
        # Return empty dataframes in case of error
        empty_stats = pd.DataFrame(columns=["start_time", "end_time", "unique_activities", 
                                         "unique_resources", "num_events", "duration", "duration_h"])
        return empty_stats, empty_stats.copy(), 0

def analyze_transition_patterns(df):
    """
    Analyze transition patterns between activities
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (transitions, transition_counts, probability_matrix)
    """
    try:
        # Create transitions dataframe
        transitions = df.copy()
        
        # Add next activity for each case
        transitions["next_task_id"] = transitions.groupby("case_id", observed=False)["task_id"].shift(-1)
        transitions["next_task_name"] = transitions.groupby("case_id", observed=False)["task_name"].shift(-1)
        
        # Remove last event in each case (no next activity)
        transitions = transitions.dropna(subset=["next_task_id"])
        
        # Count transitions
        transition_counts = transitions.groupby(["task_name", "next_task_name"], observed=False).size().reset_index(name="count")
        
        # Calculate transition probabilities
        total_per_source = transitions.groupby("task_name", observed=False).size()
        probability_matrix = transition_counts.copy()
        
        # Avoid division by zero
        probability_matrix["probability"] = probability_matrix.apply(
            lambda row: row["count"] / total_per_source[row["task_name"]] 
                if total_per_source[row["task_name"]] > 0 else 0.0, 
            axis=1
        )
        
        return transitions, transition_counts, probability_matrix
    
    except Exception as e:
        logger.error(f"Error in analyze_transition_patterns: {e}")
        # Return empty dataframes in case of error
        empty_transitions = pd.DataFrame(columns=["task_name", "next_task_name", "task_id", "next_task_id"])
        empty_counts = pd.DataFrame(columns=["task_name", "next_task_name", "count"])
        empty_probs = pd.DataFrame(columns=["task_name", "next_task_name", "count", "probability"])
        return empty_transitions, empty_counts, empty_probs

def identify_process_variants(df, max_variants=10):
    """
    Identify and analyze process variants
    
    Args:
        df: Process data dataframe
        max_variants: Maximum number of variants to return
        
    Returns:
        Tuple of (variant_stats, variant_sequences)
    """
    try:
        # Sort data by case ID and timestamp
        sorted_df = df.sort_values(["case_id", "timestamp"])
        
        # Extract variant sequences
        variant_sequences = {}
        for case_id, case_df in sorted_df.groupby("case_id", observed=False):
            # Create sequence of task names
            sequence = tuple(case_df["task_name"].values)
            variant_sequences[case_id] = sequence
        
        # Count variants
        variant_counts = {}
        for sequence in variant_sequences.values():
            variant_counts[sequence] = variant_counts.get(sequence, 0) + 1
        
        # Sort by frequency
        sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top variants
        top_variants = sorted_variants[:max_variants]
        
        # Calculate percentage
        total_cases = len(variant_sequences)
        variant_stats = pd.DataFrame(columns=["variant", "count", "percentage"])
        
        for i, (sequence, count) in enumerate(top_variants):
            variant_stats.loc[i] = {
                "variant": " -> ".join(sequence),
                "count": count,
                "percentage": count / total_cases * 100
            }
        
        # Add cumulative percentage
        variant_stats["cumulative_pct"] = variant_stats["percentage"].cumsum()
        
        return variant_stats, variant_sequences
    
    except Exception as e:
        logger.error(f"Error in identify_process_variants: {e}")
        # Return empty dataframes in case of error
        empty_stats = pd.DataFrame(columns=["variant", "count", "percentage", "cumulative_pct"])
        return empty_stats, {}

def analyze_resource_workload(df):
    """
    Analyze resource workload
    
    Args:
        df: Process data dataframe
        
    Returns:
        Resource statistics dataframe
    """
    try:
        # Count activities per resource
        resource_counts = df.groupby("resource_id", observed=False)["task_id"].count().reset_index(name="num_events")
        
        # Add percentage of total activities
        total_events = df.shape[0]
        resource_counts["percentage"] = resource_counts["num_events"] / total_events * 100
        
        # Get unique cases per resource
        unique_cases = df.groupby("resource_id", observed=False)["case_id"].nunique().reset_index(name="num_cases")
        resource_counts = resource_counts.merge(unique_cases, on="resource_id")
        
        # Get unique activities per resource
        unique_activities = df.groupby("resource_id", observed=False)["task_id"].nunique().reset_index(name="num_activities")
        resource_counts = resource_counts.merge(unique_activities, on="resource_id")
        
        # Calculate Gini coefficient to measure workload distribution
        sorted_counts = resource_counts["num_events"].sort_values().values
        n = len(sorted_counts)
        if n > 0:
            gini_coefficient = 2 * np.sum(np.arange(1, n+1) * sorted_counts) / (n * np.sum(sorted_counts)) - (n+1)/n
            resource_counts.attrs["gini_coefficient"] = float(gini_coefficient)
        
        # Sort by event count (descending)
        resource_counts = resource_counts.sort_values("num_events", ascending=False)
        
        return resource_counts
    
    except Exception as e:
        logger.error(f"Error in analyze_resource_workload: {e}")
        # Return empty dataframe in case of error
        empty_stats = pd.DataFrame(columns=["resource_id", "num_events", "percentage", "num_cases", "num_activities"])
        empty_stats.attrs["gini_coefficient"] = 0.0
        return empty_stats

def check_conformance(df):
    """
    Perform conformance checking on the process
    
    Args:
        df: Process data dataframe
        
    Returns:
        Dictionary of conformance results
    """
    try:
        # Create conformance checker
        checker = ConformanceChecker(df)
        
        # Run conformance checking
        results = checker.check_conformance()
        
        return results
    
    except Exception as e:
        logger.error(f"Error in check_conformance: {e}")
        # Return basic error result in case of failure
        return {
            "error": str(e),
            "conformance_ratio": 0.0,
            "conforming_cases": 0,
            "non_conforming_cases": 0
        }