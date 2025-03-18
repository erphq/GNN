"""
Efficient conformance checking for process mining with graceful fallback
when PM4Py is not available.
"""

import pandas as pd
import numpy as np
import time
import logging
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# Check for PM4Py availability once
PM4PY_AVAILABLE = False
try:
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.conformance.alignments import algorithm as alignments
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    PM4PY_AVAILABLE = True
except ImportError:
    logger.info("PM4Py not available. Using simplified conformance checking.")


class ViolationType(Enum):
    """Types of conformance violations"""
    WRONG_ACTIVITY = "wrong_activity"
    SKIPPED_ACTIVITY = "skipped_activity"
    DUPLICATE_ACTIVITY = "duplicate_activity"
    WRONG_SEQUENCE = "wrong_sequence"
    INCOMPLETE_CASE = "incomplete_case"
    INVALID_TRANSITION = "invalid_transition"
    OTHER = "other"


class ConformanceChecker:
    """Efficient conformance checker with graceful fallback"""
    
    def __init__(self, df: pd.DataFrame, use_token_replay: bool = True, use_alignments: bool = False):
        """
        Initialize conformance checker
        
        Args:
            df: Process dataframe
            use_token_replay: Whether to use token replay (if PM4Py available)
            use_alignments: Whether to use alignments (more accurate but slower)
        """
        self.df = df
        self.use_token_replay = use_token_replay
        self.use_alignments = use_alignments
        self.pm4py_available = PM4PY_AVAILABLE
        
        # Initialize results containers
        self.process_model = None
        self.conformance_results = None
        self.conforming_cases = []
        self.non_conforming_cases = []
        self.violations = pd.DataFrame()
        
        # Find column mappings once during initialization
        self.col_map = self._discover_column_mapping()
    
    def _discover_column_mapping(self) -> Dict[str, str]:
        """
        Discover column mapping from dataframe to standard names
        
        Returns:
            Dictionary mapping standard column names to actual column names
        """
        col_map = {}
        
        # Map for case ID
        for std_col, candidates in [
            ('case_id', ['case_id', 'case:concept:name', 'case:id', 'caseid', 'case']),
            ('task_name', ['task_name', 'concept:name', 'activity', 'event', 'task_id', 'task']),
            ('timestamp', ['timestamp', 'time:timestamp', 'time', 'date', 'datetime'])
        ]:
            for candidate in candidates:
                if candidate in self.df.columns:
                    col_map[std_col] = candidate
                    break
        
        # For PM4Py specific mapping
        if self.pm4py_available:
            pm4py_map = {}
            if 'case_id' in col_map:
                pm4py_map['case:concept:name'] = col_map['case_id']
            if 'task_name' in col_map:
                pm4py_map['concept:name'] = col_map['task_name']
            if 'timestamp' in col_map:
                pm4py_map['time:timestamp'] = col_map['timestamp']
            col_map.update(pm4py_map)
        
        return col_map

    def check_conformance(self) -> Dict[str, Any]:
        """
        Check conformance using most appropriate method
        
        Returns:
            Dictionary of conformance results
        """
        start_time = time.time()
        
        try:
            if self.pm4py_available:
                result = self._check_with_pm4py()
            else:
                result = self._check_simplified()
                
            # Add execution time
            result['execution_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Conformance checking error: {e}")
            # Return basic error result
            return {
                "error": str(e),
                "conformance_ratio": 0.0,
                "execution_time": time.time() - start_time
            }
    
    def _prepare_pm4py_dataframe(self) -> pd.DataFrame:
        """
        Prepare dataframe for PM4Py efficiently
        
        Returns:
            Formatted dataframe
        """
        # Create a copy only if we need to modify the dataframe
        needs_copy = False
        for pm4py_col, df_col in [
            ('case:concept:name', self.col_map.get('case_id')),
            ('concept:name', self.col_map.get('task_name')),
            ('time:timestamp', self.col_map.get('timestamp'))
        ]:
            if df_col and pm4py_col != df_col:
                needs_copy = True
                break
                
        df_pm = self.df.copy() if needs_copy else self.df
        
        # Rename columns for PM4Py if needed
        if needs_copy:
            rename_map = {}
            for pm4py_col, df_col in [
                ('case:concept:name', self.col_map.get('case_id')),
                ('concept:name', self.col_map.get('task_name')),
                ('time:timestamp', self.col_map.get('timestamp'))
            ]:
                if df_col and pm4py_col != df_col:
                    rename_map[df_col] = pm4py_col
            
            if rename_map:
                df_pm = df_pm.rename(columns=rename_map)
        
        # Ensure timestamp is properly formatted
        if 'time:timestamp' in df_pm.columns:
            if not pd.api.types.is_datetime64_dtype(df_pm['time:timestamp']):
                df_pm['time:timestamp'] = pd.to_datetime(df_pm['time:timestamp'])
                
            # Ensure timestamps are timezone-naive
            if hasattr(df_pm['time:timestamp'].dtype, 'tz') and df_pm['time:timestamp'].dtype.tz is not None:
                df_pm['time:timestamp'] = df_pm['time:timestamp'].dt.tz_localize(None)
        
        # Process with PM4Py functions
        df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
        
        return df_pm
    
    def _check_with_pm4py(self) -> Dict[str, Any]:
        """
        Efficient implementation of PM4Py-based conformance checking
        
        Returns:
            Dictionary of conformance results
        """
        try:
            # Prepare dataframe for PM4Py
            df_pm = self._prepare_pm4py_dataframe()
            
            # Convert to event log
            event_log = log_converter.apply(df_pm)
            
            # Discover process model with inductive miner
            process_tree = inductive_miner.apply(event_log)
            self.process_model = process_tree
            
            # Convert to Petri net
            net, im, fm = pt_converter.apply(process_tree)
            
            # Perform conformance checking
            if self.use_alignments:
                logger.info("Using alignment-based conformance checking")
                self.conformance_results = alignments.apply(event_log, net, im, fm)
                return self._process_results(is_alignment=True)
            else:
                logger.info("Using token replay-based conformance checking")
                self.conformance_results = token_replay.apply(event_log, net, im, fm)
                return self._process_results(is_alignment=False)
                
        except Exception as e:
            logger.warning(f"PM4Py conformance checking failed: {e}. Falling back to simplified method.")
            return self._check_simplified()
    
    def _process_results(self, is_alignment: bool = False) -> Dict[str, Any]:
        """
        Process conformance checking results efficiently
        
        Args:
            is_alignment: Whether results are from alignment checking
            
        Returns:
            Dictionary of conformance results
        """
        conforming = []
        non_conforming = []
        violations_list = []
        
        if is_alignment:
            # Process alignment results
            case_id_mapping = {}
            for idx, group in enumerate(self.df.groupby(self.col_map.get('case_id', 'case_id'))):
                case_id_mapping[idx] = group[0]
            
            for idx, alignment in enumerate(self.conformance_results):
                case_id = case_id_mapping.get(idx, f"Case_{idx}")
                
                # Check if alignment is perfect
                alignment_cost = alignment.get('cost', float('inf'))
                is_fit = alignment_cost == 0
                
                if is_fit:
                    conforming.append(case_id)
                else:
                    non_conforming.append(case_id)
                    
                    # Process alignment steps for violations
                    self._extract_alignment_violations(case_id, alignment, violations_list)
        else:
            # Process token replay results
            for trace_result in self.conformance_results:
                case_id = trace_result.get("trace_attributes", {}).get("concept:name", "Unknown")
                is_fit = trace_result.get("trace_is_fit", False)
                
                if is_fit:
                    conforming.append(case_id)
                else:
                    non_conforming.append(case_id)
                    
                    # Extract violations from token replay result
                    self._extract_token_replay_violations(case_id, trace_result, violations_list)
        
        # Store results
        self.conforming_cases = conforming
        self.non_conforming_cases = non_conforming
        
        if violations_list:
            self.violations = pd.DataFrame(violations_list)
        
        # Calculate statistics
        total_cases = len(conforming) + len(non_conforming)
        conformance_ratio = len(conforming) / total_cases if total_cases > 0 else 0
        
        # Count violations by type
        violations_by_type = defaultdict(int)
        for violation in violations_list:
            violations_by_type[violation['violation_type']] += 1
        
        return {
            "total_cases": total_cases,
            "conforming_cases": len(conforming),
            "non_conforming_cases": len(non_conforming),
            "conformance_ratio": conformance_ratio,
            "violations": dict(violations_by_type),
            "method": "alignments" if is_alignment else "token_replay"
        }
    
    def _extract_token_replay_violations(self, case_id: str, trace_result: Dict, violations_list: List):
        """Extract violations from token replay result"""
        # Missing tokens indicate skipped activities
        if trace_result.get("missing_tokens", 0) > 0:
            violations_list.append({
                "case_id": case_id,
                "violation_type": ViolationType.SKIPPED_ACTIVITY.value,
                "count": trace_result["missing_tokens"],
                "fitness": trace_result.get("trace_fitness", 0)
            })
        
        # Remaining tokens indicate additional activities
        if trace_result.get("remaining_tokens", 0) > 0:
            violations_list.append({
                "case_id": case_id,
                "violation_type": ViolationType.DUPLICATE_ACTIVITY.value,
                "count": trace_result["remaining_tokens"],
                "fitness": trace_result.get("trace_fitness", 0)
            })
        
        # Compare produced and consumed tokens for sequence violations
        if "produced_tokens" in trace_result and "consumed_tokens" in trace_result:
            if trace_result["produced_tokens"] > trace_result["consumed_tokens"]:
                violations_list.append({
                    "case_id": case_id,
                    "violation_type": ViolationType.WRONG_SEQUENCE.value,
                    "count": trace_result["produced_tokens"] - trace_result["consumed_tokens"],
                    "fitness": trace_result.get("trace_fitness", 0)
                })
    
    def _extract_alignment_violations(self, case_id: str, alignment: Dict, violations_list: List):
        """Extract violations from alignment result"""
        alignment_steps = alignment.get('alignment', [])
        cost = alignment.get('cost', 0)
        
        for step in alignment_steps:
            if len(step) >= 2:
                # Model move (skipped activity)
                if step[0][0] == '>>':
                    violations_list.append({
                        "case_id": case_id,
                        "violation_type": ViolationType.SKIPPED_ACTIVITY.value,
                        "activity": step[0][1] if len(step[0]) > 1 else "unknown",
                        "cost": cost
                    })
                
                # Log move (unexpected activity)
                elif step[1][0] == '>>':
                    violations_list.append({
                        "case_id": case_id,
                        "violation_type": ViolationType.WRONG_ACTIVITY.value,
                        "activity": step[1][1] if len(step[1]) > 1 else "unknown",
                        "cost": cost
                    })
    
    def _check_simplified(self) -> Dict[str, Any]:
        """
        Efficient variant-based conformance checking
        
        Returns:
            Dictionary of conformance results
        """
        logger.info("Using simplified variant-based conformance checking")
        
        # Get the necessary column names
        case_id_col = self.col_map.get('case_id', 'case_id')
        task_col = self.col_map.get('task_name', 'task_name')
        time_col = self.col_map.get('timestamp', 'timestamp')
        
        # Check if columns exist
        if not all(col in self.df.columns for col in [case_id_col, task_col]):
            raise ValueError(f"Required columns missing. Need case ID and task columns.")
        
        # Efficiently identify process variants
        # Sort the dataframe first if time column exists
        if time_col in self.df.columns:
            df_sorted = self.df.sort_values([case_id_col, time_col])
        else:
            df_sorted = self.df  # Use as-is if no timestamp
        
        # Group by case and aggregate activities to get sequences
        case_sequences = {}
        variant_counts = defaultdict(int)
        
        for case_id, group in df_sorted.groupby(case_id_col):
            # Convert sequence to tuple for hashing
            sequence = tuple(group[task_col].values)
            case_sequences[case_id] = sequence
            variant_counts[sequence] += 1
        
        # Find the most common variant (happy path)
        if variant_counts:
            happy_path = max(variant_counts.items(), key=lambda x: x[1])[0]
            
            # Identify conforming and non-conforming cases
            self.conforming_cases = []
            self.non_conforming_cases = []
            violations_list = []
            
            for case_id, sequence in case_sequences.items():
                if sequence == happy_path:
                    self.conforming_cases.append(case_id)
                else:
                    self.non_conforming_cases.append(case_id)
                    # Add violation information
                    self._add_sequence_violations(case_id, sequence, happy_path, violations_list)
            
            # Store violations
            if violations_list:
                self.violations = pd.DataFrame(violations_list)
            
            # Calculate statistics
            total_cases = len(case_sequences)
            conformance_ratio = len(self.conforming_cases) / total_cases if total_cases > 0 else 0
            
            # Count violation types
            violation_counts = defaultdict(int)
            for violation in violations_list:
                violation_counts[violation['violation_type']] += 1
            
            return {
                "total_cases": total_cases,
                "conforming_cases": len(self.conforming_cases),
                "non_conforming_cases": len(self.non_conforming_cases),
                "conformance_ratio": conformance_ratio,
                "violations": dict(violation_counts),
                "happy_path_frequency": variant_counts[happy_path],
                "total_variants": len(variant_counts),
                "method": "simplified_variant"
            }
        else:
            return {
                "total_cases": 0,
                "conforming_cases": 0,
                "non_conforming_cases": 0,
                "conformance_ratio": 0.0,
                "violations": {},
                "method": "simplified_variant"
            }
    
    def _add_sequence_violations(self, case_id, sequence, happy_path, violations_list):
        """Add violation information based on sequence comparison"""
        # Check for different types of violations
        if len(sequence) < len(happy_path):
            # Incomplete case (missing activities)
            violations_list.append({
                "case_id": case_id,
                "violation_type": ViolationType.INCOMPLETE_CASE.value,
                "count": len(happy_path) - len(sequence)
            })
        elif len(sequence) > len(happy_path):
            # Extra activities
            violations_list.append({
                "case_id": case_id,
                "violation_type": ViolationType.DUPLICATE_ACTIVITY.value,
                "count": len(sequence) - len(happy_path)
            })
        
        # Check for wrong sequence (different activities or order)
        common_length = min(len(sequence), len(happy_path))
        sequence_diff = sum(1 for i in range(common_length) if sequence[i] != happy_path[i])
        
        if sequence_diff > 0:
            violations_list.append({
                "case_id": case_id,
                "violation_type": ViolationType.WRONG_SEQUENCE.value,
                "count": sequence_diff
            })

    def get_violating_cases(self) -> pd.DataFrame:
        """
        Get events from cases with violations
        
        Returns:
            DataFrame with events from violating cases
        """
        if not self.non_conforming_cases:
            return pd.DataFrame()
        
        case_id_col = self.col_map.get('case_id', 'case_id')
        if case_id_col not in self.df.columns:
            return pd.DataFrame()
        
        return self.df[self.df[case_id_col].isin(self.non_conforming_cases)].copy()
    
    def get_conformance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of conformance checking results
        
        Returns:
            Dictionary with conformance summary
        """
        if not self.conforming_cases and not self.non_conforming_cases:
            return {"status": "Not analyzed yet"}
        
        violation_counts = {}
        if not self.violations.empty and 'violation_type' in self.violations.columns:
            violation_counts = self.violations['violation_type'].value_counts().to_dict()
        
        return {
            "conforming_cases": len(self.conforming_cases),
            "non_conforming_cases": len(self.non_conforming_cases),
            "conformance_ratio": len(self.conforming_cases) / (len(self.conforming_cases) + len(self.non_conforming_cases)),
            "violation_types": violation_counts,
            "pm4py_used": self.pm4py_available
        }