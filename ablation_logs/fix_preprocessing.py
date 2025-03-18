#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patched data preprocessing module to fix tuple handling issues
"""

import importlib.util
import sys
import os
from colorama import Fore, Style

# Load the main module
spec = importlib.util.spec_from_file_location("main", "main.py")
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)

# Fix load_and_preprocess_data_phase1 to properly handle tuple results
def patched_load_and_preprocess_data_phase1(data_path, args):
    from modules.data_preprocessing import load_and_preprocess_data, create_feature_representation, build_graph_data
    
    main.print_section_header("Loading and Preprocessing Data with Phase 1 Enhancements")
    
    # Load and preprocess data
    result = load_and_preprocess_data(
        data_path,
        use_adaptive_norm=args.adaptive_norm,
        enhanced_features=args.enhanced_features,
        enhanced_graphs=args.enhanced_graphs,
        batch_size=args.batch_size
    )
    
    # Proper type checking with diagnostic output
    if isinstance(result, tuple):
        print(f"{Fore.YELLOW}Debug: load_and_preprocess_data returned tuple of length {len(result)}{Style.RESET_ALL}")
        
        if len(result) == 4:
            # Properly returns (df, graphs, task_encoder, resource_encoder)
            return result
        elif len(result) >= 1:
            # Extract the dataframe from the first element if it's a dataframe
            candidate_df = result[0]
            if hasattr(candidate_df, 'columns'):
                print(f"{Fore.GREEN}Debug: Successfully extracted dataframe from tuple[0]{Style.RESET_ALL}")
                df = candidate_df
            else:
                print(f"{Fore.RED}Error: First element of tuple is not a dataframe{Style.RESET_ALL}")
                df = result  # Let it fail later with a clear error
        else:
            print(f"{Fore.RED}Error: Returned tuple is empty{Style.RESET_ALL}")
            df = result  # Let it fail later with a clear error
    else:
        # Just returns a dataframe or other object
        df = result
    
    # Process the dataframe normally
    if hasattr(df, 'columns'):
        # Create feature representation
        df, task_encoder, resource_encoder = create_feature_representation(df, use_norm_features=args.adaptive_norm)
        graphs = build_graph_data(df)
        return df, graphs, task_encoder, resource_encoder
    else:
        print(f"{Fore.RED}Error: df is not a dataframe, it's a {type(df)}{Style.RESET_ALL}")
        raise TypeError(f"Expected DataFrame, got {type(df)}")

# Apply our patch
main.load_and_preprocess_data_phase1 = patched_load_and_preprocess_data_phase1

# Run the main function with the arguments passed to this script
if __name__ == "__main__":
    # Pass all arguments to main function
    main.main()
