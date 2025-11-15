#!/usr/bin/env python3
"""
Model finder utility for DAgger policies

Automatically finds best model for each milestone from models directory
"""

import os
import glob
import re
from typing import Optional, Dict


def find_best_model_for_milestone(
    milestone_id: str,
    models_dir: str = "dagger_pipeline_results/models",
    prefer_final: bool = False
) -> Optional[str]:
    """
    Find best model file for a given milestone

    Searches both flat structure and subdirectory structure:
    - Flat: models/dagger_milestone_{MILESTONE}_best.pth
    - Subdir: models/{MILESTONE}/dagger_{MILESTONE}_{MILESTONE}_best.pth

    Priority (when prefer_final=False, default):
    1. dagger_milestone_{MILESTONE}_best.pth or
       dagger_{MILESTONE}_{MILESTONE}_best.pth
    2. dagger_milestone_{MILESTONE}_final.pth or
       dagger_{MILESTONE}_{MILESTONE}_final.pth
    3. dagger_milestone_{MILESTONE}_iter{N}.pth or
       dagger_{MILESTONE}_{MILESTONE}_iter{N}.pth (highest N)
    4. dagger_test_{MILESTONE}_best.pth
    5. dagger_test_{MILESTONE}_final.pth
    6. dagger_test_{MILESTONE}_iter{N}.pth (highest N)

    Priority (when prefer_final=True):
    1. dagger_milestone_{MILESTONE}_final.pth or
       dagger_{MILESTONE}_{MILESTONE}_final.pth
    2. dagger_milestone_{MILESTONE}_best.pth or
       dagger_{MILESTONE}_{MILESTONE}_best.pth
    3. dagger_milestone_{MILESTONE}_iter{N}.pth or
       dagger_{MILESTONE}_{MILESTONE}_iter{N}.pth (highest N)
    4. dagger_test_{MILESTONE}_final.pth
    5. dagger_test_{MILESTONE}_best.pth
    6. dagger_test_{MILESTONE}_iter{N}.pth (highest N)

    Args:
        milestone_id: Milestone ID (e.g., "RIVAL_HOUSE")
        models_dir: Directory containing model files
        prefer_final: Prioritize _final.pth over _best.pth

    Returns:
        Path to best model file, or None if not found
    """
    if not os.path.exists(models_dir):
        return None

    # Search locations: flat directory and milestone subdirectory
    search_dirs = [
        models_dir,  # Flat structure
        os.path.join(models_dir, milestone_id)  # Subdirectory structure
    ]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # Define patterns for best and final
        best_patterns = [
            f"dagger_milestone_{milestone_id}_best.pth",
            f"dagger_{milestone_id}_{milestone_id}_best.pth",
            f"dagger_{milestone_id}_best.pth"
        ]
        final_patterns = [
            f"dagger_milestone_{milestone_id}_final.pth",
            f"dagger_{milestone_id}_{milestone_id}_final.pth",
            f"dagger_{milestone_id}_final.pth"
        ]

        # Priority 1 & 2: Check based on prefer_final
        if prefer_final:
            # Priority 1: *_final.pth
            for pattern in final_patterns:
                final_path = os.path.join(search_dir, pattern)
                if os.path.exists(final_path):
                    return final_path

            # Priority 2: *_best.pth
            for pattern in best_patterns:
                best_path = os.path.join(search_dir, pattern)
                if os.path.exists(best_path):
                    return best_path
        else:
            # Priority 1: *_best.pth
            for pattern in best_patterns:
                best_path = os.path.join(search_dir, pattern)
                if os.path.exists(best_path):
                    return best_path

            # Priority 2: *_final.pth
            for pattern in final_patterns:
                final_path = os.path.join(search_dir, pattern)
                if os.path.exists(final_path):
                    return final_path

        # Priority 3: *_iter{N}.pth (highest N)
        for pattern in [
            f"dagger_milestone_{milestone_id}_iter*.pth",
            f"dagger_{milestone_id}_{milestone_id}_iter*.pth",
            f"dagger_{milestone_id}_iter*.pth"
        ]:
            iter_pattern = os.path.join(search_dir, pattern)
            iter_files = glob.glob(iter_pattern)
            if iter_files:
                # Extract iteration numbers and find highest
                iter_nums = []
                for f in iter_files:
                    match = re.search(r'iter(\d+)\.pth$', f)
                    if match:
                        iter_nums.append((int(match.group(1)), f))
                if iter_nums:
                    iter_nums.sort(reverse=True)  # Highest first
                    return iter_nums[0][1]

        # Priority 4 & 5: Test models (based on prefer_final)
        test_best = f"dagger_test_{milestone_id}_best.pth"
        test_final = f"dagger_test_{milestone_id}_final.pth"

        if prefer_final:
            # Priority 4: test_final
            test_final_path = os.path.join(search_dir, test_final)
            if os.path.exists(test_final_path):
                return test_final_path

            # Priority 5: test_best
            test_best_path = os.path.join(search_dir, test_best)
            if os.path.exists(test_best_path):
                return test_best_path
        else:
            # Priority 4: test_best
            test_best_path = os.path.join(search_dir, test_best)
            if os.path.exists(test_best_path):
                return test_best_path

            # Priority 5: test_final
            test_final_path = os.path.join(search_dir, test_final)
            if os.path.exists(test_final_path):
                return test_final_path

        # Priority 6: dagger_test_{MILESTONE}_iter{N}.pth (highest N)
        test_iter_pattern = os.path.join(search_dir, f"dagger_test_{milestone_id}_iter*.pth")
        test_iter_files = glob.glob(test_iter_pattern)
        if test_iter_files:
            # Extract iteration numbers and find highest
            iter_nums = []
            for f in test_iter_files:
                match = re.search(r'iter(\d+)\.pth$', f)
                if match:
                    iter_nums.append((int(match.group(1)), f))
            if iter_nums:
                iter_nums.sort(reverse=True)  # Highest first
                return iter_nums[0][1]

    return None


def discover_all_milestone_models(
    models_dir: str = "dagger_pipeline_results/models",
    prefer_final: bool = False
) -> Dict[str, str]:
    """
    Discover all available milestone models in models directory

    Supports both flat and subdirectory structures:
    - Flat: models/*.pth
    - Subdir: models/MILESTONE/*.pth

    Args:
        models_dir: Directory containing model files
        prefer_final: Prioritize _final.pth over _best.pth

    Returns:
        Dict mapping milestone_id -> best_model_path
    """
    if not os.path.exists(models_dir):
        return {}

    # Find all model files (both flat and in subdirectories)
    all_models = []
    all_models.extend(glob.glob(os.path.join(models_dir, "*.pth")))  # Flat structure
    all_models.extend(glob.glob(os.path.join(models_dir, "*", "*.pth")))  # Subdirectory structure

    # Extract milestone IDs
    milestone_ids = set()
    for model_path in all_models:
        basename = os.path.basename(model_path)
        # Match patterns:
        # - dagger_milestone_{MILESTONE}_*
        # - dagger_test_{MILESTONE}_*
        # - dagger_{MILESTONE}_{MILESTONE}_*
        match = re.search(r'dagger_(?:milestone|test)?_?([A-Z][A-Z0-9_]+?)(?:_\1)?_(?:best|final|iter)', basename)
        if match:
            milestone_ids.add(match.group(1))
        else:
            # Alternative: extract from parent directory name
            parent_dir = os.path.basename(os.path.dirname(model_path))
            if parent_dir and parent_dir != os.path.basename(models_dir):
                # Check if parent dir looks like milestone ID (uppercase with underscores)
                if re.match(r'^[A-Z][A-Z0-9_]+$', parent_dir):
                    milestone_ids.add(parent_dir)

    # Find best model for each milestone
    milestone_models = {}
    for milestone_id in milestone_ids:
        best_model = find_best_model_for_milestone(
            milestone_id, models_dir, prefer_final
        )
        if best_model:
            milestone_models[milestone_id] = best_model

    return milestone_models


def print_discovered_models(models_dir: str = "dagger_pipeline_results/models"):
    """
    Print discovered models (for debugging)

    Args:
        models_dir: Directory containing model files
    """
    milestone_models = discover_all_milestone_models(models_dir)

    if not milestone_models:
        print(f"No models found in {models_dir}/")
        return

    print(f"📦 Discovered {len(milestone_models)} milestone models:")
    for milestone_id, model_path in sorted(milestone_models.items()):
        # Show relative path from models_dir if it's a subdirectory
        if model_path.startswith(models_dir):
            rel_path = os.path.relpath(model_path, models_dir)
            print(f"   {milestone_id:<30} -> {rel_path}")
        else:
            basename = os.path.basename(model_path)
            print(f"   {milestone_id:<30} -> {basename}")


if __name__ == "__main__":
    # Test discovery
    import sys
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "dagger_pipeline_results/models"
    print_discovered_models(models_dir)
