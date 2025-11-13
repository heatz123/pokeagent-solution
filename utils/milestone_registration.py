#!/usr/bin/env python3
"""
Milestone registration helper

Centralized milestone registration logic for consistency between CodeAgent and Trainer
"""

from utils.server_milestone_definitions import SERVER_MILESTONES
from utils.custom_milestone_definitions import CUSTOM_MILESTONES


def register_all_milestones(milestone_manager):
    """
    Register all milestones (server + custom) to a MilestoneManager instance

    Args:
        milestone_manager: MilestoneManager instance to register to

    Returns:
        Number of milestones registered
    """
    total_registered = 0

    # 1. Register server milestones as custom milestones (state-based checks)
    print(f"🔧 Registering {len(SERVER_MILESTONES)} server milestones...")
    for milestone in SERVER_MILESTONES:
        milestone_manager.add_custom_milestone(
            milestone_id=milestone["id"],
            description=milestone["description"],
            insert_after=milestone["insert_after"],
            check_fn=milestone["check_fn"],
            category=milestone.get("category", "server")
        )
        total_registered += 1

    print(f"✅ Server milestones registered")

    # 2. Register additional custom milestones
    print(f"🔧 Registering {len(CUSTOM_MILESTONES)} custom milestones...")
    for milestone in CUSTOM_MILESTONES:
        milestone_manager.add_custom_milestone(
            milestone_id=milestone["id"],
            description=milestone["description"],
            insert_after=milestone["insert_after"],
            check_fn=milestone["check_fn"],
            category=milestone.get("category", "custom")
        )
        total_registered += 1

    print(f"✅ Custom milestones registered")
    print(f"✅ Total: {total_registered} milestones registered")

    return total_registered
