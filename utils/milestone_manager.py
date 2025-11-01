#!/usr/bin/env python3
"""
Milestone Manager for tracking Pokemon Emerald progression
Follows SimpleAgent's Objective pattern with descriptions
"""

class MilestoneManager:
    """Manages complete milestone order with descriptions"""

    # Complete milestone list with descriptions (from SimpleAgent objectives)
    ALL_MILESTONES = [
        # Phase 1: Game Initialization
        {
            "id": "GAME_RUNNING",
            "description": "Complete title sequence and begin the game",
            "category": "system"
        },
        {
            "id": "PLAYER_NAME_SET",
            "description": "Player has chosen their character name",
            "category": "intro"
        },
        {
            "id": "INTRO_CUTSCENE_COMPLETE",
            "description": "Complete intro cutscene with moving van",
            "category": "intro"
        },

        # Phase 2: Tutorial & Starting Town
        {
            "id": "LITTLEROOT_TOWN",
            "description": "Arrive at Littleroot Town",
            "category": "location"
        },
        {
            "id": "PLAYER_HOUSE_ENTERED",
            "description": "Enter player's house for the first time",
            "category": "location"
        },
        {
            "id": "PLAYER_BEDROOM",
            "description": "Go upstairs to player's bedroom",
            "category": "location"
        },
        {
            "id": "CLOCK_SET",
            "description": "Set the clock in player's bedroom and get out of the house",
            "category": "task"
        },
        {
            "id": "RIVAL_HOUSE",
            "description": "Visit May's house next door",
            "category": "location"
        },
        {
            "id": "RIVAL_BEDROOM",
            "description": "Visit May's bedroom on the second floor",
            "category": "location"
        },

        # Phase 3: Professor Birch & Starter
        {
            "id": "ROUTE_101",
            "description": "Travel to Route 101",
            "category": "location"
        },
        {
            "id": "STARTER_CHOSEN",
            "description": "Choose starter Pokemon",
            "category": "pokemon"
        },
        {
            "id": "BIRCH_LAB_VISITED",
            "description": "Visit Professor Birch's lab",
            "category": "location"
        },

        # Phase 4: Rival Battle
        {
            "id": "OLDALE_TOWN",
            "description": "Arrive at Oldale Town",
            "category": "location"
        },
        {
            "id": "ROUTE_103",
            "description": "Travel to Route 103",
            "category": "location"
        },
        {
            "id": "RECEIVED_POKEDEX",
            "description": "Receive Pokedex from Professor Birch",
            "category": "item"
        },

        # Phase 5: Route 102 & Petalburg
        {
            "id": "ROUTE_102",
            "description": "Travel to Route 102",
            "category": "location"
        },
        {
            "id": "PETALBURG_CITY",
            "description": "Arrive at Petalburg City",
            "category": "location"
        },
        {
            "id": "DAD_FIRST_MEETING",
            "description": "Meet Dad at Petalburg Gym",
            "category": "story"
        },
        {
            "id": "GYM_EXPLANATION",
            "description": "Receive gym explanation from Dad",
            "category": "story"
        },

        # Phase 6: Road to Rustboro
        {
            "id": "ROUTE_104_SOUTH",
            "description": "Travel to Route 104 (South)",
            "category": "location"
        },
        {
            "id": "PETALBURG_WOODS",
            "description": "Navigate through Petalburg Woods",
            "category": "location"
        },
        {
            "id": "TEAM_AQUA_GRUNT_DEFEATED",
            "description": "Defeat Team Aqua grunt",
            "category": "battle"
        },
        {
            "id": "ROUTE_104_NORTH",
            "description": "Travel to Route 104 (North)",
            "category": "location"
        },
        {
            "id": "RUSTBORO_CITY",
            "description": "Arrive at Rustboro City",
            "category": "location"
        },

        # Phase 7: First Gym
        {
            "id": "RUSTBORO_GYM_ENTERED",
            "description": "Enter Rustboro Gym",
            "category": "location"
        },
        {
            "id": "ROXANNE_DEFEATED",
            "description": "Defeat Gym Leader Roxanne",
            "category": "battle"
        },
        {
            "id": "FIRST_GYM_COMPLETE",
            "description": "Receive Stone Badge (first gym badge)",
            "category": "badge"
        }
    ]

    def __init__(self):
        """Initialize milestone manager with lookup dict"""
        # Build lookup dict for quick access by ID
        self._milestone_dict = {m["id"]: m for m in self.ALL_MILESTONES}

    def get_milestone_info(self, milestone_id: str) -> dict:
        """
        Get full milestone info by ID

        Args:
            milestone_id: Milestone ID string

        Returns:
            Dict with id, description, category or default if not found
        """
        return self._milestone_dict.get(milestone_id, {
            "id": milestone_id,
            "description": "Unknown milestone",
            "category": "unknown"
        })

    def get_next_milestone(self, completed_milestones: dict) -> str:
        """
        Get next uncompleted milestone ID

        Args:
            completed_milestones: Dict like {"GAME_RUNNING": {"completed": True, ...}}
                                  Only contains completed milestones

        Returns:
            Next milestone ID (string) or None if all complete
        """
        for milestone in self.ALL_MILESTONES:
            milestone_id = milestone["id"]
            # If not in dict, it's not completed yet
            if milestone_id not in completed_milestones:
                return milestone_id
            # If in dict but completed=False, also not completed
            if not completed_milestones[milestone_id].get('completed', False):
                return milestone_id
        return None  # All completed

    def get_next_milestone_info(self, completed_milestones: dict) -> dict:
        """
        Get full info dict for next milestone

        Args:
            completed_milestones: Dict of completed milestones

        Returns:
            Milestone info dict or None if all complete
        """
        next_id = self.get_next_milestone(completed_milestones)
        if next_id:
            return self.get_milestone_info(next_id)
        return None

    def get_next_milestone_index(self, completed_milestones: dict) -> int:
        """
        Get index of next milestone in ALL_MILESTONES list

        Args:
            completed_milestones: Dict of completed milestones

        Returns:
            Index (0-based) or len(ALL_MILESTONES) if all complete
        """
        next_id = self.get_next_milestone(completed_milestones)
        if next_id:
            return next(i for i, m in enumerate(self.ALL_MILESTONES) if m["id"] == next_id)
        return len(self.ALL_MILESTONES)  # All complete

    def get_all_with_status(self, completed_milestones: dict) -> list:
        """
        Get all milestones with completion status and descriptions

        Args:
            completed_milestones: Dict of completed milestones

        Returns:
            List of dicts with:
            - id: milestone ID
            - name: milestone ID (for backward compatibility)
            - description: human-readable description
            - category: milestone category
            - completed: bool
            - index: position in list
            - timestamp: completion timestamp or None
        """
        result = []
        for i, milestone in enumerate(self.ALL_MILESTONES):
            milestone_id = milestone["id"]
            is_completed = milestone_id in completed_milestones and \
                          completed_milestones[milestone_id].get('completed', False)

            result.append({
                "id": milestone_id,
                "name": milestone_id,  # For backward compatibility
                "description": milestone["description"],
                "category": milestone["category"],
                "completed": is_completed,
                "index": i,
                "timestamp": completed_milestones.get(milestone_id, {}).get('timestamp') if is_completed else None
            })
        return result

    def get_total_count(self) -> int:
        """Get total number of milestones"""
        return len(self.ALL_MILESTONES)
