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
            "category": "system",
            "condition": "state is not None"
        },
        {
            "id": "PLAYER_NAME_SET",
            "description": "Player has chosen their character name",
            "category": "intro",
            "condition": "state.get('player', {}).get('name', '').strip() not in ['', 'UNKNOWN', 'PLAYER']"
        },
        {
            "id": "INTRO_CUTSCENE_COMPLETE",
            "description": "Complete intro cutscene with moving van",
            "category": "intro",
            "condition": "'MOVING_VAN' in str(state.get('player', {}).get('location', '')).upper()"
        },

        # Phase 2: Tutorial & Starting Town
        {
            "id": "LITTLEROOT_TOWN",
            "description": "Arrive at Littleroot Town",
            "category": "location",
            "condition": "'LITTLEROOT' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "PLAYER_HOUSE_ENTERED",
            "description": "Enter player's house for the first time",
            "category": "location",
            "condition": "'LITTLEROOT TOWN BRENDANS HOUSE 1F' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "PLAYER_BEDROOM",
            "description": "Go upstairs to player's bedroom",
            "category": "location",
            "condition": "'LITTLEROOT TOWN BRENDANS HOUSE 2F' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "CLOCK_SET",
            "description": "Set the clock in player's bedroom and leave the house",
            "category": "task",
            "condition": "milestone_tracker.is_completed('PLAYER_BEDROOM') and 'LITTLEROOT' in str(state.get('player', {}).get('location', '')).upper() and 'HOUSE' not in str(state.get('player', {}).get('location', '')).upper() and 'LAB' not in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "RIVAL_HOUSE",
            "description": "Visit May's house next door",
            "category": "location",
            "condition": "'LITTLEROOT TOWN MAYS HOUSE 1F' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "RIVAL_BEDROOM",
            "description": "Visit May's bedroom on the second floor",
            "category": "location",
            "condition": "'LITTLEROOT TOWN MAYS HOUSE 2F' in str(state.get('player', {}).get('location', '')).upper()"
        },

        # Phase 3: Professor Birch & Starter
        {
            "id": "ROUTE_101",
            "description": "Travel to Route 101",
            "category": "location",
            "condition": "'ROUTE_101' in str(state.get('player', {}).get('location', '')).upper() or 'ROUTE 101' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "STARTER_CHOSEN",
            "description": "Choose starter Pokemon",
            "category": "pokemon",
            "condition": "len(state.get('player', {}).get('party', [])) >= 1 and any(p.get('species_name', '').strip() for p in state.get('player', {}).get('party', []))"
        },
        {
            "id": "BIRCH_LAB_VISITED",
            "description": "Visit Professor Birch's lab",
            "category": "location",
            "condition": "'LITTLEROOT TOWN PROFESSOR BIRCHS LAB' in str(state.get('player', {}).get('location', '')).upper()"
        },

        # Phase 4: Rival Battle
        {
            "id": "OLDALE_TOWN",
            "description": "Arrive at Oldale Town",
            "category": "location",
            "condition": "milestone_tracker.is_completed('LITTLEROOT_TOWN') and 'OLDALE' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "ROUTE_103",
            "description": "Travel to Route 103 and battle with May",
            "category": "location",
            "condition": "milestone_tracker.is_completed('ROUTE_101') and milestone_tracker.is_completed('STARTER_CHOSEN') and ('ROUTE_103' in str(state.get('player', {}).get('location', '')).upper() or 'ROUTE 103' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "RECEIVED_POKEDEX",
            "description": "Receive Pokedex from Professor Birch",
            "category": "item",
            "condition": "milestone_tracker.is_completed('ROUTE_103') and 'LITTLEROOT TOWN PROFESSOR BIRCHS LAB' in str(state.get('player', {}).get('location', '')).upper()"
        },

        # Phase 5: Route 102 & Petalburg
        {
            "id": "ROUTE_102",
            "description": "Travel to Route 102",
            "category": "location",
            "condition": "milestone_tracker.is_completed('RECEIVED_POKEDEX') and ('ROUTE_102' in str(state.get('player', {}).get('location', '')).upper() or 'ROUTE 102' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "PETALBURG_CITY",
            "description": "Arrive at Petalburg City",
            "category": "location",
            "condition": "milestone_tracker.is_completed('LITTLEROOT_TOWN') and milestone_tracker.is_completed('OLDALE_TOWN') and 'PETALBURG' in str(state.get('player', {}).get('location', '')).upper()"
        },
        {
            "id": "DAD_FIRST_MEETING",
            "description": "Meet Dad at Petalburg Gym",
            "category": "story",
            "condition": "milestone_tracker.is_completed('PETALBURG_CITY') and ('PETALBURG CITY GYM' in str(state.get('player', {}).get('location', '')).upper() or 'PETALBURG_CITY_GYM' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "GYM_EXPLANATION",
            "description": "Receive gym explanation from Dad",
            "category": "story",
            "condition": "milestone_tracker.is_completed('DAD_FIRST_MEETING') and ('PETALBURG CITY GYM' in str(state.get('player', {}).get('location', '')).upper() or 'PETALBURG_CITY_GYM' in str(state.get('player', {}).get('location', '')).upper())"
        },

        # Phase 6: Road to Rustboro
        {
            "id": "ROUTE_104_SOUTH",
            "description": "Travel to Route 104 (South)",
            "category": "location",
            "condition": "milestone_tracker.is_completed('PETALBURG_CITY') and ('ROUTE_104' in str(state.get('player', {}).get('location', '')).upper() or 'ROUTE 104' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "PETALBURG_WOODS",
            "description": "Navigate through Petalburg Woods",
            "category": "location",
            "condition": "milestone_tracker.is_completed('ROUTE_104_SOUTH') and ('PETALBURG_WOODS' in str(state.get('player', {}).get('location', '')).upper() or 'PETALBURG WOODS' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "TEAM_AQUA_GRUNT_DEFEATED",
            "description": "Defeat Team Aqua grunt",
            "category": "battle",
            "condition": "milestone_tracker.is_completed('PETALBURG_WOODS') and ('PETALBURG_WOODS' in str(state.get('player', {}).get('location', '')).upper() or 'PETALBURG WOODS' in str(state.get('player', {}).get('location', '')).upper()) and state.get('player', {}).get('position', {}).get('y') == 23 and state.get('player', {}).get('position', {}).get('x') in [26, 27]"
        },
        {
            "id": "ROUTE_104_NORTH",
            "description": "Travel to Route 104 (North)",
            "category": "location",
            "condition": "milestone_tracker.is_completed('PETALBURG_WOODS') and milestone_tracker.is_completed('TEAM_AQUA_GRUNT_DEFEATED') and ('ROUTE_104' in str(state.get('player', {}).get('location', '')).upper() or 'ROUTE 104' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "RUSTBORO_CITY",
            "description": "Arrive at Rustboro City",
            "category": "location",
            "condition": "milestone_tracker.is_completed('PETALBURG_CITY') and 'RUSTBORO' in str(state.get('player', {}).get('location', '')).upper()"
        },

        # Phase 7: First Gym
        {
            "id": "RUSTBORO_GYM_ENTERED",
            "description": "Enter Rustboro Gym",
            "category": "location",
            "condition": "milestone_tracker.is_completed('RUSTBORO_CITY') and ('RUSTBORO_GYM' in str(state.get('player', {}).get('location', '')).upper() or 'RUSTBORO CITY GYM' in str(state.get('player', {}).get('location', '')).upper())"
        },
        {
            "id": "ROXANNE_DEFEATED",
            "description": "Defeat Gym Leader Roxanne",
            "category": "battle",
            "condition": "milestone_tracker.is_completed('STONE_BADGE')"
        },
        {
            "id": "FIRST_GYM_COMPLETE",
            "description": "Receive Stone Badge (first gym badge)",
            "category": "badge",
            "condition": "milestone_tracker.is_completed('STONE_BADGE') and 'GYM' not in str(state.get('player', {}).get('location', '')).upper()"
        }
    ]

    def __init__(self):
        """Initialize milestone manager with lookup dict"""
        # Build lookup dict for quick access by ID
        self._milestone_dict = {m["id"]: m for m in self.ALL_MILESTONES}

        # Custom milestones (instance-level, not shared between instances)
        self.custom_milestones = []

    def add_custom_milestone(
        self,
        milestone_id: str,
        description: str,
        insert_after: str,
        check_fn: callable,
        category: str = "custom"
    ):
        """
        Add a custom milestone to this manager instance

        Args:
            milestone_id: Unique milestone ID
            description: Human-readable description for LLM
            insert_after: ID of milestone to insert after
            check_fn: Completion check function (game_state, action) -> bool
            category: Category for UI display
        """
        self.custom_milestones.append({
            "id": milestone_id,
            "description": description,
            "category": category,
            "insert_after": insert_after,
            "check_fn": check_fn
        })

    def get_ordered_milestones(self) -> list:
        """
        Get all milestones (base + custom) in correct order

        Uses insert() to place custom milestones after their specified base milestone

        Returns:
            List of milestone dicts with id, description, category
        """
        # Start with a copy of base milestones
        result = list(self.ALL_MILESTONES)

        # Insert each custom milestone after its specified base milestone
        for custom in self.custom_milestones:
            insert_after_id = custom["insert_after"]

            # Find the index of the base milestone
            for i, milestone in enumerate(result):
                if milestone["id"] == insert_after_id:
                    # Insert custom milestone right after
                    result.insert(i + 1, {
                        "id": custom["id"],
                        "description": custom["description"],
                        "category": custom["category"]
                    })
                    break  # Move to next custom milestone

        return result

    def get_custom_check_fn(self, milestone_id: str):
        """
        Get the check function for a custom milestone

        Args:
            milestone_id: Milestone ID

        Returns:
            Check function or None if not a custom milestone
        """
        for custom in self.custom_milestones:
            if custom["id"] == milestone_id:
                return custom["check_fn"]
        return None

    def get_milestone_info(self, milestone_id: str) -> dict:
        """
        Get full milestone info by ID (checks both base and custom)

        Args:
            milestone_id: Milestone ID string

        Returns:
            Dict with id, description, category or default if not found
        """
        # Check base milestones first
        if milestone_id in self._milestone_dict:
            return self._milestone_dict[milestone_id]

        # Check custom milestones
        for custom in self.custom_milestones:
            if custom["id"] == milestone_id:
                return {
                    "id": custom["id"],
                    "description": custom["description"],
                    "category": custom["category"]
                }

        # Not found
        return {
            "id": milestone_id,
            "description": "Unknown milestone",
            "category": "unknown"
        }

    def get_next_milestone(self, completed_milestones: dict) -> str:
        """
        Get next uncompleted milestone ID (includes custom milestones)

        Args:
            completed_milestones: Dict like {"GAME_RUNNING": {"completed": True, ...}}
                                  Only contains completed milestones

        Returns:
            Next milestone ID (string) or None if all complete
        """
        # Use ordered milestones which includes custom milestones
        ordered = self.get_ordered_milestones()

        for milestone in ordered:
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
