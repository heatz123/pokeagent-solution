"""
Knowledge Base for CodeAgent

Stores learned facts across runs with metadata (step, milestone, timestamp).
Persists to .pokeagent_cache/knowledge.json
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KnowledgeEntry:
    """Single knowledge entry with metadata"""

    def __init__(self, content: str, step: int, milestone: str, entry_id: str = None, evidence_text: str = ""):
        """
        Initialize a knowledge entry

        Args:
            content: The knowledge sentence (e.g., "The clock has been set")
            step: Step number when this was learned
            milestone: Milestone ID when this was learned
            entry_id: Optional entry ID (auto-generated if None)
            evidence_text: Human-readable explanation of why this knowledge is true
        """
        self.id = entry_id or f"kb_{int(time.time() * 1000)}"
        self.content = content
        self.evidence_text = evidence_text
        self.created_step = step
        self.created_milestone = milestone
        self.created_timestamp = datetime.now().isoformat()

        # Update tracking
        self.updated_step = None
        self.updated_milestone = None
        self.updated_timestamp = None

        # Evidence file paths (relative paths)
        self.evidence_screenshot_path = None
        self.evidence_state_path = None

        # Validation tracking (can store validator name, status, or notes)
        self.validation = None

        self.tags = []  # Optional: for future filtering

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "evidence_text": self.evidence_text,
            "evidence_screenshot_path": self.evidence_screenshot_path,
            "evidence_state_path": self.evidence_state_path,
            "created_step": self.created_step,
            "created_milestone": self.created_milestone,
            "created_timestamp": self.created_timestamp,
            "updated_step": self.updated_step,
            "updated_milestone": self.updated_milestone,
            "updated_timestamp": self.updated_timestamp,
            "validation": self.validation,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create KnowledgeEntry from dictionary"""
        entry = cls(
            content=data['content'],
            step=data['created_step'],
            milestone=data['created_milestone'],
            entry_id=data['id'],
            evidence_text=data.get('evidence_text', '')
        )
        entry.created_timestamp = data.get('created_timestamp', entry.created_timestamp)
        entry.updated_step = data.get('updated_step')
        entry.updated_milestone = data.get('updated_milestone')
        entry.updated_timestamp = data.get('updated_timestamp')
        entry.evidence_screenshot_path = data.get('evidence_screenshot_path')
        entry.evidence_state_path = data.get('evidence_state_path')
        entry.validation = data.get('validation')
        entry.tags = data.get('tags', [])
        return entry


class KnowledgeBase:
    """
    Knowledge Base that persists learned facts across runs

    Stores entries in JSON format with metadata for tracking when/where
    each fact was learned.
    """

    def __init__(self, filepath: str = ".pokeagent_cache/knowledge.json"):
        """
        Initialize the knowledge base

        Args:
            filepath: Path to the JSON storage file
        """
        self.filepath = filepath
        self.entries: List[KnowledgeEntry] = []
        self.load()

    def _get_next_id(self) -> str:
        """
        Generate next sequential ID based on existing entries

        Returns:
            Next ID in format kb_1, kb_2, etc.
        """
        if not self.entries:
            return "kb_1"

        # Extract numeric part from existing IDs
        max_num = 0
        for entry in self.entries:
            if entry.id.startswith("kb_"):
                try:
                    num = int(entry.id.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass

        return f"kb_{max_num + 1}"

    def load(self):
        """Load knowledge entries from file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.entries = [
                        KnowledgeEntry.from_dict(e)
                        for e in data.get('entries', [])
                    ]
                logger.info(f"Loaded {len(self.entries)} knowledge entries from {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
                self.entries = []
        else:
            logger.info(f"No existing knowledge base found at {self.filepath}, starting fresh")
            self.entries = []

    def save(self):
        """Save knowledge entries to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)

            # Write to file
            with open(self.filepath, 'w') as f:
                json.dump({
                    'entries': [e.to_dict() for e in self.entries]
                }, f, indent=2)

            logger.debug(f"Saved {len(self.entries)} knowledge entries to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def add(self, content: str, step: int, milestone: str,
            evidence_text: str = "",
            evidence_screenshot=None,
            evidence_state: dict = None) -> str:
        """
        Add a new knowledge entry with evidence

        Args:
            content: The knowledge sentence
            step: Current step number
            milestone: Current milestone ID
            evidence_text: Human-readable explanation of why this is true
            evidence_screenshot: PIL Image of game state when knowledge was learned
            evidence_state: Filtered game state dict for verification

        Returns:
            The ID of the created entry
        """
        self.load()  # Reload from file to preserve external changes

        # Generate next sequential ID
        next_id = self._get_next_id()
        entry = KnowledgeEntry(content, step, milestone, entry_id=next_id, evidence_text=evidence_text)

        # Save screenshot if provided
        if evidence_screenshot:
            screenshot_dir = os.path.join(
                os.path.dirname(self.filepath) or '.',
                "kb_evidence",
                "screenshots"
            )
            os.makedirs(screenshot_dir, exist_ok=True)

            screenshot_path = os.path.join(screenshot_dir, f"{entry.id}.png")
            evidence_screenshot.save(screenshot_path)

            # Store relative path
            entry.evidence_screenshot_path = screenshot_path
            logger.debug(f"Saved evidence screenshot to {screenshot_path}")

        # Save state if provided
        if evidence_state:
            state_dir = os.path.join(
                os.path.dirname(self.filepath) or '.',
                "kb_evidence",
                "states"
            )
            os.makedirs(state_dir, exist_ok=True)

            state_path = os.path.join(state_dir, f"{entry.id}.json")
            with open(state_path, 'w') as f:
                json.dump(evidence_state, f, indent=2, default=str)

            # Store relative path
            entry.evidence_state_path = state_path
            logger.debug(f"Saved evidence state to {state_path}")

        self.entries.append(entry)
        self.save()
        logger.info(f"Added knowledge entry {entry.id}: {content}")

        return entry.id

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all knowledge entries as dictionaries"""
        self.load()  # Reload from file to get latest external changes
        return [e.to_dict() for e in self.entries]

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent N knowledge entries

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entry dictionaries (most recent last)
        """
        self.load()  # Reload from file to get latest external changes
        return [e.to_dict() for e in self.entries[-limit:]]

    def clear(self):
        """Clear all knowledge entries (for debugging)"""
        self.entries = []
        self.save()
        logger.info("Knowledge base cleared")

    def get_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Get a specific knowledge entry by ID

        Args:
            entry_id: Knowledge entry ID (e.g., "kb_1234567890")

        Returns:
            KnowledgeEntry if found, None otherwise
        """
        self.load()  # Reload from file to get latest external changes
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def update_by_id(self, entry_id: str, new_content: str, step: int, milestone: str,
                     evidence_text: str = None,
                     evidence_screenshot=None,
                     evidence_state: dict = None) -> bool:
        """
        Update a knowledge entry by ID

        Args:
            entry_id: Knowledge entry ID (e.g., "kb_1234567890")
            new_content: New content to replace the old content
            step: Current step number
            milestone: Current milestone ID
            evidence_text: Optional new evidence text
            evidence_screenshot: Optional new screenshot
            evidence_state: Optional new state

        Returns:
            True if updated successfully, False if ID not found
        """
        self.load()  # Reload from file to get latest external changes
        for entry in self.entries:
            if entry.id == entry_id:
                # Update content and metadata
                entry.content = new_content
                entry.updated_step = step
                entry.updated_milestone = milestone
                entry.updated_timestamp = datetime.now().isoformat()

                # Update evidence text if provided
                if evidence_text is not None:
                    entry.evidence_text = evidence_text

                # Update screenshot if provided
                if evidence_screenshot:
                    screenshot_dir = os.path.join(
                        os.path.dirname(self.filepath) or '.',
                        "kb_evidence",
                        "screenshots"
                    )
                    os.makedirs(screenshot_dir, exist_ok=True)

                    screenshot_path = os.path.join(screenshot_dir, f"{entry_id}.png")
                    evidence_screenshot.save(screenshot_path)
                    entry.evidence_screenshot_path = screenshot_path
                    logger.debug(f"Updated evidence screenshot at {screenshot_path}")

                # Update state if provided
                if evidence_state:
                    state_dir = os.path.join(
                        os.path.dirname(self.filepath) or '.',
                        "kb_evidence",
                        "states"
                    )
                    os.makedirs(state_dir, exist_ok=True)

                    state_path = os.path.join(state_dir, f"{entry_id}.json")
                    with open(state_path, 'w') as f:
                        json.dump(evidence_state, f, indent=2, default=str)
                    entry.evidence_state_path = state_path
                    logger.debug(f"Updated evidence state at {state_path}")

                self.save()
                logger.info(f"Updated knowledge entry {entry_id}: {new_content}")
                return True

        logger.warning(f"Knowledge entry {entry_id} not found for update")
        return False

    def delete_by_id(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry by ID (including evidence files)

        Args:
            entry_id: Knowledge entry ID (e.g., "kb_1234567890")

        Returns:
            True if deleted successfully, False if ID not found
        """
        self.load()  # Reload from file to get latest external changes
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                deleted_content = entry.content

                # Delete evidence screenshot if exists
                if entry.evidence_screenshot_path and os.path.exists(entry.evidence_screenshot_path):
                    try:
                        os.remove(entry.evidence_screenshot_path)
                        logger.debug(f"Deleted evidence screenshot: {entry.evidence_screenshot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete screenshot: {e}")

                # Delete evidence state if exists
                if entry.evidence_state_path and os.path.exists(entry.evidence_state_path):
                    try:
                        os.remove(entry.evidence_state_path)
                        logger.debug(f"Deleted evidence state: {entry.evidence_state_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete state: {e}")

                # Delete entry from list
                del self.entries[i]
                self.save()
                logger.info(f"Deleted knowledge entry {entry_id}: {deleted_content}")
                return True

        logger.warning(f"Knowledge entry {entry_id} not found for deletion")
        return False

    def get_evidence(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Load evidence (screenshot and state) for a knowledge entry

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Dict with 'text', 'screenshot', 'state' keys, or None if not found
        """
        entry = self.get_by_id(entry_id)
        if not entry:
            return None

        evidence = {
            "text": entry.evidence_text,
            "screenshot": None,
            "state": None
        }

        # Load screenshot
        if entry.evidence_screenshot_path and os.path.exists(entry.evidence_screenshot_path):
            try:
                from PIL import Image
                evidence["screenshot"] = Image.open(entry.evidence_screenshot_path)
                logger.debug(f"Loaded evidence screenshot from {entry.evidence_screenshot_path}")
            except Exception as e:
                logger.warning(f"Failed to load screenshot: {e}")

        # Load state
        if entry.evidence_state_path and os.path.exists(entry.evidence_state_path):
            try:
                with open(entry.evidence_state_path, 'r') as f:
                    evidence["state"] = json.load(f)
                logger.debug(f"Loaded evidence state from {entry.evidence_state_path}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return evidence

    def update_validation_by_id(
        self,
        entry_id: str,
        validation: Dict[str, Any]
    ) -> bool:
        """
        Update validation field for a knowledge entry

        Args:
            entry_id: Knowledge entry ID
            validation: Validation dict to set (e.g., {"status": "verified", ...})

        Returns:
            True if updated successfully, False if ID not found
        """
        self.load()  # Reload from file to get latest external changes
        for entry in self.entries:
            if entry.id == entry_id:
                entry.validation = validation
                self.save()
                logger.info(f"Updated validation for {entry_id}: {validation.get('status')}")
                return True

        logger.warning(f"Knowledge entry {entry_id} not found for validation update")
        return False

    def __len__(self) -> int:
        """Return number of entries"""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"KnowledgeBase(entries={len(self.entries)}, filepath='{self.filepath}')"
