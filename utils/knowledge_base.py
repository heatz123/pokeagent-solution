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

    def __init__(self, content: str, step: int, milestone: str, entry_id: str = None):
        """
        Initialize a knowledge entry

        Args:
            content: The knowledge sentence (e.g., "The clock has been set")
            step: Step number when this was learned
            milestone: Milestone ID when this was learned
            entry_id: Optional entry ID (auto-generated if None)
        """
        self.id = entry_id or f"kb_{int(time.time() * 1000)}"
        self.content = content
        self.created_step = step
        self.created_milestone = milestone
        self.created_timestamp = datetime.now().isoformat()

        # Update tracking
        self.updated_step = None
        self.updated_milestone = None
        self.updated_timestamp = None

        self.tags = []  # Optional: for future filtering

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "created_step": self.created_step,
            "created_milestone": self.created_milestone,
            "created_timestamp": self.created_timestamp,
            "updated_step": self.updated_step,
            "updated_milestone": self.updated_milestone,
            "updated_timestamp": self.updated_timestamp,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create KnowledgeEntry from dictionary"""
        entry = cls(
            content=data['content'],
            step=data['created_step'],
            milestone=data['created_milestone'],
            entry_id=data['id']
        )
        entry.created_timestamp = data.get('created_timestamp', entry.created_timestamp)
        entry.updated_step = data.get('updated_step')
        entry.updated_milestone = data.get('updated_milestone')
        entry.updated_timestamp = data.get('updated_timestamp')
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

    def add(self, content: str, step: int, milestone: str) -> str:
        """
        Add a new knowledge entry

        Args:
            content: The knowledge sentence
            step: Current step number
            milestone: Current milestone ID

        Returns:
            The ID of the created entry
        """
        entry = KnowledgeEntry(content, step, milestone)
        self.entries.append(entry)
        self.save()
        logger.info(f"Added knowledge entry {entry.id}: {content}")
        return entry.id

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all knowledge entries as dictionaries"""
        return [e.to_dict() for e in self.entries]

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent N knowledge entries

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entry dictionaries (most recent last)
        """
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
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def update_by_id(self, entry_id: str, new_content: str, step: int, milestone: str) -> bool:
        """
        Update a knowledge entry by ID

        Args:
            entry_id: Knowledge entry ID (e.g., "kb_1234567890")
            new_content: New content to replace the old content
            step: Current step number
            milestone: Current milestone ID

        Returns:
            True if updated successfully, False if ID not found
        """
        for entry in self.entries:
            if entry.id == entry_id:
                # Update content and metadata
                entry.content = new_content
                entry.updated_step = step
                entry.updated_milestone = milestone
                entry.updated_timestamp = datetime.now().isoformat()
                self.save()
                logger.info(f"Updated knowledge entry {entry_id}: {new_content}")
                return True

        logger.warning(f"Knowledge entry {entry_id} not found for update")
        return False

    def delete_by_id(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry by ID

        Args:
            entry_id: Knowledge entry ID (e.g., "kb_1234567890")

        Returns:
            True if deleted successfully, False if ID not found
        """
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                deleted_content = entry.content
                del self.entries[i]
                self.save()
                logger.info(f"Deleted knowledge entry {entry_id}: {deleted_content}")
                return True

        logger.warning(f"Knowledge entry {entry_id} not found for deletion")
        return False

    def __len__(self) -> int:
        """Return number of entries"""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"KnowledgeBase(entries={len(self.entries)}, filepath='{self.filepath}')"
