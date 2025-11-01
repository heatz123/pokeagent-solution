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
        self.tags = []  # Optional: for future filtering

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "created_step": self.created_step,
            "created_milestone": self.created_milestone,
            "created_timestamp": self.created_timestamp,
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

    def to_prompt_format(self, limit: int = 20) -> str:
        """
        Format knowledge base for inclusion in LLM prompt

        Args:
            limit: Maximum number of recent entries to include

        Returns:
            Formatted string for prompt
        """
        if not self.entries:
            return "Knowledge Base is empty. You can add learnings with ADD_KNOWLEDGE: <sentence>"

        recent = self.entries[-limit:]
        lines = ["KNOWLEDGE BASE (learned facts):"]
        for i, entry in enumerate(recent, 1):
            # Shorten milestone name for readability
            milestone_short = entry.created_milestone.replace('story_', '')
            lines.append(f"{i}. {entry.content} [Step {entry.created_step}, {milestone_short}]")

        return "\n".join(lines)

    def clear(self):
        """Clear all knowledge entries (for debugging)"""
        self.entries = []
        self.save()
        logger.info("Knowledge base cleared")

    def __len__(self) -> int:
        """Return number of entries"""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"KnowledgeBase(entries={len(self.entries)}, filepath='{self.filepath}')"
