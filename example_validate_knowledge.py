#!/usr/bin/env python3
"""
Example script to validate knowledge entries with reasons

This shows how to use the validation field to mark knowledge as verified
with a reason explaining why it's correct.
"""

from datetime import datetime
from utils.knowledge_base import KnowledgeBase

def validate_knowledge_example():
    """
    Example: Validate a knowledge entry with reason
    """

    # Load knowledge base
    kb = KnowledgeBase()

    print("=" * 60)
    print("Knowledge Validation Example")
    print("=" * 60)

    # Show recent entries
    print("\nRecent knowledge entries:")
    entries = kb.get_recent(limit=5)

    if not entries:
        print("No knowledge entries found.")
        print("\nTip: Run the agent first to generate some knowledge entries.")
        return

    for entry in entries:
        validation_status = "✓" if entry.get('validation') else "○"
        print(f"{validation_status} [{entry['id']}] {entry['content']}")
        if entry.get('validation'):
            val = entry['validation']
            print(f"    Validated: {val.get('status')} - {val.get('reason', 'No reason')}")

    # Example: Validate the first entry
    if entries:
        entry_to_validate = entries[0]
        entry_id = entry_to_validate['id']

        print(f"\n{'='*60}")
        print(f"Validating: [{entry_id}]")
        print(f"Content: {entry_to_validate['content']}")
        print(f"{'='*60}")

        # Load evidence to check
        evidence = kb.get_evidence(entry_id)
        if evidence:
            print("\nEvidence available:")
            print(f"  Text: {evidence['text']}")
            print(f"  Screenshot: {'Yes' if evidence['screenshot'] else 'No'}")
            print(f"  State: {'Yes' if evidence['state'] else 'No'}")

            if evidence['state']:
                state = evidence['state']
                print(f"\n  State details:")
                if 'player' in state:
                    print(f"    Player position: {state['player'].get('position')}")
                    print(f"    Player location: {state['player'].get('location')}")

        # Validation with reason
        validation_data = {
            "status": "verified",
            "verified_by": "heatz",
            "reason": "Checked screenshot and game state - coordinates confirmed",
            "timestamp": datetime.now().isoformat()
        }

        print(f"\nValidation data:")
        print(f"  Status: {validation_data['status']}")
        print(f"  Verified by: {validation_data['verified_by']}")
        print(f"  Reason: {validation_data['reason']}")

        success = kb.update_validation_by_id(entry_id, validation_data)

        if success:
            print(f"\n✅ Successfully validated [{entry_id}]")
        else:
            print(f"\n❌ Failed to validate [{entry_id}]")

def show_validation_formats():
    """Show different validation format examples"""

    print("\n" + "="*60)
    print("Validation Format Examples")
    print("="*60)

    examples = [
        {
            "name": "Simple verification",
            "data": {
                "status": "verified",
                "verified_by": "heatz",
                "reason": "Confirmed by checking screenshot evidence",
                "timestamp": "2025-11-09T22:50:00"
            }
        },
        {
            "name": "Needs correction",
            "data": {
                "status": "incorrect",
                "verified_by": "heatz",
                "reason": "Position is wrong - should be (6,1) not (6,2)",
                "timestamp": "2025-11-09T22:51:00"
            }
        },
        {
            "name": "Uncertain",
            "data": {
                "status": "uncertain",
                "verified_by": "heatz",
                "reason": "Evidence unclear - need more testing",
                "timestamp": "2025-11-09T22:52:00"
            }
        },
        {
            "name": "Comprehensive validation",
            "data": {
                "status": "verified",
                "verified_by": "heatz",
                "reason": "Cross-checked with multiple runs - consistently true",
                "confidence": "high",
                "tested_steps": [10, 25, 42],
                "timestamp": "2025-11-09T22:53:00"
            }
        }
    ]

    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['data']}")

if __name__ == "__main__":
    print("""
This script demonstrates how to validate knowledge entries.

Usage:
1. Run the agent to generate knowledge entries
2. Run this script to see recent entries and validate them
3. Validation includes:
   - status (verified/incorrect/uncertain)
   - verified_by (your name)
   - reason (why you validated it this way)
   - timestamp (when you validated it)
    """)

    validate_knowledge_example()
    show_validation_formats()

    print("\n" + "="*60)
    print("You can also manually edit .pokeagent_cache/knowledge.json")
    print("and add validation data to any entry.")
    print("="*60)
