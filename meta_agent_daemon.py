#!/usr/bin/env python3
"""
Meta-Agent Daemon
Periodically validates knowledge base entries
"""

import time
import signal
import sys
import logging
from datetime import datetime
from meta_agent import SimpleLLMClient, MetaAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
should_exit = False

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global should_exit
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    should_exit = True

def run_daemon(interval_seconds: int = 30, max_validations_per_cycle: int = 20):
    """
    Run meta-agent validation in a loop

    Args:
        interval_seconds: Sleep interval between validation cycles
        max_validations_per_cycle: Max entries to validate per cycle
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 80)
    logger.info("🤖 META-AGENT DAEMON STARTED")
    logger.info(f"   Interval: {interval_seconds} seconds")
    logger.info(f"   Max validations per cycle: {max_validations_per_cycle}")
    logger.info("=" * 80)

    # Initialize LLM client
    try:
        llm = SimpleLLMClient(provider="openai", model="gpt-5")
        meta_agent = MetaAgent(llm_client=llm)
        logger.info("✅ Meta-agent initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize meta-agent: {e}")
        sys.exit(1)

    cycle_count = 0

    while not should_exit:
        cycle_count += 1
        cycle_start = time.time()

        logger.info("")
        logger.info(f"🔄 Cycle #{cycle_count} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            stats = meta_agent.validate_all(max_validations=max_validations_per_cycle)

            logger.info(f"✅ Cycle #{cycle_count} completed:")
            logger.info(f"   - Total checked: {stats['total']}")
            logger.info(f"   - Newly validated: {stats['newly_validated']}")
            logger.info(f"   - Already validated: {stats['already_validated']}")
            logger.info(f"   - Skipped (limit): {stats['skipped']}")

        except Exception as e:
            logger.error(f"❌ Error during validation cycle: {e}", exc_info=True)

        cycle_duration = time.time() - cycle_start
        logger.info(f"⏱️  Cycle duration: {cycle_duration:.1f}s")

        # Sleep until next cycle
        if not should_exit:
            logger.info(f"💤 Sleeping for {interval_seconds}s...")
            time.sleep(interval_seconds)

    logger.info("=" * 80)
    logger.info(f"🛑 META-AGENT DAEMON STOPPED (ran {cycle_count} cycles)")
    logger.info("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta-Agent Validation Daemon")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number for this daemon instance"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Interval between validation cycles in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-validations",
        type=int,
        default=20,
        help="Max validations per cycle (default: 20)"
    )

    args = parser.parse_args()

    run_daemon(
        interval_seconds=args.interval,
        max_validations_per_cycle=args.max_validations
    )
