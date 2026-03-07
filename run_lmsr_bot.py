#!/usr/bin/env python3
"""
LMSR Prediction Market Bot — Entry Point

Runs the Logarithmic Market Scoring Rule prediction bot that:
  1. Subscribes to market and social data via Redis
  2. Maintains Bayesian beliefs updated from real-time signals
  3. Prices outcomes using the LMSR cost function (softmax)
  4. Detects mispricings and publishes trading opportunities

Usage:
    python run_lmsr_bot.py
    python run_lmsr_bot.py --liquidity 200000
    python run_lmsr_bot.py --min-edge 0.03 --max-kelly 0.15
"""

import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime

from services.lmsr_prediction_bot import LMSRPredictionBot


def setup_logging():
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f'logs/lmsr_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMSR Prediction Market Bot"
    )
    parser.add_argument(
        '--liquidity', '-b', type=float, default=None,
        help='Liquidity parameter (b). Larger = more liquidity, higher max loss. '
             'Default from config.json.'
    )
    parser.add_argument(
        '--min-edge', type=float, default=None,
        help='Minimum edge (|belief - price|) to trigger a trade signal. '
             'Default from config.json.'
    )
    parser.add_argument(
        '--max-kelly', type=float, default=None,
        help='Maximum Kelly fraction for position sizing (0-1). '
             'Default from config.json.'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Base learning rate for Bayesian signal updates. '
             'Default from config.json.'
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info("Starting LMSR Prediction Market Bot...")

    bot = LMSRPredictionBot()

    # Apply CLI overrides
    if args.liquidity is not None:
        bot.liquidity = args.liquidity
        bot.pricing_engine.b = args.liquidity
        logging.info(f"Override: liquidity = {args.liquidity:,.0f}")

    if args.min_edge is not None:
        bot.min_edge = args.min_edge
        bot.inefficiency_detector.min_edge = args.min_edge
        logging.info(f"Override: min_edge = {args.min_edge}")

    if args.max_kelly is not None:
        bot.max_kelly_fraction = args.max_kelly
        bot.inefficiency_detector.max_kelly_fraction = args.max_kelly
        logging.info(f"Override: max_kelly = {args.max_kelly}")

    if args.learning_rate is not None:
        bot.signal_learning_rate = args.learning_rate
        logging.info(f"Override: learning_rate = {args.learning_rate}")

    # Print configuration summary
    print("\n" + "=" * 60)
    print("  LMSR Prediction Market Bot")
    print("=" * 60)
    print(f"  Liquidity (b):        {bot.liquidity:>12,.0f}")
    print(f"  Max maker loss (n=2): ${bot.pricing_engine.max_loss(2):>11,.2f}")
    print(f"  Min edge threshold:   {bot.min_edge:>12.4f}")
    print(f"  Max Kelly fraction:   {bot.max_kelly_fraction:>12.4f}")
    print(f"  Signal learning rate: {bot.signal_learning_rate:>12.4f}")
    print(f"  Max position size:    {bot.max_position_size:>12.1%}")
    print(f"  Max open positions:   {bot.max_open_positions:>12d}")
    print("=" * 60 + "\n")

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Shutting down LMSR bot...")
        bot.running = False
        print("\nLMSR Prediction Bot stopped.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Critical error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
