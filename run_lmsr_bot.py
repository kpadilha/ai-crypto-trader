#!/usr/bin/env python3
"""
LMSR Prediction Market Bot — Entry Point (Polymarket)

Runs the Logarithmic Market Scoring Rule prediction bot that:
  1. Discovers binary markets on Polymarket via the Gamma API
  2. Polls CLOB prices and builds Bayesian beliefs from price signals
  3. Detects mispricings where model beliefs diverge from market prices
  4. Optionally executes trades on Polymarket via py-clob-client
  5. Also accepts supplementary signals from Redis (market & social data)

Usage:
    python run_lmsr_bot.py
    python run_lmsr_bot.py --auto-trade --order-size 25
    python run_lmsr_bot.py --min-edge 0.03 --max-kelly 0.15
    python run_lmsr_bot.py --liquidity 200000 --poll-interval 5
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
        description="LMSR Prediction Market Bot (Polymarket)"
    )

    # LMSR parameters
    lmsr_group = parser.add_argument_group("LMSR parameters")
    lmsr_group.add_argument(
        '--liquidity', '-b', type=float, default=None,
        help='Liquidity parameter (b). Larger = more liquidity, higher max loss.'
    )
    lmsr_group.add_argument(
        '--min-edge', type=float, default=None,
        help='Minimum edge (|belief - price|) to trigger a trade signal.'
    )
    lmsr_group.add_argument(
        '--max-kelly', type=float, default=None,
        help='Maximum Kelly fraction for position sizing (0-1).'
    )
    lmsr_group.add_argument(
        '--learning-rate', type=float, default=None,
        help='Base learning rate for Bayesian signal updates.'
    )

    # Polymarket parameters
    poly_group = parser.add_argument_group("Polymarket parameters")
    poly_group.add_argument(
        '--auto-trade', action='store_true', default=None,
        help='Enable automatic order execution on Polymarket.'
    )
    poly_group.add_argument(
        '--order-size', type=float, default=None,
        help='Base order size in USDC for auto-trades.'
    )
    poly_group.add_argument(
        '--poll-interval', type=float, default=None,
        help='Seconds between Polymarket price polls.'
    )
    poly_group.add_argument(
        '--max-markets', type=int, default=None,
        help='Maximum number of Polymarket markets to track.'
    )
    poly_group.add_argument(
        '--no-websocket', action='store_true', default=False,
        help='Use REST polling instead of WebSocket streaming.'
    )

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info("Starting LMSR Prediction Market Bot (Polymarket)...")

    bot = LMSRPredictionBot()

    # Apply LMSR overrides
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

    # Apply Polymarket overrides
    if args.auto_trade is not None:
        bot.poly_auto_trade = args.auto_trade
        logging.info(f"Override: auto_trade = {args.auto_trade}")

    if args.order_size is not None:
        bot.poly_order_size = args.order_size
        logging.info(f"Override: order_size = ${args.order_size}")

    if args.poll_interval is not None:
        bot.poly_poll_interval = args.poll_interval
        logging.info(f"Override: poll_interval = {args.poll_interval}s")

    if args.max_markets is not None:
        bot.poly_max_markets = args.max_markets
        logging.info(f"Override: max_markets = {args.max_markets}")

    if args.no_websocket:
        bot.poly_use_websocket = False
        logging.info("Override: using REST polling instead of WebSocket")

    # Print configuration summary
    print("\n" + "=" * 60)
    print("  LMSR Prediction Market Bot (Polymarket)")
    print("=" * 60)
    print(f"  Liquidity (b):        {bot.liquidity:>12,.0f}")
    print(f"  Max maker loss (n=2): ${bot.pricing_engine.max_loss(2):>11,.2f}")
    print(f"  Min edge threshold:   {bot.min_edge:>12.4f}")
    print(f"  Max Kelly fraction:   {bot.max_kelly_fraction:>12.4f}")
    print(f"  Signal learning rate: {bot.signal_learning_rate:>12.4f}")
    print(f"  Max position size:    {bot.max_position_size:>12.1%}")
    print(f"  Max open positions:   {bot.max_open_positions:>12d}")
    print("-" * 60)
    feed_mode = "WebSocket (real-time)" if bot.poly_use_websocket else f"REST polling ({bot.poly_poll_interval:.0f}s)"
    print(f"  Polymarket CLOB:      {'https://clob.polymarket.com':>30s}")
    print(f"  Data feed:            {feed_mode:>30s}")
    print(f"  Poll interval:        {bot.poly_poll_interval:>11.0f}s")
    print(f"  Max tracked markets:  {bot.poly_max_markets:>12d}")
    print(f"  Auto-trade:           {'ENABLED' if bot.poly_auto_trade else 'DISABLED':>12s}")
    if bot.poly_auto_trade:
        print(f"  Order size (USDC):    ${bot.poly_order_size:>11.2f}")
    print("=" * 60 + "\n")

    if bot.poly_auto_trade and not os.getenv("POLYMARKET_PRIVATE_KEY"):
        logging.warning(
            "Auto-trade enabled but POLYMARKET_PRIVATE_KEY not set. "
            "Orders will fail. Set the env var or run without --auto-trade."
        )

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
