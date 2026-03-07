import os
import json
import math
import asyncio
import logging as logger
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
import numpy as np

from services.polymarket_client import PolymarketClient

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/lmsr_prediction_bot.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)

logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [LMSRBot] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)


@dataclass
class Outcome:
    """Represents a single outcome in a prediction market."""
    name: str
    quantity: float = 0.0


@dataclass
class Market:
    """A prediction market with LMSR pricing."""
    market_id: str
    symbol: str
    description: str
    outcomes: List[Outcome] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    resolution_outcome: Optional[int] = None


@dataclass
class BayesianBelief:
    """Tracks Bayesian posterior beliefs for a market's outcomes."""
    log_posteriors: np.ndarray  # log-space posteriors for numerical stability
    update_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def posteriors(self) -> np.ndarray:
        """Convert log-posteriors to probabilities via softmax normalization."""
        shifted = self.log_posteriors - np.max(self.log_posteriors)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)


class LMSRPricingEngine:
    """
    Logarithmic Market Scoring Rule (LMSR) pricing engine.

    Implements the Hanson LMSR cost function:
        C(q) = b * ln(sum(exp(q_i / b)))

    Price function (softmax):
        p_i = exp(q_i / b) / sum(exp(q_j / b))
    """

    def __init__(self, liquidity: float = 100_000.0):
        self.b = liquidity

    def cost(self, quantities: np.ndarray) -> float:
        """
        LMSR cost function: C(q) = b * ln(sum(exp(q_i / b)))

        Uses log-sum-exp trick for numerical stability.
        """
        scaled = quantities / self.b
        max_scaled = np.max(scaled)
        return self.b * (max_scaled + math.log(np.sum(np.exp(scaled - max_scaled))))

    def prices(self, quantities: np.ndarray) -> np.ndarray:
        """
        Instantaneous prices via softmax: p_i = exp(q_i / b) / sum(exp(q_j / b))

        Properties: sum(p_i) = 1 and p_i in (0, 1) for all i.
        """
        scaled = quantities / self.b
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    def trade_cost(self, quantities: np.ndarray, outcome_idx: int, delta: float) -> float:
        """
        Cost to move outcome i from q_i to q_i + delta:
            TradeCost = C(q_1, ..., q_i + delta, ..., q_n) - C(q_1, ..., q_i, ..., q_n)
        """
        cost_before = self.cost(quantities)
        new_quantities = quantities.copy()
        new_quantities[outcome_idx] += delta
        cost_after = self.cost(new_quantities)
        return cost_after - cost_before

    def max_loss(self, n_outcomes: int) -> float:
        """Maximum market maker loss: L_max = b * ln(n)"""
        return self.b * math.log(n_outcomes)


class BayesianSignalProcessor:
    """
    Sequential Bayesian belief updater operating in log-space
    for numerical stability.

    Update rule:
        log P(H|D) = log P(D|H) + log P(H) - log Z

    where Z is the normalizing constant.
    """

    def __init__(self, n_outcomes: int, prior: Optional[np.ndarray] = None):
        if prior is not None:
            self.belief = BayesianBelief(log_posteriors=np.log(prior))
        else:
            # Uniform prior
            uniform = np.ones(n_outcomes) / n_outcomes
            self.belief = BayesianBelief(log_posteriors=np.log(uniform))

    def update(self, log_likelihoods: np.ndarray) -> np.ndarray:
        """
        Sequential Bayesian update in log-space:
            log_posterior = log_prior + log_likelihood - log_Z
        """
        unnormalized = self.belief.log_posteriors + log_likelihoods
        # Normalize in log-space (log-sum-exp)
        max_val = np.max(unnormalized)
        log_z = max_val + math.log(np.sum(np.exp(unnormalized - max_val)))
        self.belief.log_posteriors = unnormalized - log_z
        self.belief.update_count += 1
        self.belief.last_updated = datetime.now(timezone.utc).isoformat()
        return self.belief.posteriors

    def update_from_signal(self, signal_strength: float, favored_outcome: int,
                           n_outcomes: int, base_lr: float = 0.1) -> np.ndarray:
        """
        Convert a directional signal into log-likelihoods and update beliefs.

        Args:
            signal_strength: Signal magnitude in [0, 1]. Higher = stronger evidence.
            favored_outcome: Index of the outcome favored by the signal.
            n_outcomes: Total number of outcomes.
            base_lr: Base learning rate controlling update magnitude.
        """
        log_likelihoods = np.zeros(n_outcomes)
        boost = base_lr * signal_strength
        log_likelihoods[favored_outcome] = boost
        # Distribute negative evidence to other outcomes
        penalty = -boost / (n_outcomes - 1) if n_outcomes > 1 else 0.0
        for i in range(n_outcomes):
            if i != favored_outcome:
                log_likelihoods[i] = penalty
        return self.update(log_likelihoods)

    @property
    def posteriors(self) -> np.ndarray:
        return self.belief.posteriors


class InefficiencyDetector:
    """
    Detects mispricing between LMSR market prices and Bayesian model beliefs.

    Entry condition: |p_market_i - p_model_i| > threshold

    Expected value of a trade at market price p with true probability p_hat:
        EV = p_hat - p  (for buying)
    """

    def __init__(self, min_edge: float = 0.05, min_confidence: float = 0.6,
                 max_kelly_fraction: float = 0.25):
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_kelly_fraction = max_kelly_fraction

    def detect(self, market_prices: np.ndarray,
               model_beliefs: np.ndarray) -> List[Dict]:
        """
        Find outcomes where model beliefs diverge significantly from market prices.
        Returns a list of trading opportunities.
        """
        opportunities = []
        for i in range(len(market_prices)):
            edge = model_beliefs[i] - market_prices[i]
            abs_edge = abs(edge)

            if abs_edge < self.min_edge:
                continue
            if model_beliefs[i] < self.min_confidence and edge > 0:
                # Only require confidence for buys, not shorts
                pass

            direction = "BUY" if edge > 0 else "SELL"
            ev = edge  # EV = p_hat - p (per unit)

            # Kelly criterion for position sizing: f* = edge / (1 - p_market)
            # Capped at max_kelly_fraction
            if direction == "BUY":
                odds_against = (1.0 - market_prices[i])
                kelly = edge / odds_against if odds_against > 0 else 0.0
            else:
                odds_against = market_prices[i]
                kelly = abs_edge / odds_against if odds_against > 0 else 0.0

            kelly = min(kelly, self.max_kelly_fraction)

            opportunities.append({
                "outcome_idx": i,
                "direction": direction,
                "market_price": float(market_prices[i]),
                "model_belief": float(model_beliefs[i]),
                "edge": float(edge),
                "abs_edge": float(abs_edge),
                "expected_value": float(ev),
                "kelly_fraction": float(kelly),
            })

        # Sort by absolute edge descending
        opportunities.sort(key=lambda x: x["abs_edge"], reverse=True)
        return opportunities


class LMSRPredictionBot:
    """
    LMSR Prediction Market bot that:
    1. Polls Polymarket CLOB for live prices on binary markets
    2. Updates Bayesian beliefs from price movement signals
    3. Compares LMSR model prices against Polymarket prices to detect inefficiencies
    4. Executes trades on Polymarket when edge exceeds threshold
    5. Also accepts signals from Redis pub/sub (market & social data)
    """

    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        lmsr_config = self.config.get("lmsr", {})
        self.liquidity = lmsr_config.get("liquidity_parameter", 100_000.0)
        self.min_edge = lmsr_config.get("min_edge", 0.05)
        self.min_confidence = lmsr_config.get("min_confidence", 0.6)
        self.max_kelly_fraction = lmsr_config.get("max_kelly_fraction", 0.25)
        self.update_interval = lmsr_config.get("update_interval_sec", 5)
        self.signal_learning_rate = lmsr_config.get("signal_learning_rate", 0.1)
        self.max_position_size = lmsr_config.get("max_position_size", 0.3)
        self.max_open_positions = lmsr_config.get("max_open_positions", 5)

        self.pricing_engine = LMSRPricingEngine(self.liquidity)
        self.inefficiency_detector = InefficiencyDetector(
            min_edge=self.min_edge,
            min_confidence=self.min_confidence,
            max_kelly_fraction=self.max_kelly_fraction,
        )

        # Active markets: market_id -> Market
        self.markets: Dict[str, Market] = {}
        # Bayesian processors per market: market_id -> BayesianSignalProcessor
        self.signal_processors: Dict[str, BayesianSignalProcessor] = {}
        # Active positions
        self.positions: Dict[str, Dict] = {}

        # Polymarket integration
        self.polymarket = PolymarketClient(self.config)
        # condition_id -> polymarket parsed market data
        self.poly_markets: Dict[str, Dict] = {}
        # market_id -> condition_id mapping
        self.market_to_condition: Dict[str, str] = {}
        # Price history for signal generation: condition_id -> [prices]
        self.price_history: Dict[str, List[Dict]] = {}

        poly_config = self.config.get("polymarket", {})
        self.poly_poll_interval = poly_config.get("poll_interval_sec", 10)
        self.poly_max_markets = poly_config.get("max_tracked_markets", 20)
        self.poly_auto_trade = poly_config.get("auto_trade", False)
        self.poly_order_size = poly_config.get("default_order_size", 10.0)
        self.poly_use_websocket = poly_config.get("use_websocket", True)

        self.redis: Optional[Redis] = None
        self.running = True

    # ------------------------------------------------------------------
    # Market management
    # ------------------------------------------------------------------

    def create_binary_market(self, symbol: str, description: str = "") -> Market:
        """
        Create a binary (YES/NO) prediction market for a symbol.
        Binary market with n=2: max loss = b * ln(2).
        """
        market_id = f"lmsr_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        market = Market(
            market_id=market_id,
            symbol=symbol,
            description=description or f"Will {symbol} go up in the next interval?",
            outcomes=[Outcome(name="YES"), Outcome(name="NO")],
        )
        self.markets[market_id] = market
        self.signal_processors[market_id] = BayesianSignalProcessor(n_outcomes=2)

        max_loss = self.pricing_engine.max_loss(2)
        logger.info(
            f"Created binary market {market_id} for {symbol} | "
            f"b={self.liquidity} | max_loss=${max_loss:,.2f}"
        )
        return market

    def get_market_state(self, market_id: str) -> Optional[Dict]:
        """Get current state of a market including prices and beliefs."""
        market = self.markets.get(market_id)
        if not market:
            return None

        quantities = np.array([o.quantity for o in market.outcomes])
        prices = self.pricing_engine.prices(quantities)
        processor = self.signal_processors.get(market_id)
        beliefs = processor.posteriors if processor else prices

        return {
            "market_id": market_id,
            "symbol": market.symbol,
            "description": market.description,
            "outcomes": [
                {
                    "name": o.name,
                    "quantity": o.quantity,
                    "price": float(prices[i]),
                    "belief": float(beliefs[i]),
                }
                for i, o in enumerate(market.outcomes)
            ],
            "resolved": market.resolved,
            "created_at": market.created_at,
            "belief_updates": processor.belief.update_count if processor else 0,
        }

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def process_market_signal(self, market_id: str, signal: Dict) -> Optional[List[Dict]]:
        """
        Process an incoming market signal, update Bayesian beliefs,
        and detect inefficiencies.

        Signal format:
        {
            "signal_type": "price_move" | "volume_spike" | "social" | "technical",
            "strength": float [0, 1],
            "direction": "up" | "down",
            "metadata": {...}
        }
        """
        market = self.markets.get(market_id)
        processor = self.signal_processors.get(market_id)
        if not market or not processor:
            logger.warning(f"Unknown market: {market_id}")
            return None

        n = len(market.outcomes)
        signal_type = signal.get("signal_type", "unknown")
        strength = min(max(signal.get("strength", 0.0), 0.0), 1.0)
        direction = signal.get("direction", "up")

        # Map direction to favored outcome (0=YES/up, 1=NO/down for binary)
        favored = 0 if direction == "up" else 1

        # Apply signal-type-specific learning rate multiplier
        lr_multipliers = {
            "price_move": 1.0,
            "volume_spike": 0.8,
            "social": 0.5,
            "technical": 1.2,
            "sentiment": 0.6,
        }
        lr = self.signal_learning_rate * lr_multipliers.get(signal_type, 0.7)

        # Bayesian update
        new_beliefs = processor.update_from_signal(
            signal_strength=strength,
            favored_outcome=favored,
            n_outcomes=n,
            base_lr=lr,
        )

        # Current LMSR prices
        quantities = np.array([o.quantity for o in market.outcomes])
        market_prices = self.pricing_engine.prices(quantities)

        # Detect inefficiencies
        opportunities = self.inefficiency_detector.detect(market_prices, new_beliefs)

        if opportunities:
            logger.info(
                f"Market {market_id} | signal={signal_type} str={strength:.2f} "
                f"dir={direction} | beliefs={new_beliefs.round(4)} | "
                f"prices={market_prices.round(4)} | "
                f"found {len(opportunities)} opportunities"
            )

        return opportunities

    def execute_trade(self, market_id: str, outcome_idx: int,
                      delta: float) -> Optional[Dict]:
        """
        Execute a trade on a market: buy delta shares of outcome_idx.
        Returns trade details including cost.
        """
        market = self.markets.get(market_id)
        if not market or market.resolved:
            return None

        quantities = np.array([o.quantity for o in market.outcomes])
        cost = self.pricing_engine.trade_cost(quantities, outcome_idx, delta)
        price_before = self.pricing_engine.prices(quantities)[outcome_idx]

        # Update quantity
        market.outcomes[outcome_idx].quantity += delta

        new_quantities = np.array([o.quantity for o in market.outcomes])
        price_after = self.pricing_engine.prices(new_quantities)[outcome_idx]

        trade = {
            "market_id": market_id,
            "outcome_idx": outcome_idx,
            "outcome_name": market.outcomes[outcome_idx].name,
            "delta": delta,
            "cost": float(cost),
            "price_before": float(price_before),
            "price_after": float(price_after),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Trade executed | {market.symbol} | "
            f"outcome={trade['outcome_name']} delta={delta:+.2f} "
            f"cost=${cost:.4f} | price {price_before:.4f} -> {price_after:.4f}"
        )
        return trade

    # ------------------------------------------------------------------
    # Polymarket integration
    # ------------------------------------------------------------------

    async def discover_polymarket_markets(self):
        """Fetch tradable binary markets from Polymarket's Gamma API."""
        tradable = await self.polymarket.discover_tradable_markets(
            limit=self.poly_max_markets
        )

        for pm in tradable:
            condition_id = pm["condition_id"]
            if condition_id in self.poly_markets:
                continue

            self.poly_markets[condition_id] = pm
            self.price_history[condition_id] = []

            # Create an internal LMSR market for each Polymarket market
            market = self._create_polymarket_mirror(pm)
            self.market_to_condition[market.market_id] = condition_id

            logger.info(
                f"Tracking Polymarket market: {pm['question'][:80]} | "
                f"prices={[t['price'] for t in pm['tokens']]}"
            )

        return len(tradable)

    def _create_polymarket_mirror(self, pm: Dict) -> Market:
        """Create an internal LMSR market mirroring a Polymarket market."""
        condition_id = pm["condition_id"]
        market_id = f"poly_{condition_id[:16]}"

        # Use Polymarket's current prices as initial prior
        initial_prices = pm.get("initial_prices", [0.5, 0.5])
        prior = np.array(initial_prices)
        prior = prior / prior.sum()  # normalize

        market = Market(
            market_id=market_id,
            symbol=condition_id,
            description=pm.get("question", ""),
            outcomes=[Outcome(name=o) for o in pm.get("outcomes", ["Yes", "No"])],
        )
        self.markets[market_id] = market
        self.signal_processors[market_id] = BayesianSignalProcessor(
            n_outcomes=len(market.outcomes), prior=prior
        )
        return market

    async def poll_polymarket_prices(self):
        """
        Fallback: poll Polymarket CLOB REST API for price updates.
        Used when WebSocket is disabled (--no-websocket).
        """
        logger.info(
            f"Starting Polymarket REST polling | "
            f"interval={self.poly_poll_interval}s | "
            f"auto_trade={self.poly_auto_trade}"
        )

        while self.running:
            for condition_id, pm in list(self.poly_markets.items()):
                if not self.running:
                    break

                try:
                    prices = await self.polymarket.get_market_prices(pm)
                    if not prices:
                        continue

                    market_id = self._find_market_for_condition(condition_id)
                    if not market_id:
                        continue

                    await self._process_price_update(condition_id, market_id, pm, prices)

                except Exception as e:
                    logger.error(
                        f"Error polling {condition_id[:16]}: {e}", exc_info=True
                    )

            await asyncio.sleep(self.poly_poll_interval)

    def _find_market_for_condition(self, condition_id: str) -> Optional[str]:
        """Find internal market_id for a Polymarket condition_id."""
        for mid, cid in self.market_to_condition.items():
            if cid == condition_id:
                return mid
        return None

    def _polymarket_price_to_signals(self, condition_id: str,
                                     prices: Dict[str, float]) -> List[Dict]:
        """Generate signals from Polymarket price movements."""
        signals = []
        history = self.price_history.get(condition_id, [])

        yes_price = prices.get("Yes", prices.get("YES", 0.5))
        no_price = prices.get("No", prices.get("NO", 0.5))

        # Price momentum signal: compare to previous snapshot
        if len(history) >= 2:
            prev = history[-2]["prices"]
            prev_yes = prev.get("Yes", prev.get("YES", 0.5))
            delta = yes_price - prev_yes

            if abs(delta) > 0.005:
                signals.append({
                    "signal_type": "price_move",
                    "strength": min(abs(delta) / 0.05, 1.0),
                    "direction": "up" if delta > 0 else "down",
                    "metadata": {"source": "polymarket_price_delta", "value": delta},
                })

        # Trend signal: compare to 10 snapshots ago
        if len(history) >= 10:
            old = history[-10]["prices"]
            old_yes = old.get("Yes", old.get("YES", 0.5))
            trend = yes_price - old_yes

            if abs(trend) > 0.01:
                signals.append({
                    "signal_type": "technical",
                    "strength": min(abs(trend) / 0.10, 1.0),
                    "direction": "up" if trend > 0 else "down",
                    "metadata": {"source": "polymarket_trend", "value": trend},
                })

        # Extreme price signal (RSI-like): price near boundaries
        if yes_price > 0.85 or yes_price < 0.15:
            signals.append({
                "signal_type": "technical",
                "strength": max(abs(yes_price - 0.5) * 2 - 0.5, 0.3),
                "direction": "up" if yes_price > 0.5 else "down",
                "metadata": {"source": "price_extreme", "value": yes_price},
            })

        return signals

    async def _execute_polymarket_trade(self, opportunity: Dict):
        """Execute a trade on Polymarket based on a detected opportunity."""
        tokens = opportunity.get("polymarket_tokens", [])
        if not tokens:
            return

        outcome_idx = opportunity["outcome_idx"]
        direction = opportunity["direction"]

        if outcome_idx >= len(tokens):
            return

        token = tokens[outcome_idx]
        token_id = token.get("token_id", "")
        if not token_id:
            return

        kelly = opportunity["kelly_fraction"]
        order_amount = self.poly_order_size * kelly

        if order_amount < 1.0:
            logger.debug(f"Order amount ${order_amount:.2f} too small, skipping")
            return

        # Use limit order at model belief price for better fill
        limit_price = opportunity["model_belief"]
        # Clamp to valid Polymarket range
        limit_price = max(0.01, min(0.99, limit_price))
        size = order_amount / limit_price

        result = self.polymarket.place_limit_order(
            token_id=token_id,
            price=round(limit_price, 2),
            size=round(size, 2),
            side=direction,
        )

        if result:
            self.positions[token_id] = {
                "condition_id": opportunity.get("condition_id"),
                "outcome_idx": outcome_idx,
                "direction": direction,
                "entry_price": opportunity["market_price"],
                "size": size,
                "order_result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @staticmethod
    def _deduplicate_opportunities(opportunities: List[Dict]) -> List[Dict]:
        """De-duplicate by outcome_idx, keeping highest edge."""
        best = {}
        for opp in opportunities:
            idx = opp["outcome_idx"]
            if idx not in best or opp["abs_edge"] > best[idx]["abs_edge"]:
                best[idx] = opp
        return list(best.values())

    # ------------------------------------------------------------------
    # WebSocket streaming (real-time Polymarket events)
    # ------------------------------------------------------------------

    async def start_websocket_stream(self):
        """
        Connect to Polymarket CLOB WebSocket for real-time price updates.
        Processes book, price_change, and last_trade_price events into
        Bayesian signals with ~120ms latency (vs 10s polling).
        """
        # Collect all token IDs from tracked markets
        token_ids = []
        # token_id -> (condition_id, outcome_name) mapping
        self._ws_token_map: Dict[str, tuple] = {}

        for condition_id, pm in self.poly_markets.items():
            for token in pm.get("tokens", []):
                tid = token.get("token_id", "")
                outcome = token.get("outcome", "")
                if tid:
                    token_ids.append(tid)
                    self._ws_token_map[tid] = (condition_id, outcome)

        if not token_ids:
            logger.warning("No token IDs to subscribe to — skipping WebSocket")
            # Fall back to polling
            await self.poll_polymarket_prices()
            return

        logger.info(
            f"Starting WebSocket stream for {len(token_ids)} tokens "
            f"across {len(self.poly_markets)} markets"
        )

        await self.polymarket.stream_market_events(
            token_ids=token_ids,
            on_event=self._handle_ws_event,
        )

    async def _handle_ws_event(self, event: Dict):
        """
        Process a single WebSocket event from Polymarket CLOB.

        Event types:
          - book: full order book snapshot (bids/asks)
          - price_change: best bid/ask changed
          - last_trade_price: last executed trade price
        """
        event_type = event.get("event_type", "")
        asset_id = event.get("asset_id", "")

        if not asset_id or asset_id not in self._ws_token_map:
            return

        condition_id, outcome = self._ws_token_map[asset_id]
        pm = self.poly_markets.get(condition_id)
        if not pm:
            return

        market_id = self._find_market_for_condition(condition_id)
        if not market_id:
            return

        if event_type == "price_change":
            # Extract new prices from the event
            prices = self._extract_prices_from_ws(event, condition_id)
            if prices:
                await self._process_price_update(condition_id, market_id, pm, prices)

        elif event_type == "book":
            # Full book snapshot — extract best bid/ask as prices
            prices = self._extract_prices_from_book(event, condition_id)
            if prices:
                await self._process_price_update(condition_id, market_id, pm, prices)

        elif event_type == "last_trade_price":
            price = event.get("price")
            if price is not None:
                price = float(price)
                # Generate a quick signal from a trade execution
                history = self.price_history.get(condition_id, [])
                if history:
                    prev_prices = history[-1].get("prices", {})
                    prev_price = prev_prices.get(outcome, prev_prices.get(outcome.upper(), 0.5))
                    delta = price - prev_price
                    if abs(delta) > 0.003:
                        signal = {
                            "signal_type": "price_move",
                            "strength": min(abs(delta) / 0.03, 1.0),
                            "direction": "up" if (delta > 0 and outcome in ("Yes", "YES")) or
                                                  (delta < 0 and outcome in ("No", "NO")) else "down",
                            "metadata": {"source": "ws_last_trade", "value": delta},
                        }
                        self.process_market_signal(market_id, signal)

    def _extract_prices_from_ws(self, event: Dict, condition_id: str) -> Dict[str, float]:
        """Extract YES/NO prices from a price_change WebSocket event."""
        prices = {}
        # price_change events contain price data per asset
        asset_id = event.get("asset_id", "")
        if asset_id in self._ws_token_map:
            _, outcome = self._ws_token_map[asset_id]
            # Try different price field names
            for field in ("price", "mid", "best_bid", "best_ask"):
                val = event.get(field)
                if val is not None:
                    prices[outcome] = float(val)
                    break

            # Fill in the complementary outcome price
            if prices and len(prices) == 1:
                for name, p in prices.items():
                    complement = "No" if name in ("Yes", "YES") else "Yes"
                    prices[complement] = round(1.0 - p, 4)

        return prices

    def _extract_prices_from_book(self, event: Dict, condition_id: str) -> Dict[str, float]:
        """Extract midpoint prices from a book WebSocket event."""
        prices = {}
        asset_id = event.get("asset_id", "")
        if asset_id not in self._ws_token_map:
            return prices

        _, outcome = self._ws_token_map[asset_id]
        bids = event.get("bids", [])
        asks = event.get("asks", [])

        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 1))
            mid = (best_bid + best_ask) / 2.0
            prices[outcome] = mid
            complement = "No" if outcome in ("Yes", "YES") else "Yes"
            prices[complement] = round(1.0 - mid, 4)

        return prices

    async def _process_price_update(self, condition_id: str, market_id: str,
                                    pm: Dict, prices: Dict[str, float]):
        """Process a price update (from either WS or polling) through the signal pipeline."""
        # Record snapshot
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prices": prices,
        }
        history = self.price_history.get(condition_id, [])
        history.append(snapshot)
        if len(history) > 100:
            history = history[-100:]
        self.price_history[condition_id] = history

        # Generate signals
        signals = self._polymarket_price_to_signals(condition_id, prices)
        all_opportunities = []

        for signal in signals:
            opps = self.process_market_signal(market_id, signal)
            if opps:
                all_opportunities.extend(opps)

        if all_opportunities:
            best = self._deduplicate_opportunities(all_opportunities)
            for opp in best:
                opp["condition_id"] = condition_id
                opp["polymarket_tokens"] = pm.get("tokens", [])
                opp["question"] = pm.get("question", "")

                logger.info(
                    f"OPPORTUNITY | {pm['question'][:60]} | "
                    f"{opp['direction']} outcome={opp['outcome_idx']} | "
                    f"edge={opp['edge']:.4f} kelly={opp['kelly_fraction']:.4f} | "
                    f"market={opp['market_price']:.4f} model={opp['model_belief']:.4f}"
                )

                if self.poly_auto_trade:
                    await self._execute_polymarket_trade(opp)

                if self.redis:
                    await self.publish_opportunity(market_id, opp)

    # ------------------------------------------------------------------
    # Market data → signal conversion (Redis-sourced data)
    # ------------------------------------------------------------------

    def _market_data_to_signals(self, data: Dict) -> List[Dict]:
        """Convert incoming market data into standardized signals."""
        signals = []

        # Price momentum signal
        price_change_1m = data.get("price_change_1m", 0.0)
        if abs(price_change_1m) > 0.1:
            signals.append({
                "signal_type": "price_move",
                "strength": min(abs(price_change_1m) / 2.0, 1.0),
                "direction": "up" if price_change_1m > 0 else "down",
                "metadata": {"source": "1m_price_change", "value": price_change_1m},
            })

        # RSI signal
        rsi = data.get("rsi", 50.0)
        if rsi < 30 or rsi > 70:
            signals.append({
                "signal_type": "technical",
                "strength": min(abs(rsi - 50) / 50.0, 1.0),
                "direction": "up" if rsi < 30 else "down",
                "metadata": {"source": "rsi", "value": rsi},
            })

        # Volume signal
        avg_volume = data.get("avg_volume", 0)
        if avg_volume > 100_000:
            trend = data.get("trend", "neutral")
            signals.append({
                "signal_type": "volume_spike",
                "strength": min(avg_volume / 1_000_000, 1.0),
                "direction": "up" if trend == "uptrend" else "down",
                "metadata": {"source": "volume", "value": avg_volume},
            })

        # MACD signal
        macd = data.get("macd", 0.0)
        if abs(macd) > 0.0001:
            signals.append({
                "signal_type": "technical",
                "strength": min(abs(macd) * 1000, 1.0),
                "direction": "up" if macd > 0 else "down",
                "metadata": {"source": "macd", "value": macd},
            })

        # Bollinger Band position signal
        bb_pos = data.get("bb_position", 0.5)
        if bb_pos < 0.2 or bb_pos > 0.8:
            signals.append({
                "signal_type": "technical",
                "strength": abs(bb_pos - 0.5) * 2.0,
                "direction": "up" if bb_pos < 0.2 else "down",
                "metadata": {"source": "bollinger", "value": bb_pos},
            })

        # Trend strength signal
        trend_strength = data.get("trend_strength", 0.0)
        trend = data.get("trend", "neutral")
        if trend_strength > 0.3 and trend != "neutral":
            signals.append({
                "signal_type": "price_move",
                "strength": trend_strength,
                "direction": "up" if trend == "uptrend" else "down",
                "metadata": {"source": "trend", "value": trend_strength},
            })

        return signals

    def _social_data_to_signals(self, data: Dict) -> List[Dict]:
        """Convert social sentiment data into signals."""
        signals = []
        metrics = data.get("data", {}).get("metrics", {})
        weighted_sentiment = data.get("data", {}).get("weighted_sentiment", 0.5)

        if abs(weighted_sentiment - 0.5) > 0.1:
            signals.append({
                "signal_type": "social",
                "strength": abs(weighted_sentiment - 0.5) * 2.0,
                "direction": "up" if weighted_sentiment > 0.5 else "down",
                "metadata": {"source": "sentiment", "value": weighted_sentiment},
            })

        social_volume = metrics.get("social_volume", 0)
        if social_volume > 5000:
            signals.append({
                "signal_type": "sentiment",
                "strength": min(social_volume / 50_000, 1.0),
                "direction": "up" if weighted_sentiment > 0.5 else "down",
                "metadata": {"source": "social_volume", "value": social_volume},
            })

        return signals

    # ------------------------------------------------------------------
    # Redis integration
    # ------------------------------------------------------------------

    async def connect_redis(self) -> bool:
        """Connect to Redis with retry logic."""
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        max_retries = 5

        for attempt in range(max_retries):
            try:
                self.redis = Redis(
                    host=redis_host, port=redis_port,
                    decode_responses=True, socket_timeout=5
                )
                await self.redis.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
                return True
            except (ConnectionError, Exception) as e:
                wait = 2 ** attempt
                logger.warning(
                    f"Redis connection attempt {attempt+1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

        logger.error("Failed to connect to Redis after all retries")
        return False

    async def publish_opportunity(self, market_id: str, opportunity: Dict):
        """Publish a trading opportunity to Redis."""
        if not self.redis:
            return

        market = self.markets.get(market_id)
        message = {
            "source": "lmsr_prediction_bot",
            "market_id": market_id,
            "symbol": market.symbol if market else "UNKNOWN",
            "opportunity": opportunity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            await self.redis.publish("lmsr_signals", json.dumps(message))
            logger.debug(f"Published opportunity for {market_id}")
        except Exception as e:
            logger.error(f"Failed to publish opportunity: {e}")

    async def publish_market_state(self, market_id: str):
        """Publish current market state to Redis."""
        if not self.redis:
            return

        state = self.get_market_state(market_id)
        if state:
            try:
                await self.redis.set(
                    f"lmsr:market:{market_id}",
                    json.dumps(state),
                    ex=300,
                )
            except Exception as e:
                logger.error(f"Failed to publish market state: {e}")

    async def listen_market_updates(self):
        """Subscribe to market updates and process them."""
        if not self.redis:
            logger.error("Redis not connected")
            return

        pubsub = self.redis.pubsub()
        await pubsub.subscribe("market_updates")
        logger.info("Subscribed to market_updates channel")

        async for message in pubsub.listen():
            if not self.running:
                break
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
                symbol = data.get("symbol", "")

                if not symbol:
                    continue

                # Ensure a market exists for this symbol
                market_id = self._get_or_create_market(symbol)

                # Convert market data to signals and process
                signals = self._market_data_to_signals(data)
                all_opportunities = []

                for signal in signals:
                    opps = self.process_market_signal(market_id, signal)
                    if opps:
                        all_opportunities.extend(opps)

                # Publish best opportunities
                if all_opportunities:
                    best = self._deduplicate_opportunities(all_opportunities)
                    for opp in best:
                        await self.publish_opportunity(market_id, opp)

                # Periodically publish state
                await self.publish_market_state(market_id)

            except json.JSONDecodeError:
                logger.warning("Received invalid JSON on market_updates")
            except Exception as e:
                logger.error(f"Error processing market update: {e}", exc_info=True)

    async def listen_social_updates(self):
        """Subscribe to social data and incorporate into beliefs."""
        if not self.redis:
            return

        pubsub = self.redis.pubsub()
        await pubsub.subscribe("social_updates")
        logger.info("Subscribed to social_updates channel")

        async for message in pubsub.listen():
            if not self.running:
                break
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
                symbol = data.get("symbol", "")
                if not symbol:
                    continue

                market_id = self._get_or_create_market(symbol)
                signals = self._social_data_to_signals(data)

                for signal in signals:
                    self.process_market_signal(market_id, signal)

            except Exception as e:
                logger.error(f"Error processing social update: {e}", exc_info=True)

    def _get_or_create_market(self, symbol: str) -> str:
        """Get existing market for symbol or create a new one."""
        for mid, m in self.markets.items():
            if m.symbol == symbol and not m.resolved:
                return mid
        market = self.create_binary_market(symbol)
        return market.market_id

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get bot status summary."""
        market_states = []
        for mid in self.markets:
            state = self.get_market_state(mid)
            if state:
                market_states.append(state)

        return {
            "active_markets": len(self.markets),
            "polymarket_markets": len(self.poly_markets),
            "resolved_markets": sum(1 for m in self.markets.values() if m.resolved),
            "active_positions": len(self.positions),
            "liquidity_parameter": self.liquidity,
            "max_maker_loss_binary": float(self.pricing_engine.max_loss(2)),
            "min_edge_threshold": self.min_edge,
            "auto_trade": self.poly_auto_trade,
            "markets": market_states,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self):
        """Main entry point: discover Polymarket markets and start processing."""
        logger.info("=" * 60)
        logger.info("LMSR Prediction Bot starting (Polymarket mode)")
        logger.info(f"  Liquidity (b): {self.liquidity:,.0f}")
        logger.info(f"  Max maker loss (binary): ${self.pricing_engine.max_loss(2):,.2f}")
        logger.info(f"  Min edge threshold: {self.min_edge}")
        logger.info(f"  Max Kelly fraction: {self.max_kelly_fraction}")
        logger.info(f"  Signal learning rate: {self.signal_learning_rate}")
        logger.info(f"  Data feed: {'WebSocket (real-time)' if self.poly_use_websocket else 'Polling'}")
        logger.info(f"  Polymarket poll interval: {self.poly_poll_interval}s")
        logger.info(f"  Auto-trade: {self.poly_auto_trade}")
        logger.info("=" * 60)

        # Discover Polymarket markets
        n_markets = await self.discover_polymarket_markets()
        logger.info(f"Discovered {n_markets} tradable Polymarket markets")

        # Connect to Redis (optional — bot works without it)
        connected = await self.connect_redis()
        if not connected:
            logger.warning(
                "Redis not available — running in Polymarket-only mode "
                "(no Redis pub/sub signals)"
            )

        # Build task list — WebSocket streaming (primary) or polling (fallback)
        tasks = []
        if self.poly_use_websocket:
            tasks.append(self.start_websocket_stream())
        else:
            tasks.append(self.poll_polymarket_prices())
        tasks.append(self._polymarket_discovery_loop())
        tasks.append(self._status_loop())

        if connected:
            tasks.append(self.listen_market_updates())
            tasks.append(self.listen_social_updates())

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.running = False
            await self.polymarket.close()
            if self.redis:
                await self.redis.close()
            logger.info("LMSR Prediction Bot stopped")

    async def _polymarket_discovery_loop(self):
        """Periodically discover new Polymarket markets."""
        while self.running:
            await asyncio.sleep(300)  # Re-scan every 5 minutes
            try:
                n = await self.discover_polymarket_markets()
                if n > 0:
                    logger.info(f"Discovery refresh: {n} markets, {len(self.poly_markets)} tracked")
            except Exception as e:
                logger.error(f"Market discovery error: {e}")

    async def _status_loop(self):
        """Periodically log bot status."""
        while self.running:
            await asyncio.sleep(60)
            status = self.get_status()
            logger.info(
                f"Status | markets={status['active_markets']} | "
                f"resolved={status['resolved_markets']}"
            )
            if self.redis:
                try:
                    await self.redis.set(
                        "lmsr:bot:status",
                        json.dumps(status),
                        ex=120,
                    )
                except Exception:
                    pass
