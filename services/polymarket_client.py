import os
import json
import time
import asyncio
import logging as logger
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Dict, List, Optional
import aiohttp

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

rotating_handler = RotatingFileHandler(
    'logs/polymarket_client.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)

logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [PolymarketClient] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

# Polymarket API base URLs
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
CHAIN_ID = 137  # Polygon


class PolymarketClient:
    """
    Polymarket API client that wraps both the Gamma API (market discovery)
    and CLOB API (trading) with async HTTP.

    For order execution, delegates to the official py-clob-client SDK.
    This class handles market discovery, price feeds, and bridges data
    into the LMSR bot's signal processing pipeline.
    """

    def __init__(self, config: Dict):
        poly_config = config.get("polymarket", {})
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
        self.signature_type = poly_config.get("signature_type", 0)

        self.gamma_url = poly_config.get("gamma_api_url", GAMMA_API_URL)
        self.clob_url = poly_config.get("clob_api_url", CLOB_API_URL)
        self.ws_url = poly_config.get("ws_url", WS_URL)

        # Filtering
        self.min_liquidity = poly_config.get("min_liquidity", 10_000)
        self.market_tags = poly_config.get("market_tags", [])
        self.active_only = poly_config.get("active_only", True)

        # Rate limiting
        self.request_interval = poly_config.get("request_interval_sec", 1.0)
        self._last_request_time = 0.0

        # Cached markets: condition_id -> market data
        self.markets_cache: Dict[str, Dict] = {}
        # token_id -> condition_id mapping
        self.token_to_market: Dict[str, str] = {}

        # CLOB client (initialized lazily when trading is needed)
        self._clob_client = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def _rate_limit(self):
        """Simple rate limiter."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Gamma API — Market Discovery
    # ------------------------------------------------------------------

    async def fetch_markets(self, limit: int = 100, offset: int = 0,
                            active: bool = True) -> List[Dict]:
        """
        Fetch markets from the Gamma API.
        Returns list of market dicts with token IDs, outcomes, and metadata.
        """
        await self._rate_limit()
        session = await self._get_session()

        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": "false",
        }
        if self.market_tags:
            params["tag"] = ",".join(self.market_tags)

        try:
            async with session.get(f"{self.gamma_url}/markets", params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Gamma API error: {resp.status}")
                    return []
                data = await resp.json()
                markets = data if isinstance(data, list) else data.get("data", [])

                # Cache and index
                for m in markets:
                    condition_id = m.get("condition_id", m.get("conditionId", ""))
                    if condition_id:
                        self.markets_cache[condition_id] = m
                        # Index token IDs
                        for token in m.get("tokens", []):
                            tid = token.get("token_id", "")
                            if tid:
                                self.token_to_market[tid] = condition_id

                logger.info(f"Fetched {len(markets)} markets from Gamma API")
                return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def fetch_events(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Fetch events (groups of related markets) from Gamma API."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"limit": limit, "offset": offset, "active": "true", "closed": "false"}
            async with session.get(f"{self.gamma_url}/events", params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []

    async def search_markets(self, query: str) -> List[Dict]:
        """Search markets by text query."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"query": query, "active": "true", "closed": "false"}
            async with session.get(f"{self.gamma_url}/markets", params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to search markets: {e}")
            return []

    # ------------------------------------------------------------------
    # CLOB API — Pricing (public, no auth)
    # ------------------------------------------------------------------

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token from CLOB."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"token_id": token_id}
            async with session.get(f"{self.clob_url}/midpoint", params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                mid = data.get("mid")
                return float(mid) if mid is not None else None
        except Exception as e:
            logger.error(f"Failed to get midpoint for {token_id}: {e}")
            return None

    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get price for a token from CLOB."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"token_id": token_id, "side": side}
            async with session.get(f"{self.clob_url}/price", params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                price = data.get("price")
                return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Failed to get price for {token_id}: {e}")
            return None

    async def get_order_book(self, token_id: str) -> Optional[Dict]:
        """Get full order book for a token."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"token_id": token_id}
            async with session.get(f"{self.clob_url}/book", params=params) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"Failed to get order book for {token_id}: {e}")
            return None

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price for a token."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            params = {"token_id": token_id}
            async with session.get(
                f"{self.clob_url}/last-trade-price", params=params
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                price = data.get("price")
                return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Failed to get last trade price: {e}")
            return None

    async def get_market_prices(self, market: Dict) -> Dict[str, float]:
        """
        Get YES/NO prices for a binary market.
        Returns {"YES": price, "NO": price}.
        """
        tokens = market.get("tokens", [])
        prices = {}

        for token in tokens:
            token_id = token.get("token_id", "")
            outcome = token.get("outcome", "")
            if not token_id or not outcome:
                continue

            mid = await self.get_midpoint(token_id)
            if mid is not None:
                prices[outcome] = mid

        return prices

    # ------------------------------------------------------------------
    # CLOB API — Trading (requires py-clob-client + auth)
    # ------------------------------------------------------------------

    def _init_clob_client(self):
        """Lazily initialize the official CLOB client for order execution."""
        if self._clob_client is not None:
            return

        if not self.private_key:
            logger.warning("No POLYMARKET_PRIVATE_KEY set — trading disabled")
            return

        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self.clob_url,
                key=self.private_key,
                chain_id=CHAIN_ID,
                signature_type=self.signature_type,
                funder=self.funder_address or None,
            )
            creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(creds)
            logger.info("CLOB client initialized for trading")
        except ImportError:
            logger.error(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")

    def place_limit_order(self, token_id: str, price: float, size: float,
                          side: str = "BUY") -> Optional[Dict]:
        """
        Place a GTC limit order via the CLOB client.

        Args:
            token_id: The outcome token to trade.
            price: Price in [0.01, 0.99].
            size: Number of shares.
            side: "BUY" or "SELL".
        """
        self._init_clob_client()
        if not self._clob_client:
            logger.error("CLOB client not available — cannot place order")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side == "BUY" else SELL
            order_args = OrderArgs(
                price=price,
                size=size,
                side=order_side,
                token_id=token_id,
            )
            signed_order = self._clob_client.create_order(order_args)
            result = self._clob_client.post_order(signed_order, OrderType.GTC)

            logger.info(
                f"Limit order placed | token={token_id[:16]}... "
                f"side={side} price={price} size={size} | result={result}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None

    def place_market_order(self, token_id: str, amount: float,
                           side: str = "BUY") -> Optional[Dict]:
        """
        Place a FOK market order via the CLOB client.

        Args:
            token_id: The outcome token to trade.
            amount: Dollar amount to spend.
            side: "BUY" or "SELL".
        """
        self._init_clob_client()
        if not self._clob_client:
            logger.error("CLOB client not available — cannot place order")
            return None

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY as BUY_SIDE

            order_side = BUY_SIDE if side == "BUY" else "SELL"
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=order_side,
            )
            signed_order = self._clob_client.create_market_order(order_args)
            result = self._clob_client.post_order(signed_order, OrderType.FOK)

            logger.info(
                f"Market order placed | token={token_id[:16]}... "
                f"side={side} amount=${amount} | result={result}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel a specific order."""
        self._init_clob_client()
        if not self._clob_client:
            return None

        try:
            result = self._clob_client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None

    def cancel_all_orders(self) -> Optional[Dict]:
        """Cancel all open orders."""
        self._init_clob_client()
        if not self._clob_client:
            return None

        try:
            result = self._clob_client.cancel_all()
            logger.info("All orders cancelled")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return None

    def get_balance(self) -> Optional[float]:
        """Get USDC balance (converted from wei)."""
        self._init_clob_client()
        if not self._clob_client:
            return None

        try:
            balance_wei = self._clob_client.get_balance()
            return float(balance_wei) / 1e6
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def parse_market_for_lmsr(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Gamma API market into the format needed by the LMSR bot.
        Returns None if market is not suitable (not binary, not active, etc.).
        """
        # Check if market has order book enabled
        enable_order_book = market.get("enableOrderBook", market.get("enable_order_book", False))
        if not enable_order_book:
            return None

        tokens = market.get("tokens", [])
        if len(tokens) != 2:
            # Only handle binary markets for now
            return None

        condition_id = market.get("condition_id", market.get("conditionId", ""))
        question = market.get("question", "")
        description = market.get("description", "")
        slug = market.get("slug", "")
        end_date = market.get("end_date_iso", market.get("endDate", ""))

        # Extract token details
        outcome_tokens = []
        for token in tokens:
            outcome_tokens.append({
                "token_id": token.get("token_id", ""),
                "outcome": token.get("outcome", ""),
                "price": float(token.get("price", 0.5)),
            })

        # Sort so YES is first
        outcome_tokens.sort(key=lambda t: t["outcome"] != "Yes")

        return {
            "condition_id": condition_id,
            "question": question,
            "description": description,
            "slug": slug,
            "end_date": end_date,
            "tokens": outcome_tokens,
            "outcomes": [t["outcome"] for t in outcome_tokens],
            "initial_prices": [t["price"] for t in outcome_tokens],
        }

    async def discover_tradable_markets(self, limit: int = 50) -> List[Dict]:
        """
        Discover active binary markets with order books enabled.
        Returns parsed market dicts ready for the LMSR bot.
        """
        raw_markets = await self.fetch_markets(limit=limit, active=True)
        tradable = []

        for m in raw_markets:
            parsed = self.parse_market_for_lmsr(m)
            if parsed:
                tradable.append(parsed)

        logger.info(
            f"Discovered {len(tradable)} tradable binary markets "
            f"out of {len(raw_markets)} total"
        )
        return tradable
