#!/usr/bin/env python3
"""
💰 TRANSACTION COST & SLIPPAGE MODELING
Models realistic trading costs for accurate performance assessment
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

class TransactionCostModel:
    def __init__(self):
        """Initialize transaction cost model with realistic parameters"""

        # Commission structure (Alpaca paper trading = $0, but model for live)
        self.commission_per_share = 0.005  # $0.005 per share (typical)
        self.min_commission = 1.00  # Minimum $1 per trade
        self.max_commission = 5.00  # Cap commission at $5

        # Spread costs (bid-ask spread impact)
        self.spread_costs = {
            'large_cap': 0.0005,   # 0.05% for AAPL, MSFT, etc.
            'mid_cap': 0.0010,     # 0.10% for smaller stocks
            'etf': 0.0002,         # 0.02% for SPY, QQQ
            'volatile': 0.0015     # 0.15% for high volatility stocks
        }

        # Market impact (slippage) model
        self.market_impact_base = 0.0001  # Base 0.01% market impact
        self.liquidity_factors = {
            'AAPL': 0.5, 'MSFT': 0.5, 'GOOGL': 0.7, 'AMZN': 0.8,
            'NVDA': 0.9, 'TSLA': 1.2, 'META': 0.8, 'NFLX': 1.0,
            'SPY': 0.3, 'QQQ': 0.4, 'IWM': 0.8
        }

    def calculate_trading_costs(self, symbol: str, quantity: int, price: float,
                               side: str, market_conditions: dict = None) -> dict:
        """Calculate comprehensive trading costs for an order"""
        try:
            notional = quantity * price

            # 1. Commission costs
            commission = max(
                self.min_commission,
                min(self.max_commission, quantity * self.commission_per_share)
            )

            # 2. Spread costs
            spread_cost = self._calculate_spread_cost(symbol, notional)

            # 3. Market impact (slippage)
            market_impact = self._calculate_market_impact(
                symbol, quantity, price, side, market_conditions
            )

            # 4. Total costs
            total_cost = commission + spread_cost + market_impact
            cost_bps = (total_cost / notional) * 10000  # Basis points

            cost_breakdown = {
                "commission": commission,
                "spread_cost": spread_cost,
                "market_impact": market_impact,
                "total_cost": total_cost,
                "cost_bps": cost_bps,
                "cost_percent": (total_cost / notional) * 100,
                "notional": notional,
                "effective_price": price + (market_impact / quantity) * (1 if side == 'buy' else -1)
            }

            logger.info(f"💰 Trading costs for {symbol}: ${total_cost:.2f} ({cost_bps:.1f}bps)")
            return cost_breakdown

        except Exception as e:
            logger.error(f"Error calculating trading costs: {e}")
            # Fallback conservative estimate
            return {
                "commission": 1.0,
                "spread_cost": notional * 0.001,  # 0.1%
                "market_impact": notional * 0.0005,  # 0.05%
                "total_cost": notional * 0.002,  # 0.2% total
                "cost_bps": 20.0,
                "cost_percent": 0.2,
                "notional": notional,
                "effective_price": price,
                "error": str(e)
            }

    def _calculate_spread_cost(self, symbol: str, notional: float) -> float:
        """Calculate bid-ask spread cost"""
        try:
            # Classify symbol for spread estimation
            if symbol in ['SPY', 'QQQ', 'IWM']:
                spread_rate = self.spread_costs['etf']
            elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                spread_rate = self.spread_costs['large_cap']
            elif symbol in ['TSLA', 'NVDA']:
                spread_rate = self.spread_costs['volatile']
            else:
                spread_rate = self.spread_costs['mid_cap']

            return notional * spread_rate

        except Exception as e:
            logger.warning(f"Error calculating spread cost: {e}")
            return notional * 0.001  # Default 0.1%

    def _calculate_market_impact(self, symbol: str, quantity: int, price: float,
                                side: str, market_conditions: dict = None) -> float:
        """Calculate market impact (temporary price impact from order)"""
        try:
            notional = quantity * price

            # Base impact
            liquidity_factor = self.liquidity_factors.get(symbol, 1.0)
            base_impact = self.market_impact_base * liquidity_factor

            # Size impact (square root relationship)
            # Larger orders have proportionally higher impact
            size_factor = np.sqrt(max(1, notional / 10000))  # Relative to $10k base

            # Volatility impact
            vol_factor = 1.0
            if market_conditions and 'volatility' in market_conditions:
                vol_factor = 1 + market_conditions['volatility']  # Higher vol = higher impact

            # Time-of-day impact (market open/close have higher impact)
            time_factor = 1.0
            if market_conditions and 'time_factor' in market_conditions:
                time_factor = market_conditions['time_factor']

            # Calculate total market impact
            impact_rate = base_impact * size_factor * vol_factor * time_factor
            impact_cost = notional * impact_rate

            # Cap impact at reasonable levels
            max_impact_rate = 0.005  # 0.5% maximum
            impact_cost = min(impact_cost, notional * max_impact_rate)

            return impact_cost

        except Exception as e:
            logger.warning(f"Error calculating market impact: {e}")
            return notional * 0.0005  # Default 0.05%

    def estimate_holding_costs(self, positions: list, days_held: float) -> dict:
        """Estimate costs of holding positions (financing, etc.)"""
        try:
            total_notional = sum(abs(float(pos.get('market_value', 0))) for pos in positions)

            # Financing costs (for margin/short positions)
            financing_rate = 0.05  # 5% annual rate
            daily_financing_rate = financing_rate / 365

            financing_cost = total_notional * daily_financing_rate * days_held

            # Opportunity cost (cash not earning interest)
            opportunity_rate = 0.03  # 3% annual risk-free rate
            opportunity_cost = total_notional * (opportunity_rate / 365) * days_held

            return {
                "financing_cost": financing_cost,
                "opportunity_cost": opportunity_cost,
                "total_holding_cost": financing_cost + opportunity_cost,
                "daily_cost_bps": (financing_cost + opportunity_cost) / total_notional * 10000 / days_held if days_held > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating holding costs: {e}")
            return {"financing_cost": 0, "opportunity_cost": 0, "total_holding_cost": 0, "daily_cost_bps": 0}

    def adjust_performance_for_costs(self, gross_pnl: float, trades_data: list) -> dict:
        """Adjust performance metrics for realistic trading costs"""
        try:
            total_cost = 0
            trade_count = len(trades_data)

            for trade in trades_data:
                if isinstance(trade, dict) and 'notional' in trade:
                    # Use actual cost data if available
                    total_cost += trade.get('total_cost', 0)
                else:
                    # Estimate costs for legacy trade data
                    notional = trade.get('quantity', 100) * trade.get('price', 100)
                    estimated_cost = notional * 0.002  # 0.2% estimate
                    total_cost += estimated_cost

            net_pnl = gross_pnl - total_cost
            cost_impact = (total_cost / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0

            return {
                "gross_pnl": gross_pnl,
                "total_costs": total_cost,
                "net_pnl": net_pnl,
                "cost_impact_percent": cost_impact,
                "avg_cost_per_trade": total_cost / max(1, trade_count),
                "cost_as_pct_of_notional": 0.2,  # Typical estimate
                "trades_analyzed": trade_count
            }

        except Exception as e:
            logger.error(f"Error adjusting performance for costs: {e}")
            return {
                "gross_pnl": gross_pnl,
                "total_costs": abs(gross_pnl) * 0.02,  # 2% fallback
                "net_pnl": gross_pnl * 0.98,
                "cost_impact_percent": 2.0,
                "error": str(e)
            }

# Global transaction cost model instance
transaction_cost_model = None

def initialize_transaction_cost_model():
    """Initialize the global transaction cost model"""
    global transaction_cost_model
    transaction_cost_model = TransactionCostModel()
    logger.info("💰 Transaction cost model initialized")

def get_estimated_costs(symbol: str, quantity: int, price: float, side: str) -> dict:
    """Get estimated trading costs for an order"""
    if not transaction_cost_model:
        initialize_transaction_cost_model()

    return transaction_cost_model.calculate_trading_costs(symbol, quantity, price, side)

if __name__ == "__main__":
    print("💰 Transaction Cost & Slippage Modeling System")

    # Example usage
    initialize_transaction_cost_model()

    costs = get_estimated_costs("AAPL", 100, 150.0, "buy")
    print(f"Example AAPL trade costs: {costs}")
