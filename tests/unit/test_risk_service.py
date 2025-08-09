#!/usr/bin/env python3
"""
🧪 RISK SERVICE UNIT TESTS
Test suite for risk management functionality
"""

import pytest
from backend.services.risk_service import RiskService


class TestRiskService:
    """Test cases for RiskService"""

    def setup_method(self):
        """Set up test fixtures"""
        self.risk_service = RiskService()

    def test_position_risk_approved(self):
        """Test position risk check that should be approved"""
        result = self.risk_service.check_position_risk(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            current_portfolio_value=100000.0
        )
        
        assert result["approved"] is True
        assert result["symbol"] == "AAPL"
        assert result["portfolio_heat"] == 0.015  # 15k/100k = 1.5%

    def test_position_risk_rejected_size(self):
        """Test position risk check rejected due to size"""
        result = self.risk_service.check_position_risk(
            symbol="AAPL", 
            quantity=10000,  # Too large
            price=150.0,
            current_portfolio_value=100000.0
        )
        
        assert result["approved"] is False
        assert "exceeds maximum" in result["risks"][0]

    def test_position_risk_rejected_heat(self):
        """Test position risk check rejected due to portfolio heat"""
        result = self.risk_service.check_position_risk(
            symbol="AAPL",
            quantity=1000,
            price=150.0,  # 150k position on 100k portfolio = 150%
            current_portfolio_value=100000.0
        )
        
        assert result["approved"] is False
        assert "Portfolio heat" in result["risks"][0]

    def test_portfolio_risk_within_limits(self):
        """Test portfolio risk check within limits"""
        result = self.risk_service.check_portfolio_risk(
            current_pnl=1000.0,  # Profit
            portfolio_value=100000.0
        )
        
        assert result is True

    def test_portfolio_risk_exceeds_limits(self):
        """Test portfolio risk check exceeding limits"""
        result = self.risk_service.check_portfolio_risk(
            current_pnl=-6000.0,  # 6% loss exceeds 5% limit
            portfolio_value=100000.0
        )
        
        assert result is False

    def test_calculate_position_size(self):
        """Test position size calculation"""
        position_size = self.risk_service.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            risk_per_share=5.0,  # $5 stop loss per share
            portfolio_value=100000.0
        )
        
        # 1% of 100k = 1000 risk / 5 risk per share = 200 shares
        assert position_size == 200

    def test_calculate_position_size_zero_risk(self):
        """Test position size calculation with zero risk"""
        position_size = self.risk_service.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            risk_per_share=0.0,
            portfolio_value=100000.0
        )
        
        assert position_size == 0
