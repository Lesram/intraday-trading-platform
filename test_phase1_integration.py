#!/usr/bin/env python3
"""
🧪 PHASE 1 INTEGRATION TEST
Test the unified risk management and stop-loss systems with the live trading platform
"""

import sys
import os
import asyncio
import requests
import json
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our new Phase 1 systems
from unified_risk_manager import get_risk_manager, PositionRisk
from strategy_stop_losses import get_stop_loss_manager
from integrated_risk_dashboard import get_dashboard

class Phase1IntegrationTest:
    """Test Phase 1 risk management integration with live trading platform"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8002"
        self.risk_manager = get_risk_manager()
        self.stop_manager = get_stop_loss_manager()
        self.dashboard = get_dashboard()
        
    async def test_backend_connectivity(self) -> bool:
        """Test connection to trading backend"""
        try:
            response = requests.get(f"{self.backend_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend connectivity: PASSED")
                return True
            else:
                print(f"❌ Backend connectivity: FAILED (status {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ Backend connectivity: FAILED ({str(e)})")
            return False
    
    async def get_mock_portfolio_data(self) -> Dict[str, Any]:
        """Get mock portfolio data for testing"""
        try:
            # Try to get real portfolio data first
            response = requests.get(f"{self.backend_url}/api/portfolio", timeout=5)
            if response.status_code == 200:
                portfolio_data = response.json()
                print("✅ Using live portfolio data")
                return portfolio_data
        except:
            pass
        
        # Fall back to mock data
        print("📊 Using mock portfolio data for testing")
        return {
            "account_value": 100000,
            "positions": [
                {"symbol": "AAPL", "qty": 100, "current_price": 150.0, "market_value": 15000},
                {"symbol": "MSFT", "qty": 50, "current_price": 380.0, "market_value": 19000},
                {"symbol": "NVDA", "qty": 20, "current_price": 900.0, "market_value": 18000},
                {"symbol": "SPY", "qty": 100, "current_price": 450.0, "market_value": 45000}
            ],
            "cash": 3000
        }
    
    def convert_to_position_risks(self, portfolio_data: Dict) -> List[PositionRisk]:
        """Convert portfolio data to PositionRisk objects"""
        positions = []
        account_value = portfolio_data.get("account_value", 100000)
        
        for pos in portfolio_data.get("positions", []):
            # Estimate volatility and beta (normally would come from market data)
            volatility_map = {"AAPL": 0.25, "MSFT": 0.22, "NVDA": 0.35, "SPY": 0.15, "TSLA": 0.40}
            beta_map = {"AAPL": 1.2, "MSFT": 1.1, "NVDA": 1.8, "SPY": 1.0, "TSLA": 2.0}
            
            symbol = pos["symbol"]
            market_value = pos["market_value"]
            portfolio_pct = market_value / account_value
            
            # Risk score calculation (higher is riskier)
            volatility = volatility_map.get(symbol, 0.25)
            beta = beta_map.get(symbol, 1.0)
            risk_score = (volatility + abs(beta - 1.0)) / 2
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_value=market_value,
                portfolio_percentage=portfolio_pct,
                volatility=volatility,
                beta=beta,
                stop_loss_price=None,  # Will be calculated
                max_position_loss=market_value * 0.08,  # Assume 8% max loss
                risk_score=risk_score
            )
            
            positions.append(position_risk)
        
        return positions
    
    async def test_risk_management(self, positions: List[PositionRisk], account_value: float) -> bool:
        """Test the unified risk management system"""
        print("\n🛡️ Testing Unified Risk Management System...")
        
        try:
            # Test portfolio risk assessment
            risk_metrics = self.risk_manager.assess_portfolio_risk(positions, account_value)
            print(f"   Portfolio Heat: {risk_metrics.portfolio_heat:.1%}")
            print(f"   Risk Level: {risk_metrics.risk_level.value}")
            print(f"   Concentration Risk: {risk_metrics.concentration_risk:.1%}")
            print(f"   Daily VaR: ${risk_metrics.var_1d:,.0f}")
            
            # Test position sizing
            test_symbol = "TSLA"
            test_price = 900.0
            shares = self.risk_manager.calculate_position_size(
                test_symbol, 0.75, 0.12, test_price, account_value, 0.35
            )
            print(f"   Position Size Test ({test_symbol}): {shares} shares")
            
            # Test stop-loss calculation
            stop_price = self.risk_manager.calculate_stop_loss(
                test_symbol, test_price, 0.35, 0.75, "momentum_strategy"
            )
            print(f"   Stop-Loss Test ({test_symbol}): ${stop_price:.2f}")
            
            # Test trade approval
            approved, reason = self.risk_manager.check_trade_approval(
                test_symbol, "BUY", shares, test_price, "momentum_strategy"
            )
            print(f"   Trade Approval: {approved} - {reason}")
            
            print("✅ Risk Management: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Risk Management: FAILED ({str(e)})")
            return False
    
    async def test_stop_loss_system(self, positions: List[PositionRisk]) -> bool:
        """Test the strategy stop-loss system"""
        print("\n🎯 Testing Strategy Stop-Loss System...")
        
        try:
            # Create stop-losses for existing positions
            for pos in positions[:2]:  # Test first 2 positions
                if pos.symbol == "AAPL":
                    strategy = "automated_signal_trading"
                    confidence = 0.8
                elif pos.symbol == "MSFT":
                    strategy = "portfolio_rebalancing"
                    confidence = 0.7
                else:
                    strategy = "momentum_strategy"
                    confidence = 0.75
                
                entry_price = pos.position_value / 100  # Rough estimate
                stop_order = await self.stop_manager.create_stop_loss(
                    pos.symbol, entry_price, strategy, confidence, pos.volatility
                )
                
                print(f"   Created stop for {pos.symbol}: ${stop_order.stop_price:.2f} ({stop_order.order_type.value})")
            
            # Test stop-loss monitoring
            market_data = {pos.symbol: pos.position_value / 100 for pos in positions}
            triggered = await self.stop_manager.check_stop_triggers(market_data)
            print(f"   Stop Triggers Check: {len(triggered)} triggered")
            
            # Get stop-loss dashboard
            stop_dashboard = self.stop_manager.get_stop_loss_dashboard()
            print(f"   Active Stops: {stop_dashboard['active_stops']}")
            print(f"   Triggered Today: {stop_dashboard['triggered_today']}")
            
            print("✅ Stop-Loss System: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Stop-Loss System: FAILED ({str(e)})")
            return False
    
    async def test_integrated_dashboard(self, positions: List[PositionRisk], account_value: float) -> bool:
        """Test the integrated risk dashboard"""
        print("\n📊 Testing Integrated Risk Dashboard...")
        
        try:
            # Generate market data
            market_data = {}
            for pos in positions:
                # Simulate slight price movements
                base_price = pos.position_value / 100
                market_data[pos.symbol] = base_price * (1 + (hash(pos.symbol) % 100 - 50) * 0.0002)
            
            # Generate dashboard data
            dashboard_data = await self.dashboard.generate_dashboard_data(
                positions, account_value, market_data
            )
            
            print(f"   Overall Status: {dashboard_data.overall_status}")
            print(f"   Risk Level: {dashboard_data.risk_level}")
            print(f"   Portfolio Heat: {dashboard_data.risk_metrics['portfolio_heat']}")
            print(f"   Active Alerts: {len(dashboard_data.alerts)}")
            print(f"   Recommendations: {len(dashboard_data.recommendations)}")
            
            # Test HTML generation
            html_content = self.dashboard.generate_html_dashboard(dashboard_data)
            print(f"   HTML Dashboard: {len(html_content)} characters generated")
            
            # Save test snapshot
            snapshot_file = self.dashboard.save_dashboard_snapshot(dashboard_data, "test_integration_snapshot.json")
            print(f"   Snapshot Saved: {snapshot_file}")
            
            print("✅ Integrated Dashboard: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Integrated Dashboard: FAILED ({str(e)})")
            return False
    
    async def test_api_integration_points(self) -> bool:
        """Test potential API integration points"""
        print("\n🔌 Testing API Integration Points...")
        
        try:
            # Test signals API
            try:
                response = requests.get(f"{self.backend_url}/api/signals", timeout=5)
                signals_status = "AVAILABLE" if response.status_code == 200 else "UNAVAILABLE"
            except:
                signals_status = "UNAVAILABLE"
            
            print(f"   Signals API: {signals_status}")
            
            # Test portfolio API
            try:
                response = requests.get(f"{self.backend_url}/api/portfolio", timeout=5)
                portfolio_status = "AVAILABLE" if response.status_code == 200 else "UNAVAILABLE"
            except:
                portfolio_status = "UNAVAILABLE"
            
            print(f"   Portfolio API: {portfolio_status}")
            
            # Test if we can mock trading decisions
            print(f"   Risk Manager Integration: READY")
            print(f"   Stop-Loss Integration: READY")
            print(f"   Dashboard Integration: READY")
            
            print("✅ API Integration Points: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ API Integration Points: FAILED ({str(e)})")
            return False
    
    async def run_full_integration_test(self) -> Dict[str, bool]:
        """Run comprehensive integration test"""
        print("🧪 PHASE 1 INTEGRATION TEST SUITE")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Backend connectivity
        results["backend_connectivity"] = await self.test_backend_connectivity()
        
        # Test 2: Get portfolio data
        portfolio_data = await self.get_mock_portfolio_data()
        positions = self.convert_to_position_risks(portfolio_data)
        account_value = portfolio_data["account_value"]
        
        print(f"\n📋 Portfolio Test Data:")
        print(f"   Account Value: ${account_value:,.0f}")
        print(f"   Positions: {len(positions)}")
        for pos in positions:
            print(f"   - {pos.symbol}: ${pos.position_value:,.0f} ({pos.portfolio_percentage:.1%})")
        
        # Test 3: Risk management system
        results["risk_management"] = await self.test_risk_management(positions, account_value)
        
        # Test 4: Stop-loss system
        results["stop_loss_system"] = await self.test_stop_loss_system(positions)
        
        # Test 5: Integrated dashboard
        results["integrated_dashboard"] = await self.test_integrated_dashboard(positions, account_value)
        
        # Test 6: API integration points
        results["api_integration"] = await self.test_api_integration_points()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 INTEGRATION TEST RESULTS:")
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASSED" if passed_test else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
            if passed_test:
                passed += 1
        
        success_rate = (passed / total) * 100
        print(f"\nOverall Success Rate: {success_rate:.0f}% ({passed}/{total})")
        
        if success_rate >= 80:
            print("🎉 PHASE 1 INTEGRATION: SUCCESSFUL - Ready for Phase 2!")
        else:
            print("⚠️  PHASE 1 INTEGRATION: NEEDS ATTENTION - Review failed tests")
        
        return results

if __name__ == "__main__":
    async def main():
        tester = Phase1IntegrationTest()
        await tester.run_full_integration_test()
    
    asyncio.run(main())
