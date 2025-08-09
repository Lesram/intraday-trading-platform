"""
🔥 STEP 13: GO/NO-GO CHECKLIST TESTS (Layer 1 Requirement)
Final production readiness validation - comprehensive system verification
"""
import pytest
import httpx
import asyncio
from datetime import datetime


@pytest.mark.asyncio
async def test_critical_functionality_checklist_step13(client, alpaca_mock, alpaca_urls):
    """
    Test: Critical functionality checklist (Step 13 requirement)
    Verify all critical trading functions work end-to-end.
    This is the final gate before production deployment.
    """
    print("\n🔍 CRITICAL FUNCTIONALITY CHECKLIST:")
    
    # Mock account and positions for comprehensive testing
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "10000.00",
            "cash": "5000.00",
            "portfolio_value": "10000.00",
            "equity": "10000.00",
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "10000.00",
            "daytrading_buying_power": "10000.00",
            "daytrade_count": 0,
            "pattern_day_trader": False,
            "created_at": "2024-01-01T00:00:00Z",
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "sma": "0",
            "accrued_fees": "0"
        })
    )

    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )

    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": "go_no_go_test_order_123",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "5",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    alpaca_mock.delete(f"{alpaca_urls['single_order']}go_no_go_test_order_123").mock(
        return_value=httpx.Response(204)
    )

    checklist_results = {}

    # ✅ 1. System Health Check
    print("   Checking system health...")
    health_response = await client.get("/health")
    checklist_results["system_health"] = health_response.status_code == 200
    
    if health_response.status_code == 200:
        print("   ✅ System health: PASS")
    else:
        print(f"   ❌ System health: FAIL (status {health_response.status_code})")

    # ✅ 2. Order Submission
    print("   Testing order submission...")
    order_data = {
        "symbol": "AAPL",
        "qty": 5.0,
        "side": "buy"
    }
    order_response = await client.post("/api/trading/orders/market", json=order_data)
    checklist_results["order_submission"] = order_response.status_code == 200
    
    order_ref = None
    if order_response.status_code == 200:
        print("   ✅ Order submission: PASS")
        order_data_response = order_response.json()
        order_ref = order_data_response.get("order_ref")
    else:
        print(f"   ❌ Order submission: FAIL (status {order_response.status_code})")

    # ✅ 3. Risk Management Integration
    print("   Testing risk management...")
    large_order_data = {
        "symbol": "AAPL", 
        "qty": 100.0,  # Large order to trigger risk limits
        "side": "buy"
    }
    risk_response = await client.post("/api/trading/orders/market", json=large_order_data)
    
    # Risk management should either reject or modify the order
    if risk_response.status_code == 200:
        risk_data = risk_response.json()
        risk_working = (
            risk_data.get("status") == "rejected" or 
            risk_data.get("approved_qty", 100) < 100
        )
        checklist_results["risk_management"] = risk_working
        if risk_working:
            print("   ✅ Risk management: PASS")
        else:
            print("   ❌ Risk management: FAIL (large order not blocked/modified)")
    else:
        checklist_results["risk_management"] = False
        print(f"   ❌ Risk management: FAIL (error {risk_response.status_code})")

    # ✅ 4. Order Cancellation
    print("   Testing order cancellation...")
    if order_ref:
        cancel_data = {"reason": "Go/No-Go test cancellation"}
        cancel_response = await client.post(f"/api/trading/orders/{order_ref}/cancel", json=cancel_data)
        checklist_results["order_cancellation"] = cancel_response.status_code == 200
        
        if cancel_response.status_code == 200:
            print("   ✅ Order cancellation: PASS")
        else:
            print(f"   ❌ Order cancellation: FAIL (status {cancel_response.status_code})")
    else:
        checklist_results["order_cancellation"] = False
        print("   ❌ Order cancellation: FAIL (no order to cancel)")

    # ✅ 5. Input Validation
    print("   Testing input validation...")
    invalid_order_data = {
        "symbol": "",  # Invalid empty symbol
        "qty": -5.0,   # Invalid negative quantity
        "side": "invalid"  # Invalid side
    }
    validation_response = await client.post("/api/trading/orders/market", json=invalid_order_data)
    checklist_results["input_validation"] = validation_response.status_code in [400, 422]
    
    if validation_response.status_code in [400, 422]:
        print("   ✅ Input validation: PASS")
    else:
        print(f"   ❌ Input validation: FAIL (status {validation_response.status_code})")

    # Calculate overall pass rate
    passed_checks = sum(checklist_results.values())
    total_checks = len(checklist_results)
    pass_rate = passed_checks / total_checks * 100

    print(f"\n📊 CRITICAL FUNCTIONALITY RESULTS: {passed_checks}/{total_checks} checks passed ({pass_rate:.1f}%)")

    # Critical functionality must have high pass rate
    assert pass_rate >= 80, f"Critical functionality pass rate too low: {pass_rate:.1f}%"
    
    # Certain checks are absolutely critical
    critical_checks = ["system_health", "order_submission"]
    for check in critical_checks:
        assert checklist_results[check], f"Critical check failed: {check}"


@pytest.mark.asyncio
async def test_security_readiness_checklist_step13(client):
    """
    Test: Security readiness checklist (Step 13 requirement)
    Verify security configurations and protections are in place.
    Critical security checks before production deployment.
    """
    print("\n🔒 SECURITY READINESS CHECKLIST:")
    
    security_results = {}

    # ✅ 1. Error Information Disclosure
    print("   Checking error information disclosure...")
    error_response = await client.post("/api/trading/orders/market", json={})
    error_text = error_response.text.lower()
    
    # Should not expose sensitive internal information
    sensitive_terms = ["traceback", "file path", "internal server error", "database connection"]
    info_disclosed = any(term in error_text for term in sensitive_terms)
    security_results["error_disclosure"] = not info_disclosed
    
    if not info_disclosed:
        print("   ✅ Error disclosure: PASS")
    else:
        print("   ❌ Error disclosure: FAIL (sensitive info exposed)")

    # ✅ 2. Admin Endpoint Security
    print("   Checking admin endpoint security...")
    admin_endpoints = ["/admin", "/api/admin", "/debug"]
    admin_secure = True
    
    for endpoint in admin_endpoints:
        admin_response = await client.get(endpoint)
        if admin_response.status_code == 200:
            admin_secure = False  # Admin endpoints should not be publicly accessible
            break
    
    security_results["admin_endpoints"] = admin_secure
    
    if admin_secure:
        print("   ✅ Admin endpoints: PASS")
    else:
        print("   ❌ Admin endpoints: FAIL (publicly accessible)")

    # ✅ 3. Input Sanitization
    print("   Checking input sanitization...")
    malicious_inputs = [
        {"symbol": "<script>alert('xss')</script>", "qty": 1.0, "side": "buy"},
        {"symbol": "'; DROP TABLE orders; --", "qty": 1.0, "side": "buy"},
        {"symbol": "AAPL", "qty": 1.0, "side": "buy'; DROP TABLE users; --"}
    ]
    
    sanitization_working = True
    for malicious_input in malicious_inputs:
        mal_response = await client.post("/api/trading/orders/market", json=malicious_input)
        if mal_response.status_code not in [400, 422]:
            # Should reject malicious input with validation error
            sanitization_working = False
            break
    
    security_results["input_sanitization"] = sanitization_working
    
    if sanitization_working:
        print("   ✅ Input sanitization: PASS")
    else:
        print("   ❌ Input sanitization: FAIL (malicious input accepted)")

    # ✅ 4. Response Headers Security
    print("   Checking security response headers...")
    health_response = await client.get("/health")
    
    # Check for important security headers
    security_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection"
    ]
    
    headers_present = 0
    for header in security_headers:
        if header in health_response.headers:
            headers_present += 1
    
    # At least some security headers should be present in production
    security_results["security_headers"] = headers_present >= 1
    
    if headers_present >= 1:
        print(f"   ✅ Security headers: PASS ({headers_present}/{len(security_headers)} present)")
    else:
        print("   ❌ Security headers: FAIL (none present)")

    # Calculate security pass rate
    passed_security = sum(security_results.values())
    total_security = len(security_results)
    security_pass_rate = passed_security / total_security * 100

    print(f"\n🔒 SECURITY READINESS RESULTS: {passed_security}/{total_security} checks passed ({security_pass_rate:.1f}%)")

    # Security checks should have high pass rate
    assert security_pass_rate >= 75, f"Security readiness pass rate too low: {security_pass_rate:.1f}%"


@pytest.mark.asyncio
async def test_performance_readiness_checklist_step13(client, alpaca_mock, alpaca_urls):
    """
    Test: Performance readiness checklist (Step 13 requirement)
    Verify system performance meets production requirements.
    Final performance validation before go-live.
    """
    print("\n⚡ PERFORMANCE READINESS CHECKLIST:")
    
    # Mock for performance testing
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "10000.00",
            "cash": "5000.00",
            "portfolio_value": "10000.00",
            "equity": "10000.00",
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "10000.00",
            "daytrading_buying_power": "10000.00",
            "daytrade_count": 0,
            "pattern_day_trader": False,
            "created_at": "2024-01-01T00:00:00Z",
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "sma": "0",
            "accrued_fees": "0"
        })
    )

    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )

    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": "performance_test_order",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    performance_results = {}

    # ✅ 1. Response Time Check
    print("   Checking response times...")
    import time
    
    start_time = time.time()
    health_response = await client.get("/health")
    health_time = (time.time() - start_time) * 1000
    
    performance_results["health_response_time"] = health_time < 1000  # Under 1 second
    
    if health_time < 1000:
        print(f"   ✅ Health response time: PASS ({health_time:.0f}ms)")
    else:
        print(f"   ❌ Health response time: FAIL ({health_time:.0f}ms)")

    # ✅ 2. Order Processing Speed
    print("   Checking order processing speed...")
    order_data = {"symbol": "AAPL", "qty": 1.0, "side": "buy"}
    
    start_time = time.time()
    order_response = await client.post("/api/trading/orders/market", json=order_data)
    order_time = (time.time() - start_time) * 1000
    
    performance_results["order_processing_time"] = (
        order_response.status_code == 200 and order_time < 2000  # Under 2 seconds for test
    )
    
    if order_response.status_code == 200 and order_time < 2000:
        print(f"   ✅ Order processing time: PASS ({order_time:.0f}ms)")
    else:
        print(f"   ❌ Order processing time: FAIL ({order_time:.0f}ms, status {order_response.status_code})")

    # ✅ 3. Concurrent Load Handling
    print("   Checking concurrent load handling...")
    concurrent_requests = 5  # Reduced for test environment
    
    start_time = time.time()
    tasks = [client.get("/health") for _ in range(concurrent_requests)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    successful_responses = 0
    for response in responses:
        if not isinstance(response, Exception) and response.status_code == 200:
            successful_responses += 1
    
    load_success_rate = successful_responses / concurrent_requests * 100
    performance_results["concurrent_load"] = load_success_rate >= 80
    
    if load_success_rate >= 80:
        print(f"   ✅ Concurrent load: PASS ({load_success_rate:.1f}% success rate)")
    else:
        print(f"   ❌ Concurrent load: FAIL ({load_success_rate:.1f}% success rate)")

    # ✅ 4. Resource Stability
    print("   Checking resource stability...")
    # Test multiple requests to check for resource leaks
    stability_requests = 10
    
    stable_responses = 0
    for i in range(stability_requests):
        stability_response = await client.get("/health")
        if stability_response.status_code == 200:
            stable_responses += 1
        
        # Small delay between requests
        await asyncio.sleep(0.05)
    
    stability_rate = stable_responses / stability_requests * 100
    performance_results["resource_stability"] = stability_rate >= 90
    
    if stability_rate >= 90:
        print(f"   ✅ Resource stability: PASS ({stability_rate:.1f}% stable)")
    else:
        print(f"   ❌ Resource stability: FAIL ({stability_rate:.1f}% stable)")

    # Calculate performance pass rate
    passed_performance = sum(performance_results.values())
    total_performance = len(performance_results)
    perf_pass_rate = passed_performance / total_performance * 100

    print(f"\n⚡ PERFORMANCE READINESS RESULTS: {passed_performance}/{total_performance} checks passed ({perf_pass_rate:.1f}%)")

    # Performance checks should have high pass rate
    assert perf_pass_rate >= 75, f"Performance readiness pass rate too low: {perf_pass_rate:.1f}%"


@pytest.mark.asyncio
async def test_final_go_no_go_decision_step13(client, alpaca_mock, alpaca_urls):
    """
    Test: Final Go/No-Go decision (Step 13 requirement)
    Comprehensive final check - this determines production readiness.
    ALL critical systems must be operational for GO decision.
    """
    print("\n🚦 FINAL GO/NO-GO DECISION CHECKLIST:")
    
    # Mock for comprehensive testing
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "10000.00",
            "cash": "5000.00",
            "portfolio_value": "10000.00",
            "equity": "10000.00",
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "10000.00",
            "daytrading_buying_power": "10000.00",
            "daytrade_count": 0,
            "pattern_day_trader": False,
            "created_at": "2024-01-01T00:00:00Z",
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "sma": "0",
            "accrued_fees": "0"
        })
    )

    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )

    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": "final_test_order_123",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "3",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    final_checklist = {}

    # 🔴 CRITICAL: System Health
    print("   🔴 CRITICAL: System health check...")
    health_response = await client.get("/health")
    final_checklist["critical_system_health"] = health_response.status_code == 200

    # 🔴 CRITICAL: Core Trading Function
    print("   🔴 CRITICAL: Core trading functionality...")
    order_data = {"symbol": "AAPL", "qty": 3.0, "side": "buy"}
    trade_response = await client.post("/api/trading/orders/market", json=order_data)
    final_checklist["critical_trading"] = trade_response.status_code == 200

    # 🔴 CRITICAL: Risk Management
    print("   🔴 CRITICAL: Risk management operational...")
    # Test that risk management is working (should modify/reject large orders)
    large_order = {"symbol": "AAPL", "qty": 200.0, "side": "buy"}  # Very large order
    risk_response = await client.post("/api/trading/orders/market", json=large_order)
    
    risk_working = False
    if risk_response.status_code == 200:
        risk_data = risk_response.json()
        # Risk should either reject or significantly modify the large order
        risk_working = (
            risk_data.get("status") == "rejected" or 
            risk_data.get("approved_qty", 200) < 50  # Should be significantly reduced
        )
    
    final_checklist["critical_risk_management"] = risk_working

    # 🟡 IMPORTANT: Error Handling
    print("   🟡 IMPORTANT: Error handling...")
    error_response = await client.post("/api/trading/orders/market", json={})
    final_checklist["error_handling"] = error_response.status_code in [400, 422]

    # 🟡 IMPORTANT: Input Validation
    print("   🟡 IMPORTANT: Input validation...")
    invalid_input = {"symbol": "", "qty": -1.0, "side": "invalid"}
    validation_response = await client.post("/api/trading/orders/market", json=invalid_input)
    final_checklist["input_validation"] = validation_response.status_code in [400, 422]

    # 🟢 NICE-TO-HAVE: Metrics Endpoint
    print("   🟢 NICE-TO-HAVE: Metrics endpoint...")
    metrics_response = await client.get("/metrics")
    final_checklist["metrics_endpoint"] = metrics_response.status_code in [200, 404]  # OK if not implemented

    # Calculate results
    critical_checks = ["critical_system_health", "critical_trading", "critical_risk_management"]
    important_checks = ["error_handling", "input_validation"]
    nice_to_have_checks = ["metrics_endpoint"]

    critical_passed = sum(final_checklist[check] for check in critical_checks)
    important_passed = sum(final_checklist[check] for check in important_checks)
    nice_to_have_passed = sum(final_checklist[check] for check in nice_to_have_checks)

    total_passed = sum(final_checklist.values())
    total_checks = len(final_checklist)

    # Report results
    print(f"\n📊 FINAL CHECKLIST RESULTS:")
    print(f"   🔴 CRITICAL: {critical_passed}/{len(critical_checks)} passed")
    print(f"   🟡 IMPORTANT: {important_passed}/{len(important_checks)} passed") 
    print(f"   🟢 NICE-TO-HAVE: {nice_to_have_passed}/{len(nice_to_have_checks)} passed")
    print(f"   📈 OVERALL: {total_passed}/{total_checks} passed ({total_passed/total_checks*100:.1f}%)")

    # GO/NO-GO DECISION LOGIC
    print(f"\n🚦 GO/NO-GO DECISION:")
    
    # ALL critical checks must pass for GO
    if critical_passed == len(critical_checks):
        if important_passed >= len(important_checks) * 0.8:  # 80% of important checks
            print("   🟢 DECISION: GO - System ready for production")
            go_decision = True
        else:
            print("   🟡 DECISION: CONDITIONAL GO - Critical functions work, some important checks failed")
            go_decision = True  # Still allow with warnings
    else:
        print("   🔴 DECISION: NO-GO - Critical functionality failures detected")
        go_decision = False

    # Print failed checks for debugging
    if not go_decision or total_passed < total_checks:
        print(f"\n❌ FAILED CHECKS:")
        for check, result in final_checklist.items():
            if not result:
                print(f"   - {check}")

    # Assert based on decision
    assert go_decision, f"NO-GO decision: Critical checks failed ({critical_passed}/{len(critical_checks)})"
    
    # Also assert minimum overall pass rate
    overall_pass_rate = total_passed / total_checks * 100
    assert overall_pass_rate >= 70, f"Overall pass rate too low for production: {overall_pass_rate:.1f}%"
    
    print(f"\n✅ LAYER 1 INTEGRATION TEST HARNESS COMPLETE - PRODUCTION READINESS VERIFIED")
    
    return {
        "decision": "GO" if go_decision else "NO-GO",
        "critical_passed": critical_passed,
        "critical_total": len(critical_checks),
        "overall_pass_rate": overall_pass_rate,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
