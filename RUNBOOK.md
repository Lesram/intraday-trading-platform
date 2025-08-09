# 📋 RUNBOOK - Layer 2 Trading Platform Operations

## 🚀 System Startup & Shutdown

### Standard Startup Sequence
```bash
# 1. Navigate to project directory
cd /path/to/intraday-trading-platform

# 2. Activate environment (if using virtual env)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install/update dependencies
pip install -r requirements.txt

# 4. Verify environment configuration
cp .env.example .env
# Edit .env with your actual API keys

# 5. Start the backend API
uvicorn backend.main:app --port 8002 --reload

# 6. Verify system health
curl http://localhost:8002/api/health
```

### Production Startup (Docker)
```bash
# 1. Build Docker image
docker build -t trading-platform .

# 2. Run with environment file
docker run -d \
  --name trading-platform \
  --env-file .env \
  -p 8002:8002 \
  trading-platform

# 3. Check logs
docker logs trading-platform
```

### Graceful Shutdown
```bash
# 1. Trigger graceful shutdown
curl -X POST http://localhost:8002/api/admin/shutdown

# 2. Wait for open positions to close (if any)
curl http://localhost:8002/api/trading/positions

# 3. Stop the service
kill -TERM <process_id>
# or for Docker:
docker stop trading-platform
```

## 🛑 Emergency Procedures

### Kill Switch Activation
```bash
# Immediate stop all trading operations
curl -X POST http://localhost:8002/api/emergency/kill-switch

# Verify all positions are protected
curl http://localhost:8002/api/trading/positions

# Check system status
curl http://localhost:8002/api/health
```

### Cancel All Open Orders
```bash
# Cancel all pending orders across all symbols
curl -X DELETE http://localhost:8002/api/trading/orders/all

# Verify cancellation
curl http://localhost:8002/api/trading/orders/pending
```

### Emergency Position Liquidation
```bash
# Close all open positions (EXTREME MEASURE)
curl -X POST http://localhost:8002/api/emergency/liquidate-all

# Monitor liquidation progress
curl http://localhost:8002/api/trading/positions
```

## ⚙️ Configuration Management

### Environment Variables Reference

#### Critical Trading Settings
```bash
# Set trading mode
export TRADING_MODE=paper  # or 'live' for production

# Set execution mode  
export EXECUTION_MODE=backtest  # or 'live' for real trading

# Configure risk limits
export MAX_DAILY_LOSS=-1000.0
export MAX_POSITION_SIZE=10000.0
export PORTFOLIO_HEAT_LIMIT=0.25
```

#### Safety Rails Configuration
```bash
# Enable kill switch
export KILL_SWITCH_ENABLED=true

# Enable order throttling
export THROTTLE_ENABLED=true
export MAX_ORDERS_PER_MINUTE=60

# Risk management
export MAX_DAILY_LOSS=-1000.0  # USD
export MAX_DRAWDOWN=0.10       # 10%
```

#### Observability Settings
```bash
# Enable metrics collection
export METRICS_ENABLED=true
export METRICS_PORT=8002
export LOG_LEVEL=INFO
```

### Runtime Configuration Updates
```bash
# Update risk limits (live)
curl -X PUT http://localhost:8002/api/config/risk \
  -H "Content-Type: application/json" \
  -d '{"max_daily_loss": -500.0, "portfolio_heat_limit": 0.20}'

# Update throttling settings
curl -X PUT http://localhost:8002/api/config/throttle \
  -H "Content-Type: application/json" \
  -d '{"max_orders_per_minute": 30, "enabled": true}'
```

## 📊 Monitoring & Health Checks

### System Health Verification
```bash
# Comprehensive health check
curl http://localhost:8002/api/health | jq

# Prometheus metrics
curl http://localhost:8002/metrics

# Trading system status
curl http://localhost:8002/api/trading/status | jq
```

### Key Metrics to Monitor
```bash
# Order flow metrics
curl -s http://localhost:8002/metrics | grep "orders_submitted_total"
curl -s http://localhost:8002/metrics | grep "orders_filled_total"

# Risk management
curl -s http://localhost:8002/metrics | grep "risk_blocks_total"

# Latency metrics
curl -s http://localhost:8002/metrics | grep "order_submit_latency_seconds"

# System health
curl -s http://localhost:8002/metrics | grep "broker_connection_status"
```

### Log Analysis
```bash
# Real-time log monitoring
tail -f logs/trading_platform.log

# Filter by error level
grep "ERROR" logs/trading_platform.log

# Search for specific order ID
grep "order_12345" logs/trading_platform.log

# Risk block analysis
grep "RISK_BLOCK" logs/trading_platform.log | tail -20
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Broker Connection Failed
```bash
# Check broker status
curl http://localhost:8002/api/health/broker

# Test credentials
curl -X POST http://localhost:8002/api/broker/test-connection

# Restart with fresh credentials
export ALPACA_API_KEY=new_key
export ALPACA_SECRET_KEY=new_secret
```

#### 2. Orders Not Executing
```bash
# Check order status
curl http://localhost:8002/api/trading/orders/status/ORDER_ID

# Verify risk service
curl http://localhost:8002/api/health/risk-service

# Check throttling limits
curl http://localhost:8002/api/config/throttle
```

#### 3. High Latency Issues
```bash
# Check current latency metrics
curl -s http://localhost:8002/metrics | grep latency_seconds

# Network connectivity test
ping api.alpaca.markets

# Restart with performance mode
export MAX_WORKERS=8
uvicorn backend.main:app --port 8002 --workers 4
```

#### 4. Memory Issues
```bash
# Check memory usage
curl http://localhost:8002/api/health/system

# Clear cache
curl -X POST http://localhost:8002/api/admin/clear-cache

# Restart with memory optimization
export CACHE_SIZE=500
```

## 📈 Performance Tuning

### Optimization Settings
```bash
# High-performance configuration
export MAX_WORKERS=8
export BATCH_SIZE=200
export CACHE_SIZE=2000
export ORDER_TIMEOUT=10

# Low-latency trading
export EXECUTION_MODE=live
export THROTTLE_ENABLED=false  # Only if approved
export ORDER_TIMEOUT=5
```

### Database Optimization
```bash
# SQLite (Development)
export DATABASE_URL=sqlite:///trading_platform.db?cache=shared

# PostgreSQL (Production)
export DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db?pool_size=20
```

## 🛡️ Security Checklist

### Pre-Production Security Review
- [ ] All API keys in environment variables (not code)
- [ ] .env file in .gitignore
- [ ] Kill switch functionality tested
- [ ] Risk limits properly configured  
- [ ] Throttling enabled and tested
- [ ] Logging excludes sensitive data
- [ ] HTTPS enabled for production
- [ ] Authentication/authorization implemented

### Regular Security Tasks
```bash
# Rotate API keys monthly
export ALPACA_API_KEY=new_monthly_key

# Review access logs
grep "unauthorized" logs/trading_platform.log

# Test emergency procedures quarterly
curl -X POST http://localhost:8002/api/emergency/kill-switch
```

## 📞 Escalation Procedures

### Severity Levels

#### CRITICAL (Immediate Response)
- Unauthorized trades
- System compromised
- Large unexpected losses
- **Contact**: Operations team immediately

#### HIGH (< 30 minutes)  
- Broker connection lost
- Orders not executing
- Risk limits breached
- **Contact**: Technical lead

#### MEDIUM (< 2 hours)
- Performance degradation
- Non-critical errors
- Metrics collection issues
- **Contact**: Development team

#### LOW (Next business day)
- Documentation updates
- Enhancement requests
- Non-urgent maintenance
- **Contact**: Product team

### Emergency Contacts
```
Operations Team: +1-XXX-XXX-XXXX
Technical Lead:  +1-XXX-XXX-XXXX  
Compliance:      +1-XXX-XXX-XXXX
```

## 📋 Maintenance Schedule

### Daily Tasks
- [ ] Review overnight trading activity
- [ ] Check system health metrics
- [ ] Verify risk limits compliance
- [ ] Review error logs

### Weekly Tasks  
- [ ] Performance metrics analysis
- [ ] Database maintenance
- [ ] Security log review
- [ ] Backup verification

### Monthly Tasks
- [ ] API key rotation
- [ ] System performance review
- [ ] Emergency procedure testing
- [ ] Documentation updates
