# 🌐 PORT REFERENCE - INSTITUTIONAL TRADING PLATFORM
## Generated: August 7, 2025 - 8:49 PM

### 🎯 TARGET PLATFORM CONFIGURATION (Working State from 3am)
```
✅ INSTITUTIONAL-GRADE SETUP:
├── Trading Gateway (FastAPI): port 8002
├── React Dashboard: port 3003  
├── WebSocket: ws://localhost:8002/ws
├── API Documentation: http://localhost:8002/docs
└── Dashboard URL: http://localhost:3003
```

### 📊 CURRENT PORT STATUS (All Available)
```
🟢 AVAILABLE PORTS:
├── 3000 - Available (React dev default)
├── 3001 - Available (React secondary)
├── 3002 - Available (Alternative dashboard)
├── 3003 - Available ✅ TARGET for React Dashboard
├── 3004 - Available (Alternative)
├── 5000 - Available (Flask default)
├── 8000 - Available (Alternative API)
├── 8001 - Available (Alternative API)
├── 8002 - Available ✅ TARGET for FastAPI Gateway
├── 8003 - Available (Alternative API)
```

### 🔍 SYSTEM PORTS IN USE (Windows/VS Code)
```
🔵 SYSTEM PORTS:
├── 135, 445 - Windows services
├── 5040, 7680 - Windows internal
├── 5556 - VS Code related
├── 12754, 15292, 15393, etc. - VS Code internals
├── 49664-49684 - Windows dynamic ports
```

### 🚀 PLATFORM RESTORATION PLAN
```
STEP 1: Start FastAPI Trading Gateway on port 8002
├── File: alpaca_trading_gateway.py (2904 lines)
├── Features: Full institutional trading system
├── Config: Load from config/.env
├── Expected: http://localhost:8002/docs

STEP 2: Start React Dashboard on port 3003
├── Directory: trading-dashboard/
├── Command: npm run dev (configured for port 3003)
├── Features: Professional TypeScript/Material-UI
├── Expected: http://localhost:3003

STEP 3: Verify Integration
├── API Endpoints: /api/portfolio/metrics, /api/health
├── WebSocket: Real-time data flow
├── Dashboard: Real portfolio data ($100,027.62)
├── Models: 3/4 ML models active
```

### 📋 VERIFICATION COMMANDS
```bash
# Port Check
netstat -an | findstr ":8002 :3003"

# API Health
curl http://localhost:8002/api/health

# Portfolio Data
curl http://localhost:8002/api/portfolio/metrics

# Dashboard Access
http://localhost:3003
```

---
*Reference created for institutional trading platform restoration*
