# Metrics Cardinality Review

## Overview
This document reviews our Prometheus metrics for potential high-cardinality issues that could cause memory/performance problems.

## Current Metrics Analysis

### ✅ LOW CARDINALITY (Safe)
1. **orders_submitted_total** - Labels: `symbol`, `side`, `strategy`
   - Expected cardinality: ~50 symbols × 2 sides × 5 strategies = ~500 series
   - ✅ SAFE: Well-bounded labels

2. **orders_filled_total** - Labels: `symbol`, `side` 
   - Expected cardinality: ~50 symbols × 2 sides = ~100 series
   - ✅ SAFE: Low cardinality

3. **orders_rejected_total** - Labels: `symbol`, `reason`
   - Expected cardinality: ~50 symbols × 10 reasons = ~500 series
   - ✅ SAFE: Reason is controlled enum

4. **risk_blocks_total** - Labels: `reason`, `symbol`, `severity`
   - Expected cardinality: ~10 reasons × 50 symbols × 3 severities = ~1,500 series
   - ✅ SAFE: All labels are controlled

### ⚠️ POTENTIAL ISSUES (Monitor)
5. **broker_api_latency_seconds** - Labels: `endpoint`, `method`
   - Expected cardinality: ~20 endpoints × 5 methods = ~100 series
   - ⚠️ MONITOR: Ensure endpoint URLs don't include dynamic segments

6. **market_data_lag_seconds** - Labels: `source`
   - Expected cardinality: ~5 sources = ~5 series
   - ✅ SAFE: Very low cardinality

### 📊 HISTOGRAM BUCKETS
- All histograms use reasonable fixed bucket counts (11-12 buckets)
- Total histogram series per metric = base_cardinality × buckets × 3 (count, sum, bucket)

## Recommendations

### 1. Add Cardinality Guards
```python
# Add to TradingMetrics.__init__
self._known_symbols = set()
self._known_strategies = set() 
self._known_endpoints = set()
self.MAX_SYMBOLS = 100
self.MAX_STRATEGIES = 10
self.MAX_ENDPOINTS = 50

def _validate_symbol(self, symbol: str) -> str:
    symbol = symbol.upper()
    if symbol not in self._known_symbols:
        if len(self._known_symbols) >= self.MAX_SYMBOLS:
            logger.warning(f"Max symbols reached, using 'OTHER' for {symbol}")
            return "OTHER"
        self._known_symbols.add(symbol)
    return symbol
```

### 2. Sanitize Dynamic Labels
```python
def _sanitize_endpoint(self, endpoint: str) -> str:
    """Remove dynamic segments from endpoint labels"""
    import re
    # Remove IDs, UUIDs, timestamps
    sanitized = re.sub(r'/\d+', '/{id}', endpoint)
    sanitized = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', sanitized)
    return sanitized[:50]  # Truncate long paths
```

### 3. Implement Metrics Rotation
```python
def cleanup_stale_metrics(self, max_age_hours: int = 24):
    """Remove metrics for symbols not seen recently"""
    # Implementation would track last-seen times
    pass
```

## Action Items
- [x] Document current cardinality analysis
- [ ] Implement cardinality guards in next iteration
- [ ] Add metrics monitoring dashboard with cardinality alerts
- [ ] Set up Prometheus recording rules for high-level aggregates

## Monitoring
- Current estimated total series: ~3,000-5,000
- Prometheus default limit: 1M series
- Our usage: **Well within safe limits** ✅

**Status: HEALTHY** - No immediate action required, but guards recommended for production scale.
