// Trading Platform Types
export interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'FLAT';
  confidence: number;
  timestamp: string;
  ensemble_prediction: {
    lstm: number;
    xgboost: number;
    rf: number;
    final: number;
  };
  dynamic_weights: {
    lstm: number;
    xgboost: number;
    rf: number;
  };
  market_regime: string;
  sentiment_score: number;
  kelly_fraction: number;
  recommended_position: number;
  risk_metrics: {
    var_95: number;
    max_loss_1_day: number;
    heat_contribution: number;
  };
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  kelly_size_percent: number;
  heat_contribution: number;
  stop_loss: number;
  take_profit: number;
  entry_time: string;
  holding_period: string;
}

export interface PortfolioMetrics {
  total_value: number;
  available_cash: number;
  total_pnl: number;
  total_pnl_percent: number;
  portfolio_heat: number;
  max_heat_limit: number;
  portfolio_var: number;
  max_var_limit: number;
  current_drawdown: number;
  max_drawdown_limit: number;
  sharpe_ratio: number;
  num_positions: number;
  concentration_risk: string;
  correlation_alert: boolean;
}

export interface MarketRegime {
  regime: 'BULL_MARKET' | 'BEAR_MARKET' | 'HIGH_VOLATILITY' | 'LOW_VOLATILITY' | 'UNKNOWN';
  confidence: number;
  volatility: number;
  trend_strength: number;
  market_stress: number;
  timestamp: string;
}

export interface SentimentData {
  symbol: string;
  overall_score: number;
  news_sentiment: number;
  social_sentiment: number;
  analyst_sentiment: number;
  bot_detection_score: number;
  source_quality: number;
  article_count: number;
  last_updated: string;
}

export interface SystemHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'down' | 'offline';
  response_time: number;
  memory_usage?: number;
  cpu_usage?: number;
  error_rate?: number;
  last_check?: string;
  details?: {
    [key: string]: any;
  };
}

export interface SystemIntegrityData {
  overall_health: string;
  ml_models: {
    using_real_data: boolean;
    models_loaded: number;
    ensemble_operational: boolean;
    response_time_ms: number;
  };
  trading_system: {
    api_connection: boolean;
    portfolio_accessible: boolean;
    orders_can_execute: boolean;
    response_time_ms: number;
  };
  data_pipeline: {
    market_data_fresh: boolean;
    database_accessible: boolean;
    feature_engineering_ok: boolean;
    response_time_ms: number;
  };
  performance: {
    system_responsive: boolean;
    memory_usage_ok: boolean;
    disk_space_ok: boolean;
    avg_response_time_ms: number;
  };
  summary: {
    critical_issues: number;
    warnings: number;
    recommendations: string[];
  };
}

export interface SystemAlert {
  level: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  component: string;
}

export interface MonitoringData {
  alerts: SystemAlert[];
  alert_count: number;
  critical_count: number;
  warning_count: number;
}

export interface TradeAudit {
  trade_id: string;
  symbol: string;
  signal: 'BUY' | 'SELL';
  quantity: number;
  entry_price: number;
  timestamp: string;
  regime_tag: string;
  lstm_probability: number;
  xgboost_probability: number;
  rf_probability: number;
  dynamic_weights: {
    lstm: number;
    xgboost: number;
    rf: number;
  };
  sentiment_score: number;
  kelly_fraction: number;
  position_size_percent: number;
  risk_metrics: {
    var_95: number;
    heat_contribution: number;
    correlation_impact: number;
  };
  execution_data: {
    fill_price: number;
    slippage: number;
    latency_ms: number;
    broker: string;
  };
}

export interface ChartData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface AlertData {
  id: string;
  type: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  source: string;
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  notifications_enabled: boolean;
  auto_refresh_interval: number;
  default_timeframe: string;
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive';
  dashboard_layout: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'signal' | 'position' | 'portfolio' | 'health' | 'alert' | 'market_data';
  data: any;
  timestamp: string;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

// Chart Configuration Types
export interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'gauge' | 'heatmap';
  title: string;
  height: number;
  realtime: boolean;
  update_interval: number;
  data_source: string;
}

// Dashboard Layout Types
export interface DashboardWidget {
  id: string;
  type: 'signal_monitor' | 'risk_gauge' | 'portfolio_summary' | 'system_health' | 'chart';
  title: string;
  size: 'small' | 'medium' | 'large' | 'full';
  position: {
    row: number;
    column: number;
    colspan: number;
    rowspan: number;
  };
  config: any;
}

export interface DashboardLayout {
  id: string;
  name: string;
  widgets: DashboardWidget[];
  columns: number;
  auto_refresh: boolean;
}

// Authentication Types
export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'trader' | 'risk_manager' | 'viewer' | 'auditor';
  permissions: string[];
  last_login: string;
  preferences: UserPreferences;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  loading: boolean;
}

// Risk Management Types
export interface RiskLimit {
  type: 'portfolio_heat' | 'var_limit' | 'drawdown_limit' | 'position_size' | 'correlation';
  value: number;
  current: number;
  status: 'safe' | 'warning' | 'breach';
  last_updated: string;
}

export interface CorrelationMatrix {
  symbols: string[];
  matrix: number[][];
  timestamp: string;
  alerts: {
    symbol1: string;
    symbol2: string;
    correlation: number;
  }[];
}
