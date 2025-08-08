import axios from 'axios';
import { ApiResponse, TradingSignal, Position, PortfolioMetrics, SystemHealth, SystemIntegrityData, MonitoringData, TradeAudit, MarketRegime, SentimentData } from '../types';

// API Client Configuration
const API_BASE_URL = 'http://localhost:8002/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth tokens
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export class TradingApiService {
  // Health and Status
  static async getSystemHealth(): Promise<SystemHealth[]> {
    try {
      const response = await apiClient.get('/health');
      // Handle single object response - convert to array
      if (response.data && !Array.isArray(response.data)) {
        return [response.data];
      }
      return response.data || [];
    } catch (error) {
      console.error('Failed to fetch system health:', error);
      // Return empty array on error so dashboard doesn't crash
      return [];
    }
  }

  static async getServiceStatus(service: string): Promise<SystemHealth> {
    try {
      const response = await apiClient.get(`/services/${service}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch ${service} status:`, error);
      throw error;
    }
  }

  // Comprehensive Monitoring
  static async getSystemIntegrity(): Promise<SystemIntegrityData> {
    try {
      const response = await apiClient.get('/health/integrity');
      return response.data.data;
    } catch (error) {
      console.error('Failed to fetch system integrity:', error);
      throw error;
    }
  }

  static async getHealthStatus(): Promise<any> {
    try {
      const response = await apiClient.get('/health/status');
      return response.data.data;
    } catch (error) {
      console.error('Failed to fetch health status:', error);
      throw error;
    }
  }

  static async getSystemAlerts(): Promise<MonitoringData> {
    try {
      const response = await apiClient.get('/health/alerts');
      return response.data.data;
    } catch (error) {
      console.error('Failed to fetch system alerts:', error);
      throw error;
    }
  }

  static async forceHealthCheck(): Promise<SystemIntegrityData> {
    try {
      const response = await apiClient.post('/health/monitoring/force-check');
      return response.data.data;
    } catch (error) {
      console.error('Failed to force health check:', error);
      throw error;
    }
  }

  // Trading Signals
  static async getLatestSignals(limit = 10): Promise<TradingSignal[]> {
    try {
      const response = await apiClient.get(`/signals/latest?limit=${limit}`);
      // Extract data from the response wrapper
      const data = response.data?.data || response.data;
      // Ensure we return an array
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('Failed to fetch latest signals:', error);
      // Return empty array instead of throwing
      return [];
    }
  }

  static async getTradingSignals(limit = 10): Promise<TradingSignal[]> {
    return this.getLatestSignals(limit);
  }

  static async getSignalHistory(symbol: string, timeframe: string): Promise<TradingSignal[]> {
    try {
      const response = await apiClient.get(`/signals/history?symbol=${symbol}&timeframe=${timeframe}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch signal history:', error);
      throw error;
    }
  }

  static async generateSignal(symbol: string, timeframe: string): Promise<TradingSignal> {
    try {
      const response = await apiClient.post('/signals/generate', { symbol, timeframe });
      return response.data;
    } catch (error) {
      console.error('Failed to generate signal:', error);
      throw error;
    }
  }

  // Trading Execution
  static async executeTrade(symbol: string, side: string, quantity?: number, reason?: string): Promise<any> {
    try {
      const requestBody: any = { symbol, side };
      if (quantity) requestBody.quantity = quantity;
      if (reason) requestBody.reason = reason;
      
      const response = await apiClient.post('/trading/execute', requestBody);
      return response.data;
    } catch (error) {
      console.error('Failed to execute trade:', error);
      throw error;
    }
  }

  // Trading Strategy Controls
  static async toggleAutoTrading(enabled: boolean): Promise<{ auto_trading: boolean; message: string }> {
    try {
      const response = await apiClient.post('/trading/auto/toggle', { enabled });
      return response.data;
    } catch (error) {
      console.error('Failed to toggle auto trading:', error);
      throw error;
    }
  }

  static async toggleStrategyTrading(enabled: boolean): Promise<{ strategy_trading: boolean; message: string }> {
    try {
      const response = await apiClient.post('/trading/strategy/toggle', { enabled });
      return response.data;
    } catch (error) {
      console.error('Failed to toggle strategy trading:', error);
      throw error;
    }
  }

  static async toggleRebalancing(enabled: boolean): Promise<{ rebalancing: boolean; message: string }> {
    try {
      const response = await apiClient.post('/trading/rebalance/toggle', { enabled });
      return response.data;
    } catch (error) {
      console.error('Failed to toggle rebalancing:', error);
      throw error;
    }
  }

  // Trading Status and Logs
  static async getTradingStatus(): Promise<any> {
    try {
      const response = await apiClient.get('/trading/status');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch trading status:', error);
      throw error;
    }
  }

  static async getTradeLogs(limit = 50): Promise<any[]> {
    try {
      const response = await apiClient.get(`/trading/logs?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch trade logs:', error);
      throw error;
    }
  }

  // Portfolio Management
  static async getPortfolioMetrics(): Promise<PortfolioMetrics> {
    try {
      const response = await apiClient.get('/portfolio/metrics');
      // Handle both wrapped and direct response formats
      const data = response.data?.data || response.data;
      return data;
    } catch (error) {
      console.error('Failed to fetch portfolio metrics:', error);
      // Return default portfolio metrics to prevent dashboard crash
      return {
        total_value: 0,
        available_cash: 0,
        total_pnl: 0,
        total_pnl_percent: 0,
        portfolio_heat: 0,
        max_heat_limit: 25,
        portfolio_var: 0,
        max_var_limit: 2,
        current_drawdown: 0,
        max_drawdown_limit: 6,
        sharpe_ratio: 0,
        num_positions: 0,
        concentration_risk: 'green',
        correlation_alert: false
      };
    }
  }

  static async getCurrentPositions(): Promise<Position[]> {
    try {
      const response = await apiClient.get('/portfolio/positions');
      // Extract data from the response wrapper
      const data = response.data?.data || response.data;
      // Ensure we return an array
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('Failed to fetch current positions:', error);
      // Return empty array instead of throwing to prevent dashboard crash
      return [];
    }
  }

  // Alias for getCurrentPositions for Dashboard compatibility
  static async getPositions(): Promise<Position[]> {
    return this.getCurrentPositions();
  }

  static async getPositionHistory(symbol?: string): Promise<Position[]> {
    try {
      const url = symbol ? `/portfolio/history?symbol=${symbol}` : '/portfolio/history';
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch position history:', error);
      throw error;
    }
  }

  // Risk Management
  static async getRiskMetrics(): Promise<any> {
    try {
      const response = await apiClient.get('/risk/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch risk metrics:', error);
      throw error;
    }
  }

  static async getCorrelationMatrix(): Promise<any> {
    try {
      const response = await apiClient.get('/risk/correlation');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch correlation matrix:', error);
      throw error;
    }
  }

  static async getVaRData(): Promise<any> {
    try {
      const response = await apiClient.get('/risk/var');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch VaR data:', error);
      throw error;
    }
  }

  // Market Data
  static async getMarketRegime(): Promise<MarketRegime> {
    try {
      const response = await apiClient.get('/market/regime');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch market regime:', error);
      throw error;
    }
  }

  static async getMarketData(symbol: string, timeframe: string): Promise<any> {
    try {
      const response = await apiClient.get(`/market/data?symbol=${symbol}&timeframe=${timeframe}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch market data:', error);
      throw error;
    }
  }

  // Sentiment Analysis
  static async getSentimentData(symbol: string): Promise<SentimentData> {
    try {
      const response = await apiClient.get(`/sentiment/${symbol}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch sentiment data:', error);
      throw error;
    }
  }

  static async getBulkSentiment(symbols: string[]): Promise<SentimentData[]> {
    try {
      const response = await apiClient.post('/sentiment/bulk', { symbols });
      return response.data;
    } catch (error) {
      console.error('Failed to fetch bulk sentiment:', error);
      throw error;
    }
  }

  static async executeSignalTrade(signal: TradingSignal): Promise<any> {
    try {
      const response = await apiClient.post('/execution/trade', signal);
      return response.data;
    } catch (error) {
      console.error('Failed to execute signal trade:', error);
      throw error;
    }
  }

  static async getTradeStatus(tradeId: string): Promise<any> {
    try {
      const response = await apiClient.get(`/execution/status/${tradeId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch trade status:', error);
      throw error;
    }
  }

  // Audit and Compliance
  static async getTradeAudit(limit = 50): Promise<TradeAudit[]> {
    try {
      const response = await apiClient.get(`/audit/trades?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch trade audit:', error);
      throw error;
    }
  }

  static async exportAuditData(startDate: string, endDate: string): Promise<Blob> {
    try {
      const response = await apiClient.get(
        `/audit/export?start=${startDate}&end=${endDate}`,
        { responseType: 'blob' }
      );
      return response.data;
    } catch (error) {
      console.error('Failed to export audit data:', error);
      throw error;
    }
  }

  // Configuration
  static async getConfiguration(): Promise<any> {
    try {
      const response = await apiClient.get('/config');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch configuration:', error);
      throw error;
    }
  }

  static async updateConfiguration(config: any): Promise<any> {
    try {
      const response = await apiClient.put('/config', config);
      return response.data;
    } catch (error) {
      console.error('Failed to update configuration:', error);
      throw error;
    }
  }

  // Analytics
  static async getPerformanceMetrics(period = '1d'): Promise<any> {
    try {
      const response = await apiClient.get(`/analytics/performance?period=${period}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
      throw error;
    }
  }

  static async getAttributionAnalysis(): Promise<any> {
    try {
      const response = await apiClient.get('/analytics/attribution');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch attribution analysis:', error);
      throw error;
    }
  }

  // ========================================
  // MLOPS API METHODS
  // ========================================

  // Model Registry APIs
  static async getModels(): Promise<any> {
    try {
      const response = await apiClient.get('/mlops/models');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      throw error;
    }
  }

  static async getModelDetails(modelId: string): Promise<any> {
    try {
      const response = await apiClient.get(`/mlops/models/${modelId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch model details:', error);
      throw error;
    }
  }

  static async promoteModel(modelId: string): Promise<any> {
    try {
      const response = await apiClient.post(`/mlops/models/${modelId}/promote`);
      return response.data;
    } catch (error) {
      console.error('Failed to promote model:', error);
      throw error;
    }
  }

  static async getRegistrySummary(): Promise<any> {
    try {
      const response = await apiClient.get('/mlops/registry/summary');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch registry summary:', error);
      throw error;
    }
  }

  // Champion-Challenger Framework APIs
  static async getChallengerTests(): Promise<any> {
    try {
      const response = await apiClient.get('/mlops/tests');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch challenger tests:', error);
      throw error;
    }
  }

  static async createChallengerTest(testData: any): Promise<any> {
    try {
      const response = await apiClient.post('/mlops/tests', testData);
      return response.data;
    } catch (error) {
      console.error('Failed to create challenger test:', error);
      throw error;
    }
  }

  static async getChallengerTestDetails(testId: string): Promise<any> {
    try {
      const response = await apiClient.get(`/mlops/tests/${testId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch test details:', error);
      throw error;
    }
  }

  static async stopChallengerTest(testId: string): Promise<any> {
    try {
      const response = await apiClient.post(`/mlops/tests/${testId}/stop`);
      return response.data;
    } catch (error) {
      console.error('Failed to stop test:', error);
      throw error;
    }
  }

  // Backtesting APIs
  static async getBacktestStatus(): Promise<any> {
    try {
      const response = await apiClient.get('/mlops/backtest/status');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch backtest status:', error);
      throw error;
    }
  }

  static async startBacktest(backtestData: any): Promise<any> {
    try {
      const response = await apiClient.post('/mlops/backtest/start', backtestData);
      return response.data;
    } catch (error) {
      console.error('Failed to start backtest:', error);
      throw error;
    }
  }

  static async getBacktestResults(backtestId: string): Promise<any> {
    try {
      const response = await apiClient.get(`/mlops/backtest/results/${backtestId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch backtest results:', error);
      throw error;
    }
  }
}

export default TradingApiService;
