import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Alert,
  CircularProgress,
  Button,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  Speed,
  Security,
  Timeline,
  Refresh,
} from '@mui/icons-material';
import { TradingApiService } from '../services/api';
import SimpleHealthMonitor from '../components/SimpleHealthMonitor';

// Real-time data interfaces
interface PortfolioData {
  total_value: number;
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
  cash_available: number;
  margin_used: number;
}

interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL';
  confidence: number;
  timestamp: string;
  sentiment_score: number;
  kelly_fraction: number;
  target_price?: number;
  stop_loss?: number;
}

interface SystemHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'offline';
  response_time: number;
  details?: any;
}

interface Position {
  symbol: string;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  entry_price: number;
  current_price: number;
  side: string;
  risk_contribution: number;
}

interface RiskGaugeProps {
  label: string;
  value: number;
  limit: number;
  unit?: string;
  color?: 'primary' | 'warning' | 'error';
}

const RiskGauge: React.FC<RiskGaugeProps> = ({ label, value, limit, unit = '%', color = 'primary' }) => {
  const percentage = (value / limit) * 100;
  const gaugeColor = percentage > 80 ? 'error' : percentage > 60 ? 'warning' : color;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" component="div" gutterBottom>
          {label}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography variant="h4" color={`${gaugeColor}.main`}>
            {value.toFixed(1)}{unit}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
            / {limit}{unit}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={Math.min(percentage, 100)}
          color={gaugeColor}
          sx={{ height: 8, borderRadius: 4 }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          {percentage.toFixed(1)}% of limit
        </Typography>
      </CardContent>
    </Card>
  );
};

interface SignalCardProps {
  signal: TradingSignal;
}

const SignalCard: React.FC<SignalCardProps> = ({ signal }) => {
  const signalColor = signal.signal === 'BUY' ? 'success' : 'error';
  const signalIcon = signal.signal === 'BUY' ? <TrendingUp /> : <TrendingDown />;

  return (
    <Card sx={{ mb: 1 }}>
      <CardContent sx={{ py: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ color: `${signalColor}.main`, mr: 1 }}>
              {signalIcon}
            </Box>
            <Typography variant="h6">{signal.symbol}</Typography>
            <Chip
              label={signal.signal}
              size="small"
              color={signalColor}
              sx={{ ml: 1 }}
            />
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="body2">
              Confidence: {(signal.confidence * 100).toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Kelly: {signal.kelly_fraction && !isNaN(signal.kelly_fraction) ? (signal.kelly_fraction * 100).toFixed(1) : '10.0'}%
            </Typography>
          </Box>
        </Box>
        {signal.target_price && signal.stop_loss && (
          <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption">
              Target: ${signal.target_price.toFixed(2)}
            </Typography>
            <Typography variant="caption">
              Stop: ${signal.stop_loss.toFixed(2)}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

interface SystemHealthCardProps {
  services: SystemHealth[];
}

const SystemHealthCard: React.FC<SystemHealthCardProps> = ({ services }) => {
  const healthyCount = services.filter((s: SystemHealth) => s.status === 'healthy').length;
  const totalCount = services.length;
  const healthPercentage = (healthyCount / totalCount) * 100;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" component="div" gutterBottom>
          System Health
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <CircularProgress
            variant="determinate"
            value={healthPercentage}
            size={60}
            color={healthPercentage === 100 ? 'success' : 'warning'}
          />
          <Box sx={{ ml: 2 }}>
            <Typography variant="h5" component="div">
              {healthyCount}/{totalCount}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Services Healthy
            </Typography>
          </Box>
        </Box>
        {services.map((service: SystemHealth) => (
          <Box key={service.service} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">{service.service}</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Chip
                label={service.status}
                size="small"
                color={service.status === 'healthy' ? 'success' : 'warning'}
                variant="outlined"
              />
              <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                {service.response_time}ms
              </Typography>
            </Box>
          </Box>
        ))}
      </CardContent>
    </Card>
  );
};

const TradingDashboard: React.FC = () => {
  // Real-time data state
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');

  // WebSocket connection
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Load initial data
  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Loading real-time trading data...');
      
      // Load all data in parallel
      const [healthData, signalsData, portfolioMetrics, positionsData] = await Promise.all([
        TradingApiService.getSystemHealth(),
        TradingApiService.getLatestSignals(5),
        TradingApiService.getPortfolioMetrics(),
        TradingApiService.getPositions(),
      ]);

      setSystemHealth(healthData);
      setSignals(signalsData);
      setPortfolioData(portfolioMetrics);
      setPositions(positionsData);
      setLastUpdate(new Date().toLocaleTimeString());
      
      console.log('✅ Real-time data loaded successfully');
    } catch (err: any) {
      console.error('❌ Error loading trading data:', err);
      setError(err.message || 'Failed to load trading data');
    } finally {
      setIsLoading(false);
    }
  };

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8002/ws');
        
        ws.onopen = () => {
          console.log('🔗 WebSocket connected for real-time updates');
          setSocket(ws);
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'real_time_update') {
              console.log('📡 Real-time update received');
              setSignals(data.signals || []);
              setPortfolioData(data.portfolio);
              setSystemHealth(data.health || []);
              setLastUpdate(new Date().toLocaleTimeString());
            }
          } catch (err) {
            console.error('Error parsing WebSocket message:', err);
          }
        };
        
        ws.onclose = () => {
          console.log('🔌 WebSocket disconnected');
          setSocket(null);
          // Reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
        
      } catch (err) {
        console.error('Failed to connect WebSocket:', err);
      }
    };

    // Initial data load
    loadData();
    
    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Cleanup
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);

  if (isLoading && !portfolioData) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading Trading Platform...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Institutional Trading Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Last Update: {lastUpdate}
          </Typography>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadData}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Chip
            label={socket ? 'Live' : 'Offline'}
            color={socket ? 'success' : 'error'}
            size="small"
          />
        </Box>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Portfolio Overview */}
      {portfolioData && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Portfolio Overview
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color="primary">
                  ${portfolioData.total_value.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Value
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography 
                  variant="h4" 
                  color={portfolioData.total_pnl >= 0 ? 'success.main' : 'error.main'}
                >
                  ${portfolioData.total_pnl.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total P&L ({portfolioData.total_pnl_percent.toFixed(2)}%)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4">
                  {portfolioData.sharpe_ratio.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Sharpe Ratio
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4">
                  {portfolioData.num_positions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Active Positions
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      <Grid container spacing={3}>
        {/* Risk Management */}
        <Grid item xs={12} lg={8}>
          <Typography variant="h5" gutterBottom>
            Risk Management
          </Typography>
          <Grid container spacing={2}>
            {portfolioData && (
              <>
                <Grid item xs={12} md={4}>
                  <RiskGauge
                    label="Portfolio Heat"
                    value={portfolioData.portfolio_heat}
                    limit={portfolioData.max_heat_limit}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <RiskGauge
                    label="Portfolio VaR"
                    value={portfolioData.portfolio_var}
                    limit={portfolioData.max_var_limit}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <RiskGauge
                    label="Current Drawdown"
                    value={portfolioData.current_drawdown}
                    limit={portfolioData.max_drawdown_limit}
                  />
                </Grid>
              </>
            )}
          </Grid>
        </Grid>

        {/* Comprehensive System Health Monitor */}
        <Grid item xs={12} lg={8}>
          <SimpleHealthMonitor />
        </Grid>

        {/* Trading Signals */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Latest Trading Signals
              </Typography>
              {signals.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No signals available
                </Typography>
              ) : (
                signals.map((signal: TradingSignal, index: number) => (
                  <SignalCard key={`${signal.symbol}-${index}`} signal={signal} />
                ))
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Current Positions */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Positions
              </Typography>
              {positions.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No positions currently held
                </Typography>
              ) : (
                positions.slice(0, 5).map((position: Position) => (
                  <Box
                    key={position.symbol}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      py: 1,
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Box>
                      <Typography variant="body1" fontWeight="bold">
                        {position.symbol}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {position.quantity} shares @ ${position.current_price.toFixed(2)}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography
                        variant="body1"
                        color={position.unrealized_pnl >= 0 ? 'success.main' : 'error.main'}
                      >
                        ${position.unrealized_pnl.toFixed(2)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {position.unrealized_pnl_percent.toFixed(2)}%
                      </Typography>
                    </Box>
                  </Box>
                ))
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingDashboard;
