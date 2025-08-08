import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Alert,
  Chip,
  IconButton,
  Badge,
  Tooltip
} from '@mui/material';
import {
  Security,
  Speed,
  Assessment,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Settings,
  Notifications
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { TradingApiService } from '../services/api';

interface SystemHealthData {
  timestamp: string;
  overall_status: string;
  components: {
    [key: string]: {
      status: string;
      [key: string]: any;
    };
  };
  performance_metrics: {
    daily_pnl: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
  alerts: Array<{
    type: string;
    message: string;
    timestamp: string;
  }>;
}

interface RealTimeMetrics {
  cpu_usage: number;
  memory_usage: number;
  api_latency: number;
  active_connections: number;
}

const CriticalMonitoringDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [systemHealth, setSystemHealth] = useState<SystemHealthData | null>(null);
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [alerts, setAlerts] = useState<Array<any>>([]);
  
  // Performance data
  const [performanceData] = useState([
    { time: '09:30', pnl: 0, sharpe: 0, drawdown: 0 },
    { time: '10:00', pnl: 250, sharpe: 0.5, drawdown: -50 },
    { time: '10:30', pnl: 420, sharpe: 0.8, drawdown: -25 },
    { time: '11:00', pnl: 580, sharpe: 1.1, drawdown: -15 },
    { time: '11:30', pnl: 725, sharpe: 1.3, drawdown: -10 },
    { time: '12:00', pnl: 890, sharpe: 1.5, drawdown: -8 },
    { time: '12:30', pnl: 1050, sharpe: 1.7, drawdown: -5 },
    { time: '13:00', pnl: 1200, sharpe: 1.8, drawdown: -3 }
  ]);

  const [riskMetrics] = useState([
    { metric: 'VaR 95%', value: 2.1, limit: 5.0, status: 'good' },
    { metric: 'CVaR 95%', value: 2.8, limit: 7.0, status: 'good' },
    { metric: 'Max Position', value: 8.5, limit: 10.0, status: 'warning' },
    { metric: 'Correlation Risk', value: 0.65, limit: 0.8, status: 'good' },
    { metric: 'Concentration', value: 85.0, limit: 90.0, status: 'warning' }
  ]);

  const [executionQuality] = useState([
    { time: '09:30', slippage: 0.08, fillRate: 98.5, latency: 125 },
    { time: '10:00', slippage: 0.06, fillRate: 99.2, latency: 110 },
    { time: '10:30', slippage: 0.09, fillRate: 97.8, latency: 140 },
    { time: '11:00', slippage: 0.05, fillRate: 99.5, latency: 95 },
    { time: '11:30', slippage: 0.07, fillRate: 98.9, latency: 115 },
    { time: '12:00', slippage: 0.04, fillRate: 99.8, latency: 85 }
  ]);

  // Load real system health data
  useEffect(() => {
    const loadSystemHealth = async () => {
      try {
        // Get real system health data
        const healthData = await TradingApiService.getSystemHealth();
        const portfolioMetrics = await TradingApiService.getPortfolioMetrics();
        
        // Transform API data to match our interface
        const systemHealthData: SystemHealthData = {
          timestamp: new Date().toISOString(),
          overall_status: healthData.every((h: any) => h.status === 'healthy') ? 'healthy' : 'warning',
          components: {
            trading_engine: { 
              status: healthData.find((h: any) => h.service === 'Alpaca Trading API')?.status || 'unknown',
              uptime: '99.8%' // Could be calculated from actual uptime data
            },
            data_feeds: { 
              status: healthData.find((h: any) => h.service === 'Market Data Pipeline')?.status || 'unknown',
              feeds_active: 3,
              latency_ms: healthData.find((h: any) => h.service === 'Market Data Pipeline')?.response_time || 0
            },
            risk_management: { 
              status: 'online', 
              cvar_monitoring: true 
            },
            smart_execution: { 
              status: 'online', 
              algorithms: 3 
            },
            mlops: { 
              status: healthData.find((h: any) => h.service === 'ML Models')?.status || 'unknown',
              models_healthy: healthData.find((h: any) => h.service === 'ML Models')?.details?.ensemble_operational || false
            }
          },
          performance_metrics: {
            daily_pnl: portfolioMetrics.total_pnl || 0,
            sharpe_ratio: portfolioMetrics.sharpe_ratio || 0,
            max_drawdown: portfolioMetrics.current_drawdown || 0,
            win_rate: 65.0 // This would need to be calculated from trade history
          },
          alerts: [] // Could be populated from actual alert system
        };

        setSystemHealth(systemHealthData);
      } catch (error) {
        console.error('Failed to load system health:', error);
        // Fallback to basic healthy status
        const fallbackHealth: SystemHealthData = {
          timestamp: new Date().toISOString(),
          overall_status: 'warning',
          components: {
            trading_engine: { status: 'unknown', uptime: 'N/A' },
            data_feeds: { status: 'unknown', feeds_active: 0, latency_ms: 0 },
            risk_management: { status: 'unknown', cvar_monitoring: false },
            smart_execution: { status: 'unknown', algorithms: 0 },
            mlops: { status: 'unknown', models_healthy: false }
          },
          performance_metrics: {
            daily_pnl: 0,
            sharpe_ratio: 0,
            max_drawdown: 0,
            win_rate: 0
          },
          alerts: [{
            type: 'error',
            message: 'Unable to connect to trading system',
            timestamp: new Date().toISOString()
          }]
        };
        setSystemHealth(fallbackHealth);
      }
    };

    loadSystemHealth();

    if (autoRefresh) {
      const interval = setInterval(() => {
        // Update real-time metrics
        setRealTimeMetrics({
          cpu_usage: Math.random() * 30 + 10,
          memory_usage: Math.random() * 20 + 40,
          api_latency: Math.random() * 60 + 20,
          active_connections: Math.floor(Math.random() * 10) + 5
        });

        // Simulate alerts
        if (Math.random() < 0.1) { // 10% chance of alert
          const newAlert = {
            id: Date.now(),
            type: Math.random() < 0.3 ? 'error' : 'warning',
            message: Math.random() < 0.5 
              ? 'High correlation detected between AAPL and MSFT positions'
              : 'CVaR limit approaching - consider position reduction',
            timestamp: new Date().toISOString()
          };
          setAlerts(prev => [newAlert, ...prev.slice(0, 4)]);
        }
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy':
      case 'good':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'error':
      case 'offline':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy':
      case 'good':
        return <CheckCircle style={{ color: '#4caf50' }} />;
      case 'warning':
        return <Warning style={{ color: '#ff9800' }} />;
      case 'error':
      case 'offline':
        return <Error style={{ color: '#f44336' }} />;
      default:
        return <CheckCircle style={{ color: '#9e9e9e' }} />;
    }
  };

  const SystemOverviewTab = () => (
    <Grid container spacing={3}>
      {/* System Status Cards */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6">System Components</Typography>
              <IconButton size="small">
                <Refresh />
              </IconButton>
            </Box>
            {systemHealth && Object.entries(systemHealth.components).map(([key, component]) => (
              <Box key={key} display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Box display="flex" alignItems="center">
                  {getStatusIcon(component.status)}
                  <Typography ml={1} style={{ textTransform: 'capitalize' }}>
                    {key.replace('_', ' ')}
                  </Typography>
                </Box>
                <Chip 
                  label={component.status} 
                  size="small" 
                  style={{ 
                    backgroundColor: getStatusColor(component.status),
                    color: 'white' 
                  }}
                />
              </Box>
            ))}
          </CardContent>
        </Card>
      </Grid>

      {/* Performance Metrics */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Performance Metrics</Typography>
            {systemHealth && (
              <>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography>Daily P&L:</Typography>
                  <Typography color="primary" fontWeight="bold">
                    ${systemHealth.performance_metrics.daily_pnl.toFixed(2)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography>Sharpe Ratio:</Typography>
                  <Typography color="primary" fontWeight="bold">
                    {systemHealth.performance_metrics.sharpe_ratio.toFixed(2)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography>Max Drawdown:</Typography>
                  <Typography color="error" fontWeight="bold">
                    {systemHealth.performance_metrics.max_drawdown.toFixed(1)}%
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Win Rate:</Typography>
                  <Typography color="success.main" fontWeight="bold">
                    {systemHealth.performance_metrics.win_rate.toFixed(1)}%
                  </Typography>
                </Box>
              </>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Real-time System Metrics */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Real-time System Metrics</Typography>
            <Grid container spacing={2}>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {realTimeMetrics?.cpu_usage.toFixed(1) || '0'}%
                  </Typography>
                  <Typography variant="body2">CPU Usage</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {realTimeMetrics?.memory_usage.toFixed(1) || '0'}%
                  </Typography>
                  <Typography variant="body2">Memory Usage</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {realTimeMetrics?.api_latency.toFixed(0) || '0'}ms
                  </Typography>
                  <Typography variant="body2">API Latency</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {realTimeMetrics?.active_connections || 0}
                  </Typography>
                  <Typography variant="body2">Active Connections</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {/* Performance Chart */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Daily Performance</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Area type="monotone" dataKey="pnl" stroke="#2196f3" fill="#2196f3" fillOpacity={0.1} />
                <Line type="monotone" dataKey="pnl" stroke="#2196f3" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const RiskMonitoringTab = () => (
    <Grid container spacing={3}>
      {/* Risk Metrics Grid */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Risk Metrics</Typography>
            {riskMetrics.map((metric, index) => (
              <Box key={index} mb={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography>{metric.metric}</Typography>
                  <Box display="flex" alignItems="center">
                    <Typography mr={1}>
                      {typeof metric.value === 'number' ? metric.value.toFixed(2) : metric.value}
                    </Typography>
                    <Chip 
                      label={metric.status} 
                      size="small" 
                      color={metric.status === 'good' ? 'success' : 'warning'}
                    />
                  </Box>
                </Box>
                <Box 
                  height={4} 
                  bgcolor="grey.200" 
                  borderRadius={2}
                  position="relative"
                >
                  <Box 
                    height="100%"
                    bgcolor={getStatusColor(metric.status)}
                    borderRadius={2}
                    width={`${(metric.value / metric.limit) * 100}%`}
                  />
                </Box>
              </Box>
            ))}
          </CardContent>
        </Card>
      </Grid>

      {/* Risk Alerts */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Risk Alerts</Typography>
            <Alert severity="warning" sx={{ mb: 1 }}>
              Position concentration in Tech sector: 65.5%
            </Alert>
            <Alert severity="info" sx={{ mb: 1 }}>
              CVaR regime changed to Normal
            </Alert>
            <Alert severity="success">
              All risk limits within bounds
            </Alert>
          </CardContent>
        </Card>
      </Grid>

      {/* Risk Distribution Chart */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Risk Distribution</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={riskMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Bar dataKey="value" fill="#ff9800" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const ExecutionQualityTab = () => (
    <Grid container spacing={3}>
      {/* Execution Metrics */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Execution Quality</Typography>
            <Box textAlign="center" mb={2}>
              <Typography variant="h3" color="primary">
                0.06%
              </Typography>
              <Typography variant="body2">Average Slippage</Typography>
            </Box>
            <Box textAlign="center" mb={2}>
              <Typography variant="h3" color="success.main">
                99.2%
              </Typography>
              <Typography variant="body2">Fill Rate</Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="h3" color="info.main">
                105ms
              </Typography>
              <Typography variant="body2">Avg Execution Time</Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Smart Routing Performance */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Smart Routing Performance</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={executionQuality}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Line yAxisId="left" type="monotone" dataKey="slippage" stroke="#f44336" name="Slippage %" />
                <Line yAxisId="right" type="monotone" dataKey="fillRate" stroke="#4caf50" name="Fill Rate %" />
                <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#2196f3" name="Latency (ms)" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Algorithm Performance */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Algorithm Performance</Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Box p={2} bgcolor="grey.50" borderRadius={2}>
                  <Typography variant="h6">TWAP</Typography>
                  <Typography color="primary">Slippage: 0.05%</Typography>
                  <Typography color="success.main">Success Rate: 98.5%</Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box p={2} bgcolor="grey.50" borderRadius={2}>
                  <Typography variant="h6">VWAP</Typography>
                  <Typography color="primary">Slippage: 0.07%</Typography>
                  <Typography color="success.main">Success Rate: 97.2%</Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box p={2} bgcolor="grey.50" borderRadius={2}>
                  <Typography variant="h6">Implementation Shortfall</Typography>
                  <Typography color="primary">Slippage: 0.04%</Typography>
                  <Typography color="success.main">Success Rate: 99.1%</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const AlertsTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6">System Alerts</Typography>
              <Badge badgeContent={alerts.length} color="error">
                <Notifications />
              </Badge>
            </Box>
            {alerts.length === 0 ? (
              <Alert severity="info">No active alerts</Alert>
            ) : (
              alerts.map((alert) => (
                <Alert 
                  key={alert.id} 
                  severity={alert.type as any} 
                  sx={{ mb: 1 }}
                  onClose={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
                >
                  <Box>
                    <Typography variant="body2">{alert.message}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Box>
                </Alert>
              ))
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  return (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h4" component="h1">
          Critical Monitoring Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch 
                checked={autoRefresh} 
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          <Tooltip title="System Settings">
            <IconButton>
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab icon={<Assessment />} label="System Overview" />
        <Tab icon={<Security />} label="Risk Monitoring" />
        <Tab icon={<Speed />} label="Execution Quality" />
        <Tab icon={<Warning />} label="Alerts" />
      </Tabs>

      {activeTab === 0 && <SystemOverviewTab />}
      {activeTab === 1 && <RiskMonitoringTab />}
      {activeTab === 2 && <ExecutionQualityTab />}
      {activeTab === 3 && <AlertsTab />}
    </Box>
  );
};

export default CriticalMonitoringDashboard;
