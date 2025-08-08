import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Tab,
  Tabs,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Refresh,
  AutoAwesome,
  Timeline,
  History,
} from '@mui/icons-material';
import { TradingApiService } from '../services/api';

interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL';
  confidence: number;
  timestamp: string;
  current_price: number;
  target_price?: number;
  stop_loss?: number;
  sentiment_score?: number;
  kelly_fraction?: number;
  signal_strength: string;
  risk_reward_ratio?: number;
}

interface TradeLog {
  timestamp: string;
  symbol: string;
  side: string;
  quantity: number;
  price?: number;
  order_id: string;
  status: string;
  reason: string;
  pnl?: number;
}

interface StrategyStatus {
  auto_trading: boolean;
  strategy_trading: boolean;
  rebalancing: boolean;
  last_signal_check?: string;
  total_trades_today: number;
  active_strategies: string[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`trading-tabpanel-${index}`}
      aria-labelledby={`trading-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const TradingInterface: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [tradeLogs, setTradeLogs] = useState<TradeLog[]>([]);
  const [strategyStatus, setStrategyStatus] = useState<StrategyStatus>({
    auto_trading: false,
    strategy_trading: false,
    rebalancing: false,
    total_trades_today: 0,
    active_strategies: []
  });
  
  // Manual trading state
  const [manualSymbol, setManualSymbol] = useState('');
  const [manualQuantity, setManualQuantity] = useState('');
  const [manualSide, setManualSide] = useState<'buy' | 'sell'>('buy');
  const [confirmDialog, setConfirmDialog] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' as 'success' | 'error' });
  
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [signalsData, logsData, statusData] = await Promise.all([
        TradingApiService.getTradingSignals(),
        TradingApiService.getTradeLogs(),
        TradingApiService.getTradingStatus()
      ]);
      
      setSignals(signalsData);
      setTradeLogs(logsData);
      setStrategyStatus(statusData);
    } catch (error) {
      console.error('Error fetching trading data:', error);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const executeManualTrade = async () => {
    if (!manualSymbol || !manualQuantity) {
      setSnackbar({ open: true, message: 'Please enter symbol and quantity', severity: 'error' });
      return;
    }

    setLoading(true);
    try {
      const result = await TradingApiService.executeTrade(
        manualSymbol.toUpperCase(),
        manualSide,
        parseFloat(manualQuantity),
        'Manual UI Trade'
      );

      if (result.success) {
        setSnackbar({ open: true, message: `Trade executed successfully!`, severity: 'success' });
        setManualSymbol('');
        setManualQuantity('');
        fetchData(); // Refresh data
      } else {
        setSnackbar({ open: true, message: `Trade failed: ${result.error}`, severity: 'error' });
      }
    } catch (error) {
      setSnackbar({ open: true, message: `Trade execution error: ${error}`, severity: 'error' });
    } finally {
      setLoading(false);
      setConfirmDialog(false);
    }
  };

  const toggleAutoTrading = async () => {
    try {
      const result = await TradingApiService.toggleAutoTrading(!strategyStatus.auto_trading);
      setStrategyStatus(prev => ({ ...prev, auto_trading: result.auto_trading }));
      setSnackbar({ 
        open: true, 
        message: `Auto trading ${result.auto_trading ? 'enabled' : 'disabled'}`, 
        severity: 'success' 
      });
    } catch (error) {
      setSnackbar({ open: true, message: 'Failed to toggle auto trading', severity: 'error' });
    }
  };

  const toggleStrategyTrading = async () => {
    try {
      const result = await TradingApiService.toggleStrategyTrading(!strategyStatus.strategy_trading);
      setStrategyStatus(prev => ({ ...prev, strategy_trading: result.strategy_trading }));
      setSnackbar({ 
        open: true, 
        message: `Strategy trading ${result.strategy_trading ? 'enabled' : 'disabled'}`, 
        severity: 'success' 
      });
    } catch (error) {
      setSnackbar({ open: true, message: 'Failed to toggle strategy trading', severity: 'error' });
    }
  };

  const toggleRebalancing = async () => {
    try {
      const result = await TradingApiService.toggleRebalancing(!strategyStatus.rebalancing);
      setStrategyStatus(prev => ({ ...prev, rebalancing: result.rebalancing }));
      setSnackbar({ 
        open: true, 
        message: `Rebalancing ${result.rebalancing ? 'enabled' : 'disabled'}`, 
        severity: 'success' 
      });
    } catch (error) {
      setSnackbar({ open: true, message: 'Failed to toggle rebalancing', severity: 'error' });
    }
  };

  const executeSignalTrade = async (signal: TradingSignal) => {
    setLoading(true);
    try {
      const result = await TradingApiService.executeTrade(
        signal.symbol,
        signal.signal.toLowerCase() as 'buy' | 'sell',
        undefined, // Let API calculate optimal size
        `Signal Trade (${signal.confidence.toFixed(2)} confidence)`
      );

      if (result.success) {
        setSnackbar({ open: true, message: `Signal trade executed for ${signal.symbol}!`, severity: 'success' });
        fetchData();
      } else {
        setSnackbar({ open: true, message: `Signal trade failed: ${result.error}`, severity: 'error' });
      }
    } catch (error) {
      setSnackbar({ open: true, message: `Signal trade error: ${error}`, severity: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} variant="fullWidth">
          <Tab label="Manual Trading" icon={<Timeline />} />
          <Tab label="Signal Trading" icon={<AutoAwesome />} />
          <Tab label="Strategy Trading" icon={<TrendingUp />} />
          <Tab label="Trade Logs" icon={<History />} />
        </Tabs>
      </Paper>

      {/* Manual Trading Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Execute Manual Trade
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <TextField
                    label="Symbol"
                    value={manualSymbol}
                    onChange={(e) => setManualSymbol(e.target.value.toUpperCase())}
                    placeholder="e.g., AAPL"
                    fullWidth
                  />
                  <TextField
                    label="Quantity"
                    type="number"
                    value={manualQuantity}
                    onChange={(e) => setManualQuantity(e.target.value)}
                    placeholder="Number of shares"
                    fullWidth
                  />
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant={manualSide === 'buy' ? 'contained' : 'outlined'}
                      color="success"
                      onClick={() => setManualSide('buy')}
                      startIcon={<TrendingUp />}
                      fullWidth
                    >
                      BUY
                    </Button>
                    <Button
                      variant={manualSide === 'sell' ? 'contained' : 'outlined'}
                      color="error"
                      onClick={() => setManualSide('sell')}
                      startIcon={<TrendingDown />}
                      fullWidth
                    >
                      SELL
                    </Button>
                  </Box>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => setConfirmDialog(true)}
                    disabled={!manualSymbol || !manualQuantity || loading}
                    fullWidth
                  >
                    Execute Trade
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Actions
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={() => fetchData()}
                    startIcon={<Refresh />}
                    fullWidth
                  >
                    Refresh Data
                  </Button>
                  <Alert severity="info">
                    Manual trades are executed immediately at market price. Use caution with large positions.
                  </Alert>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Signal Trading Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Trading Signals</Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={strategyStatus.auto_trading}
                        onChange={toggleAutoTrading}
                        color="primary"
                      />
                    }
                    label="Auto Trading"
                  />
                </Box>
                
                {signals.length === 0 ? (
                  <Typography color="text.secondary">No signals available</Typography>
                ) : (
                  <Grid container spacing={2}>
                    {signals.map((signal, index) => (
                      <Grid item xs={12} md={6} lg={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="h6">{signal.symbol}</Typography>
                              <Chip
                                label={signal.signal}
                                color={signal.signal === 'BUY' ? 'success' : 'error'}
                                size="small"
                              />
                            </Box>
                            <Typography variant="body2" color="text.secondary">
                              Confidence: {(signal.confidence * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Price: ${signal.current_price.toFixed(2)}
                            </Typography>
                            {signal.target_price && (
                              <Typography variant="body2" color="text.secondary">
                                Target: ${signal.target_price.toFixed(2)}
                              </Typography>
                            )}
                            <Typography variant="body2" color="text.secondary">
                              Strength: {signal.signal_strength}
                            </Typography>
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={() => executeSignalTrade(signal)}
                              disabled={loading}
                              sx={{ mt: 1 }}
                              fullWidth
                            >
                              Execute Signal
                            </Button>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Strategy Trading Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Automated Signal Trading
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={strategyStatus.auto_trading}
                      onChange={toggleAutoTrading}
                      color="primary"
                    />
                  }
                  label="Enable Auto Trading"
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Automatically execute high-confidence signals (75%+ confidence)
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Strategy-Based Trading
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={strategyStatus.strategy_trading}
                      onChange={toggleStrategyTrading}
                      color="primary"
                    />
                  }
                  label="Enable Strategy Trading"
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Momentum and mean reversion strategies
                </Typography>
                {strategyStatus.active_strategies.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {strategyStatus.active_strategies.map((strategy) => (
                      <Chip key={strategy} label={strategy} size="small" sx={{ mr: 1 }} />
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Portfolio Rebalancing
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={strategyStatus.rebalancing}
                      onChange={toggleRebalancing}
                      color="primary"
                    />
                  }
                  label="Enable Rebalancing"
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Maintain target portfolio allocations
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Strategy Status
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Trades Today
                    </Typography>
                    <Typography variant="h4">
                      {strategyStatus.total_trades_today}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Last Signal Check
                    </Typography>
                    <Typography variant="body1">
                      {strategyStatus.last_signal_check 
                        ? new Date(strategyStatus.last_signal_check).toLocaleTimeString()
                        : 'Not yet checked'
                      }
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Alert severity="info">
                      All strategies run automatically in the background every 5 minutes when enabled.
                      Position sizing is calculated using Kelly criterion and risk management.
                    </Alert>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Trade Logs Tab */}
      <TabPanel value={tabValue} index={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Trade Executions
            </Typography>
            {tradeLogs.length === 0 ? (
              <Typography color="text.secondary">No trades executed yet</Typography>
            ) : (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell>Quantity</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Reason</TableCell>
                      <TableCell>P&L</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tradeLogs.slice(0, 20).map((log, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {new Date(log.timestamp).toLocaleString()}
                        </TableCell>
                        <TableCell>{log.symbol}</TableCell>
                        <TableCell>
                          <Chip
                            label={log.side.toUpperCase()}
                            color={log.side.toLowerCase() === 'buy' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{log.quantity}</TableCell>
                        <TableCell>
                          <Chip
                            label={log.status}
                            color={log.status === 'filled' ? 'success' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{log.reason}</TableCell>
                        <TableCell>
                          {log.pnl !== undefined ? (
                            <Typography color={log.pnl >= 0 ? 'success.main' : 'error.main'}>
                              ${log.pnl.toFixed(2)}
                            </Typography>
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>
      </TabPanel>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog} onClose={() => setConfirmDialog(false)}>
        <DialogTitle>Confirm Trade</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to {manualSide.toUpperCase()} {manualQuantity} shares of {manualSymbol}?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog(false)}>Cancel</Button>
          <Button
            onClick={executeManualTrade}
            color="primary"
            variant="contained"
            disabled={loading}
          >
            {loading ? 'Executing...' : 'Confirm'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default TradingInterface;
