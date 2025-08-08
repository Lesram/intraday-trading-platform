import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  TrendingUp,
  TrendingDown,
  Info,
  Warning,
  CheckCircle,
  Cancel,
  Refresh,
  Timeline
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import SimpleMarketTiming from './SimpleMarketTiming';

interface TradingSignal {
  symbol: string;
  signal: number;
  confidence: number;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  risk_score: number;
  timestamp: string;
}

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  market_value: number;
}

interface OrderRequest {
  symbol: string;
  quantity: number;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit' | 'smart';
  limit_price?: number;
  execution_strategy: 'twap' | 'vwap' | 'is' | 'market';
}

const ProfessionalTradingInterface: React.FC = () => {
  const [tradingEnabled, setTradingEnabled] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderQuantity, setOrderQuantity] = useState(100);
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'smart'>('smart');
  const [executionStrategy, setExecutionStrategy] = useState<'twap' | 'vwap' | 'is' | 'market'>('twap');
  const [limitPrice, setLimitPrice] = useState<number | undefined>();
  
  const [currentSignals, setCurrentSignals] = useState<TradingSignal[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orderHistory, setOrderHistory] = useState<any[]>([]);
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [pendingOrder, setPendingOrder] = useState<OrderRequest | null>(null);
  
  const [priceData] = useState([
    { time: '09:30', price: 175.20 },
    { time: '09:45', price: 175.85 },
    { time: '10:00', price: 176.12 },
    { time: '10:15', price: 175.90 },
    { time: '10:30', price: 176.45 },
    { time: '10:45', price: 177.20 },
    { time: '11:00', price: 176.80 },
    { time: '11:15', price: 177.60 },
    { time: '11:30', price: 178.25 }
  ]);

  // Sample data
  // Add market timing state with current time
  const [marketInfo] = useState({
    is_open: false,
    status: 'CLOSED' as const,
    data_type: 'LAST_AVAILABLE' as const,
    local_time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }),
    eastern_time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' }) + ' ET',
    date: new Date().toLocaleDateString()
  });

  useEffect(() => {
    setCurrentSignals([
      {
        symbol: 'AAPL',
        signal: 0.75,
        confidence: 0.85,
        recommendation: 'BUY',
        risk_score: 0.25,
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'MSFT',
        signal: 0.45,
        confidence: 0.72,
        recommendation: 'BUY',
        risk_score: 0.18,
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'TSLA',
        signal: -0.35,
        confidence: 0.68,
        recommendation: 'SELL',
        risk_score: 0.55,
        timestamp: new Date().toISOString()
      }
    ]);

    setPositions([
      {
        symbol: 'AAPL',
        quantity: 100,
        avg_price: 175.50,
        current_price: 178.25,
        unrealized_pnl: 275.00,
        unrealized_pnl_pct: 1.57,
        market_value: 17825
      },
      {
        symbol: 'MSFT',
        quantity: 75,
        avg_price: 340.20,
        current_price: 345.80,
        unrealized_pnl: 420.00,
        unrealized_pnl_pct: 1.65,
        market_value: 25935
      },
      {
        symbol: 'TSLA',
        quantity: 50,
        avg_price: 220.15,
        current_price: 215.30,
        unrealized_pnl: -242.50,
        unrealized_pnl_pct: -2.20,
        market_value: 10765
      }
    ]);
  }, []);

  const handlePlaceOrder = () => {
    const order: OrderRequest = {
      symbol: selectedSymbol,
      quantity: orderQuantity,
      side: orderSide,
      order_type: orderType,
      execution_strategy: executionStrategy,
      limit_price: orderType === 'limit' ? limitPrice : undefined
    };

    setPendingOrder(order);
    setConfirmDialogOpen(true);
  };

  const confirmOrder = async () => {
    if (!pendingOrder) return;

    try {
      // Simulate order execution
      const executedOrder = {
        id: Date.now(),
        ...pendingOrder,
        status: 'executed',
        fill_price: getCurrentPrice(pendingOrder.symbol),
        timestamp: new Date().toISOString(),
        execution_quality: {
          slippage: Math.random() * 0.1,
          fill_rate: 98 + Math.random() * 2,
          execution_time_ms: Math.floor(Math.random() * 200) + 50
        }
      };

      setOrderHistory(prev => [executedOrder, ...prev]);
      
      // Update positions
      const existingPositionIndex = positions.findIndex(p => p.symbol === pendingOrder.symbol);
      if (existingPositionIndex >= 0) {
        const updatedPositions = [...positions];
        const position = updatedPositions[existingPositionIndex];
        
        if (pendingOrder.side === 'buy') {
          const newQuantity = position.quantity + pendingOrder.quantity;
          const newAvgPrice = ((position.quantity * position.avg_price) + 
                              (pendingOrder.quantity * executedOrder.fill_price)) / newQuantity;
          
          updatedPositions[existingPositionIndex] = {
            ...position,
            quantity: newQuantity,
            avg_price: newAvgPrice,
            market_value: newQuantity * position.current_price,
            unrealized_pnl: (position.current_price - newAvgPrice) * newQuantity,
            unrealized_pnl_pct: ((position.current_price - newAvgPrice) / newAvgPrice) * 100
          };
        } else {
          // Handle sell logic
          const newQuantity = Math.max(0, position.quantity - pendingOrder.quantity);
          updatedPositions[existingPositionIndex] = {
            ...position,
            quantity: newQuantity,
            market_value: newQuantity * position.current_price,
            unrealized_pnl: (position.current_price - position.avg_price) * newQuantity,
            unrealized_pnl_pct: newQuantity > 0 ? ((position.current_price - position.avg_price) / position.avg_price) * 100 : 0
          };
        }
        
        setPositions(updatedPositions);
      }

      setConfirmDialogOpen(false);
      setPendingOrder(null);
      
    } catch (error) {
      console.error('Order execution failed:', error);
    }
  };

  const getCurrentPrice = (symbol: string): number => {
    const position = positions.find(p => p.symbol === symbol);
    return position?.current_price || 100;
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'BUY': return 'success';
      case 'SELL': return 'error';
      case 'HOLD': return 'warning';
      default: return 'default';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'BUY': return <TrendingUp />;
      case 'SELL': return <TrendingDown />;
      case 'HOLD': return <Timeline />;
      default: return <Info />;
    }
  };

  return (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h4" component="h1">
          Professional Trading Interface
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch 
                checked={tradingEnabled} 
                onChange={(e) => setTradingEnabled(e.target.checked)}
                color="primary"
              />
            }
            label="Live Trading"
          />
          <Chip 
            label={tradingEnabled ? 'LIVE' : 'SIMULATION'} 
            color={tradingEnabled ? 'error' : 'warning'}
            variant="filled"
          />
        </Box>
      </Box>

      {!tradingEnabled && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Trading is in simulation mode. Enable live trading to execute real orders.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Market Timing Widget */}
        <Grid item xs={12} md={4}>
                    <SimpleMarketTiming marketInfo={marketInfo} />
        </Grid>

        {/* Order Entry Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>Order Entry</Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Symbol</InputLabel>
                <Select
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                >
                  <MenuItem value="AAPL">AAPL</MenuItem>
                  <MenuItem value="MSFT">MSFT</MenuItem>
                  <MenuItem value="TSLA">TSLA</MenuItem>
                  <MenuItem value="GOOGL">GOOGL</MenuItem>
                  <MenuItem value="AMZN">AMZN</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                margin="normal"
                label="Quantity"
                type="number"
                value={orderQuantity}
                onChange={(e) => setOrderQuantity(parseInt(e.target.value) || 0)}
              />

              <FormControl fullWidth margin="normal">
                <InputLabel>Side</InputLabel>
                <Select
                  value={orderSide}
                  onChange={(e) => setOrderSide(e.target.value as 'buy' | 'sell')}
                >
                  <MenuItem value="buy">Buy</MenuItem>
                  <MenuItem value="sell">Sell</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth margin="normal">
                <InputLabel>Order Type</InputLabel>
                <Select
                  value={orderType}
                  onChange={(e) => setOrderType(e.target.value as any)}
                >
                  <MenuItem value="market">Market</MenuItem>
                  <MenuItem value="limit">Limit</MenuItem>
                  <MenuItem value="smart">Smart Routing</MenuItem>
                </Select>
              </FormControl>

              {orderType === 'limit' && (
                <TextField
                  fullWidth
                  margin="normal"
                  label="Limit Price"
                  type="number"
                  value={limitPrice || ''}
                  onChange={(e) => setLimitPrice(parseFloat(e.target.value) || undefined)}
                />
              )}

              <FormControl fullWidth margin="normal">
                <InputLabel>Execution Strategy</InputLabel>
                <Select
                  value={executionStrategy}
                  onChange={(e) => setExecutionStrategy(e.target.value as any)}
                >
                  <MenuItem value="market">Market</MenuItem>
                  <MenuItem value="twap">TWAP</MenuItem>
                  <MenuItem value="vwap">VWAP</MenuItem>
                  <MenuItem value="is">Implementation Shortfall</MenuItem>
                </Select>
              </FormControl>

              <Box mt={2} display="flex" gap={1}>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  startIcon={<PlayArrow />}
                  onClick={handlePlaceOrder}
                  disabled={!tradingEnabled && orderType !== 'smart'}
                >
                  Place Order
                </Button>
              </Box>

              <Box mt={1}>
                <Typography variant="caption" color="text.secondary">
                  Est. Market Impact: {(Math.random() * 0.1).toFixed(3)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Signals */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6">Current Trading Signals</Typography>
                <IconButton size="small">
                  <Refresh />
                </IconButton>
              </Box>
              
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Signal</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Recommendation</TableCell>
                      <TableCell>Risk Score</TableCell>
                      <TableCell>Action</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {currentSignals.map((signal) => (
                      <TableRow key={signal.symbol}>
                        <TableCell>
                          <strong>{signal.symbol}</strong>
                        </TableCell>
                        <TableCell>
                          <Typography 
                            color={signal.signal >= 0 ? 'success.main' : 'error.main'}
                            fontWeight="bold"
                          >
                            {signal.signal.toFixed(3)}
                          </Typography>
                        </TableCell>
                        <TableCell>{(signal.confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <Chip
                            icon={getRecommendationIcon(signal.recommendation)}
                            label={signal.recommendation}
                            color={getRecommendationColor(signal.recommendation) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={`${(signal.risk_score * 100).toFixed(0)}%`}
                            size="small"
                            color={signal.risk_score < 0.3 ? 'success' : signal.risk_score < 0.5 ? 'warning' : 'error'}
                          />
                        </TableCell>
                        <TableCell>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => setSelectedSymbol(signal.symbol)}
                          >
                            Trade
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Price Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>
                {selectedSymbol} - Intraday Price
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Line type="monotone" dataKey="price" stroke="#2196f3" strokeWidth={2} />
                  <RechartsTooltip />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Positions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>Current Positions</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Qty</TableCell>
                      <TableCell align="right">Avg Price</TableCell>
                      <TableCell align="right">Current</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell>Action</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {positions.map((position) => (
                      <TableRow key={position.symbol}>
                        <TableCell><strong>{position.symbol}</strong></TableCell>
                        <TableCell align="right">{position.quantity}</TableCell>
                        <TableCell align="right">${position.avg_price.toFixed(2)}</TableCell>
                        <TableCell align="right">${position.current_price.toFixed(2)}</TableCell>
                        <TableCell align="right">
                          <Box>
                            <Typography 
                              color={position.unrealized_pnl >= 0 ? 'success.main' : 'error.main'}
                              fontWeight="bold"
                              fontSize="0.875rem"
                            >
                              ${position.unrealized_pnl.toFixed(2)}
                            </Typography>
                            <Typography 
                              color={position.unrealized_pnl >= 0 ? 'success.main' : 'error.main'}
                              fontSize="0.75rem"
                            >
                              ({position.unrealized_pnl_pct.toFixed(2)}%)
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Tooltip title="Close Position">
                            <IconButton 
                              size="small"
                              onClick={() => {
                                setSelectedSymbol(position.symbol);
                                setOrderQuantity(position.quantity);
                                setOrderSide('sell');
                              }}
                            >
                              <Cancel color="error" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Orders */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" mb={2}>Recent Order History</Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Fill Price</TableCell>
                      <TableCell>Strategy</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Execution Quality</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {orderHistory.map((order) => (
                      <TableRow key={order.id}>
                        <TableCell>
                          {new Date(order.timestamp).toLocaleTimeString()}
                        </TableCell>
                        <TableCell><strong>{order.symbol}</strong></TableCell>
                        <TableCell>
                          <Chip 
                            label={order.side.toUpperCase()} 
                            color={order.side === 'buy' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">{order.quantity}</TableCell>
                        <TableCell align="right">${order.fill_price?.toFixed(2)}</TableCell>
                        <TableCell>{order.execution_strategy.toUpperCase()}</TableCell>
                        <TableCell>
                          <Chip 
                            icon={<CheckCircle />}
                            label={order.status.toUpperCase()} 
                            color="success"
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Box>
                            <Typography variant="caption" display="block">
                              Slippage: {order.execution_quality?.slippage.toFixed(3)}%
                            </Typography>
                            <Typography variant="caption" display="block">
                              Fill: {order.execution_quality?.fill_rate.toFixed(1)}%
                            </Typography>
                            <Typography variant="caption">
                              Time: {order.execution_quality?.execution_time_ms}ms
                            </Typography>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Order Confirmation Dialog */}
      <Dialog open={confirmDialogOpen} onClose={() => setConfirmDialogOpen(false)}>
        <DialogTitle>Confirm Order</DialogTitle>
        <DialogContent>
          {pendingOrder && (
            <Box>
              <Typography><strong>Symbol:</strong> {pendingOrder.symbol}</Typography>
              <Typography><strong>Side:</strong> {pendingOrder.side.toUpperCase()}</Typography>
              <Typography><strong>Quantity:</strong> {pendingOrder.quantity}</Typography>
              <Typography><strong>Order Type:</strong> {pendingOrder.order_type.toUpperCase()}</Typography>
              <Typography><strong>Strategy:</strong> {pendingOrder.execution_strategy.toUpperCase()}</Typography>
              {pendingOrder.limit_price && (
                <Typography><strong>Limit Price:</strong> ${pendingOrder.limit_price.toFixed(2)}</Typography>
              )}
              <Typography><strong>Est. Market Value:</strong> ${(pendingOrder.quantity * getCurrentPrice(pendingOrder.symbol)).toLocaleString()}</Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={confirmOrder} autoFocus>
            Confirm Order
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProfessionalTradingInterface;
