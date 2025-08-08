import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  TrendingUp,
  Analytics,
  PieChart,
  BarChart as BarChartIcon,
  Timeline,
  Download,
  FilterList
} from '@mui/icons-material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  PieChart as RechartsPieChart, 
  Cell,
  Scatter,
  ScatterChart,
  ComposedChart,
  Legend,
  Tooltip
} from 'recharts';

interface PerformanceData {
  date: string;
  portfolio_return: number;
  benchmark_return: number;
  alpha: number;
  sharpe: number;
  drawdown: number;
  volume: number;
}

interface PositionData {
  symbol: string;
  weight: number;
  return_contribution: number;
  risk_contribution: number;
  sector: string;
  market_value: number;
}

interface FactorExposure {
  factor: string;
  exposure: number;
  benchmark: number;
  active_exposure: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const AdvancedAnalyticsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState('1M');
  const [selectedBenchmark, setSelectedBenchmark] = useState('SPY');

  // Sample performance data
  const [performanceData] = useState<PerformanceData[]>([
    { date: '2025-07-01', portfolio_return: 0.5, benchmark_return: 0.3, alpha: 0.2, sharpe: 1.2, drawdown: 0, volume: 1000000 },
    { date: '2025-07-08', portfolio_return: 1.2, benchmark_return: 0.8, alpha: 0.4, sharpe: 1.4, drawdown: -0.1, volume: 1200000 },
    { date: '2025-07-15', portfolio_return: 0.8, benchmark_return: 1.0, alpha: -0.2, sharpe: 1.3, drawdown: -0.3, volume: 900000 },
    { date: '2025-07-22', portfolio_return: 2.1, benchmark_return: 1.5, alpha: 0.6, sharpe: 1.6, drawdown: -0.1, volume: 1500000 },
    { date: '2025-07-29', portfolio_return: 1.8, benchmark_return: 1.2, alpha: 0.6, sharpe: 1.7, drawdown: 0, volume: 1300000 },
    { date: '2025-08-05', portfolio_return: 2.5, benchmark_return: 1.8, alpha: 0.7, sharpe: 1.8, drawdown: 0, volume: 1600000 }
  ]);

  // Sample position data
  const [positionData] = useState<PositionData[]>([
    { symbol: 'AAPL', weight: 25.5, return_contribution: 1.2, risk_contribution: 18.2, sector: 'Technology', market_value: 127500 },
    { symbol: 'MSFT', weight: 30.2, return_contribution: 1.8, risk_contribution: 22.1, sector: 'Technology', market_value: 151000 },
    { symbol: 'TSLA', weight: 12.8, return_contribution: -0.3, risk_contribution: 28.5, sector: 'Technology', market_value: 64000 },
    { symbol: 'JPM', weight: 15.2, return_contribution: 0.8, risk_contribution: 12.8, sector: 'Finance', market_value: 76000 },
    { symbol: 'JNJ', weight: 8.5, return_contribution: 0.4, risk_contribution: 6.2, sector: 'Healthcare', market_value: 42500 },
    { symbol: 'XOM', weight: 7.8, return_contribution: 0.6, risk_contribution: 12.2, sector: 'Energy', market_value: 39000 }
  ]);

  // Sample factor exposure data
  const [factorExposure] = useState<FactorExposure[]>([
    { factor: 'Market Beta', exposure: 0.85, benchmark: 1.0, active_exposure: -0.15 },
    { factor: 'Size Factor', exposure: -0.2, benchmark: 0.0, active_exposure: -0.2 },
    { factor: 'Value Factor', exposure: 0.1, benchmark: 0.0, active_exposure: 0.1 },
    { factor: 'Momentum', exposure: 0.35, benchmark: 0.0, active_exposure: 0.35 },
    { factor: 'Quality', exposure: 0.25, benchmark: 0.0, active_exposure: 0.25 },
    { factor: 'Volatility', exposure: -0.1, benchmark: 0.0, active_exposure: -0.1 }
  ]);

  // Risk-Return scatter data
  const [riskReturnData] = useState([
    { risk: 8.5, return: 15.2, symbol: 'AAPL', size: 25.5 },
    { risk: 12.1, return: 18.7, symbol: 'MSFT', size: 30.2 },
    { risk: 28.5, return: -2.1, symbol: 'TSLA', size: 12.8 },
    { risk: 15.2, return: 12.8, symbol: 'JPM', size: 15.2 },
    { risk: 8.8, return: 8.5, symbol: 'JNJ', size: 8.5 },
    { risk: 22.1, return: 14.2, symbol: 'XOM', size: 7.8 }
  ]);

  const PerformanceTab = () => (
    <Grid container spacing={3}>
      {/* Performance Summary Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" color="primary">Total Return</Typography>
            <Typography variant="h4">+15.2%</Typography>
            <Typography variant="body2" color="text.secondary">vs Benchmark: +11.8%</Typography>
            <LinearProgress variant="determinate" value={75} sx={{ mt: 1 }} />
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" color="primary">Alpha Generation</Typography>
            <Typography variant="h4">+3.4%</Typography>
            <Typography variant="body2" color="success.main">Outperforming benchmark</Typography>
            <LinearProgress variant="determinate" value={85} color="success" sx={{ mt: 1 }} />
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" color="primary">Sharpe Ratio</Typography>
            <Typography variant="h4">1.78</Typography>
            <Typography variant="body2" color="text.secondary">Risk-adjusted returns</Typography>
            <LinearProgress variant="determinate" value={89} color="info" sx={{ mt: 1 }} />
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" color="primary">Max Drawdown</Typography>
            <Typography variant="h4" color="error">-2.8%</Typography>
            <Typography variant="body2" color="success.main">Within limits</Typography>
            <LinearProgress variant="determinate" value={25} color="warning" sx={{ mt: 1 }} />
          </CardContent>
        </Card>
      </Grid>

      {/* Performance Chart */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6">Portfolio vs Benchmark Performance</Typography>
              <Box display="flex" gap={1}>
                <FormControl size="small">
                  <Select value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
                    <MenuItem value="1W">1W</MenuItem>
                    <MenuItem value="1M">1M</MenuItem>
                    <MenuItem value="3M">3M</MenuItem>
                    <MenuItem value="1Y">1Y</MenuItem>
                  </Select>
                </FormControl>
                <Button startIcon={<Download />} size="small">Export</Button>
              </Box>
            </Box>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Bar yAxisId="right" dataKey="volume" fill="#e3f2fd" name="Volume" />
                <Area yAxisId="left" type="monotone" dataKey="benchmark_return" stackId="1" stroke="#ff9800" fill="#ff9800" fillOpacity={0.3} name="Benchmark" />
                <Area yAxisId="left" type="monotone" dataKey="portfolio_return" stackId="2" stroke="#2196f3" fill="#2196f3" fillOpacity={0.3} name="Portfolio" />
                <Line yAxisId="left" type="monotone" dataKey="alpha" stroke="#4caf50" strokeWidth={2} name="Alpha" />
                <Tooltip />
                <Legend />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Sharpe Ratio Trend */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Sharpe Ratio Trend</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Line type="monotone" dataKey="sharpe" stroke="#9c27b0" strokeWidth={3} />
                <Tooltip />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const PositionAnalysisTab = () => (
    <Grid container spacing={3}>
      {/* Position Performance Table */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Position Analysis</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Weight %</TableCell>
                    <TableCell align="right">Return Contrib.</TableCell>
                    <TableCell align="right">Risk Contrib.</TableCell>
                    <TableCell>Sector</TableCell>
                    <TableCell align="right">Market Value</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {positionData.map((position) => (
                    <TableRow key={position.symbol}>
                      <TableCell component="th" scope="row">
                        <strong>{position.symbol}</strong>
                      </TableCell>
                      <TableCell align="right">{position.weight.toFixed(1)}%</TableCell>
                      <TableCell align="right">
                        <Typography color={position.return_contribution >= 0 ? 'success.main' : 'error.main'}>
                          {position.return_contribution.toFixed(2)}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{position.risk_contribution.toFixed(1)}%</TableCell>
                      <TableCell>
                        <Chip label={position.sector} size="small" />
                      </TableCell>
                      <TableCell align="right">${position.market_value.toLocaleString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Sector Allocation */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Sector Allocation</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RechartsPieChart>
                <RechartsPieChart data={[
                  { name: 'Technology', value: 68.5, color: '#0088FE' },
                  { name: 'Finance', value: 15.2, color: '#00C49F' },
                  { name: 'Healthcare', value: 8.5, color: '#FFBB28' },
                  { name: 'Energy', value: 7.8, color: '#FF8042' }
                ]}>
                  {[
                    { name: 'Technology', value: 68.5 },
                    { name: 'Finance', value: 15.2 },
                    { name: 'Healthcare', value: 8.5 },
                    { name: 'Energy', value: 7.8 }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </RechartsPieChart>
                <Tooltip />
              </RechartsPieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Risk-Return Scatter */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Risk-Return Analysis</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={riskReturnData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="risk" name="Risk %" />
                <YAxis type="number" dataKey="return" name="Return %" />
                <Scatter name="Positions" dataKey="return" fill="#8884d8">
                  {riskReturnData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Scatter>
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const FactorAnalysisTab = () => (
    <Grid container spacing={3}>
      {/* Factor Exposure Chart */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Factor Exposure Analysis</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={factorExposure} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="factor" />
                <Bar dataKey="exposure" fill="#2196f3" name="Portfolio Exposure" />
                <Bar dataKey="benchmark" fill="#ff9800" name="Benchmark Exposure" />
                <Tooltip />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Active Exposure */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Active Factor Exposure</Typography>
            {factorExposure.map((factor, index) => (
              <Box key={index} mb={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">{factor.factor}</Typography>
                  <Typography 
                    variant="body2" 
                    color={factor.active_exposure >= 0 ? 'success.main' : 'error.main'}
                    fontWeight="bold"
                  >
                    {factor.active_exposure.toFixed(2)}
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.abs(factor.active_exposure) * 100} 
                  color={factor.active_exposure >= 0 ? 'success' : 'error'}
                />
              </Box>
            ))}
          </CardContent>
        </Card>
      </Grid>

      {/* Attribution Analysis */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Performance Attribution</Typography>
            <Grid container spacing={3}>
              <Grid item xs={3}>
                <Box textAlign="center" p={2} bgcolor="primary.light" borderRadius={2}>
                  <Typography variant="h4" color="white">+1.8%</Typography>
                  <Typography variant="body2" color="white">Security Selection</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center" p={2} bgcolor="success.light" borderRadius={2}>
                  <Typography variant="h4" color="white">+0.9%</Typography>
                  <Typography variant="body2" color="white">Market Timing</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center" p={2} bgcolor="info.light" borderRadius={2}>
                  <Typography variant="h4" color="white">+0.55%</Typography>
                  <Typography variant="body2" color="white">Execution Alpha</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center" p={2} bgcolor="warning.light" borderRadius={2}>
                  <Typography variant="h4" color="white">+0.15%</Typography>
                  <Typography variant="body2" color="white">Factor Timing</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const ReportsTab = () => (
    <Grid container spacing={3}>
      {/* Report Generation */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Generate Reports</Typography>
            <Box display="flex" flexDirection="column" gap={2}>
              <Button variant="contained" startIcon={<Download />} fullWidth>
                Daily Performance Report
              </Button>
              <Button variant="contained" startIcon={<Download />} fullWidth>
                Risk Analysis Report
              </Button>
              <Button variant="contained" startIcon={<Download />} fullWidth>
                Factor Attribution Report
              </Button>
              <Button variant="contained" startIcon={<Download />} fullWidth>
                Execution Quality Report
              </Button>
              <Button variant="outlined" startIcon={<FilterList />} fullWidth>
                Custom Report Builder
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Key Metrics Summary */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" mb={2}>Key Metrics Summary</Typography>
            <TableContainer>
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell>Total Return (MTD)</TableCell>
                    <TableCell align="right"><strong>+3.25%</strong></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Benchmark Return (MTD)</TableCell>
                    <TableCell align="right">+2.15%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Alpha Generation</TableCell>
                    <TableCell align="right" style={{ color: '#4caf50' }}><strong>+1.10%</strong></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Sharpe Ratio</TableCell>
                    <TableCell align="right">1.78</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Information Ratio</TableCell>
                    <TableCell align="right">1.45</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Tracking Error</TableCell>
                    <TableCell align="right">4.2%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Max Drawdown</TableCell>
                    <TableCell align="right" style={{ color: '#f44336' }}>-2.8%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Win Rate</TableCell>
                    <TableCell align="right">68.5%</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  return (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h4" component="h1">
          Advanced Analytics Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <FormControl size="small">
            <InputLabel>Benchmark</InputLabel>
            <Select value={selectedBenchmark} onChange={(e) => setSelectedBenchmark(e.target.value)}>
              <MenuItem value="SPY">S&P 500 (SPY)</MenuItem>
              <MenuItem value="QQQ">NASDAQ (QQQ)</MenuItem>
              <MenuItem value="IWM">Russell 2000 (IWM)</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab icon={<TrendingUp />} label="Performance" />
        <Tab icon={<PieChart />} label="Positions" />
        <Tab icon={<Analytics />} label="Factor Analysis" />
        <Tab icon={<BarChartIcon />} label="Reports" />
      </Tabs>

      {activeTab === 0 && <PerformanceTab />}
      {activeTab === 1 && <PositionAnalysisTab />}
      {activeTab === 2 && <FactorAnalysisTab />}
      {activeTab === 3 && <ReportsTab />}
    </Box>
  );
};

export default AdvancedAnalyticsDashboard;
