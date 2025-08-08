import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Button, 
  Alert,
  Box,
  Chip,
  CircularProgress 
} from '@mui/material';
import { TradingApiService } from '../services/api';

const APITestComponent: React.FC = () => {
  const [testResults, setTestResults] = useState<any>({});
  const [loading, setLoading] = useState(false);

  const runTests = async () => {
    setLoading(true);
    const results: any = {};

    // Test each critical API endpoint
    const tests = [
      { name: 'Health Status', method: () => TradingApiService.getHealthStatus() },
      { name: 'System Health', method: () => TradingApiService.getSystemHealth() },
      { name: 'Portfolio Metrics', method: () => TradingApiService.getPortfolioMetrics() },
      { name: 'Trading Signals', method: () => TradingApiService.getLatestSignals(5) },
      { name: 'Portfolio Positions', method: () => TradingApiService.getPositions() },
      { name: 'Trading Status', method: () => TradingApiService.getTradingStatus() },
      { name: 'Trading Logs', method: () => TradingApiService.getTradeLogs() },
    ];

    for (const test of tests) {
      try {
        console.log(`🧪 Testing ${test.name}...`);
        const result = await test.method();
        results[test.name] = { success: true, data: result };
        console.log(`✅ ${test.name} passed:`, result);
      } catch (error: any) {
        results[test.name] = { success: false, error: error.message };
        console.error(`❌ ${test.name} failed:`, error);
      }
    }

    setTestResults(results);
    setLoading(false);
  };

  useEffect(() => {
    runTests();
  }, []);

  return (
    <Card sx={{ maxWidth: 800, margin: 2 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          🧪 API Connection Test Results
        </Typography>
        
        <Button 
          variant="contained" 
          onClick={runTests} 
          disabled={loading}
          sx={{ mb: 2 }}
        >
          {loading ? <CircularProgress size={20} /> : 'Retest APIs'}
        </Button>

        {Object.entries(testResults).map(([testName, result]: [string, any]) => (
          <Box key={testName} sx={{ mb: 1 }}>
            <Chip
              label={testName}
              color={result.success ? 'success' : 'error'}
              variant="outlined"
              sx={{ mr: 1, mb: 1 }}
            />
            {result.success ? (
              <Typography variant="caption" color="success.main">
                ✅ Connected - {typeof result.data === 'object' ? JSON.stringify(result.data).substring(0, 50) + '...' : result.data}
              </Typography>
            ) : (
              <Alert severity="error" sx={{ mt: 1 }}>
                ❌ Error: {result.error}
              </Alert>
            )}
          </Box>
        ))}

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            API Base URL: http://localhost:8002/api<br/>
            WebSocket URL: ws://localhost:8002/ws<br/>
            Current Time: {new Date().toLocaleTimeString()}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default APITestComponent;
