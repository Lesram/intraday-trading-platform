import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Alert,
  Button,
  Box,
  Chip,
  LinearProgress
} from '@mui/material';
import { Refresh, CheckCircle, Warning, Error } from '@mui/icons-material';
import { TradingApiService } from '../services/api';

const SimpleHealthMonitor: React.FC = () => {
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('🔍 Loading health status...');
      const response = await TradingApiService.getHealthStatus();
      console.log('✅ Health data loaded:', response);
      setHealthData(response);
    } catch (err: any) {
      console.error('❌ Health check failed:', err);
      setError(err.message || 'Failed to load health data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHealth();
    // Refresh every 30 seconds
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card sx={{ height: '400px' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔍 System Health Monitor
          </Typography>
          <LinearProgress sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            Loading comprehensive health analysis...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ height: '400px' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔍 System Health Monitor
          </Typography>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Button onClick={loadHealth} variant="outlined" startIcon={<Refresh />}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!healthData) {
    return (
      <Card sx={{ height: '400px' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔍 System Health Monitor
          </Typography>
          <Alert severity="warning">
            No health data available
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'excellent':
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'warning':
      case 'degraded':
        return <Warning color="warning" />;
      default:
        return <Error color="error" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'excellent':
      case 'healthy':
        return 'success';
      case 'warning':
      case 'degraded':
        return 'warning';
      default:
        return 'error';
    }
  };

  return (
    <Card sx={{ height: '400px', overflow: 'auto' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div">
            🔍 System Health Monitor
          </Typography>
          <Button
            size="small"
            onClick={loadHealth}
            disabled={loading}
            startIcon={<Refresh />}
          >
            Refresh
          </Button>
        </Box>

        {/* Overall Health Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          {getStatusIcon(healthData.overall_health || 'Unknown')}
          <Typography variant="h5">
            Overall Health: {healthData.overall_health || 'Unknown'}
          </Typography>
        </Box>

        {/* Health Metrics */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            System Status
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
            <Chip
              label={`ML Models: ${healthData.ml_models_ok ? '✅ OK' : '❌ Issues'}`}
              color={healthData.ml_models_ok ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`Trading API: ${healthData.trading_api_ok ? '✅ OK' : '❌ Offline'}`}
              color={healthData.trading_api_ok ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`Market Data: ${healthData.market_data_ok ? '✅ Fresh' : '⚠️ Stale'}`}
              color={healthData.market_data_ok ? 'success' : 'warning'}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>

        {/* Issues Summary */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Issues Summary
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Chip
              label={`${healthData.critical_issues || 0} Critical`}
              color={healthData.critical_issues > 0 ? 'error' : 'success'}
              size="small"
            />
            <Chip
              label={`${healthData.warnings || 0} Warnings`}
              color={healthData.warnings > 0 ? 'warning' : 'success'}
              size="small"
            />
          </Box>
        </Box>

        {/* System Operational Status */}
        <Alert 
          severity={healthData.system_operational ? 'success' : 'warning'}
          variant="outlined"
        >
          <Typography variant="body2">
            System Status: {healthData.system_operational ? 'Fully Operational' : 'Issues Detected'}
          </Typography>
          {healthData.last_check && (
            <Typography variant="caption" color="text.secondary">
              Last Check: {new Date(healthData.last_check).toLocaleTimeString()}
            </Typography>
          )}
        </Alert>

        {/* Quick Actions */}
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => {
              console.log('🔍 Full health data:', healthData);
              alert(`Health Status: ${healthData.overall_health}\nCritical Issues: ${healthData.critical_issues}\nWarnings: ${healthData.warnings}`);
            }}
          >
            View Details
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SimpleHealthMonitor;
