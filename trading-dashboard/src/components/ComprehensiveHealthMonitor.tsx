import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  Button,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  Refresh,
  CheckCircle,
  Warning,
  Error,
  Info
} from '@mui/icons-material';
import { TradingApiService } from '../services/api';
import { SystemIntegrityData, MonitoringData, SystemAlert } from '../types';

interface ComprehensiveHealthMonitorProps {
  onHealthUpdate?: (healthy: boolean) => void;
}

const ComprehensiveHealthMonitor: React.FC<ComprehensiveHealthMonitorProps> = ({ onHealthUpdate }) => {
  const [integrityData, setIntegrityData] = useState<SystemIntegrityData | null>(null);
  const [alerts, setAlerts] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [expanded, setExpanded] = useState<string | false>('overview');

  const loadHealthData = async (forceRefresh = false) => {
    setLoading(true);
    try {
      const [integrityResponse, alertsResponse] = await Promise.all([
        forceRefresh ? TradingApiService.forceHealthCheck() : TradingApiService.getSystemIntegrity(),
        TradingApiService.getSystemAlerts()
      ]);

      setIntegrityData(integrityResponse);
      setAlerts(alertsResponse);
      setLastUpdate(new Date().toLocaleTimeString());

      // Notify parent of health status
      const isHealthy = alertsResponse.critical_count === 0 && integrityResponse.summary.critical_issues === 0;
      onHealthUpdate?.(isHealthy);

      console.log('🔍 Comprehensive health data loaded:', { integrityResponse, alertsResponse });
    } catch (error) {
      console.error('Failed to load comprehensive health data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHealthData();
    // Refresh every 2 minutes
    const interval = setInterval(() => loadHealthData(), 120000);
    return () => clearInterval(interval);
  }, []);

  const handleAccordionChange = (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'excellent':
        return <CheckCircle color="success" />;
      case 'degraded':
      case 'warning':
        return <Warning color="warning" />;
      case 'critical':
      case 'offline':
        return <Error color="error" />;
      default:
        return <Info color="info" />;
    }
  };

  if (loading && !integrityData) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔍 Comprehensive System Monitor
          </Typography>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Loading comprehensive health analysis...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!integrityData || !alerts) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Alert severity="error">
            Failed to load comprehensive monitoring data. Monitoring system may be offline.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', maxHeight: '800px', overflow: 'auto' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="div">
            🔍 System Health Monitor
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Last check: {lastUpdate}
            </Typography>
            <Tooltip title="Force health check">
              <IconButton 
                size="small" 
                onClick={() => loadHealthData(true)}
                disabled={loading}
              >
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Overall Health Status */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            {getStatusIcon(integrityData.overall_health)}
            <Typography variant="h5">
              Overall Health: {integrityData.overall_health}
            </Typography>
          </Box>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={4}>
              <Chip
                label={`${alerts.critical_count} Critical`}
                color={alerts.critical_count > 0 ? 'error' : 'success'}
                size="small"
                variant="outlined"
              />
            </Grid>
            <Grid item xs={4}>
              <Chip
                label={`${alerts.warning_count} Warnings`}
                color={alerts.warning_count > 0 ? 'warning' : 'success'}
                size="small"
                variant="outlined"
              />
            </Grid>
            <Grid item xs={4}>
              <Chip
                label={`${integrityData.summary.recommendations.length} Recommendations`}
                color="info"
                size="small"
                variant="outlined"
              />
            </Grid>
          </Grid>
        </Box>

        {/* Active Alerts */}
        {alerts.alerts.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              🚨 Active Alerts
            </Typography>
            {alerts.alerts.slice(0, 3).map((alert: SystemAlert, index: number) => (
              <Alert
                key={index}
                severity={alert.level === 'critical' ? 'error' : alert.level as 'warning' | 'info'}
                sx={{ mb: 1 }}
                variant="outlined"
              >
                <Typography variant="subtitle2">{alert.title}</Typography>
                <Typography variant="body2">{alert.message}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {alert.component} • {new Date(alert.timestamp).toLocaleTimeString()}
                </Typography>
              </Alert>
            ))}
            {alerts.alerts.length > 3 && (
              <Typography variant="caption" color="text.secondary">
                +{alerts.alerts.length - 3} more alerts...
              </Typography>
            )}
          </Box>
        )}

        {/* Detailed Health Sections */}
        <Accordion
          expanded={expanded === 'overview'}
          onChange={handleAccordionChange('overview')}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography sx={{ width: '33%', flexShrink: 0 }}>
              🧠 ML Models
            </Typography>
            <Chip
              label={integrityData.ml_models.using_real_data ? 'Real Data' : 'Synthetic Data'}
              color={integrityData.ml_models.using_real_data ? 'success' : 'error'}
              size="small"
            />
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Models Loaded: {integrityData.ml_models.models_loaded}
                </Typography>
                <Typography variant="body2">
                  Ensemble Status: {integrityData.ml_models.ensemble_operational ? '✅ Operational' : '❌ Offline'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Response Time: {integrityData.ml_models.response_time_ms}ms
                </Typography>
                <Typography variant="body2">
                  Data Source: {integrityData.ml_models.using_real_data ? '✅ Real Market Data' : '⚠️ Synthetic Fallback'}
                </Typography>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Accordion
          expanded={expanded === 'trading'}
          onChange={handleAccordionChange('trading')}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography sx={{ width: '33%', flexShrink: 0 }}>
              💼 Trading System
            </Typography>
            <Chip
              label={integrityData.trading_system.api_connection ? 'Connected' : 'Offline'}
              color={integrityData.trading_system.api_connection ? 'success' : 'error'}
              size="small"
            />
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  API Connection: {integrityData.trading_system.api_connection ? '✅ Active' : '❌ Failed'}
                </Typography>
                <Typography variant="body2">
                  Portfolio Access: {integrityData.trading_system.portfolio_accessible ? '✅ Accessible' : '❌ Blocked'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Order Execution: {integrityData.trading_system.orders_can_execute ? '✅ Enabled' : '❌ Disabled'}
                </Typography>
                <Typography variant="body2">
                  Response Time: {integrityData.trading_system.response_time_ms}ms
                </Typography>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Accordion
          expanded={expanded === 'data'}
          onChange={handleAccordionChange('data')}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography sx={{ width: '33%', flexShrink: 0 }}>
              📊 Data Pipeline
            </Typography>
            <Chip
              label={integrityData.data_pipeline.market_data_fresh ? 'Fresh' : 'Stale'}
              color={integrityData.data_pipeline.market_data_fresh ? 'success' : 'warning'}
              size="small"
            />
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Market Data: {integrityData.data_pipeline.market_data_fresh ? '✅ Fresh' : '⚠️ Stale'}
                </Typography>
                <Typography variant="body2">
                  Database: {integrityData.data_pipeline.database_accessible ? '✅ Accessible' : '❌ Failed'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Feature Engineering: {integrityData.data_pipeline.feature_engineering_ok ? '✅ Working' : '❌ Failed'}
                </Typography>
                <Typography variant="body2">
                  Response Time: {integrityData.data_pipeline.response_time_ms}ms
                </Typography>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Accordion
          expanded={expanded === 'performance'}
          onChange={handleAccordionChange('performance')}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography sx={{ width: '33%', flexShrink: 0 }}>
              ⚡ System Performance
            </Typography>
            <Chip
              label={integrityData.performance.system_responsive ? 'Responsive' : 'Slow'}
              color={integrityData.performance.system_responsive ? 'success' : 'warning'}
              size="small"
            />
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  System Response: {integrityData.performance.system_responsive ? '✅ Fast' : '⚠️ Slow'}
                </Typography>
                <Typography variant="body2">
                  Memory Usage: {integrityData.performance.memory_usage_ok ? '✅ Normal' : '⚠️ High'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Disk Space: {integrityData.performance.disk_space_ok ? '✅ Available' : '⚠️ Low'}
                </Typography>
                <Typography variant="body2">
                  Avg Response: {integrityData.performance.avg_response_time_ms}ms
                </Typography>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Recommendations */}
        {integrityData.summary.recommendations.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              💡 Recommendations
            </Typography>
            {integrityData.summary.recommendations.slice(0, 3).map((rec: string, index: number) => (
              <Alert key={index} severity="info" sx={{ mb: 1 }} variant="outlined">
                <Typography variant="body2">{rec}</Typography>
              </Alert>
            ))}
          </Box>
        )}

        <Divider sx={{ my: 2 }} />
        
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => loadHealthData(true)}
            disabled={loading}
            startIcon={<Refresh />}
          >
            Force Health Check
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ComprehensiveHealthMonitor;
