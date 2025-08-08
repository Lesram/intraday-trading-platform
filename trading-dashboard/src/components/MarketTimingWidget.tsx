import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  LinearProgress
} from '@mui/material';
import { 
  Schedule as ScheduleIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AccessTime as AccessTimeIcon
} from '@mui/icons-material';

interface MarketTimingInfo {
  success: boolean;
  market_status: string;
  is_market_open: boolean;
  trading_session: string;
  data_type: string;
  local_time: string;
  eastern_time: string;
  local_time_short: string;
  eastern_time_short: string;
  date: string;
  next_open?: string;
  opens_at?: string;
  closes_at?: string;
}

const MarketTimingWidget: React.FC = () => {
  const [marketInfo, setMarketInfo] = useState<MarketTimingInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMarketTiming = async () => {
      try {
        const response = await fetch('/market/timing');
        const data = await response.json();
        
        if (data.success) {
          setMarketInfo(data);
          setError(null);
        } else {
          setError(data.error || 'Failed to fetch market timing');
        }
      } catch (err) {
        setError('Network error fetching market timing');
        console.error('Market timing fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchMarketTiming();

    // Update every 30 seconds
    const interval = setInterval(fetchMarketTiming, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (isOpen: boolean, status: string) => {
    if (isOpen) return 'success';
    if (status === 'PRE-MARKET' || status === 'AFTER-HOURS') return 'warning';
    return 'default';
  };

  const getStatusIcon = (isOpen: boolean, status: string) => {
    if (isOpen) return <TrendingUpIcon />;
    if (status === 'PRE-MARKET' || status === 'AFTER-HOURS') return <ScheduleIcon />;
    return <TrendingDownIcon />;
  };

  if (loading) {
    return (
      <Card sx={{ minHeight: 150 }}>
        <CardContent>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Loading market timing...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ minHeight: 150 }}>
        <CardContent>
          <Typography variant="body2" color="error">
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!marketInfo) return null;

  return (
    <Card sx={{ minHeight: 150 }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
            Market Status
          </Typography>
          <Chip
            icon={getStatusIcon(marketInfo.is_market_open, marketInfo.market_status)}
            label={marketInfo.market_status}
            color={getStatusColor(marketInfo.is_market_open, marketInfo.market_status)}
            variant={marketInfo.is_market_open ? "filled" : "outlined"}
          />
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <AccessTimeIcon sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                Local Time
              </Typography>
            </Box>
            <Typography variant="body1" sx={{ fontWeight: 'medium', fontFamily: 'monospace' }}>
              {marketInfo.local_time_short}
            </Typography>
          </Grid>

          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <AccessTimeIcon sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                Eastern Time
              </Typography>
            </Box>
            <Typography variant="body1" sx={{ fontWeight: 'medium', fontFamily: 'monospace' }}>
              {marketInfo.eastern_time_short}
            </Typography>
          </Grid>
        </Grid>

        <Box mt={2}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Trading Session
          </Typography>
          <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
            {marketInfo.trading_session}
          </Typography>
        </Box>

        <Box mt={1}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Data Type
          </Typography>
          <Chip
            label={marketInfo.data_type}
            size="small"
            color={marketInfo.data_type === 'LIVE' ? 'success' : 'info'}
            variant="outlined"
          />
        </Box>

        {marketInfo.next_open && (
          <Box mt={1}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Next Market Open
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {marketInfo.next_open}
            </Typography>
          </Box>
        )}

        {marketInfo.opens_at && (
          <Box mt={1}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Opens At
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {marketInfo.opens_at}
            </Typography>
          </Box>
        )}

        {marketInfo.closes_at && (
          <Box mt={1}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Closes At
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {marketInfo.closes_at}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default MarketTimingWidget;
