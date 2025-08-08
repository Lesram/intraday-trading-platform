import React from 'react';
import { Box, Chip, Typography } from '@mui/material';

interface MarketInfo {
  is_open: boolean;
  status: string;
  data_type: string;
  local_time: string;
  eastern_time: string;
  date: string;
}

interface Props {
  marketInfo: MarketInfo;
}

const SimpleMarketTiming: React.FC<Props> = ({ marketInfo }) => {
  if (!marketInfo) return null;

  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 2, 
      p: 1, 
      backgroundColor: 'background.paper',
      borderRadius: 1,
      border: '1px solid',
      borderColor: 'divider'
    }}>
      {/* Market Status */}
      <Chip
        label={marketInfo.status}
        color={marketInfo.is_open ? 'success' : 'error'}
        variant="filled"
        size="small"
      />
      
      {/* Data Type */}
      <Chip
        label={marketInfo.data_type}
        color={marketInfo.data_type === 'LIVE' ? 'primary' : 'warning'}
        variant="outlined"
        size="small"
      />
      
      {/* Times */}
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          Local: {marketInfo.local_time}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {marketInfo.eastern_time}
        </Typography>
      </Box>
    </Box>
  );
};

export default SimpleMarketTiming;
