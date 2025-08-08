import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, AppBar, Toolbar, Typography, Tabs, Tab } from '@mui/material';
import TradingDashboard from './pages/Dashboard_Updated';
import TradingInterface from './components/TradingInterface';
import MLOpsManagementDashboard from './components/MLOpsManagementDashboard';
import CriticalMonitoringDashboard from './components/CriticalMonitoringDashboard';
import AdvancedAnalyticsDashboard from './components/AdvancedAnalyticsDashboard';
import ProfessionalTradingInterface from './components/ProfessionalTradingInterface';

// Create custom theme for trading dashboard
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4', // Cyan
    },
    secondary: {
      main: '#ff9800', // Orange
    },
    success: {
      main: '#4caf50', // Green
    },
    error: {
      main: '#f44336', // Red
    },
    warning: {
      main: '#ff9800', // Orange
    },
    info: {
      main: '#2196f3', // Blue
    },
    background: {
      default: '#0a0e27',
      paper: '#162447',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e3a8a',
          borderRadius: 12,
          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e3a8a',
          borderRadius: 12,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

const App: React.FC = () => {
  const [currentTab, setCurrentTab] = React.useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', backgroundColor: 'background.default' }}>
        <AppBar position="static" sx={{ bgcolor: 'background.paper' }}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🚀 Professional Trading Platform
            </Typography>
            <Tabs value={currentTab} onChange={handleTabChange} textColor="inherit">
              <Tab label="Portfolio Dashboard" />
              <Tab label="Professional Trading" />
              <Tab label="Critical Monitoring" />
              <Tab label="Advanced Analytics" />
              <Tab label="MLOps Management" />
            </Tabs>
          </Toolbar>
        </AppBar>
        
        <Box sx={{ p: 3 }}>
          {currentTab === 0 && <TradingDashboard />}
          {currentTab === 1 && <ProfessionalTradingInterface />}
          {currentTab === 2 && <CriticalMonitoringDashboard />}
          {currentTab === 3 && <AdvancedAnalyticsDashboard />}
          {currentTab === 4 && <MLOpsManagementDashboard />}
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;
