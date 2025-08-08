import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Button, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Dialog, DialogTitle, DialogContent, DialogActions,
  LinearProgress, Alert, IconButton, Tooltip, Tabs, Tab
} from '@mui/material';
import {
  PlayArrow, Stop, Visibility, Assessment,
  Science, ModelTraining, Storage, CheckCircle,
  Refresh
} from '@mui/icons-material';

interface ModelMetadata {
  model_id: string;
  model_name: string;
  version: string;
  model_type: string;
  status: string;
  created_at: string;
  performance_metrics: {
    sharpe_ratio: number;
    win_rate: number;
    max_drawdown: number;
  };
  model_size_mb: number;
}

interface ChallengerTest {
  test_id: string;
  champion_model_id: string;
  challenger_model_id: string;
  current_phase: string;
  samples_collected: number;
  final_decision?: string;
  decision_confidence?: number;
}

interface BacktestResult {
  backtest_id: string;
  status: string;
  performance_metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
}

const MLOpsManagementDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [models, setModels] = useState<Record<string, ModelMetadata[]>>({});
  const [registrySummary, setRegistrySummary] = useState<any>(null);
  const [activeTests, setActiveTests] = useState<ChallengerTest[]>([]);
  const [testSummary, setTestSummary] = useState<any>(null);
  const [backtests, setBacktests] = useState<BacktestResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelMetadata | null>(null);
  const [createTestDialog, setCreateTestDialog] = useState(false);
  const [createBacktestDialog, setCreateBacktestDialog] = useState(false);

  // API base URL
  const API_BASE = 'http://localhost:8001';

  // Load data on component mount
  useEffect(() => {
    loadAllData();
    // Refresh data every 30 seconds
    const interval = setInterval(loadAllData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadModels(),
        loadActiveTests(),
        loadBacktests()
      ]);
    } catch (err) {
      setError('Failed to load MLOps data');
      console.error('Error loading MLOps data:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/mlops/models`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setModels(data.data.models);
        setRegistrySummary(data.data.summary);
      }
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const loadActiveTests = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/mlops/tests`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setActiveTests(data.data.active_tests);
        setTestSummary(data.data.summary);
      }
    } catch (err) {
      console.error('Error loading tests:', err);
    }
  };

  const loadBacktests = async () => {
    // This would load backtest results from the API
    // For now, simulate some data
    setBacktests([
      {
        backtest_id: 'bt_12345',
        status: 'completed',
        performance_metrics: {
          total_return: 0.25,
          sharpe_ratio: 2.1,
          max_drawdown: -0.08,
          win_rate: 0.58
        }
      }
    ]);
  };

  const setChampionModel = async (modelName: string, version: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/mlops/models/${modelName}/champion/${version}`,
        { method: 'POST' }
      );
      const data = await response.json();
      
      if (data.status === 'success') {
        await loadModels(); // Refresh models
      } else {
        setError('Failed to set champion model');
      }
    } catch (err) {
      setError('Failed to set champion model');
      console.error('Error setting champion:', err);
    }
  };

  const createChallengerTest = async (testConfig: any) => {
    try {
      const response = await fetch(`${API_BASE}/api/mlops/tests/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testConfig)
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setCreateTestDialog(false);
        await loadActiveTests();
      } else {
        setError('Failed to create challenger test');
      }
    } catch (err) {
      setError('Failed to create challenger test');
      console.error('Error creating test:', err);
    }
  };

  const stopTest = async (testId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/mlops/tests/${testId}/stop`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        await loadActiveTests();
      }
    } catch (err) {
      console.error('Error stopping test:', err);
    }
  };

  const runBacktest = async (backtestConfig: any) => {
    try {
      const response = await fetch(`${API_BASE}/api/mlops/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backtestConfig)
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        setCreateBacktestDialog(false);
        await loadBacktests();
      }
    } catch (err) {
      console.error('Error running backtest:', err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'champion': return 'success';
      case 'challenger': return 'warning';
      case 'registered': return 'info';
      case 'deprecated': return 'default';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase.toLowerCase()) {
      case 'completed': return 'success';
      case 'testing': return 'warning';
      case 'analysis': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const ModelRegistryTab = () => (
    <Box>
      {/* Registry Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Models
              </Typography>
              <Typography variant="h4" component="div">
                {registrySummary?.total_models || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Model Families
              </Typography>
              <Typography variant="h4" component="div">
                {Object.keys(registrySummary?.model_families || {}).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Storage Used
              </Typography>
              <Typography variant="h4" component="div">
                {(registrySummary?.total_storage_mb || 0).toFixed(1)} MB
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Champions Active
              </Typography>
              <Typography variant="h4" component="div">
                {registrySummary?.status_breakdown?.champion || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Models Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              <Storage sx={{ mr: 1, verticalAlign: 'middle' }} />
              Model Registry
            </Typography>
            <Button
              startIcon={<Refresh />}
              onClick={loadModels}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>

          {Object.entries(models).map(([modelName, versions]) => (
            <Box key={modelName} sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ mb: 1 }}>{modelName}</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Version</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Sharpe Ratio</TableCell>
                      <TableCell>Win Rate</TableCell>
                      <TableCell>Max DD</TableCell>
                      <TableCell>Size (MB)</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {versions.map((model) => (
                      <TableRow key={model.model_id}>
                        <TableCell>{model.version}</TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={model.status}
                            color={getStatusColor(model.status) as any}
                          />
                        </TableCell>
                        <TableCell>{model.performance_metrics.sharpe_ratio?.toFixed(2)}</TableCell>
                        <TableCell>{(model.performance_metrics.win_rate * 100)?.toFixed(1)}%</TableCell>
                        <TableCell>{(model.performance_metrics.max_drawdown * 100)?.toFixed(1)}%</TableCell>
                        <TableCell>{model.model_size_mb.toFixed(1)}</TableCell>
                        <TableCell>{new Date(model.created_at).toLocaleDateString()}</TableCell>
                        <TableCell>
                          <Tooltip title="View Details">
                            <IconButton size="small" onClick={() => setSelectedModel(model)}>
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          {model.status !== 'champion' && (
                            <Tooltip title="Set as Champion">
                              <IconButton
                                size="small"
                                onClick={() => setChampionModel(model.model_name, model.version)}
                              >
                                <CheckCircle />
                              </IconButton>
                            </Tooltip>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          ))}
        </CardContent>
      </Card>
    </Box>
  );

  const ChallengerTestingTab = () => (
    <Box>
      {/* Test Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Tests
              </Typography>
              <Typography variant="h4" component="div">
                {testSummary?.active_tests || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed Tests
              </Typography>
              <Typography variant="h4" component="div">
                {testSummary?.completed_tests || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Success Rate
              </Typography>
              <Typography variant="h4" component="div">
                {testSummary?.completed_tests > 0 
                  ? ((testSummary.completed_tests / testSummary.total_tests) * 100).toFixed(1) 
                  : 0}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Failed Tests
              </Typography>
              <Typography variant="h4" component="div">
                {testSummary?.failed_tests || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Active Tests */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              <Science sx={{ mr: 1, verticalAlign: 'middle' }} />
              Champion-Challenger Tests
            </Typography>
            <Box>
              <Button
                startIcon={<PlayArrow />}
                variant="contained"
                onClick={() => setCreateTestDialog(true)}
                sx={{ mr: 1 }}
              >
                Create Test
              </Button>
              <Button
                startIcon={<Refresh />}
                onClick={loadActiveTests}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Test ID</TableCell>
                  <TableCell>Champion</TableCell>
                  <TableCell>Challenger</TableCell>
                  <TableCell>Phase</TableCell>
                  <TableCell>Samples</TableCell>
                  <TableCell>Decision</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {activeTests.map((test) => (
                  <TableRow key={test.test_id}>
                    <TableCell>{test.test_id}</TableCell>
                    <TableCell>{test.champion_model_id.split('_')[0]}</TableCell>
                    <TableCell>{test.challenger_model_id.split('_')[0]}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={test.current_phase}
                        color={getPhaseColor(test.current_phase) as any}
                      />
                    </TableCell>
                    <TableCell>{test.samples_collected}</TableCell>
                    <TableCell>
                      {test.final_decision ? (
                        <Chip
                          size="small"
                          label={test.final_decision}
                          color={test.final_decision === 'promote' ? 'success' : 'default'}
                        />
                      ) : '-'}
                    </TableCell>
                    <TableCell>
                      {test.decision_confidence 
                        ? `${(test.decision_confidence * 100).toFixed(1)}%`
                        : '-'}
                    </TableCell>
                    <TableCell>
                      <Tooltip title="Stop Test">
                        <IconButton 
                          size="small" 
                          onClick={() => stopTest(test.test_id)}
                          disabled={test.current_phase === 'completed'}
                        >
                          <Stop />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {activeTests.length === 0 && (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography color="textSecondary">
                No active tests. Create a new champion-challenger test to get started.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );

  const BacktestingTab = () => (
    <Box>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
              Institutional Backtesting
            </Typography>
            <Button
              startIcon={<PlayArrow />}
              variant="contained"
              onClick={() => setCreateBacktestDialog(true)}
            >
              Run Backtest
            </Button>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Backtest ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Total Return</TableCell>
                  <TableCell>Sharpe Ratio</TableCell>
                  <TableCell>Max Drawdown</TableCell>
                  <TableCell>Win Rate</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {backtests.map((backtest) => (
                  <TableRow key={backtest.backtest_id}>
                    <TableCell>{backtest.backtest_id}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={backtest.status}
                        color={backtest.status === 'completed' ? 'success' : 'warning'}
                      />
                    </TableCell>
                    <TableCell>
                      {backtest.performance_metrics
                        ? `${(backtest.performance_metrics.total_return * 100).toFixed(1)}%`
                        : '-'}
                    </TableCell>
                    <TableCell>
                      {backtest.performance_metrics?.sharpe_ratio?.toFixed(2) || '-'}
                    </TableCell>
                    <TableCell>
                      {backtest.performance_metrics
                        ? `${(backtest.performance_metrics.max_drawdown * 100).toFixed(1)}%`
                        : '-'}
                    </TableCell>
                    <TableCell>
                      {backtest.performance_metrics
                        ? `${(backtest.performance_metrics.win_rate * 100).toFixed(1)}%`
                        : '-'}
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Results">
                        <IconButton size="small">
                          <Visibility />
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
    </Box>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          <ModelTraining sx={{ mr: 2, verticalAlign: 'middle' }} />
          MLOps Management
        </Typography>
        {loading && <LinearProgress sx={{ width: 200 }} />}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Model Registry" />
          <Tab label="Champion-Challenger" />
          <Tab label="Backtesting" />
        </Tabs>
      </Box>

      {activeTab === 0 && <ModelRegistryTab />}
      {activeTab === 1 && <ChallengerTestingTab />}
      {activeTab === 2 && <BacktestingTab />}

      {/* Model Details Dialog */}
      <Dialog open={!!selectedModel} onClose={() => setSelectedModel(null)} maxWidth="md" fullWidth>
        <DialogTitle>Model Details: {selectedModel?.model_id}</DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography><strong>Version:</strong> {selectedModel.version}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography><strong>Type:</strong> {selectedModel.model_type}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography><strong>Status:</strong> 
                    <Chip
                      size="small"
                      label={selectedModel.status}
                      color={getStatusColor(selectedModel.status) as any}
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography><strong>Size:</strong> {selectedModel.model_size_mb.toFixed(1)} MB</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Performance Metrics</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Typography>Sharpe Ratio: {selectedModel.performance_metrics.sharpe_ratio?.toFixed(2)}</Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography>Win Rate: {(selectedModel.performance_metrics.win_rate * 100)?.toFixed(1)}%</Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography>Max Drawdown: {(selectedModel.performance_metrics.max_drawdown * 100)?.toFixed(1)}%</Typography>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedModel(null)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Create Test Dialog */}
      <Dialog open={createTestDialog} onClose={() => setCreateTestDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create Champion-Challenger Test</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              Configure a new A/B test to compare model performance.
            </Typography>
            {/* Test configuration form would go here */}
            <Typography>Test configuration form coming soon...</Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateTestDialog(false)}>Cancel</Button>
          <Button variant="contained" disabled>Create Test</Button>
        </DialogActions>
      </Dialog>

      {/* Create Backtest Dialog */}
      <Dialog open={createBacktestDialog} onClose={() => setCreateBacktestDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Run Institutional Backtest</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              Configure backtest parameters for comprehensive analysis.
            </Typography>
            {/* Backtest configuration form would go here */}
            <Typography>Backtest configuration form coming soon...</Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateBacktestDialog(false)}>Cancel</Button>
          <Button variant="contained" disabled>Run Backtest</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MLOpsManagementDashboard;
