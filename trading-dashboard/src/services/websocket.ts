import { io, Socket } from 'socket.io-client';
import { WebSocketMessage, TradingSignal, Position, PortfolioMetrics, SystemHealth, AlertData } from '../types';

export class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;

  // Event callbacks
  private callbacks: { [key: string]: Function[] } = {};

  constructor(private url: string = 'ws://localhost:8002') {
    this.connect();
  }

  // Connect to WebSocket server
  connect(): void {
    if (this.isConnecting || (this.socket && this.socket.connected)) {
      return;
    }

    this.isConnecting = true;
    console.log('🔌 Connecting to WebSocket server...');

    try {
      this.socket = io(this.url, {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true,
      });

      this.setupEventHandlers();
    } catch (error) {
      console.error('❌ WebSocket connection failed:', error);
      this.handleReconnect();
    }
  }

  // Setup WebSocket event handlers
  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('✅ WebSocket connected successfully');
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('❌ WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, reconnect
        this.handleReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('❌ WebSocket connection error:', error);
      this.isConnecting = false;
      this.handleReconnect();
    });

    // Trading signal updates
    this.socket.on('trading_signal', (data: TradingSignal) => {
      console.log('📈 New trading signal:', data.symbol, data.signal);
      this.emit('signal', data);
    });

    // Position updates
    this.socket.on('position_update', (data: Position) => {
      console.log('💼 Position update:', data.symbol, data.unrealized_pnl);
      this.emit('position', data);
    });

    // Portfolio metrics updates
    this.socket.on('portfolio_metrics', (data: PortfolioMetrics) => {
      console.log('📊 Portfolio metrics update:', data.total_pnl_percent);
      this.emit('portfolio', data);
    });

    // System health updates
    this.socket.on('system_health', (data: SystemHealth[]) => {
      console.log('🔧 System health update');
      this.emit('health', data);
    });

    // Risk alerts
    this.socket.on('risk_alert', (data: AlertData) => {
      console.log('⚠️ Risk alert:', data.title);
      this.emit('alert', data);
    });

    // Market data updates
    this.socket.on('market_data', (data: any) => {
      this.emit('market_data', data);
    });

    // General message handler
    this.socket.on('message', (message: WebSocketMessage) => {
      console.log('📨 WebSocket message:', message.type);
      this.emit('message', message);
    });
  }

  // Handle reconnection logic
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      this.emit('connection', { status: 'failed', reason: 'max_attempts_reached' });
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff

    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.disconnect();
      this.connect();
    }, delay);
  }

  // Subscribe to events
  on(event: string, callback: Function): void {
    if (!this.callbacks[event]) {
      this.callbacks[event] = [];
    }
    this.callbacks[event].push(callback);
  }

  // Unsubscribe from events
  off(event: string, callback?: Function): void {
    if (!this.callbacks[event]) return;

    if (callback) {
      this.callbacks[event] = this.callbacks[event].filter(cb => cb !== callback);
    } else {
      this.callbacks[event] = [];
    }
  }

  // Emit events to callbacks
  private emit(event: string, data: any): void {
    if (!this.callbacks[event]) return;

    this.callbacks[event].forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in ${event} callback:`, error);
      }
    });
  }

  // Send message to server
  send(event: string, data: any): void {
    if (!this.socket || !this.socket.connected) {
      console.warn('❌ Cannot send message: WebSocket not connected');
      return;
    }

    this.socket.emit(event, data);
  }

  // Subscribe to specific symbol updates
  subscribeToSymbol(symbol: string): void {
    this.send('subscribe_symbol', { symbol });
    console.log(`📈 Subscribed to ${symbol} updates`);
  }

  // Unsubscribe from symbol updates
  unsubscribeFromSymbol(symbol: string): void {
    this.send('unsubscribe_symbol', { symbol });
    console.log(`📉 Unsubscribed from ${symbol} updates`);
  }

  // Subscribe to risk alerts
  subscribeToRiskAlerts(): void {
    this.send('subscribe_risk_alerts', {});
    console.log('⚠️ Subscribed to risk alerts');
  }

  // Subscribe to system health updates
  subscribeToSystemHealth(): void {
    this.send('subscribe_system_health', {});
    console.log('🔧 Subscribed to system health updates');
  }

  // Get connection status
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Disconnect WebSocket
  disconnect(): void {
    if (this.socket) {
      console.log('🔌 Disconnecting WebSocket...');
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnecting = false;
  }

  // Cleanup
  destroy(): void {
    this.disconnect();
    this.callbacks = {};
    console.log('🗑️ WebSocket service destroyed');
  }
}

// Create singleton instance
export const webSocketService = new WebSocketService();

export default webSocketService;
