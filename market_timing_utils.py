#!/usr/bin/env python3
"""
🕒 MARKET TIMING UTILITIES
Real-time market status and timezone management for institutional trading platform
"""

import pytz
from datetime import datetime, time
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MarketTimingManager:
    """Manage market hours, timezones, and trading session status"""
    
    def __init__(self):
        # Define timezone objects
        self.eastern = pytz.timezone('US/Eastern')
        self.local_tz = None
        self._detect_local_timezone()
        
        # NYSE market hours (Eastern Time)
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET
        
        # Pre-market and after-hours
        self.premarket_open = time(4, 0)   # 4:00 AM ET
        self.afterhours_close = time(20, 0)  # 8:00 PM ET
        
        logger.info(f"🕒 Market Timing Manager initialized (Local: {self.local_tz}, Eastern: US/Eastern)")
    
    def _detect_local_timezone(self):
        """Detect the local timezone"""
        try:
            import tzlocal
            self.local_tz = tzlocal.get_localzone()
        except ImportError:
            # Fallback to system timezone
            self.local_tz = pytz.timezone('UTC')
            logger.warning("tzlocal not available, using UTC as fallback")
    
    def get_current_times(self) -> Dict[str, Any]:
        """Get current time in both local and Eastern time"""
        now_utc = datetime.now(pytz.UTC)
        local_time = now_utc.astimezone(self.local_tz)
        eastern_time = now_utc.astimezone(self.eastern)
        
        return {
            'local_time': local_time,
            'eastern_time': eastern_time,
            'local_formatted': local_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'eastern_formatted': eastern_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'local_time_only': local_time.strftime('%H:%M:%S'),
            'eastern_time_only': eastern_time.strftime('%H:%M:%S'),
            'date': eastern_time.strftime('%Y-%m-%d')
        }
    
    def is_market_open(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if the market is currently open
        Returns: (is_open, status_message, timing_info)
        """
        current_times = self.get_current_times()
        et_now = current_times['eastern_time']
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if et_now.weekday() >= 5:  # Saturday or Sunday
            next_open = self._get_next_market_open(et_now)
            return False, "CLOSED - Weekend", {
                **current_times,
                'market_status': 'CLOSED_WEEKEND',
                'next_open': next_open,
                'session': 'Weekend'
            }
        
        current_time = et_now.time()
        
        # Market is open (9:30 AM - 4:00 PM ET)
        if self.market_open_time <= current_time <= self.market_close_time:
            return True, "OPEN - Regular Hours", {
                **current_times,
                'market_status': 'OPEN_REGULAR',
                'session': 'Regular Trading',
                'closes_at': self.market_close_time.strftime('%H:%M ET')
            }
        
        # Pre-market (4:00 AM - 9:30 AM ET)
        elif self.premarket_open <= current_time < self.market_open_time:
            return False, "PRE-MARKET", {
                **current_times,
                'market_status': 'PREMARKET',
                'session': 'Pre-Market Trading',
                'opens_at': self.market_open_time.strftime('%H:%M ET')
            }
        
        # After-hours (4:00 PM - 8:00 PM ET)
        elif self.market_close_time < current_time <= self.afterhours_close:
            return False, "AFTER-HOURS", {
                **current_times,
                'market_status': 'AFTERHOURS',
                'session': 'After-Hours Trading',
                'next_open': self._get_next_market_open(et_now)
            }
        
        # Market closed (8:00 PM - 4:00 AM ET)
        else:
            next_open = self._get_next_market_open(et_now)
            return False, "CLOSED", {
                **current_times,
                'market_status': 'CLOSED',
                'session': 'Market Closed',
                'next_open': next_open
            }
    
    def _get_next_market_open(self, current_et: datetime) -> str:
        """Calculate next market opening time"""
        # If it's Friday after close or weekend, next open is Monday
        if current_et.weekday() == 4 and current_et.time() > self.market_close_time:  # Friday after close
            days_to_add = 3  # Go to Monday
        elif current_et.weekday() >= 5:  # Weekend
            days_to_add = 7 - current_et.weekday()  # Days until Monday
        else:
            days_to_add = 1  # Tomorrow
        
        next_date = current_et.date()
        for _ in range(days_to_add):
            next_date = next_date.replace(day=next_date.day + 1)
        
        return f"{next_date.strftime('%Y-%m-%d')} {self.market_open_time.strftime('%H:%M ET')}"
    
    def get_market_session_info(self) -> Dict[str, Any]:
        """Get comprehensive market session information"""
        is_open, status, timing_info = self.is_market_open()
        
        return {
            'is_market_open': is_open,
            'market_status': status,
            'timing': timing_info,
            'data_type': 'LIVE' if is_open else 'LAST_AVAILABLE',
            'trading_session': timing_info.get('session', 'Unknown')
        }

# Global instance
market_timing_manager = MarketTimingManager()

def get_market_status() -> Dict[str, Any]:
    """Convenience function to get market status"""
    return market_timing_manager.get_market_session_info()

def is_market_open() -> bool:
    """Simple check if market is open"""
    is_open, _, _ = market_timing_manager.is_market_open()
    return is_open

def get_current_times() -> Dict[str, Any]:
    """Get formatted current times"""
    return market_timing_manager.get_current_times()
