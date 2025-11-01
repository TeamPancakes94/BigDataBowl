"""
Her 365 Friend Service
A companion utility service for the Sky Vision analytics platform.

This service provides friendly, accessible helper methods for interacting with
NFL tracking data and AFTERSNAP IQ analytics year-round (365 days).
"""

import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime


class Her365FriendService:
    """
    The first ever friend service for Sky Vision analytics.
    
    Provides helpful utilities and friendly interfaces to access player
    performance data, pillar metrics, and AFTERSNAP IQ scores.
    """
    
    def __init__(self, data_path: str = "./data"):
        """
        Initialize the Her 365 Friend Service.
        
        Args:
            data_path: Path to the data directory containing CSV files
        """
        self.data_path = data_path
        self.service_start_time = datetime.now()
        self.version = "1.0.0"
        self.name = "Her 365 Friend Service"
        
    def greet(self) -> str:
        """
        Friendly greeting from the service.
        
        Returns:
            A welcoming message
        """
        return f"Welcome to {self.name}! Your companion for NFL analytics, available 365 days a year."
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the friend service.
        
        Returns:
            Dictionary containing service metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "started_at": self.service_start_time.isoformat(),
            "data_path": self.data_path,
            "description": "A friendly companion service for Sky Vision analytics"
        }
    
    def load_player_data(self) -> Optional[pd.DataFrame]:
        """
        Load player data in a friendly way.
        
        Returns:
            DataFrame containing player information, or None if file not found
        """
        try:
            return pd.read_csv(f"{self.data_path}/players.csv")
        except FileNotFoundError:
            print(f"Friendly reminder: players.csv not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"Oops! Couldn't load player data: {e}")
            return None
    
    def load_play_data(self) -> Optional[pd.DataFrame]:
        """
        Load play data in a friendly way.
        
        Returns:
            DataFrame containing play information, or None if file not found
        """
        try:
            return pd.read_csv(f"{self.data_path}/plays.csv")
        except FileNotFoundError:
            print(f"Friendly reminder: plays.csv not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"Oops! Couldn't load play data: {e}")
            return None
    
    def load_tracking_data(self) -> Optional[pd.DataFrame]:
        """
        Load tracking data in a friendly way.
        
        Returns:
            DataFrame containing tracking information, or None if file not found
        """
        try:
            return pd.read_csv(f"{self.data_path}/tracking_week.csv")
        except FileNotFoundError:
            print(f"Friendly reminder: tracking_week.csv not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"Oops! Couldn't load tracking data: {e}")
            return None
    
    def get_pillar_info(self) -> Dict[str, str]:
        """
        Get friendly descriptions of the five pillars.
        
        Returns:
            Dictionary mapping pillar names to descriptions
        """
        return {
            "Anticipation": "Quickness of response after the snap (A)",
            "Separation": "Distance created or denied at the catch point (S)",
            "Execution": "Precision, body control, and mechanics (E)",
            "Ball Tracking": "Ability to locate, track, and adjust to the ball (E)",
            "Innovation": "Off-script adaptability under pressure (I)"
        }
    
    def get_aftersnap_iq_info(self) -> str:
        """
        Get a friendly explanation of AFTERSNAP IQ.
        
        Returns:
            Description of the AFTERSNAP IQ metric
        """
        return (
            "AFTERSNAP IQ™ is your comprehensive player performance score! "
            "It combines five technical pillars (A-S-E-E-I) weighted by importance, "
            "scaled to 0-100 to help you understand who truly won each play."
        )
    
    def help(self) -> str:
        """
        Get helpful information about using the service.
        
        Returns:
            Usage guide for the friend service
        """
        return """
Her 365 Friend Service - Usage Guide
====================================

Available Methods:
- greet(): Get a friendly welcome message
- get_service_info(): View service information
- load_player_data(): Load player information
- load_play_data(): Load play data
- load_tracking_data(): Load tracking data
- get_pillar_info(): Learn about the five pillars
- get_aftersnap_iq_info(): Understand AFTERSNAP IQ
- help(): Show this help message

Example:
    service = Her365FriendService()
    print(service.greet())
    players = service.load_player_data()
    print(service.get_pillar_info())

Your friend is here to help 365 days a year!
"""


def create_service(data_path: str = "./data") -> Her365FriendService:
    """
    Factory function to create a Her 365 Friend Service instance.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Initialized Her365FriendService instance
    """
    return Her365FriendService(data_path=data_path)


if __name__ == "__main__":
    # Demo the friend service
    print("=" * 50)
    print("Her 365 Friend Service Demo")
    print("=" * 50)
    
    service = create_service()
    print(f"\n{service.greet()}\n")
    
    print("Service Info:")
    info = service.get_service_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n{service.get_aftersnap_iq_info()}\n")
    
    print("The Five Pillars:")
    for pillar, description in service.get_pillar_info().items():
        print(f"  • {pillar}: {description}")
    
    print(f"\n{'-' * 50}")
    print("Service initialized and ready to be your friend!")
    print(f"{'-' * 50}")
