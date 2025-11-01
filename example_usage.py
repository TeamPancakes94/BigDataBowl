"""
Example usage of Her 365 Friend Service

This demonstrates how to use the friend service in your analytics workflow.
"""

from her365_friend_service import create_service


def main():
    """Demonstrate the Her 365 Friend Service."""
    print("\n" + "=" * 60)
    print(" Her 365 Friend Service - Example Usage")
    print("=" * 60 + "\n")
    
    # Create the service
    service = create_service(data_path="./data")
    
    # Get a friendly greeting
    print(service.greet())
    print()
    
    # Show service information
    print("📋 Service Information:")
    info = service.get_service_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print()
    
    # Explain AFTERSNAP IQ
    print("🎯 " + service.get_aftersnap_iq_info())
    print()
    
    # Show the five pillars
    print("📊 The Five Pillars of Sky Vision:")
    for pillar, description in service.get_pillar_info().items():
        print(f"   • {pillar}: {description}")
    print()
    
    # Try loading data (will show friendly messages if files are empty)
    print("💾 Data Loading Example:")
    print("   Attempting to load player data...")
    players = service.load_player_data()
    if players is not None and not players.empty:
        print(f"   ✓ Loaded {len(players)} players")
    else:
        print("   ℹ️  Player data is currently empty (that's okay for demo!)")
    print()
    
    print("   Attempting to load play data...")
    plays = service.load_play_data()
    if plays is not None and not plays.empty:
        print(f"   ✓ Loaded {len(plays)} plays")
    else:
        print("   ℹ️  Play data is currently empty (that's okay for demo!)")
    print()
    
    # Show help information
    print("❓ Need Help? The service has you covered!")
    print("   Run: service.help() to see all available methods")
    print()
    
    print("=" * 60)
    print(" Your friend is ready to help with analytics year-round! 🎉")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
