"""
Tests for Her 365 Friend Service

Simple tests to validate the friend service functionality.
"""

import sys
from her365_friend_service import Her365FriendService, create_service


def test_service_initialization():
    """Test that the service initializes correctly."""
    service = Her365FriendService()
    assert service.name == "Her 365 Friend Service"
    assert service.version == "1.0.0"
    assert service.data_path == "./data"
    print("✓ Service initialization test passed")


def test_greet():
    """Test the greet method."""
    service = Her365FriendService()
    greeting = service.greet()
    assert "Her 365 Friend Service" in greeting
    assert "365 days" in greeting
    print("✓ Greet test passed")


def test_service_info():
    """Test getting service info."""
    service = Her365FriendService()
    info = service.get_service_info()
    assert info["name"] == "Her 365 Friend Service"
    assert info["version"] == "1.0.0"
    assert "data_path" in info
    assert "started_at" in info
    print("✓ Service info test passed")


def test_pillar_info():
    """Test getting pillar information."""
    service = Her365FriendService()
    pillars = service.get_pillar_info()
    assert len(pillars) == 5
    assert "Anticipation" in pillars
    assert "Separation" in pillars
    assert "Execution" in pillars
    assert "Ball Tracking" in pillars
    assert "Innovation" in pillars
    print("✓ Pillar info test passed")


def test_aftersnap_iq_info():
    """Test getting AFTERSNAP IQ information."""
    service = Her365FriendService()
    info = service.get_aftersnap_iq_info()
    assert "AFTERSNAP IQ" in info
    assert "A-S-E-E-I" in info
    print("✓ AFTERSNAP IQ info test passed")


def test_help():
    """Test the help method."""
    service = Her365FriendService()
    help_text = service.help()
    assert "Her 365 Friend Service" in help_text
    assert "Usage Guide" in help_text
    assert "greet()" in help_text
    print("✓ Help test passed")


def test_factory_function():
    """Test the factory function."""
    service = create_service()
    assert isinstance(service, Her365FriendService)
    assert service.data_path == "./data"
    
    service_custom = create_service(data_path="/custom/path")
    assert service_custom.data_path == "/custom/path"
    print("✓ Factory function test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Running Her 365 Friend Service Tests")
    print("=" * 50 + "\n")
    
    tests = [
        test_service_initialization,
        test_greet,
        test_service_info,
        test_pillar_info,
        test_aftersnap_iq_info,
        test_help,
        test_factory_function,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "-" * 50)
    if failed == 0:
        print(f"All {len(tests)} tests passed! 🎉")
        print("-" * 50 + "\n")
        return 0
    else:
        print(f"{failed} test(s) failed out of {len(tests)}")
        print("-" * 50 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
