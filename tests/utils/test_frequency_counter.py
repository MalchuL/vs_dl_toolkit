import pytest
import warnings

from dl_toolkit.utils.frequency_counter import FreqCounter


def test_basic_frequency():
    counter = FreqCounter()
    
    # Test frequency of 2 (should trigger every 2nd call)
    assert counter("test_key", 2) is True  # First call should trigger
    assert counter("test_key", 2) is False  # Second call shouldn't trigger
    assert counter("test_key", 2) is True  # Third call should trigger again
    
    # Test frequency of 1 (should trigger every call)
    assert counter("always", 1) is True
    assert counter("always", 1) is True
    
    # Test frequency of 0 (should never trigger)
    assert counter("never", 0) is False
    assert counter("never", 0) is False


def test_multiple_keys():
    counter = FreqCounter()
    
    # Different keys should maintain separate counters
    assert counter("key1", 2) is True
    assert counter("key2", 3) is True
    assert counter("key1", 2) is False
    assert counter("key2", 3) is False
    assert counter("key1", 2) is True
    assert counter("key2", 3) is False


def test_update_count_flag():
    counter = FreqCounter()
    
    # Initial trigger
    assert counter("test", 2) is True
    
    # Test with update_count=False (shouldn't update internal counter)
    assert counter("test", 2, update_count=False) is False  # Should return current state
    assert counter("test", 2) is False  # Counter continues from previous state
    assert counter("test", 2) is True  # Triggers again after 2 calls


def test_get_counter():
    counter = FreqCounter()
    
    # Test with explicit name
    counter_fn = counter.get_counter(2, name="test_counter")
    assert counter_fn() is True
    assert counter_fn() is False
    assert counter_fn() is True
    
    # Test with auto-generated name
    with warnings.catch_warnings(record=True) as w:
        auto_counter = counter.get_counter(2)
        assert len(w) == 1
        assert "Name for counter was not specified" in str(w[0].message)
    
    assert auto_counter() is True
    assert auto_counter() is False


def test_get_counter_name_collision():
    counter = FreqCounter()
    
    # Create first counter
    counter.get_counter(2, name="test")
    
    # Attempt to create counter with same name should raise error
    with pytest.raises(ValueError, match="Counter with name test already exists"):
        counter.get_counter(2, name="test")
    
    # Should work with exist_name_ok=True
    counter.get_counter(2, name="test", exist_name_ok=True)


def test_float_frequency():
    counter = FreqCounter()
    
    # Test with float frequency < 1 (should never trigger)
    assert counter("float_test", 0.5) == True  # First call should trigger
    assert counter("float_test", 0.5) == True  # Second call should trigger
    
    # Test with float frequency > 1 (should work same as int)
    assert counter("float_test2", 1.5) == True
    assert counter("float_test2", 1.5) == False
    assert counter("float_test2", 1.5) == True
