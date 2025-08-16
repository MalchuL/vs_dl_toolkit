import warnings
from collections import defaultdict
from typing import Callable


class FreqCounter:
    """A utility class for managing frequency-based triggers in deep learning workflows.
    
    This class provides a flexible way to trigger actions at specific intervals or frequencies,
    which is particularly useful for:
    - Logging metrics at regular intervals during training
    - Scheduling validation steps
    - Triggering model checkpoints
    - Managing any periodic events in training pipelines
    
    The counter supports both integer and float frequencies, where:
    - frequency > 0: Triggers on first call and then every N calls
    - frequency ≤ 0: Never triggers
    
    Example usage:
        ```python
        counter = FreqCounter()
        
        # Trigger every 100 steps
        if counter("validation", 100):
            run_validation()
            
        # Create a reusable counter function
        log_counter = counter.get_counter(10, name="logging")
        if log_counter():
            log_metrics()
        ```
    """
    def __init__(self):
        """Initialize the frequency counter with an empty counter dictionary."""
        self._call_count = defaultdict(int)

    def __call__(self, key: str, freq: float | int, update_count: bool = True,
                 add_value: float | int = 1) -> bool:
        """Check if the frequency condition is met for the given key and update its counter.

        This method is the primary way to use the frequency counter. It checks if an action
        should be triggered based on the frequency and maintains the internal counter state.

        Args:
            key (str): Unique identifier for this counter. Different keys maintain separate counters.
            freq (float | int): The frequency at which to trigger.
                - If > 0: Triggers on first call and then every N calls
                - If ≤ 0: Never triggers
            update_count (bool, optional): Whether to update the internal counter.
                - If True: Updates counter and checks frequency condition
                - If False: Only checks current state without updating (useful for peeking)
                Defaults to True.
            add_value (float | int | None, optional): The value to update the counter by.
                Defaults to 1.
        Returns:
            bool: True if the frequency condition is met and action should be triggered,
                 False otherwise.

        Example:
            ```python
            counter = FreqCounter()
            
            # Trigger every 100 steps
            if counter("validation", 100):
                run_validation()
                
            # Check without updating counter
            if counter("checkpoint", 1000, update_count=False):
                print("Would trigger on next call")
            ```
        """
        if add_value < 0:
            raise ValueError("add_value must be non-negative")
        if update_count:
            if freq > 0:  # Default behavior, update counter
                result = False
                if self._call_count[key] == 0:
                    result = True

                self._call_count[key] += add_value
                if self._call_count[key] >= freq:
                    self._call_count[key] = 0
                return result
            else:  # If freq less than 1, don't trigger it
                self._call_count[key] = 0
                return False
        else:
            return self._call_count[key] == 0

    def get_counter(self, freq: float | int, name: str | None = None,
                    exist_name_ok: bool = False, init_value: float | int = 0) -> Callable[[bool], bool]:
        """Create a reusable counter function with fixed frequency and name.

        This method creates a closure that maintains a specific frequency counter,
        providing a more convenient way to use the counter repeatedly without
        specifying the frequency and name each time.

        Args:
            freq (float | int): The frequency at which the counter should trigger.
                Same rules as in __call__ method.
            name (str | None, optional): Name for this counter instance.
                - If None: Auto-generates a unique name with warning
                - If provided: Uses this name, checking for collisions
                Defaults to None.
            exist_name_ok (bool, optional): Whether to allow using an existing counter name.
                - If True: Reuses existing counter if name exists
                - If False: Raises ValueError if name exists
                Defaults to False.
            init_value (float | int, optional): The initial value of the counter.
                Defaults to 0.
        Returns:
            Callable[[bool], bool]: A function that takes an optional update_count parameter
                and returns whether the frequency condition is met.

        Raises:
            ValueError: If name already exists and exist_name_ok is False.

        Example:
            ```python
            counter = FreqCounter()
            
            # Create named counter that triggers every 10 steps
            log_counter = counter.get_counter(10, name="logging")
            
            # Use the counter repeatedly
            for step in range(100):
                if log_counter():  # Equivalent to counter("logging", 10)
                    log_metrics()
            ```
        """
        if name is None:
            i = 0
            while True:
                temp_name = 'counter_' + str(i)
                if temp_name not in self._call_count:
                    name = temp_name
                    warnings.warn('Name for counter was not specified. Defaulting to {}'.format(name))
                    break
                i += 1
        else:
            if name in self._call_count and not exist_name_ok:
                raise ValueError('Counter with name {} already exists'.format(name))
        self._call_count[name] = init_value

        def counter_fn(update_count=True, add_value=1):
            return self(name, freq, update_count=update_count, add_value=add_value)

        return counter_fn