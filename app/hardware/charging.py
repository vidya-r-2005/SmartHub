
"""SmartHub charging-session logic.

The relay starts OFF. Time is injected so charging can be tested
without real hardware or waiting for a timer.
"""


class MockRelay:
    """Relay substitute for development and tests."""

    def __init__(self):
        self.is_on = False

    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False

    def close(self):
        self.turn_off()


class ChargingController:
    """Manage a timed charging session using an injected relay."""

    def __init__(self, relay, seconds_per_credit=60, clock=None):
        if seconds_per_credit <= 0:
            raise ValueError("seconds_per_credit must be positive")

        import time

        self.relay = relay
        self.seconds_per_credit = seconds_per_credit
        self.clock = clock or time.monotonic
        self._deadline = None

        # Safe initial state: charging is disabled.
        self.relay.turn_off()

    def start(self, credit):
        if not isinstance(credit, (int, float)) or isinstance(credit, bool):
            raise ValueError("Credit must be numeric")
        if credit <= 0:
            raise ValueError("Credit must be positive")

        self.stop()
        self._deadline = self.clock() + credit * self.seconds_per_credit
        self.relay.turn_on()
        return self._deadline

    def update(self):
        """Call periodically; stop charging when the deadline is reached."""
        if self._deadline is not None and self.clock() >= self._deadline:
            self.stop()
            return True
        return False

    def stop(self):
        self.relay.turn_off()
        self._deadline = None

    def is_charging(self):
        return self._deadline is not None and self.relay.is_on

    def seconds_remaining(self):
        if not self.is_charging():
            return 0
        return max(0, int(self._deadline - self.clock()))

    def close(self):
        self.stop()
        self.relay.close()
