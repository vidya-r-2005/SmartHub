
"""SmartHub single-coin acceptor credit handling."""


class CoinAcceptor:
    """Track credit from one type of accepted coin."""

    def __init__(self, credit_per_coin=1):
        if credit_per_coin <= 0:
            raise ValueError("credit_per_coin must be positive")

        self.credit_per_coin = credit_per_coin
        self.total_credit = 0
        self.coin_count = 0

    def add_coin(self, count=1):
        """Record accepted coins and return the credit added."""
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")

        added_credit = count * self.credit_per_coin
        self.coin_count += count
        self.total_credit += added_credit
        return added_credit

    def get_credit(self):
        return self.total_credit

    def reset_credit(self):
        self.total_credit = 0
        self.coin_count = 0
