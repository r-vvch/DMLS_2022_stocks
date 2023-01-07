from binanceENUM import OrderType, OrderSide


class Deal:

    def __init__(self, client, symbol: str):
        self.quantity = None
        # To check if orders statuses
        self.buy_order_id = None
        self.sell_order_id = None

        self.client = client
        self.symbol = symbol

    def _order_status(self, order_side: OrderSide):
        """Returns status of sell or buy order"""
        if order_side == OrderSide.SELL:
            return self.client.get_order(self.symbol, self.sell_order_id)['status']
        elif order_side == OrderSide.BUY:
            return self.client.get_order(self.symbol, self.buy_order_id)['status']
        else:
            return None

    # TODO: Добавить флаг order_side, чтобы выводить сразу нужную цену, а не две
    def ticker_price(self) -> dict:
        """Returns bid and ask price"""
        # If you are buying a stock, you pay the ask price. If you sell a stock, you receive the bid price.
        prices = self.client.book_ticker(self.symbol)
        return {'bidPrice': prices['bidPrice'], 'askPrice': prices['askPrice']}
