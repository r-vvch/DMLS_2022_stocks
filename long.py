from deal import Deal
from binanceENUM import OrderSide, OrderType, OrderStatus  # , OrderTimeInForce


class Long(Deal):

    def __init__(self, client, symbol: str):
        super(Long, self).__init__(client, symbol)

    def buy(self, quantity: int):  # , time_in_force: OrderTimeInForce):
        """Creates buy order"""
        # # Test whether the crypto have been sold
        # if self._order_status(OrderSide.SELL) != OrderStatus.FILLED.name:
        #     return None
        # Zeroing all buy order params
        self.sell_order_id = None

        self.quantity = quantity
        params = {
            'symbol': self.symbol,
            'side': OrderSide.BUY.name,
            'type': OrderType.MARKET.name,
            # 'timeInForce': time_in_force.name,
            'quantity': quantity,
        }
        buy_response = self.client.new_order(**params)
        self.buy_order_id = buy_response['orderId']
        return buy_response

    def sell(self, order_type: OrderType, price: float, stop_price: float):  # , time_in_force: OrderTimeInForce):
        """Creates sell order"""
        # Test whether the crypto have been bought
        if self._order_status(OrderSide.BUY) != OrderStatus.FILLED.name:
            return None
        # Zeroing all buy order params
        self.buy_order_id = None

        params = {
            'symbol': self.symbol,
            'side': OrderSide.SELL.name,
            'type': order_type.name,
            # 'timeInForce': time_in_force.name,
            'quantity': self.quantity,
            'price': price,
            'stopPrice': stop_price
        }
        sell_response = self.client.new_order(**params)
        self.sell_order_id = sell_response['orderId']
        return sell_response

    # TODO: Мб сделать сразу с закупом нового
    def cancel_sell_order(self):
        """Cancels sell order"""
        # Test whether the crypto have been sold
        if self._order_status(OrderSide.SELL) != OrderStatus.FILLED.name:
            return None
        close_response = self.client.cansel_order(self.symbol, self.sell_order_id)
        self.sell_order_id = None
        return close_response
