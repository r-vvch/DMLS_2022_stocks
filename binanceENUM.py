from enum import Enum


# Class of order statuses
class OrderStatus(Enum):
    NEW = "NEW"  # The order has been accepted by the engine
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # A part of the order has been filled
    FILLED = "FILLED"  # The order has been completed
    CANCELED = "CANCELED"  # The order has been canceled by the user
    PENDING_CANCEL = "PENDING_CANCEL"  # Currently unused
    REJECTED = "REJECTED"  # The order was not accepted by the engine and not processed
    EXPIRED = "EXPIRED"  # The order was canceled according to the order type's rules
    # (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill)
    # or by the exchange, (e.g. orders canceled during liquidation,
    # orders canceled during maintenance)


# Class of order sides
class OrderSide(Enum):
    BUY = "BUY"  # Buy a crypto
    SELL = "SELL"  # Sell a crypto


# Class of order types
class OrderType(Enum):
    # Limit Order allows you to place an order at a specific or a better price. A buy Limit Order will be filled if
    # the price matches or is lower than your limit price, and a sell Limit Order will be filled at or higher than
    # your limit price.
    LIMIT = "LIMIT"
    # Market orders are matched immediately at the best available price.
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


# Class of order time in force
# This sets how long an order will be active before expiration
class OrderTimeInForce(Enum):
    GTC = "GTC"  # Good Til Canceled (An order will be on the book unless the order is canceled)
    IOC = "IOC"  # Immediate Or Cancel (An order will try to fill the order as much as
    # it can before the order expires)
    FOK = "FOK"  # Fill or Kill (An order will expire if the full order cannot be filled upon execution)


# Class of order response types
class OrderResponseType(Enum):
    ACK = "ACK"
    RESULT = "RESULT"
    FULL = "FULL"
