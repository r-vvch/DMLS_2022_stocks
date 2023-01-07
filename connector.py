from binance.spot import Spot

from short import Short
from long import Long

TESTNET_URL = 'https://testnet.binance.vision'


# TODO: Может ли случиться что мы закупимся дважды? Соответственно двойные продажи?
# TODO: Всегда ли изолированный марджин?
# TODO: Протестировать работоспособность
class Connector:

    def __init__(self, api_key: str, api_secret: str, test_net: bool = True):
        if test_net:
            self._client = Spot(key=api_key, secret=api_secret,
                                base_url=TESTNET_URL)  # Creating a client in SandBox (TestNet)
            print(self._client.time()['serverTime'])
        else:
            self._client = Spot(key=api_key, secret=api_secret)  # Creating a client in real Stock Market
            print(self._client.time()['serverTime'])

    def account_status(self) -> str:
        return self._client.account()

    def create_long(self, symbol: str) -> Long:
        return Long(self._client, symbol)

    def create_short(self, symbol: str, is_isolated: bool = True) -> Short:
        return Short(self._client, symbol, is_isolated)


if __name__ == '__main__':
    keys_path = ''
    with open(keys_path, 'r') as file:
        keys = [str(item).rstrip() for item in file.readlines()]

    con = Connector(keys[0], keys[1], test_net=False)
    print(con.account_status())
    long = con.create_long("BNBBTC")
    print(long.ticker_price())
