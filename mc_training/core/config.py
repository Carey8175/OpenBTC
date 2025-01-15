import os
import json
from loguru import logger


class Key:
    """
    用于存储欧意API的key信息
    """

    apikey: str
    secretkey: str
    passphrase: str
    flag: str   # live: 0, simulated: 1

    def get_params(self):
        return [self.apikey, self.secretkey, self.passphrase, self.flag]


class Config:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STATICS_PATH = os.path.join(ROOT_PATH, 'statics')
    CONFIG_PATH = os.path.join(STATICS_PATH, 'config.json')
    INST_PATH = os.path.join(STATICS_PATH, 'instruments.json')
    STRATEGIES_CONFIG_PATH = os.path.join(STATICS_PATH, 'strategy_configs')
    CANDLES_PATH = os.path.join(STATICS_PATH, 'candles')
    CACHE_DIR = os.path.join(os.path.expanduser("~"), '.cache_data')  # 定义隐藏文件夹路径

    def __init__(self):
        self.key = Key()
        self.proxy = None
        self.webhook = None
        self.database = None
        # ---------------
        self.init_config()

    def get_key(self) -> Key:
        return self.key

    def set_key(self, key: Key) -> None:
        self.key = key

    def init_config(self) -> None:
        if not os.path.exists(self.CONFIG_PATH):
            logger.error('config.json not found at {}'.format(self.CONFIG_PATH))

            return

        with open(self.CONFIG_PATH, 'r') as f:
            config = json.load(f)

        self.database = config['database']


class Order:
    """
    用于定义委托单内容
    """
    instrument_id: str  # 合约ID
    td_mode: str    # 交易模式 isolated: 逐仓, cross: 全仓
    px: float   # 价格
    sz: float   # 数量
    side: str   # 买卖方向
    ord_type: str   # 委托类型 market: 市价, limit: 限价, post_only: 只做maker单, fok: 全部成交或立即取消, ioc: 立即成交并取消剩余
    # callback_ratio: [float, None]   # 回调比例
    attach_algo_ords: [dict, None]  # 附加算法单

    def __init__(self, instrument_id, sz, side, td_mode: str = 'isolated', px: [float, None] = None, ord_type='market',
                 attach_algo_ords: [dict, None] = None):
        self.instrument_id = instrument_id
        self.td_mode = td_mode
        self.px = px
        self.sz = sz
        self.side = side
        self.ord_type = ord_type
        self.attach_algo_ords = attach_algo_ords


if __name__ == '__main__':
    config = Config()
    print(config.key.get_params())

