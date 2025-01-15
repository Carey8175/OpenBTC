import clickhouse_driver
from datetime import datetime

from mc_training.core.config import Config


class CKClient:
    def __init__(self):
        self.config = Config()
        self.client = clickhouse_driver.Client(
            host=self.config.database['host'],
            port=self.config.database['port'],
            user=self.config.database['user'],
            password=self.config.database['password'],
            database=self.config.database['database']
        )

    def execute(self, query):
        return self.client.execute(query)

    def insert(self, table, data, columns=None):
        if columns is None:
            columns = []

        col_str = "(" + ", ".join(columns) + ")" if columns else ""
        sql = f"INSERT INTO {table} {col_str} VALUES"

        self.client.execute(sql, data)

    def close(self):
        self.client.disconnect()

    def merge_table(self, table):
        sql = f"OPTIMIZE TABLE {table} FINAL;"
        self.execute(sql)

    def has_data_for_date(self, inst_id, date: datetime):
        """
        校验数据库是否已经有指定日期的数据
        :param inst_id: 合约ID
        :param date: 日期
        :return: True 表示有数据，False 表示无数据
        """
        start_ts = date.timestamp() * 1000
        end_ts = start_ts + 24 * 60 * 60 * 1000
        query = f"""
        SELECT COUNT(*) 
        FROM mc.candles 
        WHERE inst_id = '{inst_id}' 
        AND ts >= {start_ts} 
        AND ts < {end_ts}
        """
        count = self.execute(query)
        count = 0 if not count else count[0][0]
        return count >= 1440  # 1分钟K线，一天应有 1440 条记录

    def query_dataframe(self, query):
        return self.client.query_dataframe(query)


if __name__ == '__main__':
    ck = CKClient()
    ck.has_data_for_date('BTC-USDT-SWAP', datetime(2024, 3, 21))
    ck.close()