import os.path
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from mc_training.dataset.indicators import Indicators
from mc_training.core.config import Config
from mc_training.core.clickhouse import CKClient


class MCDataLoader:
    def __init__(self):
        self.schema = ['ts', 'open', 'high', 'low', 'close', 'vol']
        self.data: {str: pd.DataFrame} = {}
        self.ck_client = CKClient()
        self.indicators = Indicators()

    def load_data_local(self, inst_id: str, start_date: datetime, end_date: datetime) -> bool:
        """
        从本地加载数据
        Args:
            inst_id: 交易对
            start_date: 开始日期
            end_date: 结束日期
        """
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        file_path = os.path.join(Config.CACHE_DIR, f"{inst_id}/{start_ts}-{end_ts}.csv")
        if not os.path.exists(file_path):
            return False

        data = pd.read_csv(file_path)
        data['ts'] = data['ts'].astype(np.int64)
        if inst_id not in self.data:
            self.data[inst_id] = pd.DataFrame(columns=self.schema)
        self.data[inst_id] = data
        logger.info(f"Data loaded for {inst_id} from {start_date} to {end_date}.")

        return True

    def load_data_clickhouse(
        self,
        inst_id: str,
        start_date: datetime = datetime(2000, 1, 1),
        end_date: datetime = datetime(2099, 12, 31),
    ):
        """
        从ClickHouse加载数据
        Args:
            inst_id: 交易对
            start_date: 开始日期
            end_date: 结束日期
        """
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        query = f"""
        SELECT ts, open, high, low, close, vol
        FROM candles
        WHERE inst_id = '{inst_id}' 
        AND ts >= {start_ts} 
        AND ts < {end_ts}
        ORDER BY ts
        """
        try:
            data = self.ck_client.query_dataframe(query)
            if inst_id not in self.data:
                self.data[inst_id] = pd.DataFrame(columns=self.schema)
            self.data[inst_id] = data
            logger.info(f"Data loaded for {inst_id} from {start_date} to {end_date}.")
            save_path = os.path.join(Config.CACHE_DIR, f"{inst_id}/{start_ts}-{end_ts}.csv")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            data.to_csv(save_path, index=False)

        except Exception as e:
            raise RuntimeError(f"Failed to load data for {inst_id}: {e}")

    def validate_daily_data(self, inst_id: str):
        """
        校验每天的数据是否是1440条
        Args:
            inst_id: 交易对
        """
        if inst_id not in self.data or self.data[inst_id].empty:
            print(f"No data loaded for {inst_id}.")
            return

        # Convert timestamp to East 8 timezone date for grouping
        self.data[inst_id]['date'] = pd.to_datetime(
            self.data[inst_id]['ts'], unit='ms', utc=True
        ).dt.tz_convert('Asia/Shanghai').dt.date

        # Group by date and count entries per day
        daily_counts = self.data[inst_id].groupby('date').size()

        # Check for days with missing data
        invalid_days = daily_counts[daily_counts != 1440]
        if not invalid_days.empty:
            print("The following days have incorrect data counts:")
            print(invalid_days)
        else:
            print("All days have exactly 1440 records.")

        # Remove the 'date' column after validation
        self.data[inst_id].drop(columns=['date'], inplace=True)

    def get_data(self, inst_id: str) -> pd.DataFrame:
        """
        获取加载的数据
        Args:
            inst_id: 交易对
        Returns:
            pd.DataFrame: 加载的数据
        """
        return self.data.get(inst_id, pd.DataFrame(columns=self.schema))

    def truncate_data(self, inst_id):
        """
        从内存中删除数据
        Args:
            inst_id: 交易对
        """
        if inst_id in self.data:
            del self.data[inst_id]

    def normalize_data(self, inst_id: str, method: str = 'z-score'):
        """
        对指定交易对的数据进行标准化处理。
        Args:
            inst_id: 交易对
            method: 标准化方法，支持 'z-score' 或 'min-max'。
        """
        if inst_id not in self.data or self.data[inst_id].empty:
            raise ValueError(f"No data available for {inst_id}. Please load data first.")

        df = self.data[inst_id]

        # 不处理 `ts` 列
        columns_to_normalize = df.columns.difference(['ts', 'datetime'])

        if method == 'z-score':
            # Z-score 标准化
            df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[
                columns_to_normalize].std()
        elif method == 'min-max':
            # Min-Max 标准化
            df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (
                    df[columns_to_normalize].max() - df[columns_to_normalize].min())
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        # 更新标准化后的数据
        self.data[inst_id] = df

    def add_indicators(self, inst_id: str, indicator_list: list = None, btc_df: pd.DataFrame = None):
        """
        为指定交易对数据添加指标
        Args:
            inst_id: 交易对名称
            indicator_list: 需要计算的指标列表，例如 ['rsi', 'macd', 'mfi']
            btc_df: 比特币数据框（可选，用于计算 BTC 相关指标）
        """
        if inst_id not in self.data or self.data[inst_id].empty:
            raise ValueError(f"No data available for {inst_id}. Please load data first.")

        df = self.data[inst_id].copy()

        # 如果未指定指标列表，默认计算所有指标
        if indicator_list is None:
            indicator_list = [method for method in dir(self.indicators) if
                              callable(getattr(self.indicators, method)) and not method.startswith("_")]

        # 遍历指定指标并计算
        for indicator in indicator_list:
            try:
                # (f"Appling indicator: {indicator}")
                func = getattr(self.indicators, indicator)
                if indicator in {'btc_corr', 'btc_price_change'} and btc_df is not None:
                    df = func(df, btc_df)  # 传入 btc_df
                else:
                    df = func(df)  # 不需要 btc_df 的情况
            except Exception as e:
                print(f"Error applying {indicator}: {e}")
                pass

        # 删除包含 NaN 的行
        df.dropna(inplace=True)
        # 更新主数据结构
        self.data[inst_id] = df
        logger.info(f"Indicators added to {inst_id}. Total columns: {len(df.columns)}")

    def load_data(self,
                  inst_id: str,
                  start_date: datetime = datetime(2000, 1, 1),
                  end_date: datetime = datetime(2099, 12, 31),
                  add_indicators: bool = True,
                  add_delta: bool = True):
        """
        从 ClickHouse 加载数据
        Args:
            inst_id: 交易对
            start_date: 开始日期
            end_date: 结束日期
            add_indicators: 是否添加指标
        """
        if not self.load_data_local(inst_id, start_date, end_date):
            logger.info(f"Data not found locally for {inst_id}. Loading from ClickHouse.")
            self.load_data_clickhouse(inst_id, start_date, end_date)

        if add_indicators:
            self.add_indicators(inst_id)

        if add_delta:
            self.add_delta(inst_id)

    def add_delta(self, inst_id: str, periods=None):
        """
        高效计算每个特征列与之前多个时间窗口的平均值的差值
        Args:
            inst_id: 交易对名称
            periods: 计算差分的周期列表（如 [2, 5, 10, 20, 30]）
        """
        if inst_id not in self.data or self.data[inst_id].empty:
            raise ValueError(f"No data available for {inst_id}. Please load data first.")

        if periods is None:
            periods = [2, 5, 10, 20, 30]

        df = self.data[inst_id]

        # 忽略时间戳列，只计算数值列的差值
        numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(['ts'])

        # 使用 NumPy 和 Pandas 的矢量化计算批量处理
        for period in periods:
            # 滚动平均计算 (每个 period 一次性处理所有列)
            rolling_mean = df[numeric_columns].rolling(window=period).mean()
            delta = df[numeric_columns] - rolling_mean
            delta.columns = [f"{col}_delta{period}" for col in numeric_columns]

            # 合并到主 DataFrame
            df = pd.concat([df, delta], axis=1)

        # 删除包含 NaN 的行（由于 rolling 导致的缺失值）
        df.dropna(inplace=True)

        # 更新主数据结构
        self.data[inst_id] = df
        logger.info(f"Deltas added for {inst_id}. Total columns: {len(df.columns)}")


if __name__ == '__main__':
    dl = MCDataLoader()
    # Load data for BTC-USDT-SWAP from database
    time1 = time.time()
    dl.load_data('BTC-USDT-SWAP', datetime(2024, 1, 2), datetime(2025, 1, 4))
    # dl.add_indicators('BTC-USDT-SWAP')
    print(f"Time taken: {time.time() - time1:.2f}s")

    # Validate daily data counts
    dl.validate_daily_data('BTC-USDT-SWAP')

    # Print the data
    print(dl.get_data('BTC-USDT-SWAP').head())
