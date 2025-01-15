import numpy as np
import pandas as pd


class Indicators:
    def rsi(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 RSI 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 RSI 的周期

        Returns:
            pd.DataFrame: 包含 RSI 指标的数据框
        """
        # 计算价格变化
        df['delta'] = df['close'] - df['close'].shift(1)
        # 计算正数价格变化
        df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
        # 计算负数价格变化
        df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)
        # 计算平均增益和平均损失
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        # 计算 RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df[f'rsi_{period}'] = 100 - (100 / (1 + df['rs']))

        # 删除中间变量
        df.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)

        return df

    def macd(self, df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30, signal_period: int = 20) -> pd.DataFrame:
        """
        计算 MACD 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期

        Returns:
            pd.DataFrame: 包含 MACD 指标的数据框
        """
        # 计算快线
        df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period).mean()
        # 计算慢线
        df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period).mean()
        # 计算 MACD
        df['macd'] = df[f'ema_{fast_period}'] - df[f'ema_{slow_period}']
        # 计算信号线
        df['signal'] = df['macd'].ewm(span=signal_period).mean()
        # 计算 MACD 柱状图
        df['histogram'] = df['macd'] - df['signal']

        return df

    def mfi(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 MFI 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 MFI 的周期

        Returns:
            pd.DataFrame: 包含 MFI 指标的数据框
        """
        # 计算典型价格
        # 计算典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算资金流量（价格 * 交易量）
        df['money_flow'] = df['typical_price'] * df['vol']

        # 计算资金流入和流出
        df['positive_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
        df['negative_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)

        # 计算资金流入总和和资金流出总和
        df['positive_flow_sum'] = df['positive_flow'].rolling(window=period).sum()
        df['negative_flow_sum'] = df['negative_flow'].rolling(window=period).sum()

        # 计算资金流向比率
        df['money_flow_ratio'] = df['positive_flow_sum'] / df['negative_flow_sum']

        # 计算 MFI
        df[f'mfi_{period}'] = 100 - (100 / (1 + df['money_flow_ratio']))

        # 删除中间变量
        df.drop(columns=['typical_price', 'money_flow', 'positive_flow', 'negative_flow', 'positive_flow_sum',
                         'negative_flow_sum', 'money_flow_ratio'], inplace=True)

        return df

    def dea(self, df: pd.DataFrame, short_period: int = 10, long_period: int = 30, mid_period: int = 20) -> pd.DataFrame:
        """
        计算 DEA 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            short_period: 短周期
            long_period: 长周期
            mid_period: 中周期

        Returns:
            pd.DataFrame: 包含 DEA 指标的数据框
        """
        tmp_df = pd.DataFrame()
        # 计算快线
        tmp_df[f'ema_{short_period}'] = df['close'].ewm(span=short_period).mean()
        # 计算慢线
        tmp_df[f'ema_{long_period}'] = df['close'].ewm(span=long_period).mean()
        # 计算 MACD
        tmp_df['macd'] = tmp_df[f'ema_{short_period}'] - tmp_df[f'ema_{long_period}']
        # 计算信号线
        df['dea'] = tmp_df['macd'].ewm(span=mid_period).mean()

        return df

    def dif(self, df: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> pd.DataFrame:
        """
        计算 DIF 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            short_period: 短周期
            long_period: 长周期

        Returns:
            pd.DataFrame: 包含 DIF 指标的数据框
        """
        tmp_df = pd.DataFrame()

        # 计算快线
        tmp_df[f'ema_{short_period}'] = df['close'].ewm(span=short_period).mean()
        # 计算慢线
        tmp_df[f'ema_{long_period}'] = df['close'].ewm(span=long_period).mean()
        # 计算 DIF
        df['dif'] = tmp_df[f'ema_{short_period}'] - tmp_df[f'ema_{long_period}']

        return df

    def psy(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 PSY 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 PSY 的周期

        Returns:
            pd.DataFrame: 包含 PSY 指标的数据框
        """
        tmp_df = pd.DataFrame()

        # 计算价格变化
        tmp_df['delta'] = df['close'] - df['close'].shift(1)
        # 计算正数价格变化
        tmp_df['gain'] = (tmp_df['delta'] > 0).astype(int)
        # 计算 PSY
        df['psy'] = tmp_df['gain'].rolling(window=period).sum() / period * 100

        return df

    def bias(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 BIAS 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 BIAS 的周期

        Returns:
            pd.DataFrame: 包含 BIAS 指标的数据框
        """
        tmp_df = pd.DataFrame()

        # 计算均线
        tmp_df['ma'] = df['close'].rolling(window=period).mean()
        # 计算 BIAS
        df['bias'] = (df['close'] - tmp_df['ma']) / tmp_df['ma'] * 100

        return df

    def sma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 SMA 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 SMA 的周期

        Returns:
            pd.DataFrame: 包含 SMA 指标的数据框
        """
        df['sma'] = df['close'].rolling(window=period).mean()

        return df

    def tp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 TP 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 TP 指标的数据框
        """
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def tp_max(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 TP_MAX 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 TP_MAX 的周期

        Returns:
            pd.DataFrame: 包含 TP_MAX 指标的数据框
        """
        df['tp_max'] = df['tp'].rolling(window=period).max()

        return df

    def tp_min(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 TP_MIN 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 TP_MIN 的周期

        Returns:
            pd.DataFrame: 包含 TP_MIN 指标的数据框
        """
        df['tp_min'] = df['tp'].rolling(window=period).min()

        return df

    def skewness(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Skewness 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Skewness 的周期

        Returns:
            pd.DataFrame: 包含 Skewness 指标的数据框
        """
        df['skewness'] = df['close'].rolling(window=period).skew()

        return df

    def avg_low_shadow_length(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Low Shadow Length 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Low Shadow Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg Low Shadow Length 指标的数据框
        """
        df['low_shadow_length'] = abs(df['low'] - df[['open', 'close']].min(axis=1))
        df['avg_low_shadow_length'] = df['low_shadow_length'].rolling(window=period).mean()

        return df

    def avg_high_shadow_length(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg High Shadow Length 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg High Shadow Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg High Shadow Length 指标的数据框
        """
        df['high_shadow_length'] = abs(df['high'] - df[['open', 'close']].max(axis=1))
        df['avg_high_shadow_length'] = df['high_shadow_length'].rolling(window=period).mean()

        return df

    # def upper_shadow_exceeds_avg_multiple(self, df: pd.DataFrame, period: int = 20, multiple: int = 2) -> pd.DataFrame:
    #     """
    #     计算 Upper Shadow Exceeds Avg Multiple 指标
    #     在时间窗口内，超过上影线均值两倍的次数
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计算 Upper Shadow Exceeds Avg Multiple 的周期
    #         multiple: 上影线超过平均值的倍数
    #
    #     Returns:
    #         pd.DataFrame: 包含 Upper Shadow Exceeds Avg Multiple 指标的数据框
    #     """
    #     if 'avg_high_shadow_length' not in df.columns:
    #         df['high_shadow_length'] = abs(df['high'] - df[['open', 'close']].max(axis=1))
    #         df['avg_high_shadow_length'] = df['high_shadow_length'].rolling(window=period).mean()
    #
    #     # 当前为均值的多少倍
    #     df['upper_shadow_exceeds_avg_multiple'] = df['high_shadow_length'] / df['avg_high_shadow_length']
    #
    #     # 统计时间窗口内超过平均值两倍的次数
    #     df['upper_shadow_exceeds_avg_multiple_20'] = df['upper_shadow_exceeds_avg_multiple'].rolling(
    #         window=period).apply(
    #         lambda x: len(x[x >= multiple])
    #     )
    #
    #     return df

    # def lower_shadow_exceeds_avg_multiple(self, df: pd.DataFrame, period: int = 20, multiple: int = 2) -> pd.DataFrame:
    #     """
    #     计算 Lower Shadow Exceeds Avg Multiple 指标
    #     在时间窗口内，超过下影线���值两倍的次数
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计算 Lower Shadow Exceeds Avg Multiple 的周期
    #         multiple: 下影线超过平均值的倍数
    #
    #     Returns:
    #         pd.DataFrame: 包含 Lower Shadow Exceeds Avg Multiple 指标的数据框
    #     """
    #     if 'avg_low_shadow_length' not in df.columns:
    #         df['low_shadow_length'] = abs(df['low'] - df[['open', 'close']].min(axis=1))
    #         df['avg_low_shadow_length'] = df['low_shadow_length'].rolling(window=period).mean()
    #
    #     # 当前为均值的多少倍
    #     df['lower_shadow_exceeds_avg_multiple'] = df['low_shadow_length'] / df['avg_low_shadow_length']
    #
    #     # 统计时间窗口内超过平均值两倍的次数
    #     df['lower_shadow_exceeds_avg_multiple_20'] = df['lower_shadow_exceeds_avg_multiple'].rolling(
    #         window=period).apply(
    #         lambda x: len(x[x >= multiple])
    #     )
    #
    #     return df

    def lower_shadow_to_candle_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Lower Shadow To Candle Ratio 指标
        下影线与蜡烛实体的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Lower Shadow To Candle Ratio 指标的数据框
        """
        if 'low_shadow_length' not in df.columns:
            df['low_shadow_length'] = abs(df['low'] - df[['open', 'close']].min(axis=1))

        # 计算蜡烛实体长度
        df['candle_body_length'] = df['high'] - df['low']

        # 防止分母为 0 的情况，如果蜡烛实体为 0，则下影线比率为 0
        df['lower_shadow_to_candle_ratio'] = df['low_shadow_length'] / df['candle_body_length']
        df.loc[df['candle_body_length'] == 0, 'lower_shadow_to_candle_ratio'] = 0

        return df

    def upper_shadow_to_candle_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Upper Shadow To Candle Ratio 指标
        上影线与蜡烛实体的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Upper Shadow To Candle Ratio 指标的数据框
        """
        if 'high_shadow_length' not in df.columns:
            df['high_shadow_length'] = abs(df['high'] - df[['open', 'close']].max(axis=1))

        # 计算蜡烛实体长度
        df['candle_body_length'] = df['high'] - df['low']

        # 防止分母为 0 的情况，如果蜡烛实体为 0，则下影线比率为 0
        df['upper_shadow_to_candle_ratio'] = df['high_shadow_length'] / df['candle_body_length']
        df.loc[df['candle_body_length'] == 0, 'upper_shadow_to_candle_ratio'] = 0

        return df


    def avg_body_length(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Candle Length 指标
        平均蜡烛长度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Candle Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg Candle Length 指标的数据框
        """
        df['body_length'] = abs(df['open'] - df['close'])
        df['avg_body_length'] = df['body_length'].rolling(window=period).mean()

        return df

    def body_length_to_avg_ratio(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """
        计算 Candle Length To Avg Ratio 指标
        蜡烛长度与平均长度的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Candle Length To Avg Ratio 指标的数据框
        """
        if 'avg_body_length' not in df.columns:
            df['body_length'] = abs(df['open'] - df['close'])
            df['avg_body_length'] = df['body_length'].rolling(window=period).mean()

        df['body_length_to_avg_ratio'] = df['body_length'] / df['avg_body_length']

        return df

    def avg_high(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg High 指标
        平均最高价
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg High 的周期

        Returns:
            pd.DataFrame: 包含 Avg High 指标的数据框
        """
        df['avg_high'] = df['high'].rolling(window=period).mean()

        return df

    def avg_low(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Low 指标
        平均最低价
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Low 的周期

        Returns:
            pd.DataFrame: 包含 Avg Low 指标的数据框
        """
        df['avg_low'] = df['low'].rolling(window=period).mean()

        return df

    def slope_close(self, df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
        """
        计算 Slope Close 指标
        收盘价的斜率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Slope Close 的周期

        Returns:
            pd.DataFrame: 包��� Slope Close 指标的数据框
        """
        df[f'slope_close_{period}'] = df['close'].diff(period) / period

        return df

    def trend_continuation_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Trend Continuation Check 指标
        趋势延续检查
        是否延续之前的趋势，也就是说如果前一根k线图为跌，此K线图HIGH并不高于前一根的收盘价。若为涨则low小于前一根K线的收盘价。
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Trend Continuation Check 的周期

        Returns:
            pd.DataFrame: 包含 Trend Continuation Check 指标的数据框
        """
        df['trend_continuation_check'] = np.where(
            df['open'].shift(1) > df['close'].shift(1),  # 前一根K线为涨
            np.where(df['low'] >= df['open'].shift(1), 1, 0),  # 当前K线的low不低于前一根K线的open
            np.where(df['high'] <= df['close'].shift(1), 1, 0)  # 前一根K线为跌，当前K线的high不高于前一根K线的close
        )

        return df
    def momentum(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Momentum 指标
        动量指标
        用于测量价格变化的速度和幅度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Momentum 的周期

        Returns:
            pd.DataFrame: 包含 Momentum 指标的数据框
        """
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        return df

    def ln_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Ln Price 指标
        收盘价的自然对数
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Ln Price 指标的数据框
        """
        df['ln_price'] = np.log(df['close'])

        return df

    # def support(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    #     """
    #     支撑位 20 交易量softmax后的加权最低价
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计算支撑位的周期
    #
    #     Returns:
    #         pd.DataFrame: 包含支撑位的数据框
    #     """
    #     # 计算Softmax后的交易量权重
    #     def softmax(x):
    #         e_x = np.exp(x - np.max(x))  # 防止溢出
    #         return e_x / np.sum(e_x)  # 返回归一化后的值
    #
    #     # 使用rolling来计算每个窗口的加权最低价
    #     df['support'] = df['vol'].rolling(window=period).apply(
    #         lambda x: np.sum(softmax(np.array(x)) * df['low'].iloc[x.index]), raw=False
    #     )
    #
    #     return df

    # def resistance(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    #     """
    #     阻力位 20 交易量softmax后的加权最高价
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计���阻力位的周期
    #
    #     Returns:
    #         pd.DataFrame: 包含阻力位的数据框
    #     """
    #     # 计算Softmax后的交易量权重
    #     def softmax(x):
    #         e_x = np.exp(x - np.max(x))
    #         return e_x / np.sum(e_x)
    #
    #     # 使用rolling来计算每个窗口的加权最高价
    #     df['resistance'] = df['vol'].rolling(window=period).apply(
    #         lambda x: np.sum(softmax(np.array(x)) * df['high'].iloc[x.index]), raw=False
    #     )
    #
    #     return df

    def bollinger_band_expansion_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Bollinger Band Expansion Ratio 指标
        布林带扩张比率
        ��林带宽度与均线的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Bollinger Band Expansion Ratio 的周期

        Returns:
            pd.DataFrame: 包含 Bollinger Band Expansion Ratio 指标的数据框
        """
        # 计算中轨（Middle Band）
        df['middle_band'] = df['close'].rolling(window=period).mean()

        # 计算标准差
        df['std'] = df['close'].rolling(window=period).std()

        # 计算上轨和下轨
        df['upper_band'] = df['middle_band'] + 2 * df['std']
        df['lower_band'] = df['middle_band'] - 2 * df['std']

        # 计算布林带宽度
        df['bollinger_band_width'] = df['upper_band'] - df['lower_band']

        # 计算布林带扩张比率
        df['bollinger_band_expansion_ratio'] = df['bollinger_band_width'] / df['middle_band']

        return df

    # def three_consecutive_up_candles_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    #     """
    #     计算 Three Consecutive Up Candles Ratio 指标
    #     大于三连阳比率
    #     连续三根阳线的比率
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计算 Three Consecutive Up Candles Ratio 的周期
    #
    #     Returns:
    #         pd.DataFrame: 包含 Three Consecutive Up Candles Ratio 指标的数据框
    #     """
    #     df['up'] = df['close'] > df['open']
    #
    #     # 计算连续3根及以上阳线的比率
    #     def consecutive_down_candles_ratio(x):
    #         x = x.values  # 转换为numpy数组
    #         total_candles = 0  # 构成连续3根及以上阳线的K线总数
    #         consecutive = 0  # 当前连续阳线计数
    #         temp_count = 0  # 临时计数，用于累积可能构成连续阳线的数量
    #
    #         for i in range(len(x)):
    #             if x[i] == True:
    #                 consecutive += 1
    #                 temp_count += 1
    #             else:
    #                 if consecutive >= 3:  # 如果之前累积了3根及以上
    #                     total_candles += temp_count  # 将所有连续的K线都计入
    #                 consecutive = 0
    #                 temp_count = 0
    #
    #         # 处理最后一组（如果以连续阴线结尾）
    #         if consecutive >= 3:
    #             total_candles += temp_count
    #
    #         return total_candles / len(x)  # 返回占比
    #
    #     df['three_consecutive_up_candles_ratio'] = df['up'].rolling(window=period).apply(
    #         consecutive_down_candles_ratio, raw=False)
    #
    #     return df
    #
    # def three_consecutive_down_candles_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    #     """
    #     计算 Three Consecutive Down Candles Ratio 指标
    #     大于三连阴比率
    #     连续三根阴线的比率
    #     Args:
    #         df: 数据框，包含以下列：
    #             - ts: 时间戳
    #             - open: 开盘价
    #             - high: 最高价
    #             - low: 最低价
    #             - close: 收盘价
    #             - vol: 交易量
    #             - volCcy: 交易币量
    #             - valCcyQuote: 计价货币的量
    #         period: 计算 Three Consecutive Down Candles Ratio 的周期
    #
    #     Returns:
    #         pd.DataFrame: 包含 Three Consecutive Down Candles Ratio 指标的数据框
    #     """
    #     df['down'] = df['close'] < df['open']
    #
    #     # 计算连续3根及以上阴线的比率
    #     def consecutive_down_candles_ratio(x):
    #         x = x.values  # 转换为numpy数组
    #         total_candles = 0  # 构成连续3根及以上阴线的K线总数
    #         consecutive = 0  # 当前连续阴线计数
    #         temp_count = 0  # 临时计数，用于累积可能构成连续阴线的数量
    #
    #         for i in range(len(x)):
    #             if x[i] == True:
    #                 consecutive += 1
    #                 temp_count += 1
    #             else:
    #                 if consecutive >= 3:  # 如果之前累积了3根及以上
    #                     total_candles += temp_count  # 将所有连续的K线都计入
    #                 consecutive = 0
    #                 temp_count = 0
    #
    #         # 处理最后一组（如果以连续阴线结尾）
    #         if consecutive >= 3:
    #             total_candles += temp_count
    #
    #         return total_candles / len(x)  # 返回占比
    #
    #     df['three_consecutive_down_candles_ratio'] = df['down'].rolling(window=period).apply(
    #         consecutive_down_candles_ratio, raw=False)
    #
    #     return df

    def gap_percentage(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """
        计算 Gap Percentage 指标
        缺口百分比
        今日开盘价与昨日收盘价的差值除以昨日收盘价的绝对值
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Gap Percentage 的周期

        Returns:
            pd.DataFrame: 包含 Gap Percentage 指标的数据框
        """
        df['gap_percentage'] = (df['open'] - df['close'].shift(period)) / df['close'].shift(period)

        return df

    # 波动指数 --------------------------------------------------------------

    def tr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 TR 指标
        True Range 真实波动幅度
        TR = max(high - low, abs(high - close), abs(low - close))
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 TR 指标的数据框
        """
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        return df

    def atr(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 ATR 指标
        Average True Range 平均真实波动幅度
        ATR = EMA(TR, period)
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 ATR 的周期

        Returns:
            pd.DataFrame: 包含 ATR 指标的数据框
        """
        # 计算 TR
        if 'tr' not in df.columns:
            df = self.tr(df)

        # 计算 ATR
        df['atr'] = df['tr'].ewm(span=period).mean()

        return df

    def tr_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 TR PCT 指标
        True Range 真实波动幅度的百分比
        TR PCT = TR / close
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 TR PCT 指标的数据框
        """
        # 计算 TR PCT
        if 'tr' not in df.columns:
            df = self.tr(df)

        df['tr_pct'] = df['tr'] / df['close']

        return df

    def atr_pct(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 ATR PCT 指标
        Average True Range 平均真实波动幅度的百分比
        ATR PCT = ATR / close
        ATR = EMA(TR, period)
        TR = max(high - low, abs(high - close), abs(low - close)
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 ATR PCT 的周期

        Returns:
            pd.DataFrame: 包含 ATR PCT 指标的数据框
        """
        # 计算 ATR
        if 'atr' not in df.columns:
            df = self.atr(df, period)

        # 计算 ATR PCT
        df['atr_pct'] = df['atr'] / df['close']

        return df

    def historical_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Historical Volatility 指标
        历史波动率
        log为自然对数
        std为标准差
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Historical Volatility 的周期

        Returns:
            pd.DataFrame: 包含 Historical Volatility 指标的数据框
        """
        # 计算历史波动率
        df['historical_volatility'] = np.log(df['close'] / df['close'].shift(1)).rolling(window=period).std()

        return df

    # 交易量指标 --------------------------------------------------------------
    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 OBV 指标
        On Balance Volume 能量潮指标
        通过比较当日收盘价与前一日收盘价的大小，来判断成交量是成交量是积极的还是消极的
        如果当日收盘价高于前一日收盘价，则认为当日成交量是积极的，将当日成交量累加到OBV上
        如果当日收盘价低于前一日收盘价，则认为当日成交量是消极的，将当日成交量减去到OBV上
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 OBV 指标的数据框
        """
        # 计算 OBV
        df['obv'] = np.where(
            df['close'] > df['close'].shift(1),
            df['vol'],
            -df['vol']
        ).cumsum()

        return df

    def obv_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 OBV MA 指标
        OBV 移动平均线
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 OBV MA 的周期

        Returns:
            pd.DataFrame: 包含 OBV MA 指标的数据框
        """
        # 计算 OBV
        if 'obv' not in df.columns:
            df = self.obv(df)

        # 计算 OBV MA
        df['obv_ma'] = df['obv'].rolling(window=period).mean()

        return df

    def obv_ema(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 OBV EMA 指标
        OBV 指数移动平均线
        OBV EMA = EMA(OBV, period)
        OBV = sum(Volume * sign(close - close(-1)))
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 OBV EMA 的周期

        Returns:
            pd.DataFrame: 包含 OBV EMA 指标的数据框
        """
        # 计算 OBV
        if 'obv' not in df.columns:
            df = self.obv(df)

        # 计算 OBV EMA
        df[f'obv_ema_{period}'] = df['obv'].ewm(span=period).mean()

        return df

    def obv_macd(self, df: pd.DataFrame, period_short: int = 10, period_long: int = 30, period_signal: int = 20) -> pd.DataFrame:
        """
        计算 OBV MACD 指标
        OBV MACD 指数平滑异同移动平均线
        OBV MACD = EMA(OBV, period_short) - EMA(OBV, period_long)
        Signal = EMA(OBV MACD, period_signal)
        Histogram = OBV MACD - Signal
        OBV = sum(Volume * sign(close - close(-1)))
        OBV EMA = EMA(OBV, period)
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
                period_short: 计算 OBV MACD 的短周期
                period_long: 计算 OBV MACD 的长周期
                period_signal: 计算 OBV MACD 的 Signal 的周期

        Returns:
            pd.DataFrame: 包含 OBV MACD 指标的数据框
        """
        # 计算 OBV
        if 'obv' not in df.columns:
            df = self.obv(df)

        # 计算 OBV MACD
        df[f'obv_macd_{period_short}_{period_long}'] = df['obv'].ewm(span=period_short).mean() - df['obv'].ewm(span=period_long).mean()

        # 计算 Signal
        df[f'obv_macd_signal_{period_signal}'] = df[f'obv_macd_{period_short}_{period_long}'].ewm(span=period_signal).mean()

        # 计算 Histogram
        df[f'obv_macd_histogram_{period_short}_{period_long}_{period_signal}'] = df[f'obv_macd_{period_short}_{period_long}'] - df[f'obv_macd_signal_{period_signal}']

        return df

    def avg_vol(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Vol 指标
        平均交易量
        20日平均交易量
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Vol 的周期

        Returns:
            pd.DataFrame: 包含 Avg Vol 指标的数据框
        """
        df[f'avg_vol_{period}'] = df['vol'].rolling(window=period).mean()

        return df

    def avg_vol_pct(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Vol PCT ���标
        交易量的百分比
        交易量除以20日平均交易量
        Avg Vol PCT = vol / avg_vol
        20日平均交易量
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Vol PCT 的周期

        Returns:
            pd.DataFrame: 包含 Avg Vol PCT 指标的数据框
        """
        # 计算 Avg Vol
        if f'avg_vol_{period}' not in df.columns:
            df = self.avg_vol(df, period)

        # 计算 Avg Vol PCT
        df[f'avg_vol_pct_{period}'] = df['vol'] / df[f'avg_vol_{period}']

        return df

    def vol_pct(self, df: pd.DataFrame, period=1) -> pd.DataFrame:
        """
        计算 Vol PCT 指标
        交易量的百分比
        交易量除以前一日的交易量
        Vol PCT = vol / vol.shift(1)
        20日平均交易量
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Vol PCT 的周期

        Returns:
            pd.DataFrame: 包含 Vol PCT 指标的数据框
        """
        df['vol_pct'] = df['vol'] / df['vol'].shift(period)

        return df

    def vwap(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 VWAP 指标
        Volume Weighted Average Price 成交量加权平均价格
        VWAP = sum(close * vol) / sum(vol)
        20日VWAP
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 VWAP 的周期

        Returns:
            pd.DataFrame: 包含 VWAP 指标的数据框
        """
        # 计算 VWAP
        df['vwap'] = (df['close'] * df['vol']).rolling(window=period).sum() / df['vol'].rolling(window=period).sum()

        return df

    def vol_skewness(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Vol Skewness 指标
        交易量���度
        20日交易量偏度
        Vol Skewness = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
        20日平均交易量
        20日交易量偏度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Vol Skewness 的周期

        Returns:
            pd.DataFrame: 包含 Vol Skewness 指标的数据框
        """
        # 计算 Vol Skewness
        df['vol_skewness'] = (df['vol'] - df['vol'].rolling(window=period).mean()) / df['vol'].rolling(window=period).std()

        return df

    def vol_kurtosis(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Vol Kurtosis 指标
        交易量峰度
        20日交易量峰度
        Vol Kurtosis = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
        20日平均交易量
        20日交易量峰度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Vol Kurtosis 的周期

        Returns:
            pd.DataFrame: 包含 Vol Kurtosis 指标的数据框
        """
        # 计算 Vol Kurtosis
        df['vol_kurtosis'] = (df['vol'] - df['vol'].rolling(window=period).mean()) / df['vol'].rolling(window=period).std()

        return df

    def capital_flow_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Capital Flow Ratio 指标
        资金流向比率
        20日资金流向比率
        Capital Flow Ratio = (close - open) / (high - low)
        20日平均资金流向比率
        20日资金流向比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Capital Flow Ratio 的周期

        Returns:
            pd.DataFrame: 包含 Capital Flow Ratio 指标的数据框
        """
        # 计算资金流向
        df['capital_flow'] = np.where(df['close'] > df['open'],
                                      (df['high'] + df['low'] + df['close']) * df['vol'],
                                      -(df['high'] + df['low'] + df['close']) * df['vol'])

        # 计算资金流入/流出比率
        df['capital_flow_ratio'] = df['capital_flow'].rolling(window=period).sum() / df['vol'].rolling(
            window=period).sum()

        return df

    def volume_per_move(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Volume Per Move 指标
        每次价格变动的交易量
        20日每次价格变动的交易量
        Volume Per Move = vol / (high - low)
        20日平均每次价格变动的交易量
        20日每次价格变动的交易量
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Volume Per Move 的周期

        Returns:
            pd.DataFrame: 包含 Volume Per Move 指标的数据框
        """
        # 计算 Volume Per Move
        df['volume_per_move'] = df['vol'] / (df['high'] - df['low'])
        df.loc[df['high'] == df['low'], 'volume_per_move'] = 0

        return df

    def is_daylight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Is Daylight 指标
        是否为白天
        交易时间为白天（9:00 - 15:00）为1，否则为0
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Is Daylight 指标的数据框
        """
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        # 时区
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

        # 交易时间为白天（8:00 - 20:00）为1，否则为0
        df['is_daylight'] = np.where(
            (df['datetime'].dt.hour >= 8) & (df['datetime'].dt.hour < 20),
            1,
            0
        )

        return df

    def btc_corr(self, df: pd.DataFrame, btc_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 BTC Corr 指标
        比特币相关性
        与比特币的相关性
        BTC Corr = corr(close, btc_close)
        20日比特币相关性
        20日BTC Corr = corr(close, btc_close)
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            btc_df: 比特币数据框，包含以下列：
                - ts: 时间戳
                - close: 收盘价
            period: 计算 BTC Corr 的周期

        Returns:
            pd.DataFrame: 包含 BTC Corr 指标的数据框
        """
        # 合并 df 和 btc_df，确保它们在同一个 DataFrame 中
        merged_df = pd.merge(df[['ts', 'close']], btc_df[['ts', 'close']], on='ts', suffixes=('', '_btc'))

        # 计算 BTC Corr
        merged_df['btc_corr'] = merged_df['close'].rolling(window=period).corr(merged_df['close_btc'])
        merged_df.drop(['close', 'close_btc'], axis=1, inplace=True)

        # 将结果返回到原始 df, 确保ts相同
        df = pd.merge(df, merged_df[['ts', 'btc_corr']], on='ts', how='left')

        return df

    def btc_price_change(self, df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 BTC Price Change 指标
        比特币价格变化
        与比特币的价格变化
        BTC Price Change = (close - close.shift(1)) / close.shift(1)
        20日比特币价格变化
        20日BTC Price Change = (close - close.shift(1)) / close.shift(1)
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            btc_df: 比特币数据框，包含以下列：
                - ts: 时间戳
                - close: 收盘价
            period: 计算 BTC Price Change 的周期

        Returns:
            pd.DataFrame: 包含 BTC Price Change 指标的数据框
        """
        # 确保ts相同
        df = pd.merge(df, btc_df[['ts', 'close']], on='ts', how='left', suffixes=('', '_btc'))

        # 计算 BTC Price Change
        df['btc_price_change'] = (df['close_btc'] - df['close_btc'].shift(1)) / df['close_btc'].shift(1)

        return df

    def next_candle_trend(self, df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """
        计算 Next Candle Trend 指标
        下一个K线的趋势
        1为上涨，0为下跌
        Next Candle Trend = close.shift(-1) > close
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Next Candle Trend 的周期

        Returns:
            pd.DataFrame: 包含 Next Candle Trend 指标的数据框
        """
        # 计算 Next Candle Trend
        df['next_candle_trend'] = (df['close'].shift(-period) > df['close']).astype(int)

        return df
