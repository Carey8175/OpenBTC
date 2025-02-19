import io
import torch
import matplotlib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


matplotlib.use('Agg')


class ImageGenerator:
    IMAGE_SIZE = (2, 1)

    @staticmethod
    def candles(data: pd.DataFrame) -> torch.Tensor:
        """
        输入为窗口数据DF，输出为K线图的张量
        """
        fig, ax = plt.subplots(figsize=ImageGenerator.IMAGE_SIZE)

        for i, row in data.iterrows():
            is_up = row['close'] > row['open']
            # shadow
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=2.5)

            if is_up:
                # border white triangle
                ax.fill(
                    [i - 0.5, i, i + 0.5],
                    [row['open'], row['close'], row['open']],
                    edgecolor='black', facecolor='white', linewidth=1.5, zorder=2
                )
            else:
                # solid black triangle
                ax.fill(
                    [i - 0.5, i, i + 0.5],
                    [row['open'], row['close'], row['open']],
                    color='black', alpha=0.9, zorder=2
                )

        if 'MA_20' in data.columns:
            ax.plot(data.index, data['MA_20'], color='black', label='20-Day MA', linewidth=2)
        if 'MA_50' in data.columns:
            ax.plot(data.index, data['MA_50'], color='black', linestyle='--', label='50-Day MA', linewidth=2)

        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image_array = np.array(Image.open(buf).convert("L"))
        tensor = torch.from_numpy(image_array)

        return tensor.unsqueeze(0)

    @staticmethod
    def macd(data: pd.DataFrame) -> torch.Tensor:
        """
        绘制MACD图，快线使用虚线，慢线使用实线，柱状图为黑色填充
        """
        fig, ax = plt.subplots(figsize=ImageGenerator.IMAGE_SIZE)

        ax.plot(data.index, data['macd'], label="MACD", color='black', linestyle='--', linewidth=1.5)
        ax.plot(data.index, data['signal'], label="Signal Line", color='black', linestyle='-', linewidth=1.5)
        ax.bar(data.index, data['histogram'], label="MACD Histogram", color='black', alpha=1.0)

        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image_array = np.array(Image.open(buf).convert("L"))
        tensor = torch.from_numpy(image_array)

        return tensor.unsqueeze(0)

    @staticmethod
    def rsi(data: pd.DataFrame) -> torch.Tensor:
        """
        绘制RSI图，模仿K线图的样式，使用黑白配色
        """
        fig, ax = plt.subplots(figsize=ImageGenerator.IMAGE_SIZE)

        ax.plot(data.index, data['rsi_14'], label="RSI", color='black', linewidth=1.5)
        ax.axhline(y=70, color='black', linestyle='--', linewidth=1, label="Overbought (70)")
        ax.axhline(y=30, color='black', linestyle='--', linewidth=1, label="Oversold (30)")

        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image_array = np.array(Image.open(buf).convert("L"))
        tensor = torch.from_numpy(image_array)

        return tensor.unsqueeze(0)

    @staticmethod
    def volume(data: pd.DataFrame) -> torch.Tensor:
        """
        绘制成交量图
        """
        fig, ax = plt.subplots(figsize=ImageGenerator.IMAGE_SIZE)

        ax.bar(data.index, data['vol'], color='black', alpha=1.0)

        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image_array = np.array(Image.open(buf).convert("L"))
        tensor = torch.from_numpy(image_array)

        return tensor.unsqueeze(0)

    @staticmethod
    def gen(data: pd.DataFrame) -> torch.Tensor:
        candles_tensor = ImageGenerator.candles(data)
        macd_tensor = ImageGenerator.macd(data)
        rsi_tensor = ImageGenerator.rsi(data)
        vol_tensor = ImageGenerator.volume(data)

        return torch.cat([candles_tensor, macd_tensor, rsi_tensor, vol_tensor], dim=0)
