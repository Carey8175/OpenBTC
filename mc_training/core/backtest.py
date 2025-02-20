import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from matplotlib import rcParams
from scipy.stats import ttest_1samp,t
from scipy.stats import linregress
from sklearn.metrics import precision_score,recall_score, f1_score
from scipy.stats import skew


class Backtest:
    rcParams['font.family'] = 'SimHei'  # 或者 'SimSun'

    def __init__(self):
        self.strategy_ret = None
        self.decision_df = None

    def prepare_dataframes(self, labeled_data: pd.DataFrame):
        """
        Prepare dataframes for backtesting.
        """

        def _prepare_strategy_ret(df) -> pd.DataFrame:
            '''
            strategy_ret是df，包括以下列：
                trade_date:YYYYMMDD,str
                ts_code:str
                daily_ret:如果持有此资产，当天对应的持有平均收益率
                bs_ret:float 小数不是百分数
                ret_direction: bs_ret>0 True else False
                holding_period:float 持仓周期
                benchmark:str
                bench_ret:float 小数不是百分数
            USAGE:strategy_ret = prepare_strategy_ret(date_filter)
            '''
            strategy_ret = df

            # trade_date
            # date_range = pd.date_range(start=date_filter['start_date'], end=date_filter['end_date'], freq='D')
            # date_list = date_range.strftime('%Y-%m-%d').tolist()
            strategy_ret['trade_date'] = pd.to_datetime(strategy_ret['ts'], unit='ms').dt.strftime('%Y%m%d %H:%M:%S')

            # ts_code
            strategy_ret['ts_code'] = 'BTC-USDT-SWAP'  # 可任意，计算不会用到，用来自己识别

            # daily_ret 模拟收益率序列
            strategy_ret['daily_ret'] = strategy_ret['close'].pct_change()

            # bs_ret 模拟信号买卖点收益率，假设是未来两天的收益率
            strategy_ret['bs_ret'] = strategy_ret['close'].shift(-5) / strategy_ret['close'] - 1

            # ret_direction: bs_ret>0 True else False
            strategy_ret['ret_direction'] = [True if ret > 0 else False for ret in strategy_ret['bs_ret']]

            # holding_period:float 持仓周期
            strategy_ret['holding_period'] = 5

            # benchmark
            strategy_ret['benchmark'] = 'Bitcoin'

            # bench_ret
            strategy_ret['bench_ret'] = strategy_ret['daily_ret']

            return strategy_ret

        def _prepare_decision_df(df) -> pd.DataFrame:
            '''
               decision_df包含两列：
                   trade_date：时间戳
                   decision：True or False。是模型预测输出的信号
               usage:decision_df = prepare_decision_df(date_filter)
               '''
            decision_df = df.copy()

            # trade_date
            decision_df['trade_date'] = pd.to_datetime(decision_df['ts'], unit='ms').dt.strftime('%Y%m%d %H:%M:%S')

            # decision
            decision_df['decision'] = decision_df['prediction']
            decision_df = decision_df[['trade_date', 'decision']]

            return decision_df

        self.strategy_ret = _prepare_strategy_ret(labeled_data)
        self.decision_df = _prepare_decision_df(labeled_data)

    def curve_evaluation(self, fee_cost=0.0015, rolling_window=20, rolling_t_threshold=1, is_plot=True):
        def analyze_strategy_from_confusion_matrix(conf_matrix):
            """
            usage:
            strategy_type = analyze_strategy_from_confusion_matrix(conf_matrix)
            """
            # 提取混淆矩阵的值
            TN, FP = conf_matrix[0]
            FN, TP = conf_matrix[1]

            # 计算指标
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 精确率
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # 召回率

            # 判断风格
            if recall > precision * 1.2:
                strategy_type = "进攻型"
            elif precision > recall * 1.2:
                strategy_type = "防守型"
            else:
                strategy_type = "平衡型"
            return strategy_type

        def analyze_daily_return(ret_list, rf=0.02):
            '''
            分析日度收益率
            USAGE:result_df = analyze_daily_return(ret_list,rf=0.02)
            '''

            if len(ret_list) > 0:
                ret_series = np.array(ret_list)
                # 1.计算平均日度收益率
                avg_daily_return = np.mean(ret_series)
                # 2. 计算日度收益率的标准差（波动率）
                volatility = np.std(ret_series)
                annual_vol = volatility * np.sqrt(365*24)
                # 3.计算夏普率
                # 默认无风险收益率含有工作日
                daily_risk_free_rate = (1 + rf) ** (1 / (365*24)) - 1
                # 计算超额收益
                excess_daily_return = avg_daily_return - daily_risk_free_rate
                # 避免除以0的情况
                if volatility == 0:
                    print('sharpe分母波动率异常，请监察')
                    sharpe_ratio = np.nan
                else:
                    sharpe_ratio = (excess_daily_return / volatility) * np.sqrt(365*24)  # 年化夏普率

                # 4. 计算年化收益率
                total_return = np.prod(1 + ret_series) - 1
                try:
                    annual_return = (1 + total_return) ** (365*24 / len(ret_series)) - 1
                except:
                    annual_return = np.nan
                # 5.最大回撤
                cum_returns = (1 + ret_series).cumprod()
                # 初始化最大回撤和当前峰值
                max_drawdown = 0
                peak = cum_returns[0]
                # 遍历累积收益率序列
                for nav in cum_returns:
                    # 如果当前值大于峰值，则更新峰值
                    if nav > peak:
                        peak = nav
                        # 否则，计算回撤并更新最大回撤（如果需要）
                    else:
                        drawdown = (peak - nav) / peak
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                max_drawdown = -max_drawdown

                # 6.计算sortino比率
                # 1.计算平均日度收益率
                avg_daily_return = np.mean(ret_series)
                # 计算负收益的部分，设定为负值的部分表示下行风险
                downside_vol = np.std((np.minimum(0, ret_series)))
                # 3.计算夏普率
                # 默认无风险收益率含有工作日
                daily_risk_free_rate = (1 + rf) ** (1 / (365*24)) - 1
                # 计算超额收益
                excess_daily_return = avg_daily_return - daily_risk_free_rate
                # 避免除以0的情况
                if downside_vol == 0:
                    print('sortino分母波动率异常，请监察')
                    sortino_ratio = np.nan
                else:
                    sortino_ratio = (excess_daily_return / downside_vol) * np.sqrt(365*24)  # 年化夏普率

                # 3.结果汇总
                columns = ['avg_daily_return', 'annual_vol', 'downside_vol', 'sharpe_ratio', 'annual_return',
                           'max_drawdown', 'sortino_ratio']
                result_df = pd.DataFrame(columns=columns)
                result_df.loc[len(result_df)] = [avg_daily_return, annual_vol, downside_vol, sharpe_ratio,
                                                 annual_return, max_drawdown, sortino_ratio]
            else:
                columns = ['avg_daily_return', 'annual_vol', 'downside_vol', 'sharpe_ratio', 'annual_return',
                           'max_drawdown', 'sortino_ratio']
                result_df = pd.DataFrame(columns=columns)

            return result_df

        def curve_evaluation(strategy_ret, decision_df, fee_cost, rolling_window=20, rolling_t_threshold=0.8,
                             is_plot=True):
            '''
            report,curve_df = curve_evaluation(strategy_ret, decision_df,fee_cost,rolling_window=20,rolling_t_threshold=1, is_plot=True)
            '''

            def t_test(series):
                aggre_train_t, _ = ttest_1samp(series, popmean=0)
                return aggre_train_t

            def adjust_first_nonzero_in_nonzero_sequences(ret_sequence, fee_cost):
                # 找到连续为 0 的位置
                ret_af = ret_sequence.copy()
                is_zero = (ret_af == 0)

                # 找到非零段的起始位置
                edges = np.diff(is_zero.astype(int), prepend=1)
                start_positions = np.where(edges == -1)[0]  # 从 0 变成非 0 的位置

                # 对每个起始位置的第一个数字减 1
                for start in start_positions:
                    ret_af[start] -= fee_cost
                return ret_af

            evaluation_df = strategy_ret.merge(decision_df, on='trade_date', how='left')
            evaluation_df['decision'] = [True if v == 1 else False for v in evaluation_df['decision'].values]

            # 5.离散评
            evaluation_df_discrete = evaluation_df.dropna()
            y_pred = evaluation_df_discrete['decision']
            y_real = evaluation_df_discrete['ret_direction']

            accuracy = sum(y_pred == y_real) / len(y_real)
            precision = precision_score(y_real, y_pred)
            recall = recall_score(y_real, y_pred)
            f1 = f1_score(y_real, y_pred)

            cm = (confusion_matrix(evaluation_df_discrete['ret_direction'], evaluation_df_discrete['decision'])
                  / np.sum(
                        confusion_matrix(evaluation_df_discrete['ret_direction'], evaluation_df_discrete['decision'])))
            strategy_type = analyze_strategy_from_confusion_matrix(cm)

            # 6.连续评估
            # 计算信号对应的每日收益率
            evaluation_df['bs_ret_avg'] = [
                (evaluation_df['bs_ret'][i] + 1) ** (1 / evaluation_df['holding_period'][i]) - 1 for
                i in range(len(evaluation_df))]
            signal_list = evaluation_df['decision'].fillna(False)
            holding_period_list = evaluation_df['holding_period'].fillna(False)
            for i in range(len(evaluation_df)):
                signal_now = signal_list[i]
                holding_period_now = holding_period_list[i]
                if signal_now:
                    for j in range(1, holding_period_now):
                        signal_list[i + j] = False
            evaluation_df['signaling'] = signal_list

            evaluation_df_tmp = evaluation_df[['trade_date', 'holding_period', 'bs_ret_avg', 'signaling']].copy()
            for i in range(len(evaluation_df_tmp) - 2):  # 最后两天不计算成本和均摊收益
                is_holding_now = evaluation_df_tmp['signaling'][i]
                holding_period_now = evaluation_df_tmp['holding_period'][i]
                holding_ret = evaluation_df_tmp['bs_ret_avg'][i]
                if is_holding_now:
                    for j in range(1, 1 + holding_period_now):
                        evaluation_df_tmp.loc[i + j, 'strategy_ret'] = holding_ret
            evaluation_df_tmp['strategy_ret'] = evaluation_df_tmp['strategy_ret'].fillna(0)

            ret_sequence = evaluation_df_tmp['strategy_ret'].values
            ret_af = adjust_first_nonzero_in_nonzero_sequences(ret_sequence, fee_cost)
            evaluation_df_tmp['strategy_ret_af'] = ret_af

            # bs_ret是信号对应的未来收益率，daily_ret是资产当天持有的收益率
            curve_df_tmp = evaluation_df_tmp[['trade_date', 'bs_ret_avg']].copy().rename(
                columns={'bs_ret_avg': 'daily_ret'})
            curve_df_tmp['daily_ret'] = curve_df_tmp['daily_ret'].shift(1).fillna(0)
            curve_df_tmp['cul_ret'] = np.cumprod(curve_df_tmp['daily_ret'] + 1)

            curve_df_tmp['strategy_ret'] = evaluation_df_tmp['strategy_ret']
            curve_df_tmp['strategy_cul_ret'] = np.cumprod(curve_df_tmp['strategy_ret'] + 1)

            # 费率调整
            # 给每段连续的 True 分组
            curve_df_tmp['strategy_ret_af'] = evaluation_df_tmp['strategy_ret_af']
            curve_df_tmp['strategy_cul_ret_af'] = np.cumprod(curve_df_tmp['strategy_ret_af'] + 1)
            curve_df_tmp = curve_df_tmp.merge(strategy_ret[['trade_date', 'benchmark', 'bench_ret']], on='trade_date',
                                              how='left')
            curve_df_tmp['bench_cul_ret'] = np.cumprod(curve_df_tmp['bench_ret'] + 1)

            curve_df = curve_df_tmp.copy()

            # 评价指标
            # 信号统计
            deal_num = len(curve_df[curve_df['strategy_ret'] != 0])
            deal_rate = np.divide(deal_num, len(curve_df))
            win_num = len(curve_df[curve_df['strategy_ret'] > 0])
            win_rate = np.divide(win_num, deal_num)
            avg_signal_ret = curve_df[curve_df['strategy_ret'] != 0][
                'strategy_ret'].mean()

            # 无费率结果
            ret_list = curve_df['strategy_ret'].values
            result_df = analyze_daily_return(ret_list, rf=0.02)
            annual_return = result_df['annual_return'].values[0]
            annual_vol = result_df['annual_vol'].values[0]
            sharpe_ratio = result_df['sharpe_ratio'].values[0]
            max_drawdown = result_df['max_drawdown'].values[0]

            # 含费率结果
            ret_list = curve_df['strategy_ret_af'].values
            result_df = analyze_daily_return(ret_list, rf=0.02)
            annual_return_af = result_df['annual_return'].values[0]
            annual_vol_af = result_df['annual_vol'].values[0]
            sharpe_ratio_af = result_df['sharpe_ratio'].values[0]
            max_drawdown_af = result_df['max_drawdown'].values[0]

            # 策略稳定性
            # 偏度
            strategy_ret_af_skew = skew(curve_df['strategy_ret_af'])
            # 总体t值
            strategy_ret_af_t, _ = ttest_1samp(curve_df['strategy_ret_af'], popmean=0)
            # t值滚动
            rolling_window = rolling_window
            rolling_t_threshold = rolling_t_threshold  # 1.96太严格了

            strategy_ret_af_seq = curve_df['strategy_ret_af'][curve_df['strategy_ret_af'] != 0].copy()
            strategy_ret_af_roll_t = strategy_ret_af_seq.rolling(window=rolling_window).apply(t_test).dropna()
            t_value, _ = ttest_1samp(strategy_ret_af_roll_t, popmean=rolling_t_threshold)
            degree_f = len(strategy_ret_af_seq) - 1  # 自由度
            strategy_ret_af_rolling_p1_len = len(strategy_ret_af_roll_t)
            strategy_ret_af_rolling_p1 = np.round(1 - t.cdf(t_value, degree_f), 4)  # 右单边检验（H1: mean > mu_0）

            # 超额
            beta, alpha, r_value, p_value, std_err = linregress(x=curve_df['bench_ret'], y=curve_df['strategy_ret_af'])
            excess_ret = curve_df['strategy_ret_af'] - 1 * curve_df['bench_ret']
            avg_excess_ret = np.mean(excess_ret)
            excess_t, _ = ttest_1samp(excess_ret, popmean=0)
            excess_ret_skew = skew(excess_ret)
            # t值滚动
            excess_ret_roll_t = excess_ret.rolling(window=rolling_window).apply(t_test).dropna()
            t_value, _ = ttest_1samp(excess_ret_roll_t, popmean=rolling_t_threshold)
            degree_f = len(excess_ret) - 1  # 自由度
            excess_ret_rolling_p1_len = len(excess_ret_roll_t)
            excess_ret_rolling_p1 = np.round(1 - t.cdf(t_value, degree_f), 4)  # 右单边检验（H1: mean > mu_0）

            excess_cul_ret = np.cumprod(excess_ret + 1)
            ret_list = excess_ret.values
            result_df = analyze_daily_return(ret_list, rf=0)
            excess_annual_return = result_df['annual_return'].values[0]
            excess_sharpe_ratio = result_df['sharpe_ratio'].values[0]
            excess_max_drawdown = result_df['max_drawdown'].values[0]

            # 年化手续费
            ret_list = curve_df['strategy_ret_af'] - curve_df['strategy_ret']
            result_df = analyze_daily_return(ret_list, rf=0)
            annual_fee_cost = result_df['annual_return'].values[0]

            # 整理
            report_value = [deal_num, deal_rate, win_num, win_rate, strategy_type,
                            accuracy, precision, recall, f1, avg_signal_ret, strategy_ret_af_skew, strategy_ret_af_t,
                            strategy_ret_af_rolling_p1_len, strategy_ret_af_rolling_p1,
                            annual_return, annual_vol, sharpe_ratio, max_drawdown,
                            annual_return_af, annual_vol_af, sharpe_ratio_af, max_drawdown_af,
                            beta, avg_excess_ret, excess_ret_skew, excess_t, excess_ret_rolling_p1_len,
                            excess_ret_rolling_p1,
                            excess_annual_return, excess_sharpe_ratio, excess_max_drawdown, annual_fee_cost]

            columns = ['deal_num', 'deal_rate', 'win_num', 'win_rate', 'strategy_type',
                       'accuracy', 'precision', 'recall', 'f1', 'avg_signal_ret', 'strategy_ret_af_skew',
                       'strategy_ret_af_t',
                       'strategy_ret_af_rolling_p1_len', 'strategy_ret_af_rolling_p1',
                       'annual_return', 'annual_vol', 'sharpe_ratio', 'max_drawdown',
                       'annual_return_af', 'annual_vol_af', 'sharpe_ratio_af', 'max_drawdown_af',
                       'beta', 'avg_excess_ret', 'excess_ret_skew', 'excess_t', 'excess_ret_rolling_p1_len',
                       'excess_ret_rolling_p1',
                       'excess_annual_return', 'excess_sharpe_ratio', 'excess_max_drawdown', 'annual_fee_cost']

            report = pd.DataFrame(columns=columns)
            report.loc[len(report)] = report_value

            if is_plot:
                print('>strategy_type:', strategy_type)
                print('>accuracy:', accuracy)
                print('>precision:', precision)
                print('>recall:', recall)
                print('>f1:', f1)

                x_ticks = curve_df['trade_date'].values
                fig, axs = plt.subplots(7, 1, figsize=[12, 18], layout='constrained')
                axs[0].plot(curve_df[['bench_cul_ret']], 'k--')
                axs[0].plot(curve_df[['cul_ret']], 'orange')
                axs[0].plot(curve_df[['strategy_cul_ret']], 'g--')
                axs[0].plot(curve_df[['strategy_cul_ret_af']], 'b')

                axs[0].set_xticks(range(0, len(x_ticks), 20))
                axs[0].set_xticklabels(x_ticks[range(0, len(x_ticks), 20)], rotation=90)
                axs[0].legend(['bench_cul_ret', 'cul_ret', 'strategy_cul_ret', 'strategy_cul_ret_af'], loc='upper left')

                # 超额收益
                axs[1].set_xticks(range(0, len(x_ticks), 20))
                axs[1].set_xticklabels(x_ticks[range(0, len(x_ticks), 20)], rotation=90)

                axs[1].plot(curve_df[['bench_cul_ret']], 'k--')
                axs[1].plot(curve_df[['strategy_cul_ret_af']], 'b')
                axs[1].plot(excess_cul_ret, 'r')
                axs[1].legend(['bench_cul_ret', 'strategy_cul_ret_af', 'excess_cul_ret'], loc='upper left')

                # 灰色背景
                data = curve_df[['strategy_cul_ret_af']].copy()
                changed_idx = data.index[data['strategy_cul_ret_af'].diff().fillna(0) != 0]
                # 添加灰色透明背景条
                for idx in changed_idx:
                    axs[1].axvspan(idx, idx + 1, color='gray', alpha=0.05)

                # 收益分布直方图
                strategy_ret_control = curve_df[curve_df['strategy_ret_af'] != 0][
                    'strategy_ret_af']
                bins = int(len(strategy_ret_control) / 3)
                axs[2].hist(strategy_ret_control, bins=bins, edgecolor='k')
                axs[2].axvline(np.mean(strategy_ret_control), color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {np.mean(strategy_ret_control):.4f}')
                axs[2].axvline(np.median(strategy_ret_control), color='orange', linestyle='--', linewidth=2,
                               label=f'median: {np.median(strategy_ret_control):.4f}')

                axs[2].axvline(0, color='grey', linestyle='--', linewidth=2)
                axs[2].legend(loc='upper left')

                # 收益率序列
                data = pd.DataFrame()
                data['Rolling_Mean'] = strategy_ret_af_seq.rolling(window=rolling_window).mean()
                data['Rolling_Vol'] = strategy_ret_af_seq.rolling(window=rolling_window).std()

                axs[3].bar(range(len(strategy_ret_af_seq)), strategy_ret_af_seq, color='blue', alpha=0.6,
                           label='Daily Returns')
                axs[3].plot(range(len(strategy_ret_af_seq)), data['Rolling_Mean'], color='orange',
                            label=f'strategy_ret_af_roll_mean_{rolling_window}')
                axs[3].plot(range(len(strategy_ret_af_seq)), data['Rolling_Vol'], color='green',
                            label=f'strategy_ret_af_roll_vol_{rolling_window}')
                axs[3].legend(loc='upper left')

                # 收益率序列
                data = pd.DataFrame()
                data['Rolling_Mean'] = excess_ret.rolling(window=rolling_window).mean()
                data['Rolling_Vol'] = excess_ret.rolling(window=rolling_window).std()

                axs[4].bar(range(len(excess_ret)), excess_ret, color='red', alpha=0.6, label='Daily Returns')
                axs[4].plot(range(len(excess_ret)), data['Rolling_Mean'], color='orange',
                            label=f'excess_ret_roll_mean_{rolling_window}')
                axs[4].plot(range(len(excess_ret)), data['Rolling_Vol'], color='green',
                            label=f'excess_ret_roll_vol_{rolling_window}')
                axs[4].legend(loc='upper left')

                # t值稳健性
                axs[5].plot(range(len(strategy_ret_af_roll_t)), strategy_ret_af_roll_t, color='blue', alpha=0.6,
                            label=f'strategy_af_Rolling_t_{rolling_window}')

                axs[5].axhline(1.96, color='gray', linestyle='--', linewidth=2)
                axs[5].axhline(1, color='gray', linestyle='--', linewidth=2)
                axs[5].axhline(0, color='black', linewidth=2)
                axs[5].legend(loc='upper left')

                # t值稳健性
                axs[6].plot(range(len(excess_ret_roll_t)), excess_ret_roll_t, color='r', alpha=0.6,
                            label=f'excess_ret_t_{rolling_window}')

                axs[6].axhline(1.96, color='gray', linestyle='--', linewidth=2)
                axs[6].axhline(1, color='gray', linestyle='--', linewidth=2)
                axs[6].axhline(0, color='black', linewidth=2)
                axs[6].legend(loc='upper left')

                suptitle = (
                        f'年化收益af：{np.round(annual_return_af, 3)},夏普af：{np.round(sharpe_ratio_af, 2)}' +
                        f',beta：{np.round(beta, 2)},对冲年化af：{np.round(excess_annual_return, 3)},对冲夏普af：{np.round(excess_sharpe_ratio, 2)}' +
                        f',频率：{np.round(deal_rate, 3)},胜率：{np.round(win_rate, 3)}' +
                        f',最大回撤：{np.round(max_drawdown, 3)},年化交易成本:{np.round(annual_fee_cost, 3)}')
                print(suptitle)
                plt.suptitle(suptitle)
                plt.show()

            return report, curve_df

        curve_evaluation(self.strategy_ret, self.decision_df, fee_cost, rolling_window, rolling_t_threshold, is_plot)


    def backtest(self, labeled_data: pd.DataFrame):
        self.prepare_dataframes(labeled_data)
        self.curve_evaluation()
