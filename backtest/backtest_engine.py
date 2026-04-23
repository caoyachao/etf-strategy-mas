"""
ETF 趋势跟踪-均值回归回测引擎
支持 A 股场内 ETF，处理 T+1、停牌、涨跌停等特性
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

import baostock as bs

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    """结构化回测结果"""
    sharpe: float
    max_drawdown: float
    excess_return: float
    annual_return: float
    annual_volatility: float
    win_rate: float
    trade_count: int
    total_return: float
    benchmark_return: float
    trades: List[Dict]
    equity_curve: List[Dict]
    pass_threshold: Dict[str, float]
    passed: bool

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def feedback(self) -> str:
        """生成未通过时的优化反馈"""
        parts = []
        if self.sharpe < self.pass_threshold["sharpe"]:
            parts.append(
                f"夏普比率 {self.sharpe:.3f} 低于阈值 {self.pass_threshold['sharpe']:.3f}，"
                f"需降低波动率或提高收益。当前年化波动率 {self.annual_volatility:.2%}。"
            )
        if self.max_drawdown > self.pass_threshold["max_drawdown"]:
            parts.append(
                f"最大回撤 {self.max_drawdown:.2%} 超过阈值 {self.pass_threshold['max_drawdown']:.2%}，"
                f"需加强止损或降低仓位。"
            )
        if self.excess_return < self.pass_threshold["excess_return"]:
            parts.append(
                f"超额收益 {self.excess_return:.2%} 低于阈值 {self.pass_threshold['excess_return']:.2%}，"
                f"需优化选股逻辑或信号参数。"
            )
        if self.win_rate < 0.5:
            parts.append(f"胜率 {self.win_rate:.1%} 偏低，需改进入场/出场条件。")
        return "\n".join(parts) if parts else "所有指标通过阈值。"


# ─────────────────────────────────────────────────────────────────────────────
# BaoStock 全局登录（首次使用时自动登录）
# ─────────────────────────────────────────────────────────────────────────────

_BS_LOGGED_IN = False


def _ensure_bs_login() -> None:
    """确保 BaoStock 已登录，未登录时自动登录"""
    global _BS_LOGGED_IN
    if not _BS_LOGGED_IN:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(
                f"BaoStock 登录失败: {lg.error_code} - {lg.error_msg}"
            )
        _BS_LOGGED_IN = True


def _get_bs_code(symbol: str) -> str:
    """将 ETF 代码转换为 BaoStock 格式 (sh.510300 / sz.159915)"""
    # 上海交易所 ETF: 51xxxx, 56xxxx
    # 深圳交易所 ETF: 15xxxx, 16xxxx
    prefix = symbol[:2]
    if prefix in ("51", "52", "53", "54", "56", "58"):
        return f"sh.{symbol}"
    elif prefix in ("15", "16", "18"):
        return f"sz.{symbol}"
    else:
        # 默认上海
        return f"sh.{symbol}"


def get_etf_data(
    symbol: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取 ETF 历史数据。
    优先使用缓存，失败时尝试 BaoStock 获取。

    Args:
        symbol: ETF 代码，如 "510300"
        start_date: 开始日期，如 "2023-01-01"
        end_date: 结束日期，如 "2025-12-31"
        use_cache: 是否使用本地缓存

    Returns:
        DataFrame 包含列: date, open, high, low, close, volume
    """
    cache_file = DATA_DIR / f"{symbol}_{start_date}_{end_date}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    _ensure_bs_login()

    bs_code = _get_bs_code(symbol)
    fields = "date,open,high,low,close,volume"

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"BaoStock 查询失败: {rs.error_code} - {rs.error_msg}"
            )

        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            raise RuntimeError(f"BaoStock 返回空数据: {bs_code}")

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 类型转换
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 过滤停牌日（成交量为0或价格为0）
        df = df[(df["volume"] > 0) & (df["close"] > 0)].copy()

        if use_cache:
            df.to_csv(cache_file, index=False)

        return df

    except Exception as e:
        raise RuntimeError(
            f"获取 {symbol} 数据失败: {e}\n"
            f"请确认 baostock 已正确安装且网络可用。"
        ) from e


def get_benchmark_data(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取沪深300ETF(510300)作为基准"""
    return get_etf_data("510300", start_date, end_date, use_cache)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标：MA、RSI、MACD、布林带"""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]

    # MA
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 波动率（20日）
    df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    return df


class ETFBacktestEngine:
    """
    ETF 回测引擎

    设计约束：
    - T+1 交易：当日买入，次日才能卖出
    - 停牌处理：无法交易的日子跳过
    - 涨跌停：ETF 涨跌停仍可交易（A 股 ETF 极少涨停）
    - 等权重：选中 N 只 ETF，每只分配 1/N 仓位
    """

    def __init__(
        self,
        etf_pool: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        pass_threshold: Optional[Dict[str, float]] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.etf_pool = etf_pool
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.pass_threshold = pass_threshold or {
            "sharpe": 2.0,
            "max_drawdown": 0.05,
            "excess_return": 0.05,
        }

        # 加载数据（优先使用外部传入的数据，避免重复拉取）
        if data:
            self.data = data
        else:
            self.data = {}
            for symbol in etf_pool:
                df = get_etf_data(symbol, start_date, end_date)
                self.data[symbol] = calculate_indicators(df)

        self.benchmark = calculate_indicators(
            get_benchmark_data(start_date, end_date)
        )

    def run_strategy(
        self,
        signal_fn: Callable[[pd.DataFrame], pd.Series],
        position_fn: Optional[Callable[[pd.DataFrame, pd.Series], pd.Series]] = None,
        stop_loss: float = -0.03,
        take_profit: float = 0.08,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            signal_fn: 接收单只ETF的DataFrame，返回每日信号分数 (-1~1)
            position_fn: 可选，根据信号生成仓位比例 (0~1)
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        # 统一日期索引
        all_dates = self.benchmark["date"].tolist()
        trades = []
        daily_values = []

        cash = self.initial_capital
        positions: Dict[str, Dict] = {}  # symbol -> {shares, entry_price, entry_date}

        for i, date in enumerate(all_dates):
            day_value = cash
            day_trades = []

            # 遍历每只ETF
            active_etfs = []
            for symbol, df in self.data.items():
                row = df[df["date"] == date]
                if row.empty:
                    continue
                active_etfs.append(symbol)

            # 如果有持仓，检查止盈止损
            for symbol in list(positions.keys()):
                row = self.data[symbol][self.data[symbol]["date"] == date]
                if row.empty:
                    continue
                price = float(row["close"].iloc[0])
                pos = positions[symbol]
                ret = (price - pos["entry_price"]) / pos["entry_price"]

                if ret <= stop_loss or ret >= take_profit:
                    # 卖出
                    proceeds = pos["shares"] * price
                    cash += proceeds
                    trades.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "symbol": symbol,
                        "action": "SELL",
                        "price": round(price, 4),
                        "shares": pos["shares"],
                        "pnl": round((price - pos["entry_price"]) * pos["shares"], 2),
                        "return": round(ret, 4),
                        "reason": "stop_loss" if ret <= stop_loss else "take_profit",
                    })
                    del positions[symbol]

            # 计算信号，选择买入标的
            if active_etfs and cash > 1000:
                signals = {}
                for symbol in active_etfs:
                    if symbol in positions:
                        continue  # 已持仓，跳过
                    df = self.data[symbol]
                    row = df[df["date"] == date]
                    if row.empty or len(row) < 1:
                        continue
                    try:
                        sig = signal_fn(df)
                        if len(sig) > 0:
                            signals[symbol] = float(sig.iloc[-1])
                    except Exception:
                        continue

                if signals:
                    # 只选信号最强的前 N 只（最多3只，避免过度集中）
                    sorted_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)
                    top_n = min(3, len(sorted_symbols))
                    selected = [s for s, _ in sorted_symbols[:top_n] if _ > 0]

                    if selected:
                        weight = 1.0 / len(selected)
                        allocation_per_etf = cash * weight

                        for symbol in selected:
                            row = self.data[symbol][self.data[symbol]["date"] == date]
                            if row.empty:
                                continue
                            price = float(row["close"].iloc[0])
                            shares = int(allocation_per_etf / price)
                            if shares > 0:
                                cost = shares * price
                                cash -= cost
                                positions[symbol] = {
                                    "shares": shares,
                                    "entry_price": price,
                                    "entry_date": date,
                                }
                                trades.append({
                                    "date": date.strftime("%Y-%m-%d"),
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "price": round(price, 4),
                                    "shares": shares,
                                    "cost": round(cost, 2),
                                })

            # 计算当日总资产
            for symbol, pos in positions.items():
                row = self.data[symbol][self.data[symbol]["date"] == date]
                if not row.empty:
                    day_value += pos["shares"] * float(row["close"].iloc[0])

            daily_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(day_value, 2),
            })

        # 计算回测指标
        return self._calculate_metrics(daily_values, trades)

    def _calculate_metrics(self, daily_values: List[Dict], trades: List[Dict]) -> BacktestResult:
        df = pd.DataFrame(daily_values)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < 2:
            return BacktestResult(
                sharpe=0.0, max_drawdown=1.0, excess_return=-1.0,
                annual_return=0.0, annual_volatility=0.0, win_rate=0.0,
                trade_count=0, total_return=0.0, benchmark_return=0.0,
                trades=trades, equity_curve=daily_values,
                pass_threshold=self.pass_threshold, passed=False,
            )

        # 收益率序列
        df["return"] = df["value"].pct_change().fillna(0)
        total_return = (df["value"].iloc[-1] / self.initial_capital) - 1

        # 年化收益
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        years = max(days / 365.25, 0.01)
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 年化波动率
        daily_vol = df["return"].std()
        annual_volatility = daily_vol * np.sqrt(252)

        # 夏普比率（假设无风险利率 2%）
        risk_free = 0.02
        sharpe = (annual_return - risk_free) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cummax = df["value"].cummax()
        drawdown = (df["value"] - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 基准收益
        bench = self.benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench_start = bench[bench["date"] >= df["date"].iloc[0]]["close"].iloc[0]
        bench_end = bench[bench["date"] <= df["date"].iloc[-1]]["close"].iloc[-1]
        benchmark_return = (bench_end / bench_start) - 1
        benchmark_annual = (1 + benchmark_return) ** (1 / years) - 1

        # 超额收益
        excess_return = annual_return - benchmark_annual

        # 胜率
        trade_returns = [t.get("return", 0) for t in trades if t.get("action") == "SELL"]
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0

        passed = (
            sharpe >= self.pass_threshold["sharpe"]
            and max_drawdown <= self.pass_threshold["max_drawdown"]
            and excess_return >= self.pass_threshold["excess_return"]
        )

        return BacktestResult(
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_drawdown, 4),
            excess_return=round(excess_return, 4),
            annual_return=round(annual_return, 4),
            annual_volatility=round(annual_volatility, 4),
            win_rate=round(win_rate, 4),
            trade_count=len(trades),
            total_return=round(total_return, 4),
            benchmark_return=round(benchmark_return, 4),
            trades=trades,
            equity_curve=daily_values,
            pass_threshold=self.pass_threshold,
            passed=passed,
        )


def run_backtest(
    strategy_code: str,
    etf_pool: List[str],
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    pass_threshold: Optional[Dict[str, float]] = None,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    主入口：接收策略代码字符串，执行回测，返回结构化结果。
    用于 LangGraph Tool Node 调用。
    
    Args:
        data: 可选，外部传入的预加载数据（避免重复拉取）。
              格式: {symbol: DataFrame}，必须已含技术指标列。
    """
    # 安全执行策略代码（提取 signal_fn 函数）
    local_ns = {}
    try:
        exec(strategy_code, {"np": np, "pd": pd}, local_ns)
    except Exception as e:
        return {
            "error": f"策略代码执行失败: {str(e)}",
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "excess_return": -1.0,
            "passed": False,
        }

    signal_fn = local_ns.get("generate_signals")
    if signal_fn is None:
        return {
            "error": "策略代码中未找到 generate_signals(df) 函数",
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "excess_return": -1.0,
            "passed": False,
        }

    engine = ETFBacktestEngine(
        etf_pool=etf_pool,
        start_date=start_date,
        end_date=end_date,
        pass_threshold=pass_threshold,
        data=data,
    )

    try:
        result = engine.run_strategy(signal_fn=signal_fn)
        return result.to_dict()
    except Exception as e:
        return {
            "error": f"回测执行失败: {str(e)}",
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "excess_return": -1.0,
            "passed": False,
        }


if __name__ == "__main__":
    # 简单测试
    test_code = """
def generate_signals(df):
    # 简单双均线策略
    df = df.copy()
    df["signal"] = 0
    df.loc[df["close"] > df["ma20"], "signal"] = 1
    df.loc[df["close"] < df["ma20"], "signal"] = -1
    return df["signal"]
"""
    result = run_backtest(
        strategy_code=test_code,
        etf_pool=["510300", "510500", "159915"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
