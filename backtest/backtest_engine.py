"""
ETF 场内基金回测引擎

纯回测逻辑，不处理数据获取。
数据通过外部传入（data 参数）或内部回退拉取。

外部入口：
    run_backtest(strategy_code, etf_pool, start_date, end_date, data)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import baostock as bs


@dataclass
class BacktestResult:
    """回测结果结构"""

    sharpe: float
    max_drawdown: float
    excess_return: float
    annual_return: float
    annual_volatility: float
    win_rate: float
    trade_count: int
    total_return: float
    benchmark_return: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    pass_threshold: Dict[str, float]
    passed: bool

    def to_dict(self) -> Dict:
        return {
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "excess_return": self.excess_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "total_return": self.total_return,
            "benchmark_return": self.benchmark_return,
            "passed": self.passed,
            "pass_threshold": self.pass_threshold,
        }


class ETFBacktestEngine:
    """场内 ETF 回测引擎（支持 T+1 交易规则）"""

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

        # 加载数据：优先使用外部传入的数据
        if data:
            self.data = data
        else:
            # 回退：内部自行拉取（通过 data.fetcher）
            from data.fetcher import get_etf_data
            from data.indicators import calculate_indicators

            self.data = {}
            for symbol in etf_pool:
                df = get_etf_data(symbol, start_date, end_date)
                self.data[symbol] = calculate_indicators(df)

        self.benchmark = self._load_benchmark()

    def _load_benchmark(self) -> pd.DataFrame:
        """加载基准数据"""
        from data.fetcher import get_benchmark_data
        from data.indicators import calculate_indicators

        return calculate_indicators(
            get_benchmark_data(self.start_date, self.end_date)
        )

    def run_strategy(self, signal_fn: callable) -> BacktestResult:
        """运行策略回测"""
        trades: List[Dict[str, Any]] = []
        daily_values: List[float] = []

        # 对齐所有 ETF 的交易日
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df["date"].tolist())
        sorted_dates = sorted(all_dates)

        positions: Dict[str, int] = {s: 0 for s in self.etf_pool}
        cash = self.initial_capital
        available_cash = cash

        for date in sorted_dates:
            day_trades = []

            # 生成交易信号
            for symbol, df in self.data.items():
                day_data = df[df["date"] == date]
                if day_data.empty:
                    continue

                # 调用策略信号函数
                signal_df = signal_fn(df.copy())
                signal = signal_df[df["date"] == date].iloc[0] if not signal_df[df["date"] == date].empty else 0

                if isinstance(signal, pd.Series):
                    signal = signal.iloc[0]

                current_price = day_data["close"].iloc[0]

                # 买入信号（T+1：今天买入，明天可用资金）
                if signal > 0 and positions[symbol] == 0:
                    max_shares = int(available_cash / current_price)
                    if max_shares > 0:
                        cost = max_shares * current_price
                        if cost <= available_cash:
                            positions[symbol] = max_shares
                            available_cash -= cost
                            day_trades.append({
                                "date": date,
                                "symbol": symbol,
                                "action": "BUY",
                                "price": current_price,
                                "shares": max_shares,
                                "cost": cost,
                            })

                # 卖出信号
                elif signal < 0 and positions[symbol] > 0:
                    shares = positions[symbol]
                    revenue = shares * current_price
                    cash += revenue
                    available_cash = cash  # 卖出后资金立即可用
                    positions[symbol] = 0
                    day_trades.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "SELL",
                        "price": current_price,
                        "shares": shares,
                        "revenue": revenue,
                    })

            # 计算当日持仓市值
            portfolio_value = cash
            for symbol, shares in positions.items():
                if shares > 0:
                    day_data = self.data[symbol][self.data[symbol]["date"] == date]
                    if not day_data.empty:
                        portfolio_value += shares * day_data["close"].iloc[0]

            daily_values.append(portfolio_value)
            trades.extend(day_trades)

        return self._calculate_metrics(daily_values, trades)

    def _calculate_metrics(
        self,
        daily_values: List[float],
        trades: List[Dict[str, Any]],
    ) -> BacktestResult:
        """计算回测指标"""
        df = pd.DataFrame({"value": daily_values})
        df["date"] = pd.date_range(start=self.start_date, periods=len(daily_values), freq="B")

        total_return = (df["value"].iloc[-1] / self.initial_capital) - 1

        # 年化收益
        years = len(df) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 年化波动率
        daily_returns = df["value"].pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)

        # 夏普比率
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cummax = df["value"].cummax()
        drawdown = (df["value"] - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 基准收益
        bench = self.benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench_start = bench[bench["date"] >= df["date"].iloc[0]]["close"].iloc[0]
        bench_end = bench[bench["date"] <= df["date"].iloc[-1]]["close"].iloc[0]
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
