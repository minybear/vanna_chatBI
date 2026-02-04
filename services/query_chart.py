"""执行 SQL 并生成图表 JSON。"""
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Optional, Tuple, Any


def run_query_and_chart(
    vn,
    sql: str,
    question: Optional[str] = None,
    operator_id: Optional[str] = None,
    datasource_id: Optional[str] = None,
) -> Tuple[Any, Optional[str]]:
    """
    执行 SQL，清洗数据，按意图生成图表 JSON。
    返回 (df, chart_json)，chart_json 可为 None。
    """
    df = vn.run_sql(sql=sql, operator_id=operator_id, datasource_id=datasource_id)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

    df = df.replace([np.inf, -np.inf], None)
    df = df.replace({np.nan: None})

    potential_date_cols = [
        col for col in df.columns
        if any(kw in col.lower() for kw in ["date", "time", "dt", "day", "month"])
    ]
    for col in potential_date_cols:
        if df[col].dtype == "object":
            try:
                sample_value = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                if len(sample_value) == 8 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
                elif len(sample_value) == 10 and sample_value.isdigit():
                    df[col] = pd.to_datetime(df[col], format="%Y%m%d%H", errors="coerce")
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    try:
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            df[col] = df[col].apply(lambda v: v.isoformat() if pd.notna(v) and hasattr(v, "isoformat") else v)
    except Exception:
        pass

    chart_json = None
    if question:
        try:
            intent = vn.detect_intent(question)
            if intent.get("intent") == "detail":
                pass
            else:
                chart_recommendation = vn._recommend_chart_type(df, question)
                if chart_recommendation is not None:
                    plotly_code = vn.generate_plotly_code(
                        question=question, sql=sql, df=df, chart_recommendation=chart_recommendation
                    )
                    if plotly_code is not None:
                        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
                        if fig:
                            chart_json = fig.to_json()
        except Exception as e:
            print(f"可视化生成失败: {e}")
            import traceback
            traceback.print_exc()

    return df, chart_json
