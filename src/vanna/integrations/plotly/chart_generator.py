"""Plotly-based chart generator with automatic chart type selection."""

from typing import Dict, Any, List, cast
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


class PlotlyChartGenerator:
    """Generate Plotly charts using heuristics based on DataFrame characteristics."""

    # Vanna brand colors from landing page
    THEME_COLORS = {
        "navy": "#023d60",
        "cream": "#e7e1cf",
        "teal": "#15a8a8",
        "orange": "#fe5d26",
        "magenta": "#bf1363",
    }

    # Color palette for charts (excluding cream as it's too light for data)
    COLOR_PALETTE = ["#15a8a8", "#fe5d26", "#bf1363", "#023d60"]

    @staticmethod
    def _is_status_like_column(col_name: str, df: pd.DataFrame) -> bool:
        """判断列是否像状态/编码类，不作为 Y 轴高度（避免状态码主导刻度）。"""
        lower = (col_name or "").lower()
        status_keywords = ("status", "code", "state", "类型", "状态", "编码")
        value_keywords = ("count", "total", "amount", "sum", "num", "数量", "订单", "笔数", "value", "值")
        if any(k in lower for k in value_keywords):
            return False
        if any(k in lower for k in status_keywords):
            return True
        if lower.endswith("_id") or lower.endswith("id"):
            return True
        # 数据特征：取值种类少且多为整数码值
        s = df[col_name].dropna()
        if len(s) == 0:
            return False
        n = pd.to_numeric(s, errors="coerce").dropna()
        if len(n) == 0:
            return False
        distinct = n.astype(int).nunique()
        lo, hi = float(n.min()), float(n.max())
        if distinct <= 30 and (hi - lo > 1000 or (hi <= 9999 and lo >= 0)):
            return True
        return False

    @staticmethod
    def _is_value_metric_column(col_name: str) -> bool:
        """判断列名是否像数值/指标类，优先作为 Y 轴。"""
        lower = (col_name or "").lower()
        return any(
            k in lower
            for k in (
                "count", "total", "amount", "sum", "num", "数量", "订单", "笔数",
                "value", "值", "revenue", "金额", "avg", "mean", "平均",
            )
        )

    def _filter_value_cols_for_y(self, numeric_cols: List[str], df: pd.DataFrame) -> List[str]:
        """从数值列中筛出用于 Y 轴的列：排除状态类，优先数值指标类。"""
        if not numeric_cols:
            return numeric_cols
        value_like = [c for c in numeric_cols if self._is_value_metric_column(c)]
        not_status = [c for c in numeric_cols if not self._is_status_like_column(c, df)]
        if value_like:
            return value_like
        if not_status:
            return not_status
        return numeric_cols

    def generate_chart(self, df: pd.DataFrame, title: str = "Chart") -> Dict[str, Any]:
        """Generate a Plotly chart based on DataFrame shape and types.

        Heuristics:
        - 4+ columns: table
        - 1 numeric column: histogram
        - 2 columns (1 categorical, 1 numeric): bar chart
        - 2 numeric columns: scatter plot
        - 3+ numeric columns: correlation heatmap or multi-line chart
        - Time series data: line chart
        - Multiple categorical: grouped bar chart

        Args:
            df: DataFrame to visualize
            title: Title for the chart

        Returns:
            Plotly figure as dictionary

        Raises:
            ValueError: If DataFrame is empty or cannot be visualized
        """
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")

        # Try to convert columns to datetime if they look like dates
        # This helps in identifying time series correctly
        for col in df.columns:
            # Check if column name suggests it's a date
            if any(x in col.lower() for x in ['date', 'time', 'day', 'month', 'year', 'dt', '_dt', 'stat']):
                try:
                    temp_col = None
                    # Handle different date formats
                    if df[col].dtype in ['int64', 'int32', 'float64']:
                        # Try YYYYMMDD format (e.g., 20251110 -> 2025-11-10)
                        temp_col = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
                    elif df[col].dtype == 'object':
                        # Try automatic parsing for string dates
                        temp_col = pd.to_datetime(df[col], errors='coerce')

                    # Only apply if conversion was successful (not all NaT)
                    if temp_col is not None and not temp_col.isna().all():
                        df[col] = temp_col
                except (ValueError, TypeError, AttributeError):
                    pass

        # Identify column types (needed for 4-col multi-line check)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # 时间名列（用于无 datetime64 时仍能识别时间轴）
        time_like_col_names = [
            c for c in df.columns
            if any(kw in (c or "").lower() for kw in ["date", "time", "dt", "day", "month", "stat"])
        ]
        # 4+ 列时：若为「时间+分类+数值」结构，优先多线折线图，否则才用表格
        if len(df.columns) >= 4 and (datetime_cols or time_like_col_names) and categorical_cols and numeric_cols:
            time_col = datetime_cols[0] if datetime_cols else time_like_col_names[0]
            value_cols = self._filter_value_cols_for_y(numeric_cols, df) or numeric_cols
            val_col = value_cols[0]
            cat_col = None
            for preferred in ["status_desc", "provisioning_status", "status", "type", "category"]:
                if preferred in df.columns:
                    cat_col = preferred
                    break
            if not cat_col and categorical_cols:
                cat_col = categorical_cols[0]
            if cat_col:
                fig = self._create_grouped_time_series_chart(
                    df, time_col, cat_col, val_col, title
                )
                result = json.loads(pio.to_json(fig))
                return result
        if len(df.columns) >= 4:
            fig = self._create_table(df, title)
            result: Dict[str, Any] = json.loads(pio.to_json(fig))
            return result

        # Check for time series (including time-like column by name)
        time_like_cols = [
            c for c in df.columns
            if any(kw in (c or "").lower() for kw in ["date", "time", "dt", "day", "month", "stat"])
        ]
        is_timeseries = len(datetime_cols) > 0 or bool(time_like_cols)

        # Apply heuristics
        if is_timeseries and len(numeric_cols) > 0:
            # Time series line chart：Y 轴用数值列，排除状态类列
            value_cols = self._filter_value_cols_for_y(numeric_cols, df)
            if not value_cols:
                value_cols = numeric_cols
            time_col = datetime_cols[0] if datetime_cols else time_like_cols[0]
            if len(categorical_cols) > 0 and len(numeric_cols) == 1:
                # Case: Long format time series (Date, Category, Value)
                # e.g. stat_date, measure_type, daily_count
                fig = self._create_grouped_time_series_chart(
                    df, time_col, categorical_cols[0], numeric_cols[0], title
                )
            elif len(categorical_cols) > 0:
                # 有分类列时优先多线图（每种类型/状态一条线）
                val_col = value_cols[0] if value_cols else numeric_cols[0]
                cat_col = categorical_cols[0]
                fig = self._create_grouped_time_series_chart(
                    df, time_col, cat_col, val_col, title
                )
            else:
                # Case: Wide format time series (Date, Value1, Value2...)
                fig = self._create_time_series_chart(
                    df, time_col, value_cols, title
                )
        elif len(numeric_cols) == 1 and len(categorical_cols) == 0:
            # Single numeric column: histogram
            fig = self._create_histogram(df, numeric_cols[0], title)
        elif len(numeric_cols) == 1 and len(categorical_cols) == 1:
            # One categorical, one numeric: bar chart
            fig = self._create_bar_chart(
                df, categorical_cols[0], numeric_cols[0], title
            )
        elif len(numeric_cols) == 2 and len(categorical_cols) >= 1:
            # Two numeric columns + categorical: Y 轴优先用数值列，不用状态列
            col1, col2 = numeric_cols[0], numeric_cols[1]
            col1_is_x = any(x in col1.lower() for x in ['date', 'time', 'year', 'id', 'day', 'month'])
            col2_is_x = any(x in col2.lower() for x in ['date', 'time', 'year', 'id', 'day', 'month'])
            if col1_is_x and not col2_is_x:
                x_col, y_col = col1, col2
            elif col2_is_x and not col1_is_x:
                x_col, y_col = col2, col1
            else:
                x_col, y_col = col1, col2
            # 若当前 y 像状态列而另一列为数值列，则用数值列作 Y
            if self._is_status_like_column(y_col, df) and self._is_value_metric_column(x_col):
                x_col, y_col = y_col, x_col
            elif self._is_status_like_column(y_col, df) and not self._is_status_like_column(x_col, df):
                x_col, y_col = y_col, x_col

            fig = self._create_grouped_time_series_chart(
                df, x_col, categorical_cols[0], y_col, title
            )
        elif len(numeric_cols) == 2:
            # Two numeric columns: scatter plot
            fig = self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1], title)
        elif len(numeric_cols) >= 3:
            # Multiple numeric columns: correlation heatmap
            fig = self._create_correlation_heatmap(df, numeric_cols, title)
        elif len(categorical_cols) >= 2:
            # Multiple categorical: grouped bar chart
            fig = self._create_grouped_bar_chart(df, categorical_cols, title)
        else:
            # Fallback: show first two columns as scatter/bar
            if len(df.columns) >= 2:
                fig = self._create_generic_chart(
                    df, df.columns[0], df.columns[1], title
                )
            else:
                raise ValueError(
                    "Cannot determine appropriate visualization for this DataFrame"
                )

        # Convert to JSON-serializable dict using plotly's JSON encoder
        result = json.loads(pio.to_json(fig))
        return result

    def _apply_standard_layout(self, fig: go.Figure) -> go.Figure:
        """Apply consistent Vanna brand styling to all charts.

        Uses Vanna brand colors from the landing page for a cohesive look.

        Args:
            fig: Plotly figure to update

        Returns:
            Updated figure with Vanna brand styling
        """
        fig.update_layout(
            # paper_bgcolor='white',
            # plot_bgcolor='white',
            font={"color": self.THEME_COLORS["navy"]},  # Navy for text
            autosize=True,  # Allow chart to resize responsively
            colorway=self.COLOR_PALETTE,  # Use Vanna brand colors for data
            # Move legend to bottom to avoid overlap with modebar
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,  # Place below the chart
                xanchor="center",
                x=0.5
            ),
            # Add some margin to the top for modebar
            margin=dict(t=40, b=40),
            # Don't set width/height - let frontend handle sizing
        )
        return fig

    def _create_histogram(self, df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """Create a histogram for a single numeric column."""
        fig = px.histogram(
            df,
            x=column,
            title=title,
            color_discrete_sequence=[self.THEME_COLORS["teal"]],
        )
        fig.update_layout(xaxis_title=column, yaxis_title="Count", showlegend=False)
        self._apply_standard_layout(fig)
        return fig

    def _create_bar_chart(
        self, df: pd.DataFrame, x_col: str, y_col: str, title: str
    ) -> go.Figure:
        """Create a bar chart for categorical vs numeric data."""
        # Aggregate if needed
        agg_df = df.groupby(x_col)[y_col].sum().reset_index()
        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            title=title,
            color_discrete_sequence=[self.THEME_COLORS["orange"]],
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        self._apply_standard_layout(fig)
        return fig

    def _create_scatter_plot(
        self, df: pd.DataFrame, x_col: str, y_col: str, title: str
    ) -> go.Figure:
        """Create a scatter plot for two numeric columns."""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color_discrete_sequence=[self.THEME_COLORS["magenta"]],
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        self._apply_standard_layout(fig)
        return fig

    def _create_correlation_heatmap(
        self, df: pd.DataFrame, columns: List[str], title: str
    ) -> go.Figure:
        """Create a correlation heatmap for multiple numeric columns."""
        corr_matrix = df[columns].corr()
        # Custom Vanna color scale: navy (negative) -> cream (neutral) -> teal (positive)
        vanna_colorscale = [
            [0.0, self.THEME_COLORS["navy"]],
            [0.5, self.THEME_COLORS["cream"]],
            [1.0, self.THEME_COLORS["teal"]],
        ]
        fig = cast(
            go.Figure,
            px.imshow(
                corr_matrix,
                title=title,
                labels=dict(color="Correlation"),
                x=columns,
                y=columns,
                color_continuous_scale=vanna_colorscale,
                zmin=-1,
                zmax=1,
            ),
        )
        self._apply_standard_layout(fig)
        return fig

    def _create_grouped_time_series_chart(
        self, df: pd.DataFrame, time_col: str, cat_col: str, val_col: str, title: str
    ) -> go.Figure:
        """Create a multi-line time series chart grouped by a categorical column."""
        fig = px.line(
            df,
            x=time_col,
            y=val_col,
            color=cat_col,
            title=title,
            color_discrete_sequence=self.COLOR_PALETTE,
        )
        fig.update_traces(line=dict(shape="spline"))
        layout_kw: Dict[str, Any] = dict(
            xaxis_title=time_col,
            yaxis_title=val_col,
            hovermode="x unified",
        )
        if len(df) > 24:
            layout_kw["xaxis"] = dict(
                rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(21, 168, 168, 0.08)"),
                type="date",
            )
        fig.update_layout(**layout_kw)
        self._apply_standard_layout(fig)
        return fig

    def _create_time_series_chart(
        self, df: pd.DataFrame, time_col: str, value_cols: List[str], title: str
    ) -> go.Figure:
        """Create a time series line chart."""
        fig = go.Figure()

        for i, col in enumerate(value_cols[:5]):  # Limit to 5 lines for readability
            color = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, shape="spline"),
                )
            )

        layout_kw: Dict[str, Any] = dict(
            title=title,
            xaxis_title=time_col,
            yaxis_title="Value",
            hovermode="x unified",
        )
        if len(df) > 24:
            layout_kw["xaxis"] = dict(
                rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(21, 168, 168, 0.08)"),
                type="date",
            )
        fig.update_layout(**layout_kw)
        self._apply_standard_layout(fig)
        return fig

    def _create_grouped_bar_chart(
        self, df: pd.DataFrame, categorical_cols: List[str], title: str
    ) -> go.Figure:
        """Create a grouped bar chart for multiple categorical columns."""
        # Use first two categorical columns
        if len(categorical_cols) >= 2:
            # Count occurrences
            grouped = df.groupby(categorical_cols[:2]).size().reset_index(name="count")
            fig = px.bar(
                grouped,
                x=categorical_cols[0],
                y="count",
                color=categorical_cols[1],
                title=title,
                barmode="group",
                color_discrete_sequence=self.COLOR_PALETTE,
            )
            self._apply_standard_layout(fig)
            return fig
        else:
            # Single categorical: value counts
            counts = df[categorical_cols[0]].value_counts().reset_index()
            counts.columns = [categorical_cols[0], "count"]
            fig = px.bar(
                counts,
                x=categorical_cols[0],
                y="count",
                title=title,
                color_discrete_sequence=[self.THEME_COLORS["teal"]],
            )
            self._apply_standard_layout(fig)
            return fig

    def _create_generic_chart(
        self, df: pd.DataFrame, col1: str, col2: str, title: str
    ) -> go.Figure:
        """Create a generic chart for any two columns."""
        # Try to determine the best representation
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(
            df[col2]
        ):
            return self._create_scatter_plot(df, col1, col2, title)
        else:
            # Treat first as categorical, second as value
            fig = px.bar(
                df,
                x=col1,
                y=col2,
                title=title,
                color_discrete_sequence=[self.THEME_COLORS["orange"]],
            )
            self._apply_standard_layout(fig)
            return fig

    def _create_table(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create a Plotly table for DataFrames with 4 or more columns."""
        # Prepare header
        header_values = list(df.columns)

        # Prepare cell values (transpose to get columns)
        cell_values = [df[col].tolist() for col in df.columns]

        # Create the table
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values,
                        fill_color=self.THEME_COLORS["navy"],
                        font=dict(color="white", size=12),
                        align="left",
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color=[
                            [
                                self.THEME_COLORS["cream"] if i % 2 == 0 else "white"
                                for i in range(len(df))
                            ]
                        ],
                        font=dict(color=self.THEME_COLORS["navy"], size=11),
                        align="left",
                    ),
                )
            ]
        )

        fig.update_layout(title=title, font={"color": self.THEME_COLORS["navy"]})

        return fig
