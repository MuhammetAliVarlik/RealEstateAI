import pandas as pd
from langchain_core.tools import tool
from typing import Optional, List, Union,Dict
import io
import numpy as np
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../data")
csv_path = os.path.join(MODEL_DIR, "HouseData.csv")

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


df=pd.read_csv(csv_path)

@tool
def view_dataframe(
    show_columns: Optional[List[str]] = None,
    filter_condition: Optional[str] = None,
    sort_by: Optional[Union[str, List[str]]] = None,
    ascending: bool = True,
    group_by: Optional[List[str]] = None,
    aggregations: Optional[Dict] = None,
    show_head: Optional[int] = None,
    show_tail: Optional[int] = None,
    show_info: bool = False,
    show_na: bool = False,
    show_describe: bool = False,) -> dict:
    """
    Performs flexible viewing, filtering, sorting, and analysis on a real estate dataset (Pandas DataFrame).

    This function can be used to inspect specific parts of the dataset, obtain summary statistics, and perform grouping and sorting operations.

    Args:
        show_columns (List[str], optional): List of column names to display.
        filter_condition (str, optional): Filtering condition in Pandas `query()` format.
        sort_by (Union[str, List[str]], optional): Column name(s) to sort by.
        ascending (bool, optional): Sort order. True for ascending, False for descending.
        group_by (List[str], optional): Column names to group by.
        aggregations (Dict[str, str], optional): Aggregation operations to apply after grouping, e.g. {"price": "mean"}.
        show_head (int, optional): Number of rows to show from the top.
        show_tail (int, optional): Number of rows to show from the bottom.
        show_info (bool, optional): Returns the output of `df.info()`.
        show_na (bool, optional): Lists rows containing missing values.
        show_describe (bool, optional): Returns summary statistics using `df.describe()`.

    Returns:
        dict: A dictionary containing the results of the requested operations.  
            In case of an error, returns {"error": "Description"}.

    Examples:
        1. List 3+1 homes in Kadıköy priced under 300,000 TRY:
        ```python
        view_dataframe(
            show_columns=["district", "price", "NumberOfRooms", "address"],
            filter_condition='price < 300000 and NumberOfRooms == "3+1" and district == "kadikoy"',
            show_head=3
        )
        ```

        2. Check for records with missing values across all listings:
        ```python
        view_dataframe(show_na=True)
        ```

        3. List average prices by district, sorted ascending:
        ```python
        view_dataframe(
            group_by=["district"],
            aggregations={"price": "mean"},
            sort_by="price",
            ascending=True
        )
        ```

        4. Show basic listing info (first 3 records):
        ```python
        view_dataframe(
            show_columns=["district", "price", "address"],
            show_head=3
        )
        ```

    Note:
        This function is for viewing and analysis purposes only; it does not modify the data.
    """


    try:
        results = {}
        df_view = df.copy()
        print(df_view.head())

        if show_head is not None and show_head > 3:
            show_head = 3

        if show_tail is not None and show_tail > 3:
            show_tail = 3

        if show_info:
            import io
            buffer = io.StringIO()
            df_view.info(buf=buffer)
            results['info'] = buffer.getvalue()

        if show_na:
            results['na_rows'] = df_view[df_view.isna().any(axis=1)].to_dict(orient='records')

        if show_describe:
            results['describe'] = df_view.describe(include='all').to_dict()

        if show_columns:
            df_view = df_view[show_columns]

        if filter_condition:
            df_view = df_view.query(filter_condition)

        if sort_by:
            df_view = df_view.sort_values(by=sort_by, ascending=ascending)

        if group_by and aggregations:
            df_view = df_view.groupby(group_by).agg(aggregations).reset_index()

        if show_head:
            results['head'] = df_view.head(show_head).to_dict(orient='records')
        elif show_tail:
            results['tail'] = df_view.tail(show_tail).to_dict(orient='records')
        else:
            results['data'] = df_view.to_dict(orient='records')

        return sanitize_for_json(results)
    
    except Exception as e:
        return {"error": str(e)}
