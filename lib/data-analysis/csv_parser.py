import pandas as pd
from typing import Callable, TypeVar
T = TypeVar("T")

'''filters excel data by removing data with a specific column value, based on a predicate evaluating the input type '''
def filter_csv(file, col_name, condition: Callable[[T], bool]):
    df = pd.read_csv(file)
    filtered_df = pd.DataFrame()
    for i in range(len(df)):
        if condition(df.iloc[i][col_name]):
            copy = df.iloc[i]
            copied_row_df = pd.DataFrame([copy])
            filtered_df = pd.concat([filtered_df, copied_row_df], ignore_index = True)
    return filtered_df



# example (finds names with letter a in them)
func = lambda col: "TN" in col
yearFunc = lambda year: year >= 2010 and year <= 2023

with open('./temp_data.csv', 'r') as f:
    df = filter_csv(f, 'Year', yearFunc)
    print('df length is: ' + str(len(df)))
    df.to_csv('./range_temp_data.csv')
