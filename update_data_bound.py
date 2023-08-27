import pandas as pd
import json
# 讀取 CSV 文件

pd.set_option('display.max_columns', None)


def filter_data(df, value):
    filtered_df = df[df['Group'] == value]
    return filtered_df


def find_data_max(data):
    max_values = data.max()
    return max_values


def find_data_min(data):
    min_values = data.max()
    return min_values


def find_data_mean(data):
    mean_values = data.mean()
    return mean_values


def change_column_name(df):
    d_path = {
        '2X positive culture': 'two_positive_culture',
        'APTT': 'APTT',
        'ASA_2': 'ASA_2',
        'Age': 'Age',
        'HGB': 'Hb',
        'P.T': 'P_T',
        'PLATELET': 'PLATELET',
        'Positive Histology': 'Positive_Histology',
        'Primary, Revision\nnative hip': 'Surgery',
        'Pulurence': 'Purulence',  # 膿:Purulence
        # 'Rheumatoid Arthritis/collagen': 'Rheumatoid_Arthritis/collagen',
        'Segment (%)': 'Segment',
        'Serum CRP': 'Serum_CRP',
        'Serum ESR': 'Serum_ESR',
        'Serum WBC ': 'Serum_WBC_',
        'Single Positive culture': 'Single_Positive_culture',
        'Synovial WBC': 'Synovial_WBC',
        'Synovial_PMN': 'Synovial_PMN',
        'Total CCI': 'Total_CCI',
        'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
    }
    val_df = df.rename(columns=d_path)
    return val_df


def series_to_df(_series):
    series_values = _series.values
    series_index = _series.index
    df = pd.DataFrame([series_values], columns=series_index)
    return df


def df_to_dict(df):
    data_dict = df.iloc[0].to_dict()
    return data_dict


def dict_to_json(data_dict, keys_to_select, value):
    selected_data = {k: data_dict[k] for k in keys_to_select}
    if value == 0:
        with open('auto_boundary/nonICM_bound_N.json', 'w') as json_file:
            json.dump(selected_data, json_file, indent=4)
    elif value == 1:
        with open('auto_boundary/nonICM_bound_I.json', 'w') as json_file:
            json.dump(selected_data, json_file, indent=4)
    return selected_data


def count_nonICM_data_bound(train_index, value):
    data = pd.read_csv('PJI_Dataset/PJI_all.csv', encoding='utf-8')
    data = data.iloc[train_index, :]
    filtered_data = filter_data(data, value)
    mean_values = find_data_mean(filtered_data)
    df = series_to_df(mean_values)
    df = change_column_name(df)
    data_dict = df_to_dict(df)
    keys_to_nonICM_select = ["Age", "Segment", "Hb", "PLATELET", "Serum_WBC_", "P_T",
                             "APTT", "Total_CCI", "Total_Elixhauser_Groups_per_record", "Surgery", "ASA_2"]
    nonICM_selected_data = dict_to_json(
        data_dict, keys_to_nonICM_select, value)
    print(nonICM_selected_data)
    return nonICM_selected_data


keys_to_ICM_select = ["two_positive_culture", "Serum_CRP", "Serum_ESR", "Synovial_WBC",
                      "Single_Positive_culture", "Synovial_PMN", "Positive_Histology", "Purulence"]
keys_to_nonICM_select = ["Age", "Segment", "Hb", "PLATELET", "Serum_WBC_", "P_T",
                         "APTT", "Total_CCI", "Total_Elixhauser_Groups_per_record", "Surgery", "ASA_2"]
