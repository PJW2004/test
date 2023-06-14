import os
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy

def to_do_logic(
    input_param = None,
    ex_model_param = None,
    in_dir = None,
    out_dir = None,
    ex_out_dir = None,
    is_training = True,
    training_model_path = None,
):


    csv_file_name = f"{in_dir}{os.sep}sample_data_all.csv"
    raw_data = pd.read_csv(csv_file_name)

    missing_data = deepcopy(raw_data)

    missing_data = raw_data.dropna(subset=)

    missing_data.loc[:, ] = missing_data[].fillna(missing_data.median(numeric_only=True))

    missing_data.loc[:, ] = missing_data[].fillna(method="ffill")

    missing_data.set_index(['date'], inplace=True)
    missing_data.index = pd.to_datetime(missing_data.index)

    encoding_data = deepcopy(missing_data)

    from sklearn.preprocessing import OneHotEncoder
    column_list = 
    if is_training:
        encoded_path = f"{out_dir}{os.sep}OneHotEncoder.pkl"
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(missing_data[column_list])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_list), index=missing_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)
        joblib.dump(encoder, encoded_path)
    else:
        encoded_path = f"{in_dir}{os.sep}OneHotEncoder.pkl"
        encoder = joblib.load(encoded_path)
        encoded_data = encoder.transform(missing_data[column_list])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_list), index=missing_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)

    scaling_data = deepcopy(encoding_data)

    from sklearn.preprocessing import MinMaxScaler
    column_list = 
    if is_training:
        scaling_path = f"{out_dir}{os.sep}MinMaxScaler.pkl"
        scaler = MinMaxScaler()
        scaling_data.loc[:, column_list] = scaler.fit_transform(encoding_data[column_list])
        joblib.dump(scaler, scaling_path)
    else:
        scaled_path = f"{in_dir}{os.sep}MinMaxScaler.pkl"
        scaler = joblib.load(scaled_path)
        scaling_data.loc[:, column_list] = scaler.transform(encoding_data[column_list])

    feature_list = ['year', 'weekofyear', 'shop_id', 'item_id', 'category_id', 'item_price', 'isweekend', 'trend', 'seasonal', 'resid']
    target_list = ['item_cnt_day']
    feature_dataset = scaling_data[feature_list]
    target_dataset = scaling_data[target_list]

    train_shape = 0.8
    val_shape = 0.1
    test_shape = 0.1
    total_shape = len(feature_dataset)
    
    val_shape = val_shape + train_shape
    test_shape = test_shape + val_shape
    
    train_scale = int(total_shape * train_shape)
    val_scale = int(total_shape * val_shape)
    test_scale = int(total_shape * test_shape)
    
    X_train = feature_dataset[:train_scale]
    X_val = feature_dataset[train_scale : val_scale]
    X_test = feature_dataset[val_scale : test_scale]
    y_train = target_dataset[:train_scale]
    y_val = target_dataset[train_scale : val_scale]
    y_test = target_dataset[val_scale : test_scale]

    # 파라미터 로드
    with open(f'{in_dir}{os.sep}input_param.json', encoding='utf-8') as param_file:
        params = json.load(param_file)
    
    compile_param = params.get('compile_opt')
    fit_param = params.get('fitting_opt')
    fit_param['X'] = X_train
    fit_param['y'] = y_train
    fit_param['eval_set'] = [(X_val, y_val)]

    # XGBoost 모듈
    if is_training:
        import xgboost
        from xgboost.sklearn import XGBRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        model = XGBRegressor(**compile_param)

        model.fit(**fit_param)
        joblib.dump(model, f'{out_dir}{os.sep}xgboost.pkl')
            
        if not X_test.empty:
            pred = model.predict(X_test)
        else:
            pred = model.predict(X_val)
        pred_df = pd.DataFrame(pred, columns=['predicted'], index=y_test.index)
        pred_df = pd.concat([X_test, y_test, pred_df], axis=1)
        pred_df['predicted'] = pred_df['predicted'].apply(lambda x: int(round(x)))
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv', index=False)
    
        xgboost.plot_importance(model)
        plt.savefig(f'{out_dir}{os.sep}feature_importance.png')
    
        accuracy_json: dict = {'r_squared': 1-(1-r2_score(y_test, pred))*((len(X_test)-1) / (len(X_test)-X_test.shape[1]-1))}
        loss_json: dict = {
            'mse': mean_squared_error(y_test, pred), 
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred)
        }
        output_param: dict = {
            'r_squared': r2_score(y_test, pred),
            'mse': mean_squared_error(y_test, pred),
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred)
        }
        file_list: list = [
			'output.csv',
            'xgboost.pkl',
            'loss.png',
            'feature_importance.png',
        ]
        return accuracy_json, loss_json, output_param, file_list
    else:
        model = joblib.load(training_model_path)
        pred = model.predict(X_test)
        pred_df = pd.DataFrame(
            data = pred,
            index = X_test.index,
            columns = ['predict']
        )
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv')
        output_param = pred_df.to_dict()
        file_list: list = [
            'output.csv',
        ]
        return None, None, output_param, file_list
    
    


if __name__ == "__main__":
    in_dir = "."
    out_dir = "."
    training_model_path = "model_name"
    result = to_do_logic(
        in_dir = in_dir,
        out_dir = out_dir,
        is_training = True,
        training_model_path = training_model_path
    )
    print(result)

