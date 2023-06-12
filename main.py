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

    raw_data.set_index(['date'], inplace=True)

    feature_list = ['year', 'weekofyear', 'shop_id', 'item_id', 'category_id', 'item_price', 'isweekend', 'trend', 'seasonal', 'resid']
    target_list = ['item_cnt_day']
    feature_dataset = raw_data[feature_list]
    target_dataset = raw_data[target_list]

    missing_data = deepcopy(feature_dataset)

    missing_data = feature_dataset.dropna(subset=['item_price', 'isweekend'])

    missing_data.loc[:, ['resid', 'trend']] = feature_dataset[['resid', 'trend']].fillna(feature_dataset.median(numeric_only=True))

    missing_data.loc[:, ['year']] = feature_dataset[['year']].fillna(method="ffill")

    scaling_data = deepcopy(missing_data)

    from sklearn.preprocessing import MinMaxScaler
    column_names = ['trend', 'seasonal', 'resid']
    if is_training:
        scaling_path = f"{out_dir}{os.sep}MinMaxScaler.pkl"
        scaler = MinMaxScaler()
        scaling_data.loc[:, column_names] = scaler.fit_transform(missing_data[column_names])
        joblib.dump(scaler, scaling_path)
    else:
        scaled_path = f"{in_dir}{os.sep}MinMaxScaler.pkl"
        scaler = joblib.load(scaled_path)
        scaling_data.loc[:, column_names] = scaler.transform(missing_data[column_names])

    encoding_data = deepcopy(scaling_data)

    from sklearn.preprocessing import OneHotEncoder
    column_names = ['item_id', 'year']
    if is_training:
        encoded_path = f"{out_dir}{os.sep}OneHotEncoder.pkl"
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(scaling_data[column_names])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_names), index=scaling_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)
        joblib.dump(encoder, encoded_path)
    else:
        encoded_path = f"{in_dir}{os.sep}OneHotEncoder.pkl"
        encoder = joblib.load(encoded_path)
        encoded_data = encoder.transform(scaling_data[column_names])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_names), index=scaling_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)

    train_shape = 0.8
    val_shape = 0.1
    test_shape = 0.1
    total_shape = len(encoding_data)
    
    val_shape = val_shape + train_shape
    test_shape = test_shape + val_shape
    
    train_scale = int(total_shape * train_shape)
    val_scale = int(total_shape * val_shape)
    test_scale = int(total_shape * test_shape)
    
    X_train = encoding_data[:train_scale]
    X_val = encoding_data[train_scale : val_scale]
    X_test = encoding_data[val_scale : test_scale]
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
    fit_param['eval_set'] = (X_val, y_val)



    # LightGBM 모듈
    if is_training:
        import lightgbm
        from lightgbm.sklearn import LGBMRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        model = LGBMRegressor(**compile_param)


        model.fit(**fit_param)
        joblib.dump(model, f'{out_dir}{os.sep}lightgbm.pkl')
            
        if not X_test.empty:
            pred = model.predict(X_test)
        else:
            pred = model.predict(X_val)
        pred_df = pd.DataFrame(pred, columns=['predicted'], index=y_test.index)
        pred_df = pd.concat([X_test, y_test, pred_df], axis=1)
        pred_df['predicted'] = pred_df['predicted'].apply(lambda x: int(round(x)))
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv', index=False)
    
        lightgbm.plot_importance(model)
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
            'lightgbm.pkl',
            'loss.png',
            'feature_importance.png',
            ]
        return accuracy_json, loss_json, output_param, file_list
    else:
        model = joblib.load(f'{in_dir}{os.sep}lightgbm.pkl')
        pred = model.predict(X_test)
        pred_df = pd.DataFrame(
            data = pred,
            index = X_test.index,
            columns = ['predict']
        )
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv', index=False)
    
    


if __name__ == "__main__":
    in_dir = "."
    out_dir = "."
    result = to_do_logic(
        in_dir = in_dir,
        out_dir = out_dir,
        is_training = True,
    )
    print(result)

