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

    feature_list = ['year', 'weekofyear', 'shop_id', 'item_id', 'category_id', 'item_price', 'isweekend', 'trend', 'seasonal', 'resid', 'item_cnt_day', 'date']
    raw_data = raw_data[feature_list]

    missing_data = deepcopy(raw_data)

    # 결측치 처리
    missing_data = raw_data.dropna(subset=['item_price', 'isweekend'])

    # 결측치 처리
    missing_data.loc[:, ['resid', 'trend']] = missing_data[['resid', 'trend']].fillna(missing_data.median(numeric_only=True))

    # 결측치 처리
    missing_data.loc[:, ['year']] = missing_data[['year']].fillna(method="ffill")

    outlier_data = deepcopy(missing_data)

    # 이상치 처리
    if is_training:
        for _, group in outlier_data.groupby(['shop_id', 'item_id']):
            q1 = group["item_cnt_day"].quantile(0.25)
            q3 = group["item_cnt_day"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_idx = group[(group["item_cnt_day"] < lower_bound) | (group["item_cnt_day"] > upper_bound)]["item_cnt_day"].index
            outlier_data = outlier_data.drop(outlier_idx, axis=0)

    # 이상치 처리
    if is_training:
        for _, group in outlier_data.groupby(['shop_id', 'item_id']):
            z_scores = (group["item_cnt_day"] - group["item_cnt_day"].mean()) / group["item_cnt_day"].std()
            norm_dist = group[np.abs(z_scores) <= 3]["item_cnt_day"]
            outlier_idx = group[np.abs(z_scores) > 3]["item_cnt_day"].index
            for i in outlier_idx:
                outlier_data["item_cnt_day"][i] = min(norm_dist)

    outlier_data['date'] = pd.to_datetime(outlier_data['date'])
    outlier_data.set_index(['date'], inplace=True)

    engineering_data = deepcopy(outlier_data)

    # trend, seasonal, resid 파생 변수 생성
    import statsmodels.api as sm
    seasonal_data = pd.DataFrame(columns=["trend", "seasonal", "resid"])
    for _, g_dataframe in outlier_data.groupby(['shop_id', 'item_id']):
        seasonal = sm.tsa.seasonal_decompose(
            g_dataframe["item_cnt_day"], model="additive", period=7, extrapolate_trend="freq"
        )
        seasonal_temp = pd.DataFrame([seasonal.trend, seasonal.seasonal, seasonal.resid])
        seasonal_temp = seasonal_temp.transpose()
        seasonal_data = pd.concat([seasonal_data, seasonal_temp], axis=0)
    engineering_data["trend"] = seasonal_data["trend"].tolist()
    engineering_data["seasonal"] = seasonal_data["seasonal"].tolist()
    engineering_data["resid"] = seasonal_data["resid"].tolist()

    # date 별 파생 변수 생성
    column_list = ['year', 'month']
    for column in column_list:
        engineering_data[column] = getattr(engineering_data.index, column)

    encoding_data = deepcopy(engineering_data)

    # onehot 인코딩
    from sklearn.preprocessing import OneHotEncoder
    column_list = ['item_id', 'year']
    if is_training:
        encoded_path = f"{out_dir}{os.sep}OneHotEncoder.pkl"
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(engineering_data[column_list])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_list), index=engineering_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)
        joblib.dump(encoder, encoded_path)
    else:
        encoded_path = f"{in_dir}{os.sep}OneHotEncoder.pkl"
        encoder = joblib.load(encoded_path)
        encoded_data = encoder.transform(engineering_data[column_list])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_list), index=engineering_data.index)
        encoding_data = pd.concat([encoding_data, encoded_data], axis=1)

    scaling_data = deepcopy(encoding_data)

    # 스케일링 처리
    from sklearn.preprocessing import MinMaxScaler
    column_list = ['trend', 'seasonal', 'resid']
    if is_training:
        scaling_path = f"{out_dir}{os.sep}MinMaxScaler.pkl"
        scaler = MinMaxScaler()
        scaling_data.loc[:, column_list] = scaler.fit_transform(encoding_data[column_list])
        joblib.dump(scaler, scaling_path)
    else:
        scaled_path = f"{in_dir}{os.sep}MinMaxScaler.pkl"
        scaler = joblib.load(scaled_path)
        scaling_data.loc[:, column_list] = scaler.transform(encoding_data[column_list])

    target_list = ['item_cnt_day']
    target_dataset = scaling_data[target_list]

    # 데이터 분할
    train_shape = 0.8
    val_shape = 0.1
    test_shape = 0.1
    total_shape = len(scaling_data)
    
    val_shape = val_shape + train_shape
    test_shape = test_shape + val_shape
    
    train_scale = int(total_shape * train_shape)
    val_scale = int(total_shape * val_shape)
    test_scale = int(total_shape * test_shape)
    
    X_train = scaling_data[:train_scale]
    X_val = scaling_data[train_scale : val_scale]
    X_test = scaling_data[val_scale : test_scale]
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

    # RandomForest 모듈
    if is_training:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        model = RandomForestRegressor(**compile_param)

        model.fit(**fit_param)
        joblib.dump(model, f'{out_dir}{os.sep}randomforest.pkl')
            
        # X_test를 사용할 경우와 X_validation을 사용할 경우
        if not X_test.empty: pred = model.predict(X_test)
        else: pred = model.predict(X_val)

        # 예측한 데이터 저장
        pred_df = pd.DataFrame(pred, columns=['predicted'], index=y_test.index)
        pred_df = pd.concat([X_test, y_test, pred_df], axis=1)
        pred_df['predicted'] = pred_df['predicted'].apply(lambda x: int(round(x)))
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv', index=False)
    
        # Feature Importance 계산
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X_train.columns
        sorted_feature_names = [feature_names[i] for i in indices]

        # Feature Importance 시각화 및 저장
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.plot(range(X_train.shape[1]), importances[indices], marker='o')
        plt.xticks(range(X_train.shape[1]), sorted_feature_names, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(f'{out_dir}{os.sep}feature_importance.png')
        
        # aiflow 값 반환
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
            'randomforest.pkl',
            'loss.png',
            'feature_importance.png',
        ]
        return accuracy_json, loss_json, output_param, file_list

    else:
        # 모델 불러오기
        model = joblib.load(training_model_path)
        pred = model.predict(X_test)
        
        # 예측한 값 반환
        pred_df = pd.DataFrame(
            data = pred,
            index = X_test.index.strftime("%Y-%m-%d"),
            columns = ['predict']
        )
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv')

        # aiflow 값 반환
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

