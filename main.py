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
    csv_file_name = f"{in_dir}{os.sep}fraud2.csv"
    raw_data = pd.read_csv(csv_file_name) 
    feature_list = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'isFraud']
    raw_data = raw_data[feature_list]

    encoding_data = deepcopy(raw_data)
    encoding_data = pd.get_dummies(encoding_data, columns=['type', 'nameOrig', 'nameDest'], drop_first = True)

    target_list = ['isFraud']
    target_dataset = encoding_data[target_list]
    encoding_data.drop(columns=target_list, inplace=True)

    scaling_data = deepcopy(encoding_data)
    from sklearn.preprocessing import StandardScaler
    column_list = encoding_data.columns
    if is_training:
        scaling_path = f"{out_dir}{os.sep}StandardScaler.pkl"
        scaler = StandardScaler()
        scaling_data.loc[:, column_list] = scaler.fit_transform(encoding_data[column_list])
        joblib.dump(scaler, scaling_path)
    else:
        scaled_path = f"{in_dir}{os.sep}StandardScaler.pkl"
        scaler = joblib.load(scaled_path)
        scaling_data.loc[:, column_list] = scaler.transform(encoding_data[column_list])

    from sklearn.model_selection import train_test_split
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        scaling_data, 
        target_dataset, 
        test_size=(0.15 + 0.15), 
        stratify=target_dataset
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, 
        y_val_test, 
        test_size=(0.15 / (0.15 + 0.15)), 
        stratify=y_val_test
    )

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
        from xgboost.sklearn import XGBClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, f1_score
        model = XGBClassifier(**compile_param)

        model.fit(**fit_param)
        joblib.dump(model, f'{out_dir}{os.sep}xgboost.pkl')
            
        # X_test를 사용할 경우와 X_validation을 사용할 경우
        if not X_test.empty:
            pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)[:, 1]
        else: 
            pred = model.predict(X_val)
            pred_proba = model.predict_proba(X_val)[:, 1]

        # 예측한 데이터 저장
        pred_df = pd.DataFrame(pred, columns=['predicted'])
        pred_df = pd.concat([X_test, y_test, pred_df], axis=1)
        pred_df['predicted'] = pred_df['predicted']
        pred_df.to_csv(f'{out_dir}{os.sep}output.csv', index=False)

        acc_score = accuracy_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred_proba)
        f1 = f1_score(y_test, pred)
    
        # Feature Importance 계산
        xgboost.plot_importance(model)
        plt.savefig(f'{out_dir}{os.sep}feature_importance.png')

        # 시각화
        # Confusion Matrix
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, 
            pred,
            values_format="d", 
            cmap="Blues"
        )
        plt.savefig(f'{out_dir}{os.sep}confusion_matrix.png')

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
        auc = round(roc_auc, 3)
        plt.figure(figsize = (15, 8))
        plt.plot(fpr, tpr, label=f"XGBoost Classifier, AUC={auc}", color="b")
        plt.plot([0, 1], [0, 1], color = "g")
        plt.title("ROC Curve")
        plt.legend(loc="upper right")
        plt.savefig(f"{out_dir}{os.sep}roc_curve.png")
    
        # aiflow 값 반환
        accuracy_json: dict = {
            'accuracy_score': acc_score
            }
        loss_json: dict = {
            'roc_auc_score': roc_auc, 
            'f1_score': f1
        }
        output_param: dict = {
            'accuracy_score': acc_score,
            'roc_auc_score': roc_auc, 
            'f1_score': f1
        }
        file_list: list = [
			'output.csv',
            'xgboost.pkl',
            'confusion_matrix.png',
            'roc_curve.png'
        ]
        return accuracy_json, loss_json, output_param, file_list

    else:
        # 모델 불러오기
        model = joblib.load(training_model_path)
        pred = model.predict(X_test)
        
        # 예측한 값 반환
        pred_df = pd.DataFrame(
            data = pred,
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
