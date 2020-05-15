from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from utils.utils import getUserAvgSeq,getUserMaxSeq
import numpy as np
import pandas as pd
import lightgbm as lgb


def base_train(x_train, y_train, x_test, y_test, params, cate_cols=None,job="classification"):
    # create dataset for lightgbm
    if not cate_cols:
        cate_cols = 'auto'
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cate_cols)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, categorical_feature=cate_cols)
    print('Starting training...')

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10)
    y_pred_prob = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    if job == 'classification':
        res_auc = roc_auc_score(y_test, y_pred_prob)
    return gbm

def base_predict (gbm_age,gbm_gender,df,weight,isSave=False,method="avg") :
    df_res = df[['user_id']]
    seq_test = df["creative_id_seq"].tolist()
    
    if method == "avg":
        test_data = getUserAvgSeq(seq_test, weight)
    elif method == "max":
        test_data = getUserMaxSeq(seq_test, weight)
    else:
        test_avg = getUserAvgSeq(seq_test, weight)
        test_max = getUserMaxSeq(seq_test, weight)
        test_data = [np.append(x,y) for x,y in zip(data_avg,data_max)]
    dddf = pd.DataFrame(np.array(test_data))
    df_res['predicted_gender'] = gbm_gender.predict(dddf)
    df_res.loc[df_res['predicted_gender'] >= 0.5, 'predicted_gender'] = 2
    df_res.loc[df_res['predicted_gender'] < 0.5, 'predicted_gender'] = 1
    df_res["predicted_gender"] = df_res["predicted_gender"].map(lambda x: int(x))
    df_res['predicted_age'] = np.argmax(gbm_age.predict(dddf), axis=1) + 1
    if isSave:
        df_res.to_csv("submission.csv", index=False)
    return df_res



