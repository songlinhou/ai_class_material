import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np



def get_ckd_dataset(normalized=True,test_ratio=0.2):
    ckd_df = pd.read_csv('ckd.csv')
    numerical_columns = set(['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc'])
    nominal_columns = set(['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class'])
    all_columns = set(ckd_df.columns)
    numeric_ckd_names = list(all_columns.intersection(numerical_columns))
    nominal_ckd_names = list(all_columns.intersection(nominal_columns))

    for col_name in numeric_ckd_names:
        col = ckd_df[col_name]
        col.fillna(np.mean(col),inplace=True)

    for col_name in nominal_ckd_names:
        col = ckd_df[col_name]
        col = col.apply(str)

    # to replace ? in each column with the random values following the value distribution of that column
    for nominal_name in nominal_ckd_names:
        norminal_col =  ckd_df[nominal_name]
        counts = norminal_col.value_counts()
        counts_keys = counts.keys()
        valid_value_list = [k for k in counts_keys if k != '?']
        valid_value_count = np.array([counts[k] for k in counts_keys if k != '?'])
        valid_value_prob = valid_value_count * 1.0 / np.sum(valid_value_count)
        num_of_unknow_records = len(norminal_col[norminal_col == '?'])
        norminal_col[norminal_col == '?'] = np.random.choice(valid_value_list,num_of_unknow_records,replace=True,p=valid_value_prob)

    nominal_data = ckd_df[nominal_ckd_names].drop('class',axis=1)
    true_categories = ckd_df['class']
    true_categories.replace('ckd',1,inplace=True)
    true_categories.replace('notckd',0,inplace=True)
    nominal_data_encoded = pd.get_dummies(nominal_data,drop_first=True)
    numeric_data = ckd_df[numeric_ckd_names]

    X_ckd = pd.concat([numeric_data,nominal_data_encoded],axis=1)
    X_ckd = X_ckd.astype(float)
    y_ckd = true_categories.values
    y_ckd = y_ckd.reshape((1,len(y_ckd)))
    y_ckd = y_ckd.astype(float)
    
    if not normalized:
      X_train, X_test, y_train, y_test = train_test_split(X_ckd, y_ckd.T, test_size=test_ratio, random_state=100)
      return X_train, X_test, y_train, y_test
    
    scaler = StandardScaler()
    X_ckd_normalized = scaler.fit_transform(X_ckd)
    X_train, X_test, y_train, y_test = train_test_split(X_ckd_normalized, y_ckd.T, test_size=test_ratio, random_state=100)
    
    return X_train, X_test, y_train, y_test

