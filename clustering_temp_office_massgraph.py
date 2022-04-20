import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
import collections
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model


def read_csv(filedir):
    final_result = []
    f = open(filedir, 'r', encoding='ANSI')
    rdr = csv.reader(f)
    for line in rdr:
        final_result.append(line)
    f.close()
    return final_result

def write_csv(filedir, content):
    f = open(filedir, 'w', newline='')
    wr = csv.writer(f)
    for item in content:
        wr.writerow(item)
    f.close()


def clustering(square_array):
    df = pd.DataFrame(square_array, columns=['name', 'elec'])
    print(df)
    df_npy = np.array(df)
    name_npy = np.array([np.array(x) for x in df_npy[:, 0]])  # 노선의 이름을 저장하는 데이터 변수입니다.
    feature_npy = np.array([np.array(x) for x in df_npy[:, 1]])  # 노선의 시계열 데이터를 저장하는 데이터 변수입니다.

    def normalization_axis(feature):  # 시계열 데이터의 효율적인 클러스터링을 위해 normalization을 진행합니다.
        sum_npy = np.array([x / np.sum(x) for x in feature])
        return sum_npy


    feature_norm_npy = normalization_axis(feature_npy)
    feature_npy = np.array([np.array(x) for x in df_npy[:, 1]])
    print(feature_npy)
    print(sum(feature_norm_npy[0]))

    route_total = str(feature_norm_npy[~np.isnan(feature_norm_npy).any(axis=1)].shape[0])
    print("Item number : " + str(feature_norm_npy.shape[0]))
    print("Item number with NaN removed : " + route_total)
    feature_norm = feature_norm_npy[~np.isnan(feature_norm_npy).any(axis=1)]
    feature = feature_npy[~np.isnan(feature_norm_npy).any(axis=1)]
    name = name_npy[~np.isnan(feature_norm_npy).any(axis=1)]

    #하이퍼패러미터 테스트
    objective_function = []
    for i in range(1, 10):
        clustering= TimeSeriesKMeans(n_clusters=i, init='k-means++')
        clustering.fit(feature_norm_npy)
        objective_function.append(clustering.inertia_)
    print(objective_function)
    plt.plot(range(1, 10), objective_function)
    plt.title('elbow method-hyperparameter')
    plt.xlabel('# of clusteres')
    plt.ylabel('objective_function')
    plt.show()

    n_cluster = 4  # 결정된 수 사용
    km = TimeSeriesKMeans(n_clusters=n_cluster, metric="euclidean", max_iter=150).fit(feature_norm_npy)  # normalization 시계열 데이터를 활용해 euclidean 기반 클러스터링을 진행합니다.
    rlt_tsm = km.predict(feature_norm_npy)

    print(rlt_tsm)
    temp_list_shell = []
    for item in rlt_tsm:
        temp_item = [item]
        temp_list_shell.append(temp_item)
    print(temp_list_shell)
    write_csv('./cluster_office/cluster_result_for_temp.csv', temp_list_shell)

    # 클러스터별 분포도를 시각화하기 위해 전처리 작업입니다.
    labels = []
    sizes = []
    for i in range(n_cluster):
        labels.append("cluster_" + str(i))
        sizes.append(collections.Counter(rlt_tsm)[i])

    plt.figure(figsize=(10, 5))
    plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    plt.title("Cluster Distribution (number of data:" + str(len(df['elec'])) + ")", position=(0.5, 1.2), fontsize=20)
    plt.show()
    plt.savefig('./cluster_office/Cluster_distribution.png')

    pca = PCA(n_components=2)  # pca를 진행해 클러스터가 얼마나 잘됐는지 검토합니다.
    rlt_pca = pca.fit_transform(feature_norm_npy[~np.isnan(feature_norm_npy).any(axis=1)])
    for i in range(n_cluster):
        label_name = "cluster " + str(i)
        plt.scatter(rlt_pca[[rlt_tsm == i]][:, 0], rlt_pca[[rlt_tsm == i]][:, 1], label=label_name)
    plt.legend()
    #plt.show()
    plt.savefig('./cluster_office/Cluster_scatterplot.png')

    def show_cluster_dist(num_cluster, num_sample=50):
        plt.figure(figsize=(12, 4))
        size = sizes[num_cluster]
        if (size > num_sample):
            size = num_sample
        for i in range(size):
            name = "cluster" + str(i)
            plt.plot(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                     feature_norm[[rlt_tsm == num_cluster]][i])
        #plt.axvline(5.8, color='red', label='Test', linewidth=1)
        #plt.axvspan(5.8, 9.5, color='red', alpha=0.2)
        plt.xlabel("month")
        plt.ylabel("EUI")
        plt.title("Usage pattern of cluster" + str(num_cluster), fontsize=30)
        plt.legend()
        plt.savefig('./cluster_office/Cluster'+str(num_cluster)+'.png')
        #plt.show()

    for i in range(n_cluster):
        show_cluster_dist(i)

    def hyperparam_tuning(path_train, path_test):
        train = pd.read_csv(path_train)
        train = train.drop(["index"], axis=1)
        train.fillna("NAN", inplace=True)

        test = pd.read_csv(path_test)
        test = test.drop(["index"], axis=1)
        test.fillna("NAN", inplace=True)

        train_ohe = pd.get_dummies(train)
        test_ohe = pd.get_dummies(test)

        X = train_ohe.drop(["credit"], axis=1)
        y = train["credit"]
        X_test = test_ohe.copy()

        def objective(trial: Trial) -> float:
            params_lgb = {
                "random_state": 42,
                "verbosity": -1,
                "learning_rate": 0.05,
                "n_estimators": 10000,
                "objective": "multiclass",
                "metric": "multi_logloss",
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "subsample": trial.suggest_float("subsample", 0.3, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "max_bin": trial.suggest_int("max_bin", 200, 500),
            }

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

            model = LGBMClassifier(**params_lgb)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=False,
            )

            lgb_pred = model.predict_proba(X_valid)
            log_score = log_loss(y_valid, lgb_pred)

            return log_score

        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name="lgbm_parameter_opt",
            direction="minimize",
            sampler=sampler,
        )
        study.optimize(objective, n_trials=10)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

        #시각화
        optuna.visualization.plot_optimization_history(study)
        #파라미터들과의 관계
        optuna.visualization.plot_parallel_coordinate(study)
        # 각 파라미터들의 상관관계
        optuna.visualization.plot_contour(
            study,
            params=[
                "max_depth",
                "num_leaves",
                "colsample_bytree",
                "subsample",
                "subsample_freq",
                "min_child_samples",
                "max_bin",
            ],
        )
        # 하이퍼파라미터 중요도
        optuna.visualization.plot_param_importances(study)

def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

#=========================================================================================================
#여기부터는 신경망 관련된 것들

#plcc(피어슨 상관계수)에 대한 loss함수 추가 - Ref: https://bskyvision.com/741
from tensorflow.keras import backend as K

def plcc_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym)))) + 1e-12
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def plcc_metric(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym)))) + 1e-12
    return r_num / r_den

def createModel():
    #직접 모델을 작성
    #쓸 일은 없을 것 같지만, 혹시 몰라서 작성해 둠.
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', plcc_metric])
    return model

def fitModel(model, datafile_name):
    #csv파일을 로드해 모델을 학습함
    #csv파일의 내부데이터 순서는 [['motionsensor_1', 'motionsensor_2', 'decibel_living', 'decibel_study', 'decibel_table']] 여야 함.
    data = pd.read_csv('./'+str(datafile_name)+'.csv')
    x_train = data[['motionsensor_1', 'motionsensor_2', 'decibel_living', 'decibel_study', 'decibel_table']]
    y_train = data['switch']
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=10)
    return model

import time

#중복온도값 처리 코드(사용X)
'''data = pd.read_csv('temp_daily_data_for_fill.csv')
print(data)
data_mod = data.groupby('Temp')
result = data_mod.mean()
#.to_excel('temp_daily_data_result.xlsx')'''

#보간 코드(사용X)
'''for col_index in result.keys():
    temp_df = result[[col_index]]
    temp_df = temp_df.interpolate(method='linear')
    #print(temp_df)
    result[col_index] = temp_df

result.to_excel('temp_daily_data_result2.xlsx')'''

#데이터를 가지고 다항회귀 곡선을 구함
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#그래프 무지성 뽑기
'''data = pd.read_csv('temp_daily_data2.csv')
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = data.keys()
print(k)

for i in range(16):
    #ax = plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=0.4)
    X = data[k[0]].to_numpy()
    y = data[k[i+1]].to_numpy()
    temp_x = []
    for item in X:
        temp_x.append([item])
    X = temp_x
    temp_y = []
    for item in y:
        temp_y.append([item])
    y = temp_y
    plt.scatter(X, y)
    plt.title("Degree"+str(i))
    plt.show()
plt.show()

for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.subplots_adjust(hspace=0.4)
    X = data[k[0]].to_numpy()
    y = data[k[i+17]].to_numpy()
    temp_x = []
    for item in X:
        temp_x.append([item])
    X = temp_x
    temp_y = []
    for item in y:
        temp_y.append([item])
    y = temp_y
    plt.scatter(X, y)
    plt.title("Degree"+str(i+16))

plt.show()'''




#회귀식
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#data = pd.read_csv('temp_day.csv')
data = pd.read_csv('./cluster_office/day_temp_graph_scatter.csv')
degrees = [-7.8, -6.2, -5.6, -5.3, -5.2, -4.9, -4.8, -3.7, -3.5, -3.2, -3, -2.9, -2.6, -2.5, -2.4, -2, -1.8, -1.7, -1.5, -1.4, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, 0, 0.2, 0.3, 0.4, 0.5, 0.9, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.2, 2.3, 2.4, 2.5, 2.9, 3, 3.3, 3.4, 3.5, 3.6, 3.8, 4.1, 4.4, 4.6, 4.7, 4.8, 4.9, 5, 5.2, 5.3, 5.5, 5.9, 6.1, 6.3, 6.4, 6.5, 6.7, 6.9, 7, 7.1, 7.3, 7.5, 7.6, 7.8, 8, 8.1, 8.2, 8.3, 8.5, 8.7, 9, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8, 10, 10.1, 10.2, 10.4, 10.6, 11, 11.2, 11.3, 11.4, 11.6, 11.7, 11.9, 12.3, 12.4, 12.7, 12.8, 12.9, 13, 13.4, 13.6, 13.7, 13.8, 13.9, 14.1, 14.2, 14.3, 14.4, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.5, 16.2, 16.6, 16.7, 16.9, 17, 17.1, 17.6, 17.7, 17.8, 18, 18.1, 18.4, 18.6, 18.7, 18.8, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 20, 20.1, 20.3, 20.5, 20.6, 20.8, 20.9, 21, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 22, 22.4, 22.6, 22.7, 22.8, 22.9, 23, 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 23.8, 23.9, 24, 24.1, 24.2, 24.3, 24.4, 24.5, 24.7, 24.9, 25, 25.1, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9, 26, 26.1, 26.3, 26.4, 26.5, 26.7, 26.8, 26.9, 27, 27.2, 27.3, 27.4, 27.9, 28, 28.1, 28.3, 28.4, 28.5, 28.8, 29.2, 29.4, 29.8, 29.9, 30.3, 30.7, 31.4, 31.7]
k = data.keys()
print(k)

#단일출력용
for i in range(len(k)):
    plt.subplots_adjust(hspace=0.4)
    X = data[k[0]].to_numpy()
    y = data[k[i + 1]].to_numpy()
    temp_x = []
    for item in X:
        temp_x.append([item])
    X = temp_x
    temp_y = []
    for item in y:
        temp_y.append([item])
    y = temp_y

    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    X_new = np.linspace(-8, 32, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    if i == 0:
        plt.plot(X, y, "b.")
        plt.plot(X_new, y_new, "r-", linewidth=2, label="cluster_0")
    if i == 1:
        plt.plot(X, y, "g.")
        plt.plot(X_new, y_new, "y-", linewidth=2, label="cluster_1")
        plt.title(k[i + 1])
        #plt.xlabel("$x_1$", fontsize=18)
        #plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.legend(loc="upper left", fontsize=14)
        plt.show()

'''
#매스그래프 용
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=0.4)
    X = data[k[0]].to_numpy()
    y = data[k[i + 1]].to_numpy()
    temp_x = []
    for item in X:
        temp_x.append([item])
    X = temp_x
    temp_y = []
    for item in y:
        temp_y.append([item])
    y = temp_y

    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    X_new = np.linspace(-8, 32, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    #plt.legend(loc="upper left", fontsize=14)
    plt.title(k[i + 1])

plt.show()

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=0.4)
    X = data[k[0]].to_numpy()
    y = data[k[i + 17]].to_numpy()
    temp_x = []
    for item in X:
        temp_x.append([item])
    X = temp_x
    temp_y = []
    for item in y:
        temp_y.append([item])
    y = temp_y

    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    X_new = np.linspace(-8, 32, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    #plt.legend(loc="upper left", fontsize=14)
    plt.title(k[i + 17])

plt.show()
'''



def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range (1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "validation set")
    plt.xlabel("size of train set")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

# 개별 degree 별로 Polynomial 변환합니다.
for i in degrees:
    poly_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    lin_reg.fit(X_poly, y)

    X_new = np.linspace(-8, 32, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.show()



'''# 교차 검증으로 다항 회귀를 평가합니다.
scores = cross_val_score(pipeline, X.reshape(-1, 1), y, scoring="neg_mean_squared_error", cv=10)
# Pipeline을 구성하는 세부 객체를 접근하는 named_steps['객체명']을 이용해 회귀 계수 추출
coefficients = pipeline.named_steps['linear_regression'].coef_
print('\nDegree {0} 회귀 계수는 {1} 입니다. '.format(degrees[i], np.round(coefficients, 2)))
print('Degree {0} MSE는 {1} 입니다.'.format(degrees[i], -1 * np.mean(scores)))

X_test = np.linspace(-1, 1, 100)
# 예측값 곡선
plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
# 실제 값 곡선
plt.scatter(X.reshape(-1, 1), y, edgecolor='b', s=20, label="Samples")

plt.xlabel("x");
plt.ylabel("y");
#plt.xlim((0, 1));
#plt.ylim((-2, 2));
plt.legend(loc="best")
plt.legend(loc="best")
plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))
plt.show()'''

time.sleep(1000)


#클러스터링 부분 코드
raw_data = read_csv('./cluster_office/온도별사용량_fixed.csv')
#pricedata = []
timedata = []
i = 0
for item in raw_data:
    if i > 0:
        #timedata.append([item[0], [float(item[27]), float(item[28]), float(item[29]), float(item[30]), float(item[31]), float(item[32]), float(item[33]), float(item[34]), float(item[35]), float(item[36]), float(item[37]), float(item[38])]])
        temp_arr = []
        for iter in range(220):
            temp_arr.append(float(item[iter+1]))
        timedata.append([item[0], temp_arr])
        #pricedata.append(float(item[26]))
    i = i + 1
print(timedata)

clustering(timedata)


'''
#신경망 부분 코드
#불러온 데이터를 나눈다.(종류별로)
data = pd.read_csv('./cluster_20210911/8 clusters_good/data_encode_20210913ver.csv', encoding='ANSI')
data_cluster_0 = data.loc[(data['Result']==0),:]
data_cluster_1 = data.loc[(data['Result']==1),:]
data_cluster_2 = data.loc[(data['Result']==2),:]
data_cluster_3 = data.loc[(data['Result']==3),:]
data_cluster_4 = data.loc[(data['Result']==4),:]
data_cluster_5 = data.loc[(data['Result']==5),:]
data_cluster_6 = data.loc[(data['Result']==6),:]
data_cluster_7 = data.loc[(data['Result']==7),:]
print(data_cluster_0)


value_list = [ 'Priv/Ho', 'age', 'Temp_avg', 'pblntfPclnd']
value_cluster_0 = [data_cluster_0[value_list].values.tolist(), data_cluster_0['Elec'].values.tolist()]
value_cluster_1 = [data_cluster_1[value_list].values.tolist(), data_cluster_1['Elec'].values.tolist()]
value_cluster_2 = [data_cluster_2[value_list].values.tolist(), data_cluster_2['Elec'].values.tolist()]
value_cluster_3 = [data_cluster_3[value_list].values.tolist(), data_cluster_3['Elec'].values.tolist()]
value_cluster_4 = [data_cluster_4[value_list].values.tolist(), data_cluster_4['Elec'].values.tolist()]
value_cluster_5 = [data_cluster_5[value_list].values.tolist(), data_cluster_5['Elec'].values.tolist()]
value_cluster_6 = [data_cluster_6[value_list].values.tolist(), data_cluster_6['Elec'].values.tolist()]
value_cluster_7 = [data_cluster_7[value_list].values.tolist(), data_cluster_7['Elec'].values.tolist()]
#print(value_cluster_0)
#클러스터별로 학습 데이터와 검증 데이터를 분리
def k_folding_validation(data, fold_num, model):
    # //는 몫만을 구하는 나누기 연산
    num_validation = len(data[0]) // fold_num

    #구조를 바꾼 뒤 셔플
    temp_data = []
    for i in range(len(data[0])):
        temp_data.append([data[0][i], data[1][i]])
    np.random.shuffle(temp_data)
    validation_scores = []

    def data_split(data):
        #row로 된 데이터를 받아서 column으로 바꿔주는 느낌.
        #하면서 X, Y로도 나눠 준다.
        temp_x = []
        temp_y = []
        #print(data)
        for i in range(len(data)):
            temp_x.append(data[i][0])
            temp_y.append(data[i][1])
        #print(temp_x)
        return np.array(temp_x), np.array(temp_y)

    print(num_validation)
    for fold in range(fold_num):
        validation_data = temp_data[num_validation * fold:num_validation * (fold + 1)]
        # 리스트 + 리스트는 연결된 하나의 리스트를 생성한다
        train_data = temp_data[:num_validation * fold] + temp_data[num_validation * (fold + 1):]
        #print(train_data)
        #print(validation_data)
        #print(train_data)
        model = createModel()
        train_x, train_y = data_split(train_data)
        valid_x, valid_y = data_split(validation_data)
        model.fit(train_x, train_y, epochs=100, batch_size=20)
        val_score = model.evaluate(valid_x, valid_y, batch_size=128)
        validation_scores.append(val_score)

    validation_score = np.average(validation_scores)
    return validation_score, validation_scores

model_0 = createModel()
p = k_folding_validation(value_cluster_0, 5, model_0)
print(p)

'''


















#공시지가 플로팅
'''for item in timedata:
    if item[0] ==  '계단식':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color=rgb_to_hex(item[2]/max_price_data*255, 0, (max_price_data-item[2])/max_price_data*255), alpha=0.1)
plt.show()'''
'''
#지가종류 플로팅
for item in timedata:
    if item[0] ==  'HG':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='red', alpha=0.1)
        #plt.show()
    elif item[0] ==  'LG':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='blue', alpha=0.1)
        #plt.show()=
plt.show()

#복도종류 플로팅
for item in timedata:
    if item[0] ==  '계단식':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='red', alpha=0.1)
        #plt.show()
    elif item[0] ==  '타워식':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='blue', alpha=0.1)
        #plt.show()
    elif item[0] ==  '복도식':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='green', alpha=0.1)
        #plt.show()
    elif item[0] ==  '혼합식':
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], item[1], color='yellow', alpha=0.1)
        #plt.show()

plt.show()

'''



'''sns.set(style="white")
df = sns.load_dataset("iris")

g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)

plt.show();'''


#ax = sns.scatterplot(x=data['decibel_living'], y=data['decibel_table'], hue=data['switch'])
#plt.show()

