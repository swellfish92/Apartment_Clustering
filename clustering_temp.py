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
    '''for i in range(1, 10):
        clustering= TimeSeriesKMeans(n_clusters=i, init='k-means++')
        clustering.fit(feature_norm_npy)
        objective_function.append(clustering.inertia_)
    print(objective_function)
    plt.plot(range(1, 10), objective_function)
    plt.title('elbow method-hyperparameter')
    plt.xlabel('# of clusteres')
    plt.ylabel('objective_function')
    plt.show()'''

    n_cluster = 8  # 결정된 수 사용
    km = TimeSeriesKMeans(n_clusters=n_cluster, metric="euclidean", max_iter=150).fit(feature_norm_npy)  # normalization 시계열 데이터를 활용해 euclidean 기반 클러스터링을 진행합니다.
    rlt_tsm = km.predict(feature_norm_npy)

    print(rlt_tsm)
    temp_list_shell = []
    for item in rlt_tsm:
        temp_item = [item]
        temp_list_shell.append(temp_item)
    print(temp_list_shell)
    write_csv('cluster_result.csv', temp_list_shell)

    # 클러스터별 분포도를 시각화하기 위해 전처리 작업입니다.
    labels = []
    sizes = []
    for i in range(n_cluster):
        labels.append("cluster_" + str(i))
        sizes.append(collections.Counter(rlt_tsm)[i])

    plt.figure(figsize=(10, 5))
    plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    plt.title("Cluster Distribution (number of data:" + str(len(df['elec'])) + ")", position=(0.5, 1.2), fontsize=20)
    #plt.show()
    plt.savefig('./cluster_20210911/Cluster_distribution.png')

    pca = PCA(n_components=2)  # pca를 진행해 클러스터가 얼마나 잘됐는지 검토합니다.
    rlt_pca = pca.fit_transform(feature_norm_npy[~np.isnan(feature_norm_npy).any(axis=1)])
    for i in range(n_cluster):
        label_name = "cluster " + str(i)
        plt.scatter(rlt_pca[[rlt_tsm == i]][:, 0], rlt_pca[[rlt_tsm == i]][:, 1], label=label_name)
    plt.legend()
    #plt.show()
    plt.savefig('./cluster_20210911/Cluster_scatterplot.png')

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
        plt.savefig('./cluster_20210911/Cluster'+str(num_cluster)+'.png')
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


#클러스터링 부분 코드
'''raw_data = read_csv('elec_2020_temp.csv')
#pricedata = []
timedata = []
i = 0
for item in raw_data:
    if i > 0:
        timedata.append([item[0], [float(item[13]), float(item[14]), float(item[15]), float(item[16]), float(item[17]), float(item[18]), float(item[19]), float(item[20]), float(item[21]), float(item[22]), float(item[23]), float(item[24])]])
        #pricedata.append(float(item[26]))
    i = i + 1

clustering(timedata)'''

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

