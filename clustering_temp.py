import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import csv
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

    n_cluster = 6  # 결정된 수 사용
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
    plt.show()

    pca = PCA(n_components=2)  # pca를 진행해 클러스터가 얼마나 잘됐는지 검토합니다.
    rlt_pca = pca.fit_transform(feature_norm_npy[~np.isnan(feature_norm_npy).any(axis=1)])
    for i in range(n_cluster):
        label_name = "cluster " + str(i)
        plt.scatter(rlt_pca[[rlt_tsm == i]][:, 0], rlt_pca[[rlt_tsm == i]][:, 1], label=label_name)
    plt.legend()
    plt.show()

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
        plt.show()

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

raw_data = read_csv('elec_2020_temp.csv')
#pricedata = []
timedata = []
i = 0
for item in raw_data:
    if i > 0:
        timedata.append([item[0], [float(item[13]), float(item[14]), float(item[15]), float(item[16]), float(item[17]), float(item[18]), float(item[19]), float(item[20]), float(item[21]), float(item[22]), float(item[23]), float(item[24])]])
        #pricedata.append(float(item[26]))
    i = i + 1

clustering(timedata)
























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

