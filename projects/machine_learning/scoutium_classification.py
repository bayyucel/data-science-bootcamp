
"""YAPAY ÖĞRENME İLE YETENEK AVCILIĞI SINIFLANDIRMA

İş Problemi

Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

Veriseti Hikayesi

Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

attributes: Oyuncuları değerlendiren kullanıcıların bir maçta izleyip değerlendirdikleri her oyuncunun özelliklerine verdikleri puanları içeriyor. (bağımsız değişkenler)

potential_labels: Oyuncuları değerlendiren kullanıcıların her bir maçta oyuncularla ilgili nihai görüşlerini içeren potansiyel etiketlerini içeriyor. (hedef değişken)

9 Değişken, 10730 Gözlem, 0.65 mb
"""

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_column", None)
pd.set_option("display.width", 500)

# Adım1:  scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

df_attributes = pd.read_csv("datasets/scoutium_attributes.csv", delimiter=";")
df_potential = pd.read_csv("datasets/scoutium_potential_labels.csv", delimiter=";")

# Adım2:  Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
#("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
df_attributes.head()
df_potential.head()
df_attributes.shape
df_potential.shape
df = pd.merge(df_attributes, df_potential,
              on = ["task_response_id", "match_id", "evaluator_id", "player_id"],
              how = "inner")

df.info()
df.head()
#Adım3:  position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df = df[df["position_id"] != 1]

#Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df = df[df["potential_label"] != "below_average"]

# Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

#Adım1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların
# oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
df = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values = "attribute_value")

#Adım2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
df = df.reset_index()
df.columns.name = None
df.columns = df.columns.map(str)


#Adım6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = label_encoder(df, "potential_label")
# alfabatik sıraya göre önce 0 ve sonra 1 gelecek şekilde encode yaptı
# average = 0, highlighted = 1

# Adım7:  Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.

num_cols = [col for col in df.columns if col not in ["player_id", "position_id", "potential_label"]]
df.head()

# Adım8:  Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

scaled_cols = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(scaled_cols, columns=df[num_cols].columns)

#Adım9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir
# makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
X = df.drop(["potential_label", "player_id"], axis = 1)
y = df["potential_label"]

def base_models(X, y, scoring = "roc_auc"):
    print("Base Models...")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose = -1)),
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring = "accuracy")

######################################################
# Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbose = -1), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]



def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

######################################################
# Stacking & Ensemble Learning
######################################################


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1","precision","recall", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

df["potential_label"].value_counts()

# Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:50])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("lightgbm_importances.png")


lgbm_model = LGBMClassifier()
params = best_models["LightGBM"].get_params()
final_model = lgbm_model.set_params(**params).fit(X,y)
plot_importance(final_model, X)