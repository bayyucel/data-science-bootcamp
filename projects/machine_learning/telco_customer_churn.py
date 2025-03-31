#################################
#### TELCO CHURN  PREDICTION ####
#################################

###################
## İş Problemi ####
###################

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

############################
#### Veri Seti Hikayesi ####
############################

#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir
#telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.classifier import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#### Görev 1: Keşifçi Veri Analizi ####

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()
df.head()
df.shape
df.info()
df.isnull().any()
df.describe().T

#Adım 1: Numerik ve kategorik değişkenleri yakalayın

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th]

    #cat_cols, cat_but_car
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

#Total Charges değişkeninin tipi hatalı. Dönüşüm yapılırken bazı değerlerin " " (boş) olduğu görüldü.
#Bu değerleden dolayı astype(float) gibi bir fonksiyon işe yaramadı.
#Bu boş değerlere sahip müşterilerin "tenure" değerinin de sıfır olduğu görüldü. Yani bu müşteriler yeni aboneler ve bu yüzden henüz bir fatura ödemediler.
#Bu sebeple bu değerler boş gelmekte.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") #Bu fonksiyonda errors="coerce" argümanı, eğer dönüşemeyen bir değer varsa bunu NaN olarak değiştirir.

df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

#Kategorik değişkenlerin dağılımı
for col in cat_cols:
    print(df[col].value_counts())
    print("############################")

#Sayısal değişkenlerin dağılımı
df[num_cols].describe([0.01, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
#TotalCharges değişkeninin standart sapması çok yüksek çıktı. Buna, dağılımın geniş olduğu ve müşteriler arasında çok fark olduğu yorumu yapılabilir.
#Bu da aslında beklenebilir. Çünkü kimi müşteri çok uzun süredir abone olup kimisi yeni müşteri olabilir.

plt.figure(figsize=(8,5))
sns.histplot(data = df["TotalCharges"], bins =50, kde=True)
plt.show()

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

for col in cat_cols:
    print(df.groupby(col)["Churn"].value_counts(normalize = True))
    print("######################################")

#InternetService Fiber optic kullananların churn oranı yüksek görünüyor. Fiber opticle ilgili bir memnuniyetsizlikten kaynaklı olabilir. Altyapı vs. Bu fark fiyatlardan kaynaklı da olabilir.
df[~(df["InternetService"] == "Fiber optic") & (df["Churn"] == "Yes")]["TotalCharges"].mean()
df[~(df["InternetService"] == "Fiber optic") & (df["Churn"] == "Yes")]["MonthlyCharges"].mean()#InternetService kullanmayanların churn oranı çok düşük. Bu da anlaşılır bir durum. Sadece telefon hattı kullanan bir müşterinin bununla ilgili memnuniyetsizlik veya sorun yaşama şansı çok düşüktür.

#OnlineSecurity,online backup, deviceprotection, techsupport, streamintv, StreamingMovies kullanmayanların churn oranı yüksek. Bunu firma ürünlerini kullanmayanların churn oranı yüksek şeklinde yorumlayabilir miyiz? Ama neden böyle bir fark çıkmış olabilir.
#Bu ürünleri alan müşteriler teknolojiye daha hakim olduklarından bunu daha iyi yönetiyor olabilirler.
#Bir de bu ürünleri alanları yaş sınıfı düşük çıktı. SeniorCitizen olanların churn oranının yüksek olmasıyla aynı sonucu vermesi paralellik gösterdi.
extra_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df.groupby("SeniorCitizen")[extra_services].apply(lambda x: (x == "Yes").mean())

#Aydan aya fatura ödemesi yapan müşterilerin de churn oranı yüksek. Bu da anlaşılır, aylık ödeme yaptıklarından herhangi bir bağlayıcılık yok.
df.loc[~((df["Churn"] == "Yes") & (df["Contract"] == "Month-to-month")), "tenure"].mean()
df.loc[((df["Churn"] == "Yes") & (df["Contract"] == "Month-to-month")), "tenure"].mean()

#İlginç PaperlessBilling Yes olan kullanıcıların churn oranı yüksek görünüyor. Tabi bu anlamlı bir fark mı? Belki bir hipotez testi yapmak gerekebilir.
#Kağıtsız fatura kullanan müşteriler daha genç olabilir mi? Teknolojiye daha alışkın bir kitle. Sonuç öyle olursa diğer analizler ile paralellik gösterecektir.
pd.crosstab(df["PaperlessBilling"], df["Churn"]) #PaperlessBilling'in churn üzerine etkisi yüksek görünüyor. Bunu özellik mühendisliğinde kullabilirim.



#Electronic check ile ödeme yapanların da churn oranı yüksek görünüyor. Bu ödeme yöntemiyle ilgili yaşanan bir sıkıntıdan olabilir.
pd.crosstab(df["PaymentMethod"], df["Churn"], normalize=True)
#Fark net görünüyor. Özellik mühendisliğinde Electronic check kullananlara ayrı bir flag koyabilirim.

#SeniorCitizen 1 olan yani yaşı ileri olan müşterilerin churn oranı yüksek görünüyor. Bu sayı anlamlı mı acaba, ne ifade ediyor olabilir.
pd.crosstab(df["SeniorCitizen"], df["Churn"])
#Yaşlı müşterilerin ödeme yöntemleri özellik mühendisliğinde kullanılabilir. Kontrat süreleri de bunun için kullanılabilir.
#Ayrıca yaşlı müşteriler teknolojiyi iyi kullanamıyor ve firmadan yeteri kadar destek alamıyor olabilir.
pd.crosstab(df["SeniorCitizen"], df["TechSupport"]) # Hem SeniorCitizen hem de TechSupport = No olan müşteriler için yeni bir değişken türetilebilir.

# def target_summary_with_cat(dataframe, target, categorical_col):
#     print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

#Korelasyon matrisi
corr_matrix = df[num_cols].corr()

#  Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

outlier_thresholds(df, "tenure")

#Aykırı gözlem bulunmamaktadır.

# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()

#TotalCharges değişkeni eksik veri içeriyor.

#### Görev 2 : Feature Engineering ####
# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
#Aykırı değerlere bir işlem yapılamyacak.

#Eksik verilerin eksiklik sebebi yeni müşterilerin henüz fatura ödememesinden kaynaklı TotalCharges değerlerinin olmaması.
#Bu değerleri mean median ile doldurmak veya veriden tamamen silmek yerine "0" sıfır ile doldurmanın daha doğru olacağını düşünüyorum.

df["TotalCharges"].fillna(value = 0, inplace = True)

#Ayrıca "No internet service"  ve "No phone service" değişkenleri direkt No ile değiştirilebilir. Zaten "InternetService" ve "PhoneService" değişkenimiz var.
for col in df.columns:
    df[col] = df[col].replace("No internet service", "No")

for col in df.columns:
    df[col] = df[col].replace("No phone service", "No")

#Adım 2: Yeni değişkenler oluşturunuz.

additional_services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

#df["SENIOR_NO_TECH_SUPPORT"] = ((df["SeniorCitizen"] == 1) & (df["TechSupport"] == "No")).astype(int) #Yaşlı olup teknik destek alamayanların churn oranı yüksek gibi bir çıkarım yapmıştık.
df["SHORT_TERM_CONTRACT"] = (df["Contract"] == "Month-to-month").astype(int)
df["LONG_TERM_CONTRACT"] = (~(df["Contract"] == "Month-to-month")).astype(int)
df = df.drop("Contract", axis = 1) #Short ve Long değişkenleri türettiğimizden contract değişkenini drop ettik.
#df["Senior_Paperless"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 1 and x["PaperlessBilling"] == "Yes") else 0, axis=1)
df["PAYMENT_RATIO"] = (df["TotalCharges"] + df["MonthlyCharges"]) / ((df["tenure"] + 1) * (df["MonthlyCharges"]))
df["TENURE_SQUARE"] = df["tenure"] ** 2
df["ADDITIONAL_SERVICES_USAGE_RATIO"] = (df[additional_services].apply(lambda x: (x=="Yes").sum(), axis = 1)) / len(additional_services) #Satın alınan toplam servis sayısı / toplam ekstra servis sayısı
df["MONTHLY_CHARGES_ADD_SERVICES_RATIO"] = df["MonthlyCharges"] / ((df[additional_services].apply(lambda x: (x=="Yes").sum(), axis = 1))).apply(lambda x:1 if x==0 else x) #Aylık ödeme /

#Partner ve Dependents ile yeni değişken oluşturma. Öneriyi ChatGPT verdi.
# df.loc[((df["Partner"] == "No") & (df["Dependents"] == "No")), "FAMILY_STATUS" ] = "Single"
# df.loc[((df["Partner"] == "Yes") & (df["Dependents"] == "No")), "FAMILY_STATUS" ] = "Couple"
# df.loc[((df["Partner"] == "No") & (df["Dependents"] == "Yes")), "FAMILY_STATUS" ] = "Parent"
# df.loc[((df["Partner"] == "Yes") & (df["Dependents"] == "Yes")), "FAMILY_STATUS" ] = "Family"


# Adım 3:  Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 6)
ohe_cols = [col for col in cat_cols if df[col].nunique() > 2]
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

#Yeni oluşan verilerde aykırı değer olup olmadığını kontrol ediyorum.
for col in num_cols:
    print(col, check_outlier(df, col))

outlier_thresholds(df, "PAYMENT_RATIO")

replace_with_thresholds(df, "PAYMENT_RATIO")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)




# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

df[num_cols] = MinMaxScaler(feature_range=(0,1)).fit_transform(df[num_cols])
df.head()

sns.lmplot(x='tenure', y='Churn', data=df, logistic=True, ci=None)
plt.show()

#### Görev 3 : Modelleme  ####

#Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 48)

####################################
# 1. Model - Logistic Regression
####################################

log_model = LogisticRegression()
log_model.fit(X_train,y_train)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

confusion_matrix(y_test, y_pred)
#array([[930, 113],
#       [158, 208]]
accuracy_score(y_pred, y_test)

print(classification_report(y_test, y_pred))
# Accuracy: 0.81
# Precision: 0.65
# Recall: 0.57
# F1-score: 0.61

importance = log_model.coef_[0]
# Değişken isimleri ile DataFrame oluştur
feature_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(importance)})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_imp)
plt.title("Feature Importance - Logistic Regression")
plt.show()

# ROC AUC
y_prob = log_model.predict_proba(X_test)[:, 1]
roc_auc_score(y_train, y_prob)
#0.84

y_pred = [1 if value > 0.45 else 0 for value in y_prob ]

####################################
#2. Model - Random Forest Classifier
####################################

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

confusion_matrix(y_test, y_pred)
# array([[925, 118],
#        [184, 182]]
accuracy_score(y_pred, y_test)
#0.78

print(classification_report(y_test, y_pred))
# Accuracy: 0.79
# Precision: 0.61
# Recall: 0.50
# F1-score: 0.55

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

####################################
#3. Model - SVM
####################################

from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear', random_state = 0)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
confusion_matrix(y_test, y_pred)
# array([[957,  86],
#        [185, 181]]
accuracy_score(y_pred, y_test)
# 0.80
print(classification_report(y_test, y_pred))
# Accuracy: 0.81
# Precision: 0.68
# Recall: 0.49
# F1-score: 0.57
importance = svm_model.coef_[0]
feature_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(importance)})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)


####################################
#4. Model - Naive Bayes Model
####################################

from sklearn.naive_bayes import GaussianNB
bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)
y_pred = bayes_model.predict(X_test)

confusion_matrix(y_test, y_pred)
# array([[771, 272],
#        [ 83, 283]]
accuracy_score(y_pred, y_test)
#0.74
print(classification_report(y_test, y_pred))
# Accuracy: 0.75
# Precision: 0.51
# Recall: 0.77
# F1-score: 0.61

y_prob = bayes_model.predict_proba(X_test)[:, 1]
y_pred = [1 if value > 0.9 else 0 for value in y_prob ]

####################################
#5. Model - K-Neighbors Classifier
####################################

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_model.fit(X_train, y_train)
y_pred   = knn_model.predict(X_test)

confusion_matrix(y_test, y_pred)
# array([[891, 152],
#        [168, 198]]
accuracy_score(y_pred, y_test)
#0.77
print(classification_report(y_test, y_pred))
# Accuracy: 0.76
# Precision: 0.57
# Recall: 0.54
# F1-score: 0.55

y_prob = knn_model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_prob)

#############################
#Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.
#############################

from sklearn.model_selection import GridSearchCV

# 1. Model - Logistic Regression Final Model

log_model.get_params()

log_model_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,2000,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

log_model_best = GridSearchCV(log_model,
                              log_model_params,
                              cv = 5,
                              n_jobs= -1,
                              verbose=1).fit(X_train,y_train)

log_model_best.best_params_

log_model_final = log_model.set_params(**log_model_best.best_params_).fit(X_train,y_train)

y_pred = log_model_final.predict(X_test)

confusion_matrix(y_test, y_pred)
#array([[935, 108],
#       [158, 208]]
accuracy_score(y_pred, y_test)

print(classification_report(y_test, y_pred))
# Accuracy: 0.81
# Precision: 0.66
# Recall: 0.57
# F1-score: 0.61

y_prob = log_model.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_prob)
#0.84

#Optuna ile best params
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

#Amaç fonksiyonu tanımlanıyor.
def objective(trial):
    C = trial.suggest_loguniform("C", 0.01, 100)  # C için geniş bir aralık belirle
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
    penalty = trial.suggest_categorical("penalty", ['l1', 'l2'])# Farklı çözücüler
    max_iter = trial.suggest_int("max_iter", 100, 1000)  # Maksimum iterasyon sayısı

    # Modeli tanımla
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)

    # 5 katlı çapraz doğrulama (cross-validation)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    return score  # Optuna en iyi skoru maksimize etmeye çalışır


# Optuna çalışma sürecini başlat
study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=50)  # 50 farklı hiperparametre kombinasyonu dene

# En iyi hiperparametreleri göster
print("En iyi parametreler:", study.best_params)
print("En iyi skor:", study.best_value)

##2. Model - Random Forest Classifier için Hiperparametre Optimizasyonu
rf_model.get_params()

rf_model_params = {"n_estimators": [500, 1000],
                   "max_features": ["sqrt", "log2", None, "int"],
                   "criterion": ["gini", "entropy"]}

rf_model_best = GridSearchCV(rf_model,
                             rf_model_params,
                             cv = 5,
                             n_jobs=-1,
                             verbose=1).fit(X_train,y_train)

rf_model_best.best_params_
rf_model_final = rf_model.set_params(**rf_model_best.best_params_).fit(X_train,y_train)
y_pred = rf_model_final.predict(X_test)


confusion_matrix(y_test, y_pred)
# array([[930, 113],
#        [176, 190]]
accuracy_score(y_pred, y_test)
#0.0.79

print(classification_report(y_test, y_pred))
# Accuracy:  0.79
# Precision: 0.63
# Recall: 0.52
# F1-score: 0.57

y_prob = rf_model_final.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_prob)
#0.83


#3. Model - SVM için Hiperparametre Optimizasyonu

svm_model.get_params()


#Optuna ile best params
def objective(trial):
    C = trial.suggest_loguniform("C", 0.01, 100)  # C için log ölçekli değerler
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_loguniform("gamma", 0.001, 10) if kernel in ["rbf", "poly"] else "scale"
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, class_weight=class_weight)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# Optuna çalışma sürecini başlat
study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=50)

# En iyi hiperparametreleri göster
print("En iyi parametreler:", study.best_params)
print("En iyi skor:", study.best_value)

svm_model_final = svm_model.set_params(**study.best_params).fit(X_train, y_train)
y_pred = svm_model_final.predict(X_test)



confusion_matrix(y_test, y_pred)
# array([[944,  99],
#        [176, 190]]
accuracy_score(y_pred, y_test)
#0.80

print(classification_report(y_test, y_pred))
# Accuracy:  0.79
# Precision: 0.66
# Recall: 0.52
# F1-score: 0.58

y_prob = svm_model_final.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_prob)

#4. Model - Naive Bayes Model için hiperparametre optimizasyonu

bayes_model.get_params()

bayes_model_param = {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
             }

bayes_model_best = GridSearchCV(bayes_model,
                                bayes_model_param,
                                cv = 5,
                                n_jobs=-1,
                                verbose=1).fit(X_train, y_train)

bayes_model_best.best_params_

bayes_model_final = bayes_model.set_params(**bayes_model_best.best_params_).fit(X_train, y_train)

y_pred = bayes_model_final.predict(X_test)

confusion_matrix(y_test, y_pred)
# array([[783, 260],
#        [ 86, 280]]
accuracy_score(y_pred, y_test)
#0.75

print(classification_report(y_test, y_pred))
# Accuracy:  0.75
# Precision: 0.52
# Recall: 0.77
# F1-score: 0.62

y_prob = bayes_model_final.predict_proba(X_test)[: , 1]
roc_auc_score(y_test,y_prob)
# 0.82


#5. Model - K-Neighbors Classifier içi Hiperparametre Optimizasyonu
knn_model.get_params()

knn_model_params = {"n_neighbors": range(1,50),
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski", "chebyshev"],
                    "p": [1,2]}

knn_model_best = GridSearchCV(knn_model,
                              knn_model_params,
                              cv = 5,
                              n_jobs=-1,
                              verbose=1).fit(X_train, y_train)

knn_model_best.best_params_

knn_model_final = knn_model.set_params(**knn_model_best.best_params_).fit(X_train, y_train)

y_pred = knn_model_final.predict(X_test)

confusion_matrix(y_test, y_pred)
# array([[912, 131],
#        [159, 207]]
accuracy_score(y_test, y_pred)
#0.79
y_prob = knn_model_final.predict_proba(X_train)[: , 1]
roc_auc_score(y_train, y_prob)


print(classification_report(y_test,y_pred))

#KNN

# array([[912, 131],
#        [159, 207]]

# Accuracy:  0.79
# Precision: 0.61
# Recall: 0.57
# F1-score: 0.59
# AUC: 0.84

#Bayes

# array([[783, 260],
#        [ 86, 280]]

# Accuracy:  0.75
# Precision: 0.52
# Recall: 0.77
# F1-score: 0.62
# AUC: 0.82


#SVM

# array([[944,  99],
#        [176, 190]]


# Accuracy:  0.79
# Precision: 0.66
# Recall: 0.52
# F1-score: 0.58

#Random Forest

# array([[930, 113],
#        [176, 190]]

# Accuracy:  0.79
# Precision: 0.63
# Recall: 0.52
# F1-score: 0.57
# AUC: 0.83

#LogisticRegression

#array([[935, 108],
#       [158, 208]]

# Accuracy: 0.81
# Precision: 0.66
# Recall: 0.57
# F1-score: 0.61
# AUC: 0.84

