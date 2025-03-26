#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float", lambda x: '%.2f' %x)

df = sns.load_dataset("titanic")

def check_df(dataframe, head = 5):
    """
    Verilen DataFrame hakkında genel bir özet bilgi sunar.

    Args:
        dataframe (pd.DataFrame): İncelenecek veri seti.
        head (int, optional): İlk ve son kaç satırın gösterileceği. Default 5.

    Returns:
        None

    Prints:
        - Satır ve sütun sayısı
        - Sütun veri tipleri
        - İlk ve son gözlemler
        - Eksik değer sayısı
        - Tanımlayıcı istatistikler (quantile'lar dahil)
    """
    print("############# SHAPE ############")
    print(dataframe.shape)
    print("############# TYPES ############")
    print(dataframe.dtypes)
    print("############# HEAD ############")
    print(dataframe.head(head))
    print("############# TAIL ############")
    print(dataframe.tail(head))
    print("############# NA ############")
    print(dataframe.isnull().sum())
    print("############# QUANTILES ############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

dff = sns.load_dataset("flights")
check_df(dff)

#############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#############################################

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].dtypes not in cat_cols and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if
                   str(dataframe[col].dtypes) in ["category", "object"] and dataframe[col].nunique() > 20 ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################

def cat_summary(dataframe, cat_col):
    """
    Kategorik bir değişkenin sınıf sayılarını ve oranlarını yazdırır.

    Args:
        dataframe (pd.DataFrame): İncelenecek veri seti.
        cat_col (str): Kategorik değişkenin sütun adı.

    Returns:
        None

    Prints:
        - Kategori frekansları
        - Yüzdelik oranları (%)
    """
    print(pd.DataFrame({cat_col: dataframe[cat_col].value_counts(),
          "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe)}))
    print("#########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

def num_summary(dataframe, numerical_col):
    """
    Sayısal bir değişkenin özet istatistiklerini yazdırır.
    Args:
        dataframe (pd.DataFrame): İncelenecek veri seti.
        numerical_col (int, float): Sayısal değişkenin sütun adı.

    Returns:
        None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[[numerical_col]].describe(quantiles).T)
    print("#################################################################################################")

for col in num_cols:
    num_summary(df, col)

#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################

def target_summary_with_cat(dataframe, target, cat_col):
    """
    Hedef değişkenin kategorik değişkene göre ortalamasını yazdırır.
    Args:
        dataframe (pd.DataFrame): İncelenecek veri seti.
        target: Hedef değişken
        cat_col: Kategorik değişken

    Returns:
        None
    """
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(cat_col)[target].mean()}), end = "\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################

def target_summary_with_num(dataframe, target, num_col):
    """
    Sayısal değişkenlerin hedef değişkene göre ortalaması
    Args:
        dataframe: İncelenecek veri seti
        target: Hedef değişken
        num_col: Sayısal değişken

    Returns:
        None
    """
    print(dataframe.groupby(target).agg({num_col: "mean"}), end = "\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

corr = df[num_cols].corr()

sns.set(rc = {"figure.figsize": (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

corr_matrix = df[num_cols].corr().abs()

#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
corr_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot = False, corr_th = 0.90):
    """
    Yüksek korelasyona sahip değişkenleri tespit eder ve isimlerini döner.

    Args:
        dataframe (pd.DataFrame): Korelasyon analizi yapılacak veri seti.
        plot (bool, optional): Korelasyon matrisini ısı haritası (heatmap) ile görselleştir. Default False.
        corr_th (float, optional): Korelasyon eşik değeri. Bu değerin üzerinde korelasyona sahip sütunlar listelenir. Default 0.90.

    Returns:
        list: Belirtilen eşik değerine göre yüksek korelasyonlu değişken isimlerinden oluşan liste.

    Notes:
        - Korelasyon mutlak değere göre değerlendirilir (pozitif/negatif fark etmez).
        - Üst üçgen matris yöntemiyle tekrar eden korelasyonlar filtrelenir.
        - plot=True ise korelasyon matrisi görsel olarak da gösterilir (Seaborn ile).
    """
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
