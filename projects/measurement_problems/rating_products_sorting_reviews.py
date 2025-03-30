
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_column", None)
pd.set_option("display.max_row", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
df_ = pd.read_csv("datasets/amazon_review.csv")
df = df_.copy()

df.head()
df.shape
df.isnull().sum()
df.info()
df.nunique()

df["overall"].mean()
## tüm veri setinin average ratıng'i 4.587589

df.sort_values(by = "day_diff", ascending = False).head()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

def time_based_weighted_average(database, w1 = 0.3, w2 = 0.25, w3 = 0.18, w4= 0.14, w5 = 0.08, w6 = 0.05):
    p1 = database.loc[database["day_diff"] <= 30, "overall"].mean()
    p2 = database.loc[(database["day_diff"] > 30) & (database["day_diff"] <= 90), "overall"].mean()
    p3 = database.loc[(database["day_diff"] > 90) & (database["day_diff"] <= 180), "overall"].mean()
    p4 = database.loc[(database["day_diff"] > 180) & (database["day_diff"] <= 360), "overall"].mean()
    p5 = database.loc[(database["day_diff"] > 360) & (database["day_diff"] <= 720), "overall"].mean()
    p6 = database.loc[df["day_diff"] > 720, "overall"].mean()

    print(f"1. zaman dilimi ortalaması : {p1}\n"
          f"2. zaman dilimi ortalaması : {p2}\n"
          f"3. zaman dilimi ortalaması : {p3}\n"
          f"4. zaman dilimi ortalaması : {p4}\n"
          f"5. zaman dilimi ortalaması : {p5}\n"
          f"6. zaman dilimi ortalaması : {p6}\n")

    return p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4 + p5 * w5 + p6 * w6

time_based_weighted_average(df)
#4.697157

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################



###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def score_pos_neg_diff(pos, neg):
    return pos - neg

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis = 1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis = 1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis = 1)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values(by = "wilson_lower_bound", ascending = False)[ : 20]

# ilk satıra baktıgımızda kısının wılson lower boundu 0.95 gelmiş. total votes 2020, helpful yes 1952, helpful no 68,
# 2022 toplam oy, kalabalığın bilgeliğini yansıtıyor. buraya oldukca fazla kısı yorum yapmıs ve kalabalıgın bılgelıgı dıkkate alınmıs
# average ratınge gore sıralasaydık ve 0.91 average ratınge sahıp satıra baksaydık, burada total vote a baktıgımızda
# sadece 49 tane yorum aldıgını gormekteyız. kısı sayısı az oldugundan genellebılırlık kaygısı var. topluluğun bilgeliği kavramı yok.
