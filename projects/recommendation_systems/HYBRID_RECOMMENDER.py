
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#  Adım 1:   movie, rating veri setlerini okutunuz.
movie = pd.read_csv("datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("datasets/movie_lens_dataset/rating.csv")

#  Adım 2:  ratingveri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
rating = pd.merge(rating, movie[["movieId","title", "genres"]], how = "left", on = "movieId")

# Adım3:  Toplam oy kullanılmasayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
comment_counts = pd.DataFrame(rating["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] < 10000].index
common_movies = rating[~rating["title"].isin(rare_movies)]
len(common_movies)
#  Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index = "userId",
                                          columns= "title",
                                          values = "rating")

#  Adım5:  Yapılantümişlemleri fonksiyonlaştırınız

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("datasets/movie_lens_dataset/movie.csv")
    rating = pd.read_csv("datasets/movie_lens_dataset/rating.csv")
    rating = pd.merge(rating, movie[["movieId", "title", "genres"]], how="left", on="movieId")
    comment_counts = pd.DataFrame(rating["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] < 10000].index
    common_movies = rating[~rating["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index="userId",
                                              columns="title",
                                              values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id’si seçiniz.
random_user = 28941

# Adım 2:  Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım3:  Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
# Adım 1:  Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
movies_watched_df = user_movie_df[movies_watched]

# Adım 2:  Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe
# oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adım3:  Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste
# oluşturunuz
percentage = 0.6

users_same_movies = user_movie_count[user_movie_count["movie_count"]>(len(movies_watched) * percentage)]["userId"]
len(users_same_movies)

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################
# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df
# dataframe’ini filtreleyiniz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)]])
final_df.reset_index()
final_df[final_df.index == random_user]
# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()
corr_df.columns = ['user_id_1', 'user_id_2', 'corr']
# Adım3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe
# oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"]>0.65)][["user_id_2", "corr"]].reset_index(drop = True)
# Adım4:  top_users dataframe’ine rating veri seti ile merge ediniz.
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner', on = "userId")
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################
# Adım 1:   Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Adım 2:  Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adındayeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# Adım3:  recommendation_df içerisindeweighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(by = "weighted_rating",ascending = False)

# Adım4:  movie verisetindenfilm isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])[0:5]

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 28941
# Adım 1:   movie, rating veri setlerini okutunuz.
movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

# Adım 2:  Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
last_movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5)]\
    .sort_values(by = "timestamp",ascending = False).iloc[0, 1]

# Adım3:  User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
df = pd.merge(rating, movie[["movieId","title"]], how = "left", on = "movieId")
last_movie_name = df.loc[df["movieId"] == last_movie_id, "title"].iloc[0]
filtered_df = user_movie_df[user_movie_df[last_movie_name].notnull()]

# Adım 4:  Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
correlations = filtered_df.corrwith(filtered_df[last_movie_name])
correlations_df = pd.DataFrame(correlations.sort_values(ascending=False)).reset_index()

# Adım5:  Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
recommended_movies = correlations_df[1:6]






