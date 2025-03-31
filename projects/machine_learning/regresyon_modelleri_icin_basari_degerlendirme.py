import numpy as np
import pandas as pd


df = pd.DataFrame({"YEARS_OF_EXPERIENCE": [5,7,3,3,2,7,3,10,6,4,8,1,1,9,1],
                   "SALARY": [600,900,550,500,400,950,540,1200,900,550,1100,460,400,1000,380]})

X = df["YEARS_OF_EXPERIENCE"]
y = df["SALARY"]

#1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
# Bias=275,Weight=90(y’=b+wx)
# y' = 275 + 90*Xi
bias = 275
weight = 90

#2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
df["y'"] = bias + weight*X #

#3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız.
df["ERROR"] = df["SALARY"] - df["y'"]
df["ERROR_SQUARES"] = df["ERROR"] ** 2
df["ABSOLUTE_ERROR"] = np.absolute(df["ERROR"])

MSE = df["ERROR_SQUARES"].mean()
RMSE = np.sqrt(MSE)
MAE = df["ABSOLUTE_ERROR"].mean()