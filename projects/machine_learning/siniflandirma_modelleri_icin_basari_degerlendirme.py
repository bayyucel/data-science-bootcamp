#SINIFLANDIRMA MODELİ DEĞERLENDİRME

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

####################################
#### GÖREV 1 #######################
####################################

df = pd.DataFrame({"ACTUAL_VALUE": [1,1,1,1,1,1,0,0,0,0],
                   "PROBABILITY": [0.7,0.8,0.65,0.9,0.45,0.5,0.55,0.35,0.4,0.25] })

# Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma modeli oluşturulmuştur.
# 10 test verisi gözleminin gerçek değerleri ve modelin tahmin ettiği olasılık değerleri verilmiştir.
# Eşik değerini 0.5 alarak confusion matrix oluşturunuz.
# - Accuracy,Recall, Precision, F1 Skorlarını hesaplayınız.

th = 0.5 #Threshold

df["PREDICTION"] = [1 if value >= th else 0 for value in df["PROBABILITY"].values]
#df["PREDICTION"] = (df["PROBABILITY"] >= 0.5).astype(int)

y = df["ACTUAL_VALUE"] #Gerçek değerler
y_pred = df["PREDICTION"] #Tahmin edilen değerler

cm = confusion_matrix(y, y_pred) #Karışıklık matrisi

print(classification_report(y, y_pred))
#Accuracy: 0.80
#Precision: 0.83
#Recall: 0.83
#F1-Score: 0.83

##################
#### GÖREV 2 #####
##################

# Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur. %90.5 doğruluk
# oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır. Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi
# olmamış, iş birimi modelin başarısız olduğunu iletmiştir. Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir.
# Buna göre;
# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.
# - Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.

TP = 5 #True Pozitif
FN = 5 #False Negatif
FP = 90 #False Pozitif
TN = 900 #True Negatif

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1_score = 2*(precision*recall)/(precision+recall)

#Dengesiz veri olduğundan accuracy değeri her ne kadar yüksek olsa da precision ve recall değerlerine bakıldığında modelin o kadar da başarılı olmadığı görünüyor.
# False Pozitif değeri gerçekte dolandırılık olmayan ama bizim dolandırıcılık olarak grupladığımızı gösterir. 90 kişiyi hatalı sınıflandırmışız. Veri setinin toplamına bakıldığında bu sayı yüksek kabul edilebilir.
# recall değeri 0.5 çıkması gerçekte fraud olan işlemlerin yarısını doğru tahmin ettiğimizi yarısını gözden kaçırdığımızı gösteriyor. Bu dolandırıcılık gibi kritik bir işlem için çok düşük bir oran olarak kabul edilir.
# Yeni veri ekleme, veri ön işleme adımının gözden geçirilmesi, detaylı özellik mühendisliğinin tekrar yapılması ve modelin optimize edilmesiyle (yeni th oranları belirlenebilir) model tahmini yükseltilebilir.