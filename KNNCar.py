import pandas as pd # import thư viện pandas để dùng dataframe
import numpy as np #import thư viện numpy
import matplotlib.pyplot as plt #import thư viện matplotlip
import seaborn as sns
df = pd.read_csv("car.csv",names = ['Price','Maint','Doors','Persons','Luggage','Safety','Acceptibility'])
#đọc dữ liệu từ file CSV bằng thư viện pandas và đặt tên tương ứng với các cột dữ liệu
df.head()# hiện thỉ các giá trị đầu tiên của dataframe
df.info()# hiển thị thông tin của bảng dữ liệu
df.describe()
df.columns
x = df[['Price', 'Maint', 'Doors', 'Persons', 'Luggage', 'Safety']]
y = df['Acceptibility']
# gán X là giá trị của các cột dữ liệu  và Y là cột giá trị để thực hiện vẽ
sns.countplot(x='Price',hue = 'Acceptibility',data=df)
#cột Price với khả năng chấp nhận tương ứng
sns.countplot(x='Maint',hue='Acceptibility',data=df)
#cột Maint với khả năng chấp nhận tương ứng
sns.countplot(x='Doors',hue='Acceptibility',data=df)
#cột Doors với khả năng chấp nhận tương ứng
sns.countplot(x='Luggage',hue='Acceptibility',data=df)
#cột Luggage với khả năng chấp nhận tương ứng
sns.countplot(x='Persons',hue='Acceptibility',data=df)
#cột Persons với khả năng chấp nhận tương ứng
sns.countplot(x='Safety',hue='Acceptibility',data=df)
#cột Safety với khả năng chấp nhận tương ứng
df.Acceptibility.replace(('unacc','acc','good','vgood'),(0,1,2,3),inplace=True)
#chuyển dữ liệu cột Accceptibility về dạng số tương ứng Unacc = 0, Acc =1 , Good =2, Vgood = 3.
df.head(5)
df.Luggage.replace(('small','med','big'),(0,1,2),inplace=True)
#chuyển dữ liệu cột Luggage về dạng số tương ứng small = 0, med =1 , big =2.
df.head(5)
df.Safety.replace(('low','med','high'),(0,1,2),inplace=True)
#chuyển dữ liệu cột safety về dạng số tương ứng low = 0, med =1 , high =2.
df.head(5)
df.Maint.replace(('low','med','high','vhigh'),(0,1,2,3),inplace=True)
#chuyển dữ liệu cột  Maint về dạng số tương ứng low = 0, med =1 , high =2, vhigh = 3.
df.head(5)
df.Price.replace(('low','med','high','vhigh'),(0,1,2,3),inplace=True)
#chuyển dữ liệu cột Price về dạng số tương ứng low = 0, med =1 , high =2, vhigh = 3.
df.head(5)
df.Persons.replace(('more'),5,inplace=True)
#chuyển dữ liệu cột Persons về dạng số tương ứng More = 5.
df.head(15)
df.Doors.replace(('5more'),5,inplace=True)
#chuyển dữ liệu cột Doors về dạng số tương ứng 5More = 5.

df.corr()# thể hiện mối tương quan giữa các cột với nhau
sns.heatmap(df.corr(), annot=True)
x = df.iloc[:,:6]
y = df.iloc[:,6]
print("Shape of x:-",x.shape)
print("Shape of y:-",y.shape)
#chọn dữ liệu
from sklearn.model_selection import train_test_split
#chia dữ liệu thành dữ liệu test vói 30% và dữ liệu train . và xét random dữ liệu =1 .
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)
print("Shape of x Test",x_test.shape)
print("Shape of x Train",x_train.shape)
print("Shape of y Train",y_train.shape)
print("Shape of y test",y_test.shape)
# biến đổi dữ liệu
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier# Bộ phân loại thực hiện  bỏ phiếu KNN gần nhất
from sklearn.metrics import confusion_matrix#thư viện ma trận nhầm lẫn
#tạo mô hình
knn = KNeighborsClassifier(n_neighbors=3)# số lượng giá trị xung quanh để sử dụng cho việc xác định khả năng xác nhận
#cung cấp dữ liệu đào tạo vào mô hình
knn.fit(x_train, y_train)# điều chỉnh mô hình bằng cách  sử dụng X làm dữ liệu  đào tạo và y làm giá trị mục tiêu
#Dự đoán giá trị cho x_test
prediction = knn.predict(x_test)# dự đoán giá trị mục tiêu cho dữ liệu test
print("Training Accuracy:",knn.score(x_train,y_train)) # Trả lại độ chính xác trung bình trên các nhãn và dữ liệu train.
print("Testing Accuracy:",knn.score(x_test,y_test)) # Trả lại độ chính xác trung bình trên các nhãn và dữ liệu train
print ("Predicted labels: ", prediction[1:20])
print ("Ground truth    : ", y_test[1:20])
cm = confusion_matrix(y_test, prediction)
print(cm)