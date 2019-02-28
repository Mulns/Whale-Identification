#knn+pca

from sklearn.decomposition import PCA
from sklearn.neighbors  import KNeighborsClassifier  
 
whale = 

#get train and test  x:data;y:label
x_train = 
x_test = 
y_train = 
y_test = 

#train PCA model
PCA=PCA(n_components=100).fit(x_train)

#return data after pca
x_train_pca = PCA.transform(x_train)
x_test_pca = PCA.transform(x_test)
 
#knn core  
knn=KNeighborsClassifier(n_neighbors=6) 

#train the model using train dataset 
knn.fit(x_train_pca ,y_train)                       
 
#test the data
y_test_predict=knn.predict(x_test_pca)
 
#predict the accuracy rate
print(knn.score(x_test_pca, y_test))                          

