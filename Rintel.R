#Caricamento librerie
library(keras)
library(tidyverse)
library(stringr)
library(imager)
library(caret)

#### Caricamento dei dati #####
buildings <- list.files(path = "c:/data/intel/seg_train/buildings")
forest <- list.files(path = "c:/data/intel/seg_train/forest")
glacier <- list.files(path = "c:/data/intel/seg_train/glacier")
mountain <- list.files(path = "c:/data/intel/seg_train/mountain")
sea <- list.files(path = "c:/data/intel/seg_train/sea")
street <- list.files(path = "c:/data/intel/seg_train/street")

buildings_test <- list.files(path = "c:/data/intel/seg_test/buildings")
forest_test <- list.files(path = "c:/data/intel/seg_test/forest")
glacier_test <- list.files(path = "c:/data/intel/seg_test/glacier")
mountain_test <- list.files(path = "c:/data/intel/seg_test/mountain")
sea_test <- list.files(path = "c:/data/intel/seg_test/sea")
street_test <- list.files(path = "c:/data/intel/seg_test/street")

img_path <- "c:/data/intel/seg_train/buildings/17903.jpg"
img <- image_load(img_path, target_size = c(150, 150))                 
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor/255
dim(img_tensor) 
plot(as.raster(img_tensor[1,,,]))

train <- c(
  buildings[1:2000], 
  forest[1:2000],
  glacier[1:2000],
  mountain[1:2000],
  sea[1:2000],
  street[1:2000]
)
train <- sample(train)
length(train)
str(train)


evaluation <- c(
  buildings[2001:2191], 
  forest[2001:2191],
  glacier[2001:2191],
  mountain[2001:2191],
  sea[2001:2191],
  street[2001:2191]
)
evaluation <- sample(evaluation)

test <- c(buildings_test, forest_test, glacier_test, mountain_test, sea_test, street_test)
test <- sample(test)

data_prep <- function(immagini, size, canali, path, list_img){
  
  n_elementi <- length(immagini)
  #inizializiamo l'oggetto dove salvare l'output (array)
  reference_array <- array(NA, dim=c(n_elementi,size, size, canali))
  
  for (i in seq(length(immagini))) {
    #lista delle possibili categorie 
    folder_list <- list("buildings", "forest", "glacier", "mountain", "sea", "street")
    for(j in 1:length(folder_list)) {
      if(immagini[i] %in% list_img[[j]]) {
        #se l'immagine (formato jpg) apparteniene a una categoria allora crea 
        #determinato percorso
        img_path <- paste0(path, folder_list[[j]], "/", immagini[i])
        break
      }
    }
    #carichiamo le immagini appartenenti ai vari percorsi
    img <- image_load(path = img_path, target_size = c(size,size))
    #trasformiamo in array
    img_arr <- image_to_array(img)
    #rimodelliamo gli array secondo la dimensione voluta
    img_arr <- array_reshape(img_arr, c(1, size, size, canali))
    #salviamo in reference array
    reference_array[i,,,] <- img_arr
  }
  return(reference_array)
}

label_prep <- function(immagini, list_img) {
  #inializiamo il vettore 
  y <- c()
  for(i in 1:length(immagini)) {
    #stessa lista di prima delle categorie
    folder_list <- list("buildings", "forest", "glacier", "mountain", "sea", "street")
    for(j in 1:length(folder_list)) {
      if(immagini[i] %in% list_img[[j]]) {
        y <- append(y, j-1)
        break
      }
    }
  }
  return(y)
}

#Creiamo le liste degli oggetti di train e di test
list_img_train <- list(buildings, forest, glacier, mountain, sea, street)
list_img_test <- list(buildings_test, forest_test, glacier_test, mountain_test, sea_test, street_test)

size = 150
channels = 3
X_train <- data_prep(train, size, channels, "c:/data/intel/seg_train/", list_img_train)
X_evaluation <- data_prep(evaluation, size, channels, "c:/data/intel/seg_train/", list_img_train)
X_test <- data_prep(test, size, channels, "c:/data/intel/seg_test/", list_img_test)

y_train <- to_categorical(label_prep(train, list_img_train))
y_evaluation <- to_categorical(label_prep(evaluation, list_img_train))
y_test <- to_categorical(label_prep(test, list_img_test))
summary(y_train)
summary(list_img_train)


rm(list_img_train, list_img_test, buildings, forest, glacier, mountain, sea, street, buildings_test, forest_test, glacier_test, mountain_test, sea_test, street_test)

###### Data augmentation #####
#riscalo le immagini in 0,1 poi successivante applico trasformazioni alle immagini 
#per apprendere più possibile, sposto faccio zoom flippo e riempio i vuoti 
#dell' immagine con il pixel più vicino
train_datagen <- image_data_generator(rescale = 1/255,
                                      width_shift_range = 0.25,
                                      height_shift_range = 0.25,
                                      shear_range = 0.25,
                                      zoom_range = 0.35,
                                      horizontal_flip = TRUE,
                                      fill_mode = "nearest")

validation_datagen <- image_data_generator(rescale = 1/255)   
test_datagen <- image_data_generator(rescale = 1/255)

###### data generator #####
train_generator <- flow_images_from_data(
  x = X_train, 
  y = y_train,
  generator = train_datagen                                                                                       
  #batch_size = 32
  #shuffle= TRUE
)

validation_generator <- flow_images_from_data(
  x = X_evaluation, 
  y = y_evaluation,
  generator = validation_datagen                                                                                       
  #batch_size = 32
  #shuffle= TRUE
)

test_generator <- flow_images_from_data(
  x = X_test,
  y = y_test,
  generator = test_datagen
  #batch_size = 32
  #shuffle= TRUE
)

rm(X_train, y_train, X_evaluation, y_evaluation)

##### Modello cnn #####
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32,kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(0.43)  %>% 
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 6, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator
  )
dir.create("model")
model %>% save_model_hdf5("./model/Intel_classification.h5")
load_model <- load_model_hdf5("./model/Intel_classification.h5")
plot(history)

##### Previsione #####
model %>% evaluate(test_generator, steps = 50)
img_path2 <- "c:/data/intel/seg_test/sea/24301.jpg"
img_test <- image_load(img_path2, target_size = c(150, 150))                 
img_tensor_test <- image_to_array(img_test)
img_tensor_test <- array_reshape(img_tensor_test, c(1, 150, 150, 3))
img_tensor_test <- img_tensor_test/255
predictions <-predict(model,img_tensor_test) #classifica mare 
classi <- c("buildings", "forest", "glacier", "mountain", "sea", "street")
colnames(predictions) <- classi
predictions #classifica mare 
plot(as.raster(img_tensor_test[1,,,]))

valutazione <- function(percorso,gruppo){
  test1 <- list.files(path = percorso)
  img_path1 <- paste0("c:/data/intel/seg_test/",gruppo,"/",test1)
  probabilità <- matrix(nrow = length(test1),ncol = 6)
  classe_predetta <- matrix(nrow = length(test1),ncol = 2)
  for (i in 1:nrow(probabilità)) {
    img_test1 <- image_load(img_path1[i], target_size = c(150, 150))
    img_tensor_test1 <- image_to_array(img_test1)
    img_tensor_test1 <- array_reshape(img_tensor_test1, c(1, 150, 150, 3))
    img_tensor_test1<- img_tensor_test1/255
    prediction1 <-predict(model,img_tensor_test1)
    probabilità[i,] <- prediction1
  }
  for (i in 1:nrow(probabilità)) {
    classe_predetta[i,1] <- which.max(probabilità[i,])
    classe_predetta[i,2] <- max(probabilità[i,])
  }
  return(classe_predetta)
}

prop_class_corretta <- function(risultati,classe_corretta){
  count <- 0
  for (i in 1:nrow(risultati)){
    if(risultati[i,1]==classe_corretta){
      count <- count+1
      percentuale_corretti <- count/nrow(risultati)
    }
  }
  return(percentuale_corretti)
}
risultati_buildings <- valutazione(percorso="c:/data/intel/seg_test/buildings",gruppo="buildings")
percentuale_corretti_buildings <- prop_class_corretta(risultati_buildings,1)
risultati_forest <- valutazione(percorso="c:/data/intel/seg_test/forest",gruppo="forest")
percentuale_corretti_forest <- prop_class_corretta(risultati_forest,2)
risultati_glacier <- valutazione(percorso="c:/data/intel/seg_test/glacier",gruppo="glacier")
percentuale_corretti_glacier <- prop_class_corretta(risultati_glacier,3)
risultati_mountain <- valutazione(percorso="c:/data/intel/seg_test/mountain",gruppo="mountain")
percentuale_corretti_mountain <- prop_class_corretta(risultati_mountain,4)
risultati_sea <- valutazione(percorso="c:/data/intel/seg_test/sea",gruppo="sea")
percentuale_corretti_sea <- prop_class_corretta(risultati_sea,5)
risultati_street <- valutazione(percorso="c:/data/intel/seg_test/street",gruppo="street")
percentuale_corretti_street <- prop_class_corretta(risultati_street,6)

###### Matrice di confusione ####
Matrice_di_confusione <- function(Matrice_Di_Confusione,risultati,z){
  for (i in 1:6) {
    count <- 0
    for(j in 1:nrow(risultati))
    if(risultati[j,1]==i){
      count<-count+1
      Matrice_Di_Confusione[i,z] <- count
    }
  }
  for(i in 1:6){
    if(is.na(Matrice_Di_Confusione[i,z])){
      Matrice_Di_Confusione[i,z]<-0
    }
  }
  Matrice_Di_Confusione
}


Matrice_Di_Confusione <- matrix(nrow = 6,ncol = 6)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_buildings,1)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_forest,2)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_glacier,3)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_mountain,4)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_sea,5)
Matrice_Di_Confusione <- Matrice_di_confusione(Matrice_Di_Confusione,risultati_street,6)
colnames(Matrice_Di_Confusione) <- c("buildings", "forest", "glacier", "mountain", "sea", "street")
rownames(Matrice_Di_Confusione) <- c("buildings", "forest", "glacier", "mountain", "sea", "street")
Matrice_Di_Confusione



