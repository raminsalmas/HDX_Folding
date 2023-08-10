library(caret)
library(gbm)

# Define the DataProcessor class
DataProcessor <- setRefClass(
  "DataProcessor",
  fields = list(
    file_path = "character"
  ),
  methods = list(
    load_data = function() {
      data <- read.csv(self$file_path, header = TRUE)
      return(data)
    },
    preprocess_data = function(data_1) {
      lnp <- as.numeric(data_1[, 3])
      sasa <- as.numeric(data_1[, 4])
      phi <- as.numeric(data_1[, 5])
      psi <- as.numeric(data_1[, 6])
      r_matrix <- as.numeric(data_1[, 7])
      SS <- data_1[, 2]
      k_int <- as.numeric(data_1[, 8])
      k_obs <- as.numeric(data_1[, 9])
      res <- data_1[, 1]
      L <- data_1[, 11]
      R <- data_1[, 12]
      LL <- data_1[, 13]
      RR <- data_1[, 14]
      
      le <- labelEncoder()
      res <- le$fit_transform(res)
      L <- le$fit_transform(L)
      R <- le$fit_transform(R)
      LL <- le$fit_transform(LL)
      RR <- le$fit_transform(RR)
      
      Kio <- cbind(res, k_int, r_matrix, k_obs)
      
      aa <- cbind(phi, t(psi), t(sasa))
      
      aa_phi_psi <- cbind(phi, t(psi))
      
      aa_phi_psi_lnp <- cbind(phi, t(psi), t(lnp))
      
      aa_phi_psi_sasa_lnp <- cbind(phi, t(psi), t(sasa), t(lnp))
      
      aa_phi_psi_sasa <- cbind(phi, t(psi), t(sasa))
      
      return(list(Kio, aa_phi_psi, aa_phi_psi_lnp, aa_phi_psi_sasa_lnp, aa_phi_psi_sasa))
    }
  )
)

# Define the ModelEvaluator class
ModelEvaluator <- setRefClass(
  "ModelEvaluator",
  fields = list(
    model = "ANY"
  ),
  methods = list(
    evaluate_model = function(X, y) {
      folds <- createMultiFolds(y, k = 5, times = 1)
      predicted_targets <- c()
      actual_targets <- c()
      predicted_targets_prob <- c()
      
      for (i in 1:length(folds)) {
        train_ix <- folds[[i]]$train
        test_ix <- folds[[i]]$test
        
        train_x <- X[train_ix,]
        train_y <- y[train_ix]
        test_x <- X[test_ix,]
        test_y <- y[test_ix]
        
        classifiers <- self$model
        classifiers <- train(
          classifiers,
          x = train_x,
          y = train_y,
          method = "gbm"
        )
        predicted_labels <- predict(classifiers, newdata = test_x)
        predicted_prob <- predict(classifiers, newdata = test_x, type = "response")
        
        predicted_targets <- c(predicted_targets, predicted_labels)
        predicted_targets_prob <- c(predicted_targets_prob, predicted_prob)
        actual_targets <- c(actual_targets, test_y)
      }
      
      return(list(predicted_targets, actual_targets, predicted_targets_prob))
    }
  )
)

# Load required libraries
library(caret)
library(gbm)

# Define the file path
file_1 <- "data5.csv"

# Initialize DataProcessor and load data
data_processor <- DataProcessor(file_path = file_1)
data_1 <- data_processor$load_data()

# Preprocess data
preprocessed_data <- data_processor$preprocess_data(data_1)
Kio <- preprocessed_data[[1]]
aa_phi_psi <- preprocessed_data[[2]]

# Create model
hparam <- list(
  learning_rate = 0.2,
  n_trees = 500,
  interaction.depth = 1,
  n.minobsinnode = 10,
  shrinkage = 1,
  n.cores = 1,
  distribution = "bernoulli",
  bag.fraction = 0.5,
  train.fraction = 1,
  cv.folds = 0
)
model_gb <- train(
  x = Kio,
  y = aa_phi_psi,
  method = "gbm",
  trControl = trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 1,
    verboseIter = TRUE
  ),
  tuneGrid = hparam
)

# Evaluate model
model_evaluator <- ModelEvaluator(model = model_gb)
evaluation_results <- model_evaluator$evaluate_model(Kio, aa_phi_psi)
predicted_targets <- evaluation_results[[1]]
actual_targets <- evaluation_results[[2]]
predicted_targets_prob <- evaluation_results[[3]]
