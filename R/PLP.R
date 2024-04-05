
#' Apollo Finetuner
#' @param numEpochs Number of epochs to train the model.
#' @param numFreezeEpochs Number of epochs to freeze the pretrained model.
#' @param learningRate Learning rate for the optimizer.
#' @param weightDecay Weight decay for the optimizer.
#' @param batchSize Batch size for training.
#' @param predictionHead The type of prediction head to use. Options are "lstm" and "linear".
#' @param pretrainedModelFolder The folder containing the pretrained model.
#' @param personSequenceFolder The folder where the person sequence data was written by the 
#' `processCdmData()` function. 
#' @param maxCores The maximum number of CPU cores to use during fine-tuning. 
#' @param device The device to use for fine-tuning. Options are "cuda:x" where x is a number or "cpu".
#'@export
createApolloFinetuner <- function(numEpochs = 1,
                                  numFreezeEpochs = 1,
                                  learningRate = 3e-4,
                                  weightDecay = 1e-5,
                                  batchSize = 32,
                                  predictionHead = "lstm",
                                  pretrainedModelFolder = "path/to/model",
                                  personSequenceFolder = NULL,
                                  maxCores = 1,
                                  device = "cuda") {
  # check inputs with checkmate
  errorMessages <- checkmate::makeAssertCollection()
  checkmate::assert_numeric(numEpochs, add = errorMessages)
  checkmate::assert_numeric(numFreezeEpochs, add = errorMessages)
  checkmate::assert_numeric(learningRate, add = errorMessages)
  checkmate::assert_numeric(weightDecay, add = errorMessages)
  checkmate::assert_numeric(batchSize, add = errorMessages)
  checkmate::assert_character(predictionHead, len = 1, add = errorMessages)
  checkmate::assert_character(pretrainedModelFolder, len = 1, add = errorMessages)
  pretrainedModelFolder <- normalizePath(pretrainedModelFolder)
  checkmate::assert_directory_exists(pretrainedModelFolder, add = errorMessages)
  if (!is.null(personSequenceFolder)) {
    personSequenceFolder <- normalizePath(personSequenceFolder)
    checkmate::assert_directory_exists(personSequenceFolder, add = errorMessages)
  }
  checkmate::assert_numeric(maxCores, add = errorMessages)
  checkmate::assert_character(device, len = 1, add = errorMessages)
  
  checkmate::reportAssertions(errorMessages)
  
  # parameters to use in gridSearch if more than one
  paramGrid <- list(
    learningRate = learningRate,
    weightDecay = weightDecay,
    predictionHead = predictionHead,
    numFreezeEpochs = numFreezeEpochs
  )
  param <- PatientLevelPrediction::listCartesian(paramGrid)
  
  results <- list(
    fitFunction = "ApolloR::finetune",
    predictFunction = "ApolloR::finetunePredict",
    modelFolder = pretrainedModelFolder,
    sequenceFolder = personSequenceFolder,
    maxCores = maxCores,
    device = device,
    batchsize = batchSize,
    numEpochs = numEpochs,
    param = param,
    saveType = "file",
    modelParamNames = c("learningRate", "weightDecay", "predictionHead", "numFreezeEpochs"),
    modelType = "Apollo"
  )
  attr(results$param, "settings")$modelType <- results$modelType
  
  class(results) <- "modelSettings"
  return(results)
}

#' Apollo Simple Model
#' @param numEpochs Number of epochs to train the model.
#' @param learningRate Learning rate for the optimizer.
#' @param weightDecay Weight decay for the optimizer.
#' @param batchSize Batch size for training.
#' @param personSequenceFolder The folder where the person sequence data was written by the
#' `processCdmData()` function.
#' @param maxCores The maximum number of CPU cores to use during fine-tuning.
#' @param device The device to use for fine-tuning. Options are "cuda:x" where x is a number or "cpu".
#' @export
createApolloSimpleModel <- function(numEpochs = 1,
                                    learningRate = 3e-4,
                                    weightDecay = 1e-5,
                                    batchSize = 512,
                                    personSequenceFolder = NULL,
                                    maxCores = 1,
                                    device = "cuda",
                                    maxSequenceLength = 512L
) {
  # check inputs with checkmate
  errorMessages <- checkmate::makeAssertCollection()
  checkmate::assert_numeric(numEpochs, add = errorMessages)
  checkmate::assert_numeric(learningRate, add = errorMessages)
  checkmate::assert_numeric(weightDecay, add = errorMessages)
  checkmate::assert_numeric(batchSize, add = errorMessages)
  checkmate::assert_character(pretrainedModelFolder, len = 1, add = errorMessages)
  checkmate::assert_directory_exists(pretrainedModelFolder, add = errorMessages)
  if (!is.null(personSequenceFolder)) {
    personSequenceFolder <- normalizePath(personSequenceFolder)
    checkmate::assert_directory_exists(personSequenceFolder, add = errorMessages)
  }
  checkmate::assert_numeric(maxCores, add = errorMessages)
  checkmate::assert_character(device, len = 1, add = errorMessages)
  
  checkmate::reportAssertions(errorMessages)
  
  # parameters to use in gridSearch if more than one
  paramGrid <- list(
    learningRate = learningRate,
    weightDecay = weightDecay,
    numFreezeEpochs = 0
  )
  param <- PatientLevelPrediction::listCartesian(paramGrid)
  
  results <- list(
    fitFunction = "ApolloR::finetune",
    predictFunction = "ApolloR::finetunePredict",
    sequenceFolder = personSequenceFolder,
    maxCores = maxCores,
    device = device,
    batchsize = batchSize,
    numEpochs = numEpochs,
    simpleRegressionModel = TRUE,
    modelSettings = list(max_sequence_length = maxSequenceLength,
                         concept_embedding = TRUE,
                         visit_concept_embedding = FALSE,
                         visit_order_embedding = FALSE,
                         segment_embedding = FALSE,
                         age_embedding = FALSE,
                         date_embedding = FALSE
                         ),
    param = param,
    saveType = "file",
    modelParamNames = c("learningRate", "weightDecay"),
    modelType = "Simple"
  )
  attr(results$param, "settings")$modelType <- results$modelType
  
  class(results) <- "modelSettings"
  return(results)
}


#' Apollo Finetuner
#' @param trainData The covariate data to use for fine-tuning the model.
#' @param modelSettings The model settings to use for fine-tuning the model.
#' @param analysisId The analysis ID to use for fine-tuning the model.
#' @param analysisPath The path to save the analysis to.
#' @export
finetune <- function(
    trainData,
    modelSettings,
    analysisId,
    analysisPath,
    ...
) {
  start <- Sys.time()
  if (!is.null(trainData$folds)) {
    trainData$labels <- merge(trainData$labels, trainData$fold, by = "rowId")
  }
  # highest non-test fold is validation, rest of positive folds is training
  isTrainingCondition <- (trainData$labels$index != max(trainData$labels$index)) & (trainData$labels$index > 0)
  labels <- trainData$labels %>% 
    dplyr::mutate(is_training = isTrainingCondition) %>%
    dplyr::select(.data$rowId, .data$outcomeCount, .data$is_training) %>%
    dplyr::rename(rowId = "rowId")
  workingDir <- normalizePath(file.path(analysisPath, "workingDir"))
  if (is.null(modelSettings$sequenceFolder)) {
    modelSettings$dataFolder <- attr(trainData$covariateData, "metaData")$parquetRootFolder
    modelSettings$dataFolder <- normalizePath(modelSettings$dataFolder)
    generateCdmSequences(modelSettings, labels, workingDir)
  } else {
    if (dir.exists(file.path(workingDir, "sequences"))) {
      ParallelLogger::logInfo("Found problem sequences, reusing")
    }
  }  
  param <- modelSettings$param[[1]]
  trainingSettings <- list(
    train_fraction = "plp",
    num_epochs = as.integer(modelSettings$numEpochs),
    num_freeze_epochs = as.integer(param$numFreezeEpochs),
    learning_rate = param$learningRate,
    weight_decay = param$weightDecay
  )
  trainModelSettings <- list(
    system = list(
      sequence_data_folder = normalizePath(file.path(workingDir, "sequences")),
      output_folder = workingDir,
      pretrained_model_folder = modelSettings$modelFolder,
      batch_size = as.integer(modelSettings$batchsize),
      checkpoint_every = as.integer(1),
      writer = "json"
    ),
    learning_objectives = list(
      truncate_type = "tail",
      predict_new = FALSE,
      label_prediction = tolower(param$predictionHead) != "lstm",
      lstm_label_prediction = tolower(param$predictionHead) == "lstm"
    ),
    training = trainingSettings
  )
  if (!is.null(param$predictionHead)) {
    if (param$predictionHead == "lstm") {
      trainModelSettings$learning_objectives$lstm_label_prediction <- TRUE
    } else {
      trainModelSettings$learning_objectives$label_prediction <- TRUE
    } 
  } else {
    trainModelSettings$learning_objectives$label_prediction <- TRUE
  }
  
  if (!is.null(modelSettings$modelFolder)) {
    preTrainedModelSettings <- yaml::read_yaml(file.path(modelSettings$modelFolder, "model.yaml"))
    trainModelSettings["model"] <- list(preTrainedModelSettings)
  } else {
    trainModelSettings["model"] <- list(modelSettings$modelSettings)
  }
  
  if (modelSettings$modelType == "Simple") {
    trainModelSettings$learning_objectives$simple_regression_model <- TRUE
  }
  yamlFileName <- tempfile("model_trainer", fileext = ".yaml")
  on.exit(unlink(yamlFileName))
  yaml::write_yaml(trainModelSettings,
                   yamlFileName)
  ensurePythonFolderSet()
  trainModelModule <- reticulate::import("training.train_model")
  trainModelModule$main(c(yamlFileName, ""))
  metrics <- ParallelLogger::convertJsonToSettings(file.path(workingDir, "metrics.json"))
  # find best epoch
  bestEpoch <- which.max(lapply(metrics, function(x) x$metrics$validation$AUC))
  predictions <- predictValidation(
    workingDir = workingDir,
    param = param,
    modelSettings = modelSettings,
    preTrainedModelSettings = trainModelSettings$model,
    epoch = bestEpoch
  )
  
  # get validation predictions
  valPredictions <- predictions %>% 
    dplyr::inner_join(trainData$labels %>% dplyr::filter(index == 3), 
                      by = dplyr::join_by("person_id" == "subjectId")) %>%
    dplyr::rename(subjectId = "person_id", 
                  value = prediction) %>%
    dplyr::select(-c("observation_period_id")) %>% 
    dplyr::mutate(index = .data$index - 2)
  
  attr(valPredictions, "metaData") <- list(modelType = "binary")
  gridSearchPredictions <- list()
  gridSearchPredictions[[1]] <- list(
    prediction = valPredictions,
    param = param
  )
  paramGridSearch <- lapply(gridSearchPredictions, 
                            function(x) do.call(PatientLevelPrediction::computeGridPerformance, x))
  
  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))
  
  finalParam <- paramGridSearch[[optimalParamInd]]$param
  
  cvPrediction <- gridSearchPredictions[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- 'CV'
  
  trainPrediction <-  predictions %>% 
    dplyr::inner_join(trainData$labels %>% dplyr::filter(index %in% c(1,2)), 
                      by = dplyr::join_by("person_id" == "subjectId")) %>%
    dplyr::rename(subjectId = "person_id", 
                  value = prediction) %>%
    dplyr::select(-c("observation_period_id"))

  trainPrediction$evaluationType <- "Train"
  
  totalPrediction <- rbind(cvPrediction, trainPrediction)
  # model name is of the form checkpoint_00x.pth where x is the bestEpoch
  modelName <- paste0("checkpoint_", sprintf("%03d", bestEpoch), ".pth")
  finalModel <- file.path(workingDir, modelName)
   
  variableImportance <- NULL
  cvResult <- list(
    model = finalModel, 
    prediction = totalPrediction,
    finalParam = finalParam,
    paramGridSearch = paramGridSearch,
    variableImportance = variableImportance )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  
  prediction <- cvResult$prediction
  
  variableImportance <- cvResult$variableImportance
  
  covariateRef <- trainData$covariateData$covariateRef %>% dplyr::collect()
  covariateRef$covariateValue <- 0
  covariateRef$included <- 0
  
  comp <- start - Sys.time()
  
  result <- list(
    model = cvResult$model,
    
    preprocessing = list(
      featureEngineering = attr(trainData, "metaData")$featureEngineering,
      tidyCovariates = attr(trainData$covariateData, "metaData")$tidyCovariateDataSettings, 
      requireDenseMatrix = F
    ),
    
    prediction = prediction,
    
    modelDesign = PatientLevelPrediction::createModelDesign(
      targetId = attr(trainData, "metaData")$targetId,
      outcomeId = attr(trainData, "metaData")$outcomeId,
      restrictPlpDataSettings = attr(trainData, "metaData")$restrictPlpDataSettings,
      covariateSettings = attr(trainData, "metaData")$covariateSettings,
      populationSettings = attr(trainData, "metaData")$populationSettings,
      featureEngineeringSettings = attr(trainData$covariateData, "metaData")$featureEngineeringSettings,
      preprocessSettings = attr(trainData$covariateData, "metaData")$preprocessSettings,
      modelSettings = modelSettings,
      splitSettings = attr(trainData, "metaData")$splitSettings,
      sampleSettings = attr(trainData, "metaData")$sampleSettings
    ),
    
    trainDetails = list(
      analysisId = analysisId,
      analysisSource = '', #TODO add from model
      developmentDatabase = attr(trainData, "metaData")$cdmDatabaseName,
      developmentDatabaseSchema = attr(trainData, "metaData")$cdmDatabaseSchema, 
      attrition = attr(trainData, "metaData")$attrition, 
      trainingTime = paste(as.character(abs(comp)), attr(comp,'units')),
      trainingDate = Sys.Date(),
      modelName = "Apollo", 
      finalModelParameters = cvResult$finalParam,
      hyperParamSearch = hyperSummary
    ),
    
    covariateImportance = covariateRef
  )
  
  class(result) <- "plpModel"
  attr(result, "predictionFunction") <- modelSettings$predictFunction
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- modelSettings$saveType
  
  return(result)
}
  
#' Apollo Finetuner prediction function
#' @param plpModel The model to use for prediction.
#' @param data The covariate data to use for prediction.
#' @param cohort The cohort to use for prediction.
#' @export
finetunePredict <- function(plpModel,
                            data, 
                            cohort){
  labels <- cohort %>% 
    dplyr::mutate(is_training = FALSE) %>%
    dplyr::select(.data$rowId, .data$outcomeCount, .data$is_training) %>%
    dplyr::rename(rowId = "rowId")
  workingDir <- dirname(plpModel$model)
  modelSettings <- plpModel$modelDesign$modelSettings
  modelSettings$dataFolder <- attr(data$covariateData, "metaData")$parquetRootFolder
  modelSettings$dataFolder <- normalizePath(modelSettings$dataFolder)
  generateCdmSequences(modelSettings, labels, workingDir,
                       sequenceFolderName = "test_sequences")
  
  epoch <- as.integer(substr(basename(plpModel$model), 12, 14))
  param <- plpModel$trainDetails$finalModelParameters
  testSettings <- list(
    system = list(
      sequence_data_folder = normalizePath(file.path(workingDir, "test_sequences")),
      output_folder = workingDir,
      pretrained_model_folder = modelSettings$modelFolder,
      batch_size = as.integer(modelSettings$batchsize),
      checkpoint_every = as.integer(1),
      writer = "json",
      finetuned_epoch = as.integer(epoch),
      prediction_output_file = file.path(workingDir, "predictions.csv")
    ),
    learning_objectives = list(
      truncate_type = "tail",
      predict_new = TRUE
    ),
    training = list(
      train_fraction = 0,
      num_epochs = 1,
      learning_rate = 3e-4
    )
  ) 
  if (!is.null(param$predictionHead)) {
    if (param$predictionHead == "lstm") {
      testSettings$learning_objectives$lstm_label_prediction <- TRUE
    } else {
      testSettings$learning_objectives$label_prediction <- TRUE
    } 
  } else {
    testSettings$learning_objectives$label_prediction <- TRUE
  } 
  
if (modelSettings$modelType == "Simple") {
    testSettings$learning_objectives$simple_regression_model <- TRUE
  }
  
  if (!is.null(modelSettings$modelFolder)) {
    preTrainedModelSettings <- yaml::read_yaml(file.path(modelSettings$modelFolder, "model.yaml"))
    testSettings["model"] <- list(preTrainedModelSettings)
  } else {
    testSettings["model"] <- list(modelSettings$modelSettings)
  }
 
  
  ensurePythonFolderSet()
  trainModelModule <- reticulate::import("training.train_model")
  yamlFileName <- tempfile("model_test", fileext = ".yaml")
  on.exit(unlink(yamlFileName))
  yaml::write_yaml(testSettings,
                   yamlFileName)
  trainModelModule$main(c(yamlFileName, ""))
  
  # load csv with test predictions
  preds <- readr::read_csv(file.path(workingDir, "predictions.csv"))
  predictions <- cohort %>% 
    dplyr::inner_join(preds, by = dplyr::join_by("subjectId" == "person_id")) %>% 
    dplyr::select(-c("observation_period_id")) %>%
    dplyr::rename(value = prediction)
  return(predictions)
}

predictValidation <- function(workingDir,
                              param,
                              modelSettings,
                              preTrainedModelSettings,
                              epoch
                              ) {
  # load model from best epoch and get predictions from validation set
    validationSettings <- list(
      system = list(
        sequence_data_folder = normalizePath(file.path(workingDir, "sequences")),
        output_folder = workingDir,
        pretrained_model_folder = modelSettings$modelFolder,
        batch_size = as.integer(modelSettings$batchsize),
        checkpoint_every = as.integer(1),
        writer = "json",
        finetuned_epoch = as.integer(epoch),
        prediction_output_file = file.path(workingDir, "predictions.csv")
      ),
      learning_objectives = list(
        truncate_type = "tail",
        predict_new = TRUE,
        label_prediction = tolower(param$predictionHead) != "lstm",
        lstm_label_prediction = tolower(param$predictionHead) == "lstm"
      ),
      training = list(
        train_fraction = 0,
        num_epochs = 1,
        learning_rate = 3e-4
      ),
      model = preTrainedModelSettings
  )
  if (!is.null(param$predictionHead)) {
    if (param$predictionHead == "lstm") {
      validationSettings$learning_objectives$lstm_label_prediction <- TRUE
    } else {
      validationSettings$learning_objectives$label_prediction <- TRUE
    } 
  } else {
    validationSettings$learning_objectives$label_prediction <- TRUE
  }
  
  if (modelSettings$modelType == "Simple") {
    validationSettings$learning_objectives$simple_regression_model <- TRUE
  }
  
  ensurePythonFolderSet()
  trainModelModule <- reticulate::import("training.train_model")
  yamlFileName <- tempfile("model_validation", fileext = ".yaml")
  on.exit(unlink(yamlFileName))
  yaml::write_yaml(validationSettings,
                   yamlFileName)
  trainModelModule$main(c(yamlFileName, ""))
  
  # load csv with validation predictions
  predictions <- readr::read_csv(file.path(workingDir, "predictions.csv"))
  return(predictions)  
}

generateCdmSequences <- function(modelSettings, labels, workingDir,
                                 sequenceFolderName="sequences") {
  ParallelLogger::logInfo("Generating sequences for CDM data")
  labelFolder <- tempfile()
  writeLabelsToParquet(labels = labels, 
                       parquetRootFolder = modelSettings$dataFolder,
                       labelFolder = labelFolder)
  if (is.null(modelSettings$modelFolder)) {
    mappingSettings <- list(
      map_drugs_to_ingredients = FALSE,
      concepts_to_remove = c(0, 900000010)
    )  
  } else {
    mappingSettings <- yaml::read_yaml(file.path(modelSettings$modelFolder, "cdm_mapping.yaml"))
  }
  modelSettings$sequenceFolder <- tempfile()
  processCdmData(cdmDataPath = modelSettings$dataFolder, 
                 personSequenceFolder = modelSettings$sequenceFolder,
                 mappingSettings = mappingSettings,
                 labels = labelFolder,
                 maxCores = modelSettings$maxCores)
  outDir <- file.path(workingDir, sequenceFolderName)
  if (!dir.exists(outDir)) {
    dir.create(outDir, recursive = TRUE)      
  }
  status <- file.copy(from = file.path(modelSettings$sequenceFolder, dir(modelSettings$sequenceFolder)), 
                      to = outDir, 
                      recursive = TRUE)
  if (!all(status)) {
      stop("Failed to copy sequence folder to analysis path")
  }
}
  
filterProblemSequences <- function(modelSettings, labels, workingDir, 
                                   name = "sequences") {
  ensurePythonFolderSet()
  utils <- reticulate::import("cdm_processing.cdm_processor_utils")
  ParallelLogger::logInfo("Filtering sequences to prediction problem") 
  utils$filter_prediction_problem(modelSettings$sequenceFolder,
                                  reticulate::r_to_py(labels),
                                  workingDir,
                                  name)
  file.copy(from = file.path(modelSettings$sequenceFolder, "cdm_mapping.yaml"),
            to = file.path(workingDir, name,"cdm_mapping.yaml"),
            overwrite = TRUE)
}