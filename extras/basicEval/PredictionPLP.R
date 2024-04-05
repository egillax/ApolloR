# Code for evaluating in the context of 3 patient-level prediction problems
# Assumes a model was already pretrained on the entire database.
library(dplyr)
library(ApolloR)

targetId1 <- 301    # People aged 45-65 with a visit in 2013, no prior cancer
outcomeId1 <- 298   # Lung cancer
targetId2 <- 10460  # People aged 10- with major depressive disorder
outcomeId2 <- 10461 # Bipolar disorder
targetId3 <- 11931   # People aged 55=85 with a visit in 2012-2014, no prior dementia
outcomeId3 <- 6243  # Dementia

createCohort <- FALSE
rootFolder <- "~/projects/preTrainingEHR/"
preTrainedModel <- "model2"
pretrainedModelFolder <- file.path(rootFolder, preTrainedModel)
# ducky
if (createCohort) {
  connectionDetails <- DatabaseConnector::createConnectionDetails(
    dbms = "duckdb",
    server = "~/database/database.duckdb")
  cdmDatabaseSchema <- "main"
  cohortDatabaseSchema <- "cohorts"
  cohortTable <- "strategus_cohort_table"

# Get cohort definitions -------------------------------------------------------
cohortDefinitionSet <- readRDS("extras/basicEval/cohortDefinitionSet.rds")
# Generate cohorts -------------------------------------------------------------
connection <- DatabaseConnector::connect(connectionDetails)
cohortTableNames <- CohortGenerator::getCohortTableNames(cohortTable)
CohortGenerator::createCohortTables(
  connection = connection,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTableNames = cohortTableNames
)
CohortGenerator::generateCohortSet(
  connection = connection,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTableNames = cohortTableNames,
  cohortDefinitionSet = cohortDefinitionSet
)
DatabaseConnector::disconnect(connection)
}
#  Extract data ----------------------------------------------------------------
# Focusing on example 2 for now:
targetId <- targetId3
outcomeId <- outcomeId3
predictionFolder <- file.path(rootFolder, "pred3", "data")
if (!dir.exists(predictionFolder)) {
  dir.create(predictionFolder)
}

if (createCohort) {

databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cdmDatabaseName = "ducky",
  cdmDatabaseId = "ducky_v1",
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = cohortTable,
  outcomeDatabaseSchema = cohortDatabaseSchema,
  outcomeTable = cohortTable,
  targetId = targetId,
  outcomeIds = outcomeId
  )

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = 
    ApolloR::createCdmCovariateSettings(folder = file.path(predictionFolder, 
                                                           "CdmCovsFolder")),
  restrictPlpDataSettings = PatientLevelPrediction::createRestrictPlpDataSettings()
)
PatientLevelPrediction::savePlpData(plpData, file.path(predictionFolder, "plpData"))
}
plpData <- PatientLevelPrediction::loadPlpData(file.path(predictionFolder, "plpData"))
attr(plpData$covariateData, "metaData")$parquetRootFolder <- "~/projects/preTrainingEHR/pred3/data/CdmCovsFolder/c_-1_webtmqikjl/"

outcomeId <- unique(plpData$outcomes$outcomeId)

modelSettings <- ApolloR::createApolloFinetuner(numEpochs = 10,
                                                 numFreezeEpochs = 0,
                                                 learningRate = 3e-3,
                                                 weightDecay = 1e-5,
                                                 batchSize = 8,
                                                 predictionHead = "linear",
                                                 pretrainedModelFolder = pretrainedModelFolder,
                                                 maxCores = 1)
# 
# modelSettings <- ApolloR::createApolloSimpleModel(
# numEpochs = 50,
# learningRate = 3e-4,
# weightDecay = 1e-5,
# batchSize = 512,
# maxCores = 1
# )

populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
    binary = T, 
    includeAllOutcomes = T, 
    firstExposureOnly = T, 
    washoutPeriod = 365, 
    removeSubjectsWithPriorOutcome = F, 
    priorOutcomeLookback = 99999, 
    requireTimeAtRisk = T, 
    minTimeAtRisk = 1,  
    riskWindowStart = 1, 
    startAnchor = 'cohort start', 
    endAnchor = 'cohort start', 
    riskWindowEnd = 1825
    )

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = outcomeId,
  populationSettings = populationSettings
)

reticulate::use_virtualenv("apollo")

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = outcomeId,
  populationSettings = populationSettings
)
set.seed(42)
# sample with n from the outcome from population
# n <- 1000
# positive <- population %>%
#   dplyr::filter(outcomeCount == 1) %>%
#   sample_n(n)
# negative <- population %>%
#   dplyr::filter(outcomeCount == 0) %>%
#   sample_n(n)
# population <- dplyr::bind_rows(positive, negative)
# plpData$population <- population

plpResults <- PatientLevelPrediction::runPlp(
  plpData = plpData,
  outcomeId = outcomeId,
  modelSettings = modelSettings,
  populationSettings = populationSettings,
  sampleSettings = PatientLevelPrediction::createSampleSettings("underSample",
                                                                sampleSeed = 42),
  analysisId = 1,
  analysisName = "ApolloTest",
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42),
  executeSettings = PatientLevelPrediction::createExecuteSettings(
    runSplitData = TRUE,
    runSampleData = TRUE,
    runPreprocessData = FALSE,
    runModelDevelopment = TRUE,
  ),
  saveDirectory = "./plpResultsDementia/")
