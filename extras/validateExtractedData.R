#' Validate that extracted Parquet data includes all required columns.
#'
#' Usage:
#'   Rscript extras/validateExtractedData.R /path/to/extracted/data [/path/to/tableColumnsToExtract.csv]
#'
#' The first argument must point to the folder created by `extractCdmToParquet()`.
#' Optionally provide a second argument with the path to `tableColumnsToExtract.csv`.

suppressPackageStartupMessages({
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop(
      "The 'arrow' package is required. Please install it before running this script."
    )
  }
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop(
    "Usage: Rscript extras/validateExtractedData.R /path/to/extracted/data [/path/to/tableColumnsToExtract.csv]"
  )
}

dataFolder <- args[[1]]
if (!dir.exists(dataFolder)) {
  stop(sprintf("Data folder '%s' does not exist.", dataFolder))
}

if (length(args) >= 2) {
  specPath <- args[[2]]
} else {
  specPath <- file.path("inst", "tableColumnsToExtract.csv")
}

if (!file.exists(specPath)) {
  packageSpec <- system.file("tableColumnsToExtract.csv", package = "ApolloR")
  if (packageSpec == "") {
    stop(sprintf(
      "Cannot find specification file at '%s' and no installed package copy found.",
      specPath
    ))
  }
  specPath <- packageSpec
}

spec <- utils::read.csv(specPath, stringsAsFactors = FALSE)
requiredTables <- unique(spec$cdmTableName)

issues <- list()

recordIssue <- function(table, issueType, details) {
  issues[[length(issues) + 1]] <<- list(
    table = table,
    issue = issueType,
    details = details
  )
}

for (tableName in requiredTables) {
  tableFolder <- file.path(dataFolder, tableName)
  if (!dir.exists(tableFolder)) {
    recordIssue(
      tableName,
      "missing_table_folder",
      sprintf("Folder '%s' not found.", tableFolder)
    )
    next
  }
  parquetFiles <- list.files(
    tableFolder,
    pattern = "\\.parquet$",
    full.names = TRUE
  )
  if (length(parquetFiles) == 0) {
    recordIssue(
      tableName,
      "missing_parquet_files",
      "No parquet files found in folder."
    )
    next
  }
  dataset <- tryCatch(
    arrow::open_dataset(tableFolder, format = "parquet"),
    error = function(err) {
      recordIssue(tableName, "open_dataset_failed", conditionMessage(err))
      return(NULL)
    }
  )
  if (is.null(dataset)) {
    next
  }
  availableColumns <- dataset$schema$names
  expectedColumns <- spec$cdmFieldName[spec$cdmTableName == tableName]
  missingColumns <- setdiff(expectedColumns, availableColumns)
  if (length(missingColumns) > 0) {
    recordIssue(
      tableName,
      "missing_columns",
      sprintf(
        "Missing columns: %s",
        paste(sort(missingColumns), collapse = ", ")
      )
    )
  }
}

if (length(issues) == 0) {
  message(sprintf(
    "Success: all required columns were found in '%s'.",
    normalizePath(dataFolder)
  ))
} else {
  message("Validation issues detected:")
  for (item in issues) {
    message(sprintf(
      " - Table '%s' [%s]: %s",
      item$table,
      item$issue,
      item$details
    ))
  }
  stop(sprintf("Validation failed for %d table(s).", length(issues)))
}
