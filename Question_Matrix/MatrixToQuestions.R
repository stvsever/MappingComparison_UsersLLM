### Context
#  This script takes a matrix of barriers (rows) and coping strategies (columns)
# and creates sets of coping strategies for each barrier. Each set contains up to 10 coping strategies,
# of which up to 9 are relevant coping strategies and at least 1 is an irrelevant coping strategy
# 
# Written By Maya Braun
####################################
library(readxl)
library(writexl)
setwd("//files.ugent.be/mabraun/shares/iop_caps/Studies/8 Evaluation Which Solutions when")

MatrixInputR <- read_excel("//files.ugent.be/mabraun/shares/iop_caps/Studies/8 Evaluation Which Solutions when/MatrixInputR2.xlsx")

# Case 1: Only chosen barriers (according to matrix) ####

## Create sets of all coping strategies ####

CS <- colnames(MatrixInputR)[2:65]
CS_mot <- CS[1:10]

# Create full set per barrier
CS_rel  <- list(
  Barrier = rep("", 50),
  Relevant_Solutions = rep(list(character(0)), 50),
  Irrelevant_Solutions = rep(list(character(0)), 50),
  Amount_Q = rep(numeric(), 50)
)


for(i in c(1:50)){
 relevantsolutions <- c() 
 for(cop in CS){
  if(is.na(MatrixInputR[i, cop]) == FALSE){
  relevantsolutions <- c(relevantsolutions, cop)
   }
 }
 CS_rel$Barrier[i]<- MatrixInputR[i, "Barrier"]
 CS_rel$Relevant_Solutions[i] <- list(relevantsolutions)
 CS_rel$Irrelevant_Solutions[i] <- list(setdiff(CS, relevantsolutions))
 CS_rel$Amount_Q[i] <- ifelse(length(CS_rel$Relevant_Solutions[i][[1]]) %% 10 == 0, 
       length(CS_rel$Relevant_Solutions[i][[1]])/10 + 1,
       ifelse(ceiling(length(CS_rel$Relevant_Solutions[i][[1]])/10) > (10-(length(CS_rel$Relevant_Solutions[i][[1]]) %% 10)),
              ceiling(length(CS_rel$Relevant_Solutions[i][[1]])/10)+1,
              ceiling(length(CS_rel$Relevant_Solutions[i][[1]])/10)))
}

saveRDS(CS_rel, "Overview_Relevant_Solutions.rds")
## Make sets of answer options for each barrier ####

amount_Q <- sum(CS_rel$Amount_Q)

question_sets <- list(Barrier = rep("", amount_Q),
                      Number = rep("", amount_Q),
                      Solutions_relevant = rep(list(character(0)), amount_Q), 
                      Solutions_irrelevant = rep(list(character(0)), amount_Q),
                      Solutions_all = rep(list(character()), amount_Q))

# For each set: Choose random relevant solutions
count <- 1

for(barr in c(1:length(CS_rel$Barrier))){
 
 # define relevant and irrelevant solutions for the barrier 
 relevantsolutions <- CS_rel$Relevant_Solutions[[barr]]
 irrelevantsolutions <- CS_rel$Irrelevant_Solutions[[barr]]
 
 # calculate how many relevant solutions we need in each question set (rounded up)
 relevant_per_set <- round(length(CS_rel$Relevant_Solutions[barr][[1]]) / CS_rel$Amount_Q[barr])

 # Loop over the amount of questions for the barrier and define barrier and number
 for(Q in c(1:CS_rel$Amount_Q[barr])){
  question_sets$Barrier[count] = CS_rel$Barrier[[barr]]
  question_sets$Number[count] = Q

 # If this is not the last question for this barrier, remove a random set of relevant solutions 
 # and add them to this questions
 # otherwise use the remaining set of solutions
  if(Q != CS_rel$Amount_Q[barr]){
     relevant_per_set <- ifelse((length(relevantsolutions) - relevant_per_set != 10), relevant_per_set, relevant_per_set + 1)
     
     question_sets$Solutions_relevant[count] <- list(sample(relevantsolutions, relevant_per_set))
     question_sets$Solutions_irrelevant[count] <- list(sample(irrelevantsolutions, 10-relevant_per_set))
     
     relevantsolutions <- setdiff(relevantsolutions, unlist(question_sets$Solutions_relevant[count]))
     irrelevantsolutions <- setdiff(irrelevantsolutions, unlist(question_sets$Solutions_irrelevant[count]))
     
  } else {
   question_sets$Solutions_relevant[count] <- list(relevantsolutions)
   question_sets$Solutions_irrelevant[count] <- list(sample(irrelevantsolutions, 10-length(relevantsolutions)))
  }
  question_sets$Solutions_all[count] <- list(c(question_sets$Solutions_relevant[count][[1]], question_sets$Solutions_irrelevant[count][[1]]))
 # Count up
  count <- count+1
  }
}

data_frame <- data.frame(
  Barrier = unlist(question_sets$Barrier),
  Number = unlist(question_sets$Number),
  t(sapply(question_sets$Solutions_all, unlist))
)

# Specify the Excel file name
file_name <- "2023-11-07_qualtrics_questions.xlsx"

# Write the data frame to an Excel file
write_xlsx(data_frame, file_name)
