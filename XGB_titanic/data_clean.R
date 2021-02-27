# Data preProcess()
library(zoo)
library(tidyverse)
train = read.csv("./data/train.csv")
tests = read.csv("./data/test.csv")

train$title = str_extract(train$Name, '\\w+\\.')
tests$title = str_extract(tests$Name, '\\w+\\.')

train = merge(train, train %>% group_by(title) %>% summarise(n()), by = 'title')
tests = merge(tests, tests %>% group_by(title) %>% summarise(n()), by = 'title')

train$title[train$`n()` < 8] = 'rare'
tests$title[tests$`n()` < 8] = 'rare'

#--------------------------------------------

train = train[, !names(train) %in% c("PassengerId", "n()", "Name")]
tests = tests[, !names(tests) %in% c("PassengerId", "n()", "Name")]

train$alone = "0"
train$alone[train$SibSp == 0] = "1"
tests$alone = "0"
tests$alone[tests$SibSp == 0] = "1"
#--------------------------------------------
train$Cabin2 = c(train$Cabin %>% str_sub(1, 1))
train$Cabin2[which((train$Cabin2) == "")] = "N"

tests$Cabin2 = c(tests$Cabin %>% str_sub(1, 1))
tests$Cabin2[which((tests$Cabin2) == "")] = "N"
#--------------------------------------------
is.na(str_extract(train$Ticket, '^\\d'))
str_extract(train$Ticket, '[[:alpha:]]')
