library(caTools)
library(rpart)
library(rpart.plot)

data <- read.csv("voice.csv")

set.seed(777)
spl <- sample.split(data$label, 0.7)
train <- subset(data, spl == TRUE)
test <- subset(data, spl == FALSE)

print("Baseline model")

table(train$label)
1109/nrow(train)

table(test$label)
475/nrow(test)

print("Logistic Regression Model")

genderLog <- glm(label ~ ., data = train, family = 'binomial')

predictLog <- predict(genderLog, type = 'response')
table(train$label, predictLog >= 0.5)
(1073+1081)/nrow(train)

predictLog2 <- predict(genderLog, newdata = test, type = 'response')
table(test$label, predictLog2 >= 0.5)
(462+468)/nrow(test)


