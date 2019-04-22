rm(list=ls(all=T))
setwd("C:/Users/Pranab/MyDataScience/Edwisor/Project 1 Bike rentals")

bikes <- read.csv('day.csv', header = TRUE)

par(mfrow=c(1,3))
boxplot(bikes$temp, ylab='Temperature')
boxplot(bikes$hum, ylab='Humidity')
boxplot(bikes$windspeed, ylab='Wind-speed')

plot(density(bikes$temp))
plot(density(bikes$hum))
plot(density(bikes$windspeed))

for (col in c('hum', 'windspeed')) {

  p25 = quantile(bikes[,col], probs = c(0.25))
  p75 = quantile(bikes[,col], probs = c(0.75))
  iqr = p75 - p25
  max = p75 + 1.5*iqr
  min = p25 - 1.5*iqr
  indxl = which(bikes[,col] <= min)
  indxg = which(bikes[,col] >= max)
  if(length(indxl) != 0){
    bikes <- bikes[-c(indxl), ]}
  if(length(indxg) !=0){
    bikes <- bikes[-c(indxg), ]}
}

library(ggplot2)

par(mfrow=c(2,3))
for(col in c('mnth','weekday','weathersit', 'season', 'holiday', 'workingday')){
  boxplot(bikes[,'cnt']~bikes[,col])
}

par(mfrow=c(1,3))
for(col in c('temp','hum','windspeed')){
  plot(bikes[,col], bikes[,'cnt'])
}

corr_cont <- round(cor(bikes[,c('temp', 'hum', 'windspeed','cnt')]), 2)
install.packages('ggcorrplot')

library(ggcorrplot)
ggcorrplot(corr_cont, lab=TRUE)

install.packages('dummies')
library(dummies)

bikes.dummy <- dummy.data.frame(bikes, names = c('mnth','weekday','weathersit','season'), sep='_', all = FALSE)

bikes <- cbind(bikes, bikes.dummy)

all_vars <- c('yr', 'holiday', 'workingday','mnth_1', 'mnth_2',
            'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9',
            'mnth_10', 'mnth_11', 'mnth_12', 'weekday_0', 'weekday_1', 'weekday_2',
            'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weathersit_1',
            'weathersit_2', 'weathersit_3', 'season_1', 'season_2', 'season_3',
            'season_4', 'temp','cnt')
corr_all <- round(cor(bikes[,all_vars]), 2)[,'cnt']

caoo_df <- data.frame(features = all_vars, coeff = corr_all)

caoo_df <- caoo_df[order(caoo_df$coeff),]

par(mfrow=c(1,1))
barplot(caoo_df$coeff, names.arg = caoo_df$features, las=2)

features <- subset(caoo_df, coeff > 0.25 | coeff < -0.25)
features1 <- features[, 'features']

as.character(features1)

corr_fea <- round(cor(bikes[,c("season_1",  "mnth_1",   "mnth_2",   "weathersit_1", "season_3",   "yr",   "temp",   "cnt")]), 2)
par(mfrow=c(1,1))
ggcorrplot(corr_fea, lab=TRUE)

fea_2 <- c('yr', 'temp','weathersit_1','season_1', 'cnt')

library(rpart)
library(MASS)

train_index = sample(1:nrow(bikes), 0.75 * nrow(bikes))
train = bikes[train_index,]
test = bikes[-train_index,]

train = train[,fea_2]
test = test[,fea_2]

#library(usdm)
#install.packages(usdm)

linearMod <- lm(cnt ~ temp + weathersit_1 + yr + season_1, data=train)  # build linear regression model on full data
summary(linearMod)

predictions = predict(linearMod, test[,c('yr', 'temp','weathersit_1','season_1')])

install.packages('Metrics')
library(Metrics)

rmse(test$cnt, predictions)
