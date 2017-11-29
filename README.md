# FactorizationMachine
https://github.com/skpenn/FactorizationMachine

FactorizationMachine is an implementation of the model factorization machine using tensorflow.
it allows you to solve the problems about regression, classififcation and prediction.

# Dependency

* [TensorFlow](https://www.tensorflow.org)
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)

# Usage
simplest usage:
```shell
python train.py -d=your_data_path
```
## Add train dataset
```shell
-d=your_data_path, --train_data_path
```

## Add test dataset
```shell
-t=your_test_data, --test_data_path=your_test_data
```
Under this command, the model will test on the dataset after each epoch of train.
and show the loss.

## Batch size
```shell
-b=BATCH_SIZE, --batch_size=BATCH_SIZE
```
Defines the input data size at each step of SGD train.

## Train epoch
```shell
-e=TRAIN_EPOCH, --train_epoch=TRAIN_EPOCH
```
Defines the times that the train data used in optimization.

## Learning rate
```shel
-r=LEARNING_RATE, --learning_rate=LEARNING_RATE
```
Defines the learing rate of SGD optimizer.

## Factor dimension
```shell
-f=FACTOR_DIM, --factor_dim=FACTOR_DIM
```
The dimension of factor vectors that representing each feature.

## Cross entropy loss
```shell
-x, --use_cross_entropy
```
If this parameter added, cross entropy loss will be used as optimization loss,
otherwise, MSE loss will be used. Available only on 2-class classification and
0-1 prediction.

## Dump factor vectors
```shell
-o=PATH_TO_DUMP, --dump_factors_path=PATH_TO_DUMP
```
Under this command, the factor vectors in the last train will be dumped into
the given path with text format.

# Input data format
The model allows a data format of CSV only, with comma as delimiter. The first row descripts
the feature name, others record the values. all values must be numerical. The first column is
the result to be predicted.
For example, a movie rating dataset that predicts a user's score to a new movie, may have a
train dataset as follow type:
> rating,userId,movieId,RatingOfMovie=TI,RatingOfMovie=SW,LastRate=TI,LastRate=SW,RateTime  
> 9,116,20090811,9,6,1,0,14  
> 5,117,20090811,8,10,0,1,13  
> ...

