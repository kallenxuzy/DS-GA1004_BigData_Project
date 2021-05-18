#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
$ spark-submit --driver-memory=8g --executor-memory=8g --conf "spark.blacklist.enabled=false" param_train_2nd.py hdfs:/user/te2049/train_index_downsample.parquet hdfs:/user/bm106/pub/MSD/cf_validation.parquet hdfs:/user/te2049/indexer_downsample.parquet
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, PipelineModel
import random
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import itertools
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, expr

def main(spark, train_path, val_path, indexer_model):
    '''
    '''
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    user_index = PipelineModel.load(indexer_model)
    val = user_index.transform(val)
    val = val.select('user_idx','track_idx','count')

    #train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    #val.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    user_id = val.select('user_idx').distinct()
    true_tracks = val.select('user_idx', 'track_idx').orderBy('user_idx',"count",ascending=False)\
                .groupBy('user_idx')\
                .agg(expr('collect_list(track_idx) as tracks'))

    rank_val = [10, 20, 30] #default is 10
    reg_val = [ .01, .1, 1] #default is 1
    alpha_val = [ 1, 10, 20] #default is 1
    param_grid = itertools.product(rank_val, reg_val, alpha_val)

    maps=[]
    precs=[]
    ndcgs=[]
    rmses=[]

    for i in param_grid:
        als = ALS(maxIter=10, userCol ='user_idx', itemCol = 'track_idx', implicitPrefs = True,
        nonnegative=True, ratingCol = 'count', rank = i[0], regParam = i[1], alpha = i[2],numUserBlocks = 50, numItemBlocks = 50, seed=123)
        model = als.fit(train)

        pred_tracks = model.recommendForUserSubset(user_id,500)
        pred_tracks = pred_tracks.select("user_idx", col("recommendations.track_idx").alias("tracks")).sort('user_idx')

        tracks_rdd = pred_tracks.join(F.broadcast(true_tracks), 'user_idx', 'inner') \
                    .rdd.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(tracks_rdd)
        map = metrics.meanAveragePrecision
        prec = metrics.precisionAt(500)
        ndcg = metrics.ndcgAt(500)

        maps.append(map)
        precs.append(prec)
        ndcgs.append(ndcg)
        print('params-rank,reg,alpha: ', i )
        print('meanAveragePrecision: ', map, 'precisionAt: ', prec, 'ndcg: ', ndcg )

        preds = model.transform(val)
        reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction")
        rmse = reg_evaluator.evaluate(preds)

        rmses.append(rmse)
        print('rmse: ', rmse)

    print('param_grid-rank,reg,alpha: ', param_grid)
    print('maps: ', maps)
    print('precs: ', precs)
    print('ncdgs: ', ndcgs)
    print('rmses: ', rmses)



# Only enter this block if we're in main
if __name__ == "__main__":
    #conf = SparkConf()
    #conf.set("spark.executor.memory", "16G")
    #conf.set("spark.driver.memory", '16G')
    #conf.set("spark.executor.cores", "4")
    #conf.set('spark.executor.instances','10')
    #conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #conf.set("spark.default.parallelism", "40")
    #conf.set("spark.sql.shuffle.partitions", "40")
    #spark = SparkSession.builder.config(conf=conf).appName('first_train').getOrCreate()

    spark = SparkSession.builder.appName('first_step').getOrCreate()
    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    indexer_model = sys.argv[3]

    main(spark, train_path, val_path, indexer_model)
