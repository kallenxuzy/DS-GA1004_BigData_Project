#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import random
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import itertools
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, expr

def main(spark, train_path, val_path, test_path):
    '''
    '''
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)

    #train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    #val.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    rank_val = [10, 20, 30] #default is 10
    reg_val = [ .01, .1, 1] #default is 1
    alpha_val = [ .5, 1, 10] #default is 1
    param_grid = itertools.product(rank_val, reg_val, alpha_val)
    user_id = val.select('user_idx').distinct()

    #true_tracks = val.select('user_idx', 'track_idx')\
    #            .groupBy('user_idx')\
    #            .agg(expr('collect_list(track_idx) as tracks'))

    for i in param_grid:
        als = ALS(maxIter=1, userCol ='user_idx', itemCol = 'track_idx', implicitPrefs = True,
        nonnegative=True, ratingCol = 'count', rank = i[0], regParam = i[1], alpha = i[2])
        model = als.fit(train)

        rec_result = model.recommendForUserSubset(user_id,500)
        pred_tracks = rec_result.select('user_idx','recommendations.track_idx').withColumnRenamed('recommendations.track_idx', 'tracks')

        w = Window.partitionBy('user_idx').orderBy(col('count').desc())
        true_tracks = val.select('user_idx', 'track_idx', 'count', F.rank().over(w).alias('rank')) \
                        .where('rank <= {0}'.format(500)).groupBy('user_idx') \
                        .agg(expr('collect_list(track_idx) as tracks'))
        pred_RDD = pred_tracks.join(true_tracks, 'user_idx').rdd.map(lambda row: (row[1], row[2]))
        ranking_metrics = RankingMetrics(pred_RDD)
        map_ = metrics.meanAveragePrecision
        mpa = metrics.precisionAt(500)
        ndcg = metrics.ndcgAt(500)
        print(i, 'map score: ', map_, 'ndcg score: ', ndcg, 'map score: ', mpa)



# Only enter this block if we're in main
if __name__ == "__main__":
    conf = SparkConf()
    conf.set("spark.executor.memory", "16G")
    conf.set("spark.driver.memory", '16G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances','10')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.default.parallelism", "40")
    conf.set("spark.sql.shuffle.partitions", "40")
    spark = SparkSession.builder.config(conf=conf).appName('first_train').getOrCreate()

    #spark = SparkSession.builder.appName('first_step').getOrCreate()
    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]

    main(spark, train_path, val_path, test_path)
