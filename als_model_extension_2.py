#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit --driver-memory=4g --executor-memory=4g --conf "spark.blacklist.enabled=false" learning_curve.py hdfs:/user/bm106/pub/MSD/cf_train_new.parquet hdfs:/user/bm106/pub/MSD/cf_validation.parquet hdfs:/user/bm106/pub/MSD/cf_test.parquet
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
from time import time

def main(spark, train_path, val_path, test_path):

    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    test = spark.read.parquet(test_path)

    #sampling fraction
    data_size=[]
    precs=[]

    for numrow in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]: # 0.1%, 0.5%, 1%, 5%, 10%, 15%, 20%, 30% of all training records
        train_sampled = train.sample(numrow)

        indexer_user = StringIndexer(inputCol="user_id", outputCol="user_idx",handleInvalid='skip')
        indexer_track = StringIndexer(inputCol="track_id", outputCol="track_idx",handleInvalid='skip')

        pipeline = Pipeline(stages=[indexer_user,indexer_track])
        indexer_all = pipeline.fit(train_sampled)

        train_idx=indexer_all.transform(train_sampled)
        val_idx = indexer_all.transform(val)

        #train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        #val.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        user_id = val_idx.select('user_idx').distinct()
        true_tracks = val_idx.select('user_idx', 'track_idx').groupBy('user_idx')\
                    .agg(expr('collect_list(track_idx) as tracks'))

        st = time()
        als = ALS(maxIter=1, userCol ='user_idx', itemCol = 'track_idx', implicitPrefs = True, \
            nonnegative=True, ratingCol = 'count', rank = 150, regParam = 0.05, alpha = 1)
        model = als.fit(train_idx)
        t = round(time()-st, 5)

        pred_tracks = model.recommendForUserSubset(user_id,500)
        pred_tracks = pred_tracks.select("user_idx", col("recommendations.track_idx").alias("tracks")).sort('user_idx')

        tracks_rdd = pred_tracks.join(F.broadcast(true_tracks), 'user_idx', 'inner') \
                    .rdd.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(tracks_rdd)
        #map = metrics.meanAveragePrecision
        prec = metrics.precisionAt(500)
        #ndcg = metrics.ndcgAt(500)
        print('precisionAt: ', prec)

        #preds = model.transform(val_idx)
        #reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction")
        #rmse = reg_evaluator.evaluate(preds)
        #print('frac:', frac)
        #print('rmse: ', rmse)
        print('time:', t)

        data_size.append(int)
        precs.append(prec)


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
    test_path = sys.argv[3]

    main(spark, train_path, val_path, test_path)
