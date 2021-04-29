#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit one_train.py hdfs:/user/te2049/train_index.parquet hdfs:/user/bm106/pub/MSD/cf_test.parquet hdfs:/user/te2049/indexer.parquet
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf
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

    #train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    #val.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    user_id = val.select('user_idx').distinct()
    #true_tracks = val.select('user_idx', 'track_idx')\
    #            .groupBy('user_idx')\
    #            .agg(expr('collect_list(track_idx) as tracks'))

    als = ALS(maxIter=1, userCol ='user_idx', itemCol = 'track_idx', implicitPrefs = True, \
        nonnegative=True, ratingCol = 'count', rank = 10, regParam = 1, alpha = 1)
    model = als.fit(train)

    rec_result = model.recommendForUserSubset(user_id,500)
    #pred_tracks = rec_result.select('user_idx','recommendations.track_idx')\
                #.withColumnRenamed('recommendations.track_idx', 'tracks')

    #true_tracks = val.select('user_idx', 'track_idx')\
    #                .groupBy('user_idx')\
    #                .agg(expr('collect_list(track_idx) as true_item'))
    w = Window.partitionBy('user_idx').orderBy(col('count').desc())
    true_tracks = val.select('user_idx', 'track_idx', 'count', F.rank().over(w).alias('rank')) \
                    .where('rank <= {}'.format(500)).groupBy('user_idx') \
                    .agg(expr('collect_list(track_idx) as true_track_idx'))
    #true_tracks = true_tracks.select('user_idx', '')
    pred_RDD = rec_result.join(true_tracks, 'user_idx').rdd.map(lambda row: ([rec.track_idx for rec in row["recommendations"]],row["true_track_idx"])) #(row['tracks'], row['true_track_idx']))
    ranking_metrics = RankingMetrics(pred_RDD)
    #map_ = ranking_metrics.meanAveragePrecision
    #mpa = ranking_metrics.precisionAt(500)
    #ndcg = ranking_metrics.ndcgAt(500)
    #print('map score: ', map_, 'ndcg score: ', ndcg, 'map score: ', mpa)



# Only enter this block if we're in main
if __name__ == "__main__":
    #conf = SparkConf()
    #conf.set("spark.executor.memory", "4g")
    #conf.set("spark.driver.memory", '4g')
    #conf.set("spark.executor.cores", "4")
    #conf.set('spark.executor.instances','10')
    #conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #conf.set("spark.default.parallelism", "40")
    #conf.set("spark.sql.shuffle.partitions", "40")
    #conf.set("spark.blacklist.enabled", "False")
    #spark = SparkSession.builder.config(conf=conf).appName('first_train').getOrCreate()
    #sc = spark.sparkContext

    spark = SparkSession.builder.appName('first_step').getOrCreate()
    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    indexer_model = sys.argv[3]

    main(spark, train_path, val_path, indexer_model)
