#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
$ spark-submit  --driver-memory=4g --executor-memory=4g --conf "spark.blacklist.enabled=false" sample_indexer.py hdfs:/user/bm106/pub/MSD/cf_train_new.parquet hdfs:/user/bm106/pub/MSD/cf_validation.parquet hdfs:/user/bm106/pub/MSD/cf_test.parquet
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

def main(spark, train_path, val_path, test_path):
    '''
    '''
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    test = spark.read.parquet(test_path)

    # list of unique user_ids in train, val, test
    # test, val no overlap
    user_train = set(row['user_id'] for row in train.select('user_id').distinct().collect())
    user_val = set(row['user_id'] for row in val.select('user_id').distinct().collect())
    user_test = set(row['user_id'] for row in test.select('user_id').distinct().collect())
    # combine user_ids for train and val
    user_test=user_test.union(user_val)
    user_to_sample = user_train.difference(user_test)

    #0.5 from learning curve for downsample
    frac=0.5
    k = int(frac * len(user_to_sample))
    user_sampled = random.sample(user_to_sample, k)
    train = train[train.user_id.isin(list(user_test)+user_sampled)]

    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_idx",handleInvalid='skip')
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_idx",handleInvalid='skip')

    pipeline = Pipeline(stages=[indexer_user,indexer_track])
    indexer_all = pipeline.fit(train)

    train_idx = indexer_all.transform(train)
    indexer_all.write().overwrite().save('hdfs:/user/te2049/indexer_downsample.parquet')

    train_idx.repartition(5000,'user_idx')
    train_idx.write.mode('overwrite').parquet('hdfs:/user/te2049/train_index_downsample.parquet')

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
