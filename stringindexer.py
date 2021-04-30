#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''stringindexer
Usage:
    $ spark-submit stringindexer.py hdfs:/user/bm106/pub/MSD/cf_train.parquet hdfs:/user/bm106/pub/MSD/cf_validation.parquet hdfs:/user/bm106/pub/MSD/cf_test.parquet
'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def main(spark, train_path, val_path, test_path):

    # Create dataframes for all the input files
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    test = spark.read.parquet(test_path)

    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_idx")
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_idx")

    pipeline = Pipeline(stages=[indexer_user,indexer_track])
    all_data=train.union(val).union(test)
    indexer_all = pipeline.fit(all_data)

    train_idx = indexer_all.transform(train)
    train_idx.repartition(5000,'user_idx')
    train_idx.write.mode('overwrite').parquet('hdfs:/user/te2049/train_index.parquet')

    # Validation
    #val_idx = indexer_all.transform(val)
    #val_idx.repartition('user_idx')
    #val_idx.write.mode('overwrite').parquet('hdfs:/user/te2049/val_index.parquet')

    # Test
    #test_idx = indexer_all.transform(test)
    #test_idx.repartition('user_idx')
    #test_idx.write.mode('overwrite').parquet('hdfs:/user/te2049/test_index.parquet')

    print(train_idx.head(3))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    #spark = SparkSession.builder.appName('stringindexer').getOrCreate()

    conf = SparkConf()
    conf.set("spark.executor.memory", "16G")
    conf.set("spark.driver.memory", '16G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances','10')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.default.parallelism", "40")
    conf.set("spark.sql.shuffle.partitions", "40")
    spark = SparkSession.builder.config(conf=conf).appName('stringindexer').getOrCreate()

    # Get the filenames from the command line
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]

    # Call our main routine
    main(spark,train_path, val_path, test_path)
