#!/bin/bash
java -Xms5G -Xmx7G -XX:+UseConcMarkSweepGC -XX:+CMSParallelRemarkEnabled -XX:CMSInitiatingOccupancyFraction=60 -jar -Dspring.profiles.active=$1 -Dlogging.config=logback-spring.xml RandomForest-0.0.1.jar