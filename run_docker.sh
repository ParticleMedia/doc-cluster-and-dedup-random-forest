#!/bin/bash
java -jar -Dspring.profiles.active=$1 -Dlogging.config=logback-spring.xml RandomForest-0.0.1.jar