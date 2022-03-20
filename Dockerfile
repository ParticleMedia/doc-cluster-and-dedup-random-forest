FROM registry2.nb.com/ubuntu/java8:vimcron

RUN mkdir -p /home/services/bin \
    && groupadd -g 1001 services \
    && useradd  -g 1001 -u 1001 -m services

WORKDIR /home/services/bin

COPY run_docker.sh ./
COPY target/RandomForest-0.0.1.jar ./
COPY src/main/resources/logback-spring.xml ./

RUN chown -R services:services /home/services

USER services

ENTRYPOINT [ "sh", "run_docker.sh" ] 