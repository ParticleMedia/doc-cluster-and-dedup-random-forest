FROM registry2.nb.com/ubuntu/java8:vimcron

RUN mkdir -p /home/services/bin \
    && groupadd -g 1001 services \
    && useradd  -g 1001 -u 1001 -m services

WORKDIR /home/services/bin

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-4.9 g++-4.9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9 \
    && update-alternatives --config gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY run_docker.sh ./
COPY target/RandomForest-0.0.2.jar ./
COPY src/main/resources/logback-spring.xml ./

RUN chown -R services:services /home/services

USER services

ENTRYPOINT [ "sh", "run_docker.sh" ] 