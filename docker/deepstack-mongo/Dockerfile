FROM sensetheflow/deepstack:latest
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Install docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
RUN curl -fsSL https://apt.dockerproject.org/gpg | apt-key add -
RUN add-apt-repository "deb https://apt.dockerproject.org/repo/ ubuntu-$(lsb_release -cs)  main"
RUN cat /etc/apt/sources.list
RUN apt-get update && apt-get -y install docker-engine=1.11.2-0~xenial

# Entry point
RUN echo "#!/bin/bash\n\
echo 'Starting mongodb subdocker'\n\
mkdir -p /data/db\n\
docker run --net=$DOCKER_NETWORK --name mongo-db -d -t -v /data/db:/data/db mongo\n\
/opt/anaconda/run_jupyter.sh" > /opt/mongo_entrypoint.sh.tpl
RUN sed 's/ *$//' /opt/mongo_entrypoint.sh.tpl > /opt/mongo_entrypoint.sh
RUN chmod +x /opt/mongo_entrypoint.sh

# Pymongo library
RUN pip install pymongo

# Entry point #
###############
ENTRYPOINT ["/opt/mongo_entrypoint.sh"]
