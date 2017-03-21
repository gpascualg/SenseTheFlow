FROM sensetheflow/jupyter
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Update + dependencies #
#########################
RUN pip install xgboost

ENV ENABLE_GOMP_HACK=1

# Entry point #
###############
ENTRYPOINT ["/opt/anaconda/run_jupyter.sh"]
