FROM sensetheflow/jupyter
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Update + dependencies #
#########################
RUN apt-get update && apt-get install -y apt-transport-https
RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list && \
	 apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
RUN apt-get update && apt-get install -y scala=2.11.6-6 sbt

#WORKDIR /opt/
#RUN git clone https://github.com/p2t2/figaro.git
#WORKDIR /opt/figaro
#RUN sbt compile
#RUN sbt package
#RUN echo 'addSbtPlugin("com.cra" % "figaro" % "4.1.0.0")' >> ~/.sbt/0.13/plugins/build.sbt

RUN mkdir -p ~/.sbt/0.13/plugins/
RUN echo 'addSbtPlugin("io.get-coursier" % "sbt-coursier" % "1.0.0-M15")' >> ~/.sbt/0.13/plugins/build.sbt
RUN curl -L -o coursier https://git.io/vgvpD && chmod +x coursier && mv coursier /usr/local/bin/

RUN coursier fetch com.cra.figaro::figaro:4.1.0.0

WORKDIR /opt/
RUN git clone https://github.com/alexarchambault/jupyter-scala.git
WORKDIR /opt/jupyter-scala
RUN ./jupyter-scala

WORKDIR /root/
RUN curl -sLO http://downloads.typesafe.com/scalaide-pack/4.5.0-vfinal-neon-211-20161213/scala-SDK-4.5.0-vfinal-2.11-linux.gtk.x86_64.tar.gz
RUN tar -xf scala-SDK-4.5.0-vfinal-2.11-linux.gtk.x86_64.tar.gz && rm scala-SDK-4.5.0-vfinal-2.11-linux.gtk.x86_64.tar.gz

# Display
RUN apt-get update && apt-get install -y libxext-dev libxrender-dev libxtst-dev libgtk2.0-0 x11-apps
ENV DISPLAY :0

# Entry point #
###############
ENTRYPOINT ["/opt/anaconda/run_jupyter.sh"]
