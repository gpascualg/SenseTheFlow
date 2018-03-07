#!/bin/sh

### BEGIN INIT INFO
# Provides:		deepstack-control
# Required-Start:	$docker
# Required-Stop:	$docker
# Default-Start:	2 3 4 5
# Default-Stop:		0 1 6
# Short-Description:	Deepstack dockers
### END INIT INFO

set -e

BASEDIR=/opt/python-libraries/SenseTheFlow
SCRIPTDIR=$BASEDIR/docker/template

. /lib/lsb/init-functions

case "$1" in
    start)
        log_daemon_msg "Starting Dockers" "deepstack-control" || true
	${SCRIPTDIR}/run_all.sh
        ;;

    stop)
        log_daemon_msg "No stop script yet" "deepstack-control" || true
        ;;

    reload)
        log_daemon_msg "No reload script yet" "deepstack-control" || true
        ;;

    restart)
        log_daemon_msg "No restart script yet" "deepstack-control" || true
        ;;

    *)
        log_action_msg "Usage: /etc/init.d/dockercompose {start|stop|restart|reload}" || true
        exit 1
        ;;
esac

exit 0
