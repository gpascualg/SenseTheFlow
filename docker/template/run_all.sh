#!/bin/bash

RUNPATH=$(realpath $(dirname "$0"))

. /lib/lsb/init-functions

log_daemon_msg "$(ls $RUNPATH/run_files/)" "deepstack-control" || true; echo

for f in $(ls $RUNPATH/run_files/)
do
	if [ $f != "run_template.sh" ]
	then
		log_daemon_msg "Running [$f]" "deepstack-control" || true; echo
	        $RUNPATH/run_files/$f
	fi
done

