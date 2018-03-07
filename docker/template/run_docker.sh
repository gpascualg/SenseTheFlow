#!/bin/bash

getopt --test > /dev/null
if [[ $? -ne 4 ]]; then
    echo "I'm sorry, `getopt --test` failed in this environment."
    exit 1
fi

OPTIONS=s:v:p:e:
LONGOPTIONS=ssh-key:,volume:,port:,env:,name:

# -temporarily store output to be able to check for errors
# -activate advanced mode getopt quoting e.g. via “--options”
# -pass arguments only via   -- "$@"   to separate them correctly
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # e.g. $? == 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# use eval with "$PARSED" to properly handle the quoting
eval set -- "$PARSED"

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -p|--port)
            p="$p -p $2"
            shift 2
            ;;
        -v|--volume)
            v="$v -v $2"
            shift 2
            ;;
	-e|--env)
	    e="$e -e $2"
	    shift 2
	    ;;
        -s|--ssh-key)
            key="$2"
            s=y
            shift 2
            ;;
	--name)
	    n="--name $2"
	    name="$2"
	    shift 2
	    ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# handle non-option arguments
if [[ $# -ne 1 ]]; then
    echo "$0: A tag is required."
    exit 4
fi

# Stop and remove if it already exists
docker stop $name &>/dev/null
docker rm $name &>/dev/null

# Execute docker detached
docker run --runtime=nvidia -tdi $n $p $v $e $1

if [[ $? -ne 0 ]]; then
    echo "Fatal, could not start docker"
    exit 1
fi

# If ssh was active
if [[ $s == y ]]; then
    docker exec -i $name /bin/bash -c "cat  >> ~/.ssh/authorized_keys" < $key
fi
