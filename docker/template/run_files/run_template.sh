RAW_PASS="<password>"
SHA_PASS=$(python -c "from notebook.auth import passwd; print(passwd('$RAW_PASS'))" 2>/dev/null)

if [[ $? -ne 0 ]]; then
    SHA_PASS=$(docker run -i continuumio/anaconda3 python -c "from notebook.auth import passwd; print(passwd('$RAW_PASS'))")

    if [[ $? -ne 0 ]]; then
        echo 'Fatal error, could not find anaconda python installed'
        exit 1
    fi
fi

RUNPATH=$(realpath $(dirname "$0")/../)
$RUNPATH/run_docker.sh --name <username> -s ../ssh_keys/<username>.pub -p 8888:8888 -p 6006:6006 -p 2200:22 -v /data:/data -v /notebooks:/notebooks -e JUPYTER_PASSWORD='$SHA_PASS' -e FETCH_TF_CONTRIB=1 deepstack
