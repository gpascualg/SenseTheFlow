for f in $(ls ./run_files/)
do
	if [ $f != "run_template.sh" ]
	then
		echo "Running [$f]"
	        $(./run_files/$f)
	fi
done

