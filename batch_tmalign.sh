tmalign=./main.py
pdb_dir=test_pdbs
for i in `ls ${pdb_dir}/1`
do
	for j in `ls ${pdb_dir}/2`
	do
		echo ${i} ${j}
		${tmalign} ${pdb_dir}/1/${i} ${pdb_dir}/2/${j} | grep -e "TM-score" > batch_tm_res_greped/${i}_${j}.tmalign_res 
	done
done
