# ACTIVATE CONDA ENV BEFORE RUNNING
jn="noise_action"

script=run_agent.sbatch

jid[1]=$(./run_help.sh ${jn} ${script} | tr -dc '0-9')
echo ${jid[1]}

# for j in {2..2}
# do
#   jid[${j}]=$(./run_help2.sh ${jn}_${j} ${jid[$((j-1))]} ${script} | tr -dc '0-9')
#   echo ${jid[$((j))]}
# done
