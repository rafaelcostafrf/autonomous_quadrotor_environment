
vblank_mode=0 gnome-terminal --tab --title="Mother Process" -- python3 vldg_training.py

for((i=1;i<=$1;i++))
do
	sleep 3
	vblank_mode=0 gnome-terminal --tab --title="Child Process" -- python3 vldg_training.py -c
done

gnome-terminal --tab --title="NVIDIA STATS" -- watch -n 1 nvidia-smi
gnome-terminal --tab --title="PC STATS" -- htop
