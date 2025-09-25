# conda create -n roach python=3.7 -y
python3 -m pip install loguru
python3 -m pip install networkx
python3 -m pip install msgpack
python3 -m pip install pyzmq
python3 -m pip install msgpack-numpy

#################### Download the code bases ####################
echo "[x] Compiling the planners..."
###### Build the FrenetOptimalTrajectory Planner ######
echo "[x] Compiling the Frenet Optimal Trajectory planner..."
cd $PYLOT_HOME/dependencies/
rm -rf frenet_optimal_trajectory_planner
git clone https://github.com/erdos-project/frenet_optimal_trajectory_planner.git frenet_optimal_trajectory_planner
cd frenet_optimal_trajectory_planner/
bash build.sh

# ###### Build the RRT* Planner ######
echo "[x] Compiling the RRT* planner..."
cd $PYLOT_HOME/dependencies/
rm -rf rrt_star_planner
git clone https://github.com/erdos-project/rrt_star_planner.git rrt_star_planner
cd rrt_star_planner/
bash build.sh

# ###### Build the Hybrid A* Planner ######
echo "[x] Compiling the Hybrid A* planner..."
cd $PYLOT_HOME/dependencies/
rm -rf hybrid_astar_planner
git clone https://github.com/erdos-project/hybrid_astar_planner.git hybrid_astar_planner
cd hybrid_astar_planner/
bash build.sh