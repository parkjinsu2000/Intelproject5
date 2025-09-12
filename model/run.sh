export GST_DEBUG=1
#export GST_DEBUG_FILE=/tmp/gst_debug.log
#export GST_DEBUG_NO_COLOR=1

#export TAPPAS_LOG_LEVEL=3
#export HAILO_DEBUG_LEVEL=3
# gstreamer dump (later), profiler for time estimation(later, performance comparison)

####################################
# .npz pre-processed cache files are not compatable for each versions. remove it manually.
####################################

#source venv_py/bin/activate
#python3 py/main.py vid/ref.mp4 --source vid/ref.mp4 --no-mirror --disp-scale 0.5 
#python3 py/main.py vid/duo.mp4 --source vid/duo.mp4 --no-mirror --disp-scale 0.5
#python3 py/main.py vid/ref.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5 
#python3 py/main.py vid/solo.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5

cd ../hailo-rpi5-examples
source setup_env.sh
cd -
python3.11 -u py-hailo/main.py vid/ref.mp4 --source vid/ref.mp4 --no-mirror #--every 20
#python3 py-hailo/main.py vid/duo.mp4 --source vid/duo.mp4 --no-mirror --disp-scale 0.5
#python3 py-hailo/main.py vid/ref.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5
#python3 py-hailo/main.py vid/solo.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5
