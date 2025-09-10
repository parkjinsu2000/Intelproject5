#python3 py/main.py vid/ref.mp4 --source vid/ref.mp4 --no-mirror --disp-scale 0.5 
#python3 py/main.py vid/duo.mp4 --source vid/duo.mp4 --no-mirror --disp-scale 0.5
#python3 py/main.py vid/ref.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5 
#python3 py/main.py vid/solo.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5

python3 -u py-hailo/main.py vid/ref.mp4 --source vid/ref.mp4 --no-mirror --disp-scale 0.5
#python3 py-hailo/main.py vid/duo.mp4 --source vid/duo.mp4 --no-mirror --disp-scale 0.5
#python3 py-hailo/main.py vid/ref.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5
#python3 py-hailo/main.py vid/solo.mp4 --source vid/duo3.mp4 --no-mirror --disp-scale 0.5
