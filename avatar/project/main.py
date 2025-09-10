# main.py
from avatar_ui import run_app

OPTIONS = {
    1: ("dance_poses.json", "naruto_parts"),
    2: ("dance_poses.json", "mannequin_parts"),
    3: ("dance_poses.json", "naruto_parts_alt"),
    4: ("dance_poses.json", "naruto_parts"),
}

if __name__ == "__main__":
    run_app(OPTIONS)
