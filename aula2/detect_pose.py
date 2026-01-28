import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import PoseLandmarksConnections, FaceLandmarksConnections, HandLandmarksConnections
from tqdm import tqdm


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    height, width = annotated_image.shape[:2]
    for pose_landmarks in pose_landmarks_list:
        draw_skeleton_lines(annotated_image, height, pose_landmarks, width)
        draw_landmarks(annotated_image, height, pose_landmarks, width)

    return annotated_image

def draw_hand_landmarks_on_image(rgb_image, hand_landmarks):
    hand_landmarks_list = hand_landmarks
    annotated_image = np.copy(rgb_image)
    height, width = annotated_image.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        hands_draw_skeleton_lines(annotated_image, height, hand_landmarks, width)
        draw_hand_landmarks(annotated_image, height, hand_landmarks, width)

    return annotated_image

def draw_skeleton_lines(annotated_image: np.ndarray, image_height: int, pose_landmarks, image_width: int):
    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
        start_landmark = pose_landmarks[connection.start]
        end_landmark = pose_landmarks[connection.end]

        """This code converts normalized landmark coordinates to pixel coordinates for drawing on the image.
            Normalized coordinates (0.0 to 1.0):
            start_landmark.x and start_landmark.y are normalized values representing positions relative to the image dimensions
            0.0 means the left/top edge, 1.0 means the right/bottom edge
            Pixel coordinates (actual positions):
            Multiply normalized x by image width to get the horizontal pixel position
            Multiply normalized y by image height to get the vertical pixel position
            Convert to integers since pixel positions must be whole numbers
            Example: If start_landmark.x = 0.5 and width = 640:
            int(0.5 * 640) = 320 (center of the image horizontally)
            These pixel coordinates are then used by cv2.line() to draw the skeleton connections on the annotated image.
        """
        start_point = (int(start_landmark.x * image_width), int(start_landmark.y * image_height))
        end_point = (int(end_landmark.x * image_width), int(end_landmark.y * image_height))

        # Draw white skeleton lines
        cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

def hands_draw_skeleton_lines(annotated_image: np.ndarray, image_height: int, hand_landmarks, image_width: int):
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start_landmark = hand_landmarks[connection.start]
        end_landmark = hand_landmarks[connection.end]

        """This code converts normalized landmark coordinates to pixel coordinates for drawing on the image.
            Normalized coordinates (0.0 to 1.0):
            start_landmark.x and start_landmark.y are normalized values representing positions relative to the image dimensions
            0.0 means the left/top edge, 1.0 means the right/bottom edge
            Pixel coordinates (actual positions):
            Multiply normalized x by image width to get the horizontal pixel position
            Multiply normalized y by image height to get the vertical pixel position
            Convert to integers since pixel positions must be whole numbers
            Example: If start_landmark.x = 0.5 and width = 640:
            int(0.5 * 640) = 320 (center of the image horizontally)
            These pixel coordinates are then used by cv2.line() to draw the skeleton connections on the annotated image.
        """
        start_point = (int(start_landmark.x * image_width), int(start_landmark.y * image_height))
        end_point = (int(end_landmark.x * image_width), int(end_landmark.y * image_height))

        # Draw white skeleton lines
        cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

def draw_landmarks(annotated_image: np.ndarray, height: int, pose_landmarks, width: int):
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)

def draw_hand_landmarks(annotated_image: np.ndarray, height: int, hand_landmarks, width: int):
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)

def detect_pose(video_path, video_output):
    # model_path = 'pose_landmarker_full.task'
    model_path = 'hand_landmarker.task'
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    # url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
    if not os.path.exists(model_path):
        print(f"Downloading model file '{model_path}'...")
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded successfully.")
    base_options = python.BaseOptions(model_asset_path=model_path)

    # options = vision.PoseLandmarkerOptions(
    #     base_options=base_options,
    #     min_pose_detection_confidence=0.7,
    #     min_tracking_confidence=0.8,
    #     min_pose_presence_confidence=0.8,
    #     running_mode=vision.RunningMode.VIDEO)

    hand_option = vision.HandLandmarkerOptions(
        base_options=base_options,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )

    # landmarker = vision.PoseLandmarker.create_from_options(options)

    hand_landmarker = vision.HandLandmarker.create_from_options(hand_option)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    frame_timestamp_ms = 0
    for _ in tqdm(range(total_frames), desc="Processando video"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if pose_landmarker_result.hand_landmarks:
            annotated_image = draw_hand_landmarks_on_image(rgb_frame, pose_landmarker_result.hand_landmarks)
            frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_timestamp_ms += int(1000 / fps)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()


def is_arm_up(landmarks: list[list[NormalizedLandmark]]):
    left_eye = landmarks[FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE[0].start]
    #TODO implement it
    print(f'landmark>>> {landmarks[0]}')

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
input_video_path = "../tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
output_video_path = "../tech-challenge/output.mp4"
detect_pose(input_video_path, output_video_path)
