from typing import Optional

from pydantic import BaseModel, Field

from .base_models import ExecutionError, BaseInputModel


class BodyLandmarks:
    def __init__(self, landmarks):
        self.left_shoulder = landmarks[11]
        self.right_shoulder = landmarks[12]
        self.left_hip = landmarks[23]
        self.right_hip = landmarks[24]
        self.left_wrist = landmarks[15]
        self.right_wrist = landmarks[16]
        self.left_elbow = landmarks[13]
        self.right_elbow = landmarks[14]
        self.left_ankle = landmarks[27]
        self.right_ankle = landmarks[28]
        self.left_knee = landmarks[25]
        self.right_knee = landmarks[26]

    def _get_camera_orientation(self):
        shoulder_width = abs(self.left_shoulder.x - self.right_shoulder.x)
        hip_width = abs(self.left_hip.x - self.right_hip.x)

        avg_width = (shoulder_width + hip_width) / 2

        if avg_width > 0.2:
            return "frontal"
        else:
            return "sideways"

    def is_walking_pose(self, threshold=0.05):
        orientation = self._get_camera_orientation()

        if orientation == "frontal":
            return self._detect_walking_frontal(threshold)
        else:
            return self._detect_walking_sideways(threshold)

    def _detect_walking_frontal(self, threshold):
        avg_hip_z = (self.left_hip.z + self.right_hip.z) / 2

        left_knee_forward = self.left_knee.z < (avg_hip_z - threshold)
        right_knee_forward = self.right_knee.z < (avg_hip_z - threshold)
        left_knee_behind = self.left_knee.z > (avg_hip_z + threshold)
        right_knee_behind = self.right_knee.z > (avg_hip_z + threshold)

        return (left_knee_forward and right_knee_behind) or (right_knee_forward and left_knee_behind)

    def _detect_walking_sideways(self, threshold=0.08):
        avg_hip_x = (self.left_hip.x + self.right_hip.x) / 2

        left_knee_forward = self.left_knee.x < (avg_hip_x - threshold)
        right_knee_forward = self.right_knee.x < (avg_hip_x - threshold)
        left_knee_behind = self.left_knee.x > (avg_hip_x + threshold)
        right_knee_behind = self.right_knee.x > (avg_hip_x + threshold)

        knee_separation = abs(self.left_knee.x - self.right_knee.x)
        has_stride = knee_separation > threshold

        return has_stride and ((left_knee_forward and right_knee_behind) or (right_knee_forward and left_knee_behind))

    def is_standing_still(self):
        orientation = self._get_camera_orientation()

        if orientation == "frontal":
            avg_hip_z = (self.left_hip.z + self.right_hip.z) / 2
            left_aligned = abs(self.left_knee.z - avg_hip_z) < 0.03
            right_aligned = abs(self.right_knee.z - avg_hip_z) < 0.03
            return left_aligned and right_aligned
        else:
            avg_hip_x = (self.left_hip.x + self.right_hip.x) / 2
            avg_knee_x = (self.left_knee.x + self.right_knee.x) / 2
            knee_separation = abs(self.left_knee.x - self.right_knee.x)
            return abs(avg_knee_x - avg_hip_x) < 0.05 and knee_separation < 0.08

    def is_jumping_pose(self, threshold=0.15):
        avg_hip_y = (self.left_hip.y + self.right_hip.y) / 2
        avg_ankle_y = (self.left_ankle.y + self.right_ankle.y) / 2
        avg_knee_y = (self.left_knee.y + self.right_knee.y) / 2

        feet_elevated = avg_ankle_y < (avg_hip_y - threshold)
        knees_bent = avg_knee_y < avg_hip_y
        both_feet_together = abs(self.left_ankle.y - self.right_ankle.y) < 0.1

        return feet_elevated and knees_bent and both_feet_together

    def is_crouching_pose(self, threshold=0.2):
        avg_hip_y = (self.left_hip.y + self.right_hip.y) / 2
        avg_knee_y = (self.left_knee.y + self.right_knee.y) / 2
        avg_shoulder_y = (self.left_shoulder.y + self.right_shoulder.y) / 2

        knees_bent_significantly = (avg_knee_y - avg_hip_y) > threshold
        torso_lowered = (avg_shoulder_y - avg_hip_y) < 0.3

        return knees_bent_significantly and torso_lowered

    def get_arm_extension(self, side='left'):
        if side == 'left':
            shoulder = self.left_shoulder
            wrist = self.left_wrist
        else:
            shoulder = self.right_shoulder
            wrist = self.right_wrist

        distance = ((wrist.x - shoulder.x) ** 2 + (wrist.y - shoulder.y) ** 2) ** 0.5
        return distance

    def is_arm_extended(self, side='left', threshold=0.3):
        return self.get_arm_extension(side) > threshold

    def is_hand_raised(self, side='left'):
        avg_shoulder_y = (self.left_shoulder.y + self.right_shoulder.y) / 2
        wrist = self.left_wrist if side == 'left' else self.right_wrist
        return wrist.y < avg_shoulder_y

    def is_hand_down(self, side='left'):
        avg_hip_y = (self.left_hip.y + self.right_hip.y) / 2
        wrist = self.left_wrist if side == 'left' else self.right_wrist
        return wrist.y > avg_hip_y

    def is_hand_at_waist(self, side='left'):
        avg_shoulder_y = (self.left_shoulder.y + self.right_shoulder.y) / 2
        avg_hip_y = (self.left_hip.y + self.right_hip.y) / 2
        wrist = self.left_wrist if side == 'left' else self.right_wrist
        return avg_shoulder_y < wrist.y < avg_hip_y

    def is_hand_forward(self, side='left', threshold=0.1):
        if side == 'left':
            shoulder_z = self.left_shoulder.z
            wrist_z = self.left_wrist.z
        else:
            shoulder_z = self.right_shoulder.z
            wrist_z = self.right_wrist.z

        return wrist_z < (shoulder_z - threshold)

    def are_arms_crossed(self, threshold=0.1):
        left_wrist_crosses = self.left_wrist.x > (self.right_shoulder.x - threshold)
        right_wrist_crosses = self.right_wrist.x < (self.left_shoulder.x + threshold)
        both_at_chest = (self.left_wrist.y > self.left_shoulder.y and
                         self.right_wrist.y > self.right_shoulder.y)

        return (left_wrist_crosses or right_wrist_crosses) and both_at_chest

    def detect_hand_activity(self):
        left_raised = self.is_hand_raised('left')
        right_raised = self.is_hand_raised('right')
        left_down = self.is_hand_down('left')
        right_down = self.is_hand_down('right')
        left_extended = self.is_arm_extended('left')
        right_extended = self.is_arm_extended('right')
        left_forward = self.is_hand_forward('left')
        right_forward = self.is_hand_forward('right')

        if self.are_arms_crossed():
            return "arms_crossed"

        if left_raised and right_raised:
            if left_extended and right_extended:
                return "both_arms_extended_up"
            return "hands_raised"

        if left_extended and right_extended:
            return "both_arms_extended"

        if left_forward and right_forward:
            return "hands_forward"

        if left_raised and not right_raised:
            if left_extended:
                return "left_arm_extended"
            return "left_hand_raised"

        if right_raised and not left_raised:
            if right_extended:
                return "right_arm_extended"
            return "right_hand_raised"

        if left_extended and not right_extended:
            return "left_arm_extended"

        if right_extended and not left_extended:
            return "right_arm_extended"

        if left_forward and not right_forward:
            return "left_hand_forward"

        if right_forward and not left_forward:
            return "right_hand_forward"

        if left_down and right_down:
            return "hands_down"

        if self.is_hand_at_waist('left') and self.is_hand_at_waist('right'):
            return "hands_at_waist"

        return "hands_neutral"


class ActivityDetectionInput(BaseInputModel):
    media_pipe_model: str = Field(description="MediaPipe model for pose detection. Valid values: 'LITE', 'FULL', 'HEAVY'")


class Activity(BaseModel):
    hands_activity: str = Field(description="hands movement activity")
    movement_activity: str = Field(description="movement activity")

class ActivityDetection(BaseModel):
    frame :int = Field(description="Frame number")
    timestamp : str = Field(description="Timestamp")
    activities : list[Activity] = Field(description="Activities")


class ActivityAnomaly(BaseModel):
    frame : int = Field(description="Frame number")
    timestamp : float = Field(description="Timestamp")
    type: str = Field(description="Anomaly type")
    details : Optional[str] = Field(None, description="Anomaly details")


class ActivityDetectionResult(BaseModel):
    frames_analyzed: int = Field(description="Frames analyzed")
    detections: list[ActivityDetection] = Field(description="Activities detected")
    activity_summary: dict[str, int] = Field(description="Activity summary")
    pose_detections : int = Field(description="Total Poses detected")
    error : Optional[ExecutionError] = Field(None, description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "frames_analyzed": 100,
                "activities": [
                    {
                        "frame": 1,
                        "timestamp": 0.03,
                        "activities": ["standing", "hands_down"]
                    }
                ],
                "activity_summary": {
                    "standing": 60,
                    "sitting": 25,
                    "moving": 15,
                    "hands_raised": 40,
                    "hands_down": 60
                },
                "pose_detections": 98,
                "anomalies": []
            }
        }