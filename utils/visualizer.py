import random
from pathlib import Path

import cv2
import numpy as np

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class Visualizer:
    def __init__(self, output_dir, save=False):
        self.color_dict = {}
        self.output_dir = Path(output_dir)
        self.save = save
        self.cnt = 0

        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_predictions(self, frame, predictions, instance_ids):
        output_frame = frame.copy()

        for i in range(predictions.shape[0]):
            mask = predictions[i]
            instance_id = instance_ids[i]

            if instance_id not in self.color_dict:
                self.color_dict[instance_id] = random_color()

            color = self.color_dict[instance_id]

            colored_mask = np.zeros_like(output_frame)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]

            alpha = 0.5
            output_frame = cv2.addWeighted(output_frame, 1, colored_mask, alpha, 0)

            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            cv2.putText(output_frame, str(instance_id), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.save:
            output_path = self.output_dir / f'{str(self.cnt).zfill(6)}.png'
            cv2.imwrite(str(output_path), output_frame)

        self.cnt += 1
        return output_frame

    def visualize_frame(self, frame):
        output_frame = frame.copy()

        if self.save:
            output_path = self.output_dir / f'{str(self.cnt).zfill(6)}.png'
            cv2.imwrite(str(output_path), output_frame)

        self.cnt += 1
        return output_frame