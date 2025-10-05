import cv2
import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import read_stub, save_stub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self, team_1_class_name="white shirt", team_2_class_name="blue shirt"):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
        """
        self.team_colors = {}
        self.player_team_dict = {}

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        
        # Performance optimization: GPU support
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None


    def load_model(self):
        """
        Load the pre-trained CLIP model and processor for visual classification.
        """
        if self.model is not None:
            return
            
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}.")
        
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        logger.info("Processor loaded successfully.")


    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color of a player's jersey from the given frame and bounding box.

        Args:
            frame (numpy.ndarray): The image frame containing the player.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) of the player.

        Returns:
            str: The dominant color of the player's jersey.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        classes = [self.team_1_class_name, self.team_2_class_name]

        # Performance optimization: GPU inference with no_grad
        with torch.no_grad():
            inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        logger.info(f"Player color detected: {class_name}")
        return class_name
    
    def get_player_color_batch(self, frame, bboxes):
        """
        Extract dominant colors for multiple players in batch (performance optimization).

        Args:
            frame (numpy.ndarray): The image frame containing players.
            bboxes (list): List of bounding box coordinates.

        Returns:
            list: List of dominant colors for each player.
        """
        images = []
        for bbox in bboxes:
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(image))

        classes = [self.team_1_class_name, self.team_2_class_name]

        # Performance optimization: batch processing
        with torch.no_grad():
            inputs = self.processor(text=classes, images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        colors = [classes[probs[i].argmax()] for i in range(len(images))]
        return colors
    
    def get_player_team(self, frame, player_id, bbox):  # BUG FIX: correct parameter order
        """
        Assign a team to a player based on their jersey color.

        Args:
            frame (numpy.ndarray): The image frame containing the player.
            player_id (int): The unique identifier of the player.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) of the player.

        Returns:
            int: The team assignment for the player (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, bbox)

        team = 2
        if color == self.team_1_class_name:
            team = 1

        self.player_team_dict[player_id] = team
        logger.info(f"Player {player_id} assigned to {team}.")
        return team


    def get_player_team_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Assign teams to players across multiple video frames.

        Args:
            video_frames (list): List of video frames (numpy arrays).
            player_tracks (dict): Dictionary mapping player IDs to their bounding boxes in each frame.
            read_from_stub (bool): Flag indicating whether to read from a stub file.
            stub_path (str): Path to the stub file.

        Returns:
            dict: Dictionary mapping player IDs to their team assignments.
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                logger.info("Loaded player assignments from stub.")
                return player_assignment

        # Performance optimization: load model once
        self.load_model()

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            
            if frame_num % 50 == 0:
                self.player_team_dict = {}

            # Performance optimization: batch processing for new players
            new_players = [pid for pid in player_track.keys() if pid not in self.player_team_dict]
            
            if new_players:
                new_bboxes = [player_track[pid]['bbox'] for pid in new_players]
                colors = self.get_player_color_batch(video_frames[frame_num], new_bboxes)
                
                for player_id, color in zip(new_players, colors):
                    team = 2 if color == self.team_2_class_name else 1
                    self.player_team_dict[player_id] = team
                    logger.info(f"Player {player_id} assigned to {team}.")

            # Assign teams for all players in frame
            for player_id in player_track.keys():
                player_assignment[frame_num][player_id] = self.player_team_dict.get(player_id, 0)
        
        save_stub(stub_path, player_assignment)
        logger.info("Player assignments saved to stub.")

        return player_assignment