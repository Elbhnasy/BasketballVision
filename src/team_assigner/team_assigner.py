import cv2
import logging
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Tuple, Optional, Union
from utils import read_stub, save_stub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained CLIP model to classify players into two teams based on their
    jersey appearance. It maintains consistent team assignment for each player across frames
    with confidence-based filtering and batch processing for improved performance.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
        confidence_threshold (float): Minimum confidence score for team assignment.
        device (str): Computing device ('cuda' or 'cpu').
        batch_size (int): Number of images to process in a single batch.
    """
    
    def __init__(
        self, 
        team_1_class_name: str = "white shirt",
        team_2_class_name: str = "blue shirt",
        confidence_threshold: float = 0.6,
        batch_size: int = 16
    ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name: Description of Team 1's jersey appearance.
            team_2_class_name: Description of Team 2's jersey appearance.
            confidence_threshold: Minimum confidence score (0-1) for team assignment.
            batch_size: Number of players to process in parallel for efficiency.
        """
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_confidence_scores = {}
        
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"TeamAssigner initialized with device: {self.device}")


    def load_model(self) -> None:
        """
        Load the pre-trained CLIP model and processor for visual classification.
        Uses GPU if available for faster inference.
        """
        if self.model is not None:
            logger.info("Model already loaded, skipping...")
            return
            
        try:
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Model loaded successfully on {self.device}.")
            
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            logger.info("Processor loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


    def unload_model(self) -> None:
        """
        Unload model from memory to free up resources.
        """
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded from memory.")


    def _validate_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validate that the bounding box is within frame dimensions and has reasonable size.

        Args:
            frame: The image frame.
            bbox: Bounding box coordinates (x1, y1, x2, y2).

        Returns:
            True if bbox is valid, False otherwise.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Check boundaries
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # Check minimum size (avoid tiny boxes)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
            
        # Check maximum size (avoid full-frame boxes)
        if (x2 - x1) > w * 0.9 or (y2 - y1) > h * 0.9:
            return False
            
        return True


    def _extract_player_image(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """
        Extract and preprocess player image from frame using bounding box.

        Args:
            frame: The image frame containing the player.
            bbox: The bounding box coordinates (x1, y1, x2, y2) of the player.

        Returns:
            PIL Image of the player or None if extraction fails.
        """
        try:
            if not self._validate_bbox(frame, bbox):
                logger.warning(f"Invalid bounding box: {bbox}")
                return None
                
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Extract region and focus on upper body (jersey area)
            image = frame[y1:y2, x1:x2]
            h, w = image.shape[:2]
            
            # Focus on upper 60% where jersey is most visible
            upper_portion = int(h * 0.6)
            image = image[:upper_portion, :]
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
            
        except Exception as e:
            logger.warning(f"Failed to extract player image: {e}")
            return None


    def get_player_color_batch(
        self, 
        frame: np.ndarray, 
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[str, float]]:
        """
        Extract dominant colors for multiple players in batch for efficiency.

        Args:
            frame: The image frame containing players.
            bboxes: List of bounding box coordinates.

        Returns:
            List of tuples (class_name, confidence_score) for each player.
        """
        images = []
        valid_indices = []
        
        # Extract valid images
        for idx, bbox in enumerate(bboxes):
            img = self._extract_player_image(frame, bbox)
            if img is not None:
                images.append(img)
                valid_indices.append(idx)
        
        if not images:
            return [("unknown", 0.0)] * len(bboxes)
        
        # Two-class classification: Team 1 vs Team 2
        classes = [self.team_1_class_name, self.team_2_class_name]
        
        try:
            # Batch processing for efficiency
            with torch.no_grad():
                inputs = self.processor(
                    text=classes,
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Extract results
            results = [("unknown", 0.0)] * len(bboxes)
            for i, valid_idx in enumerate(valid_indices):
                class_idx = probs[i].argmax().item()
                confidence = probs[i][class_idx].item()
                class_name = classes[class_idx]
                
                results[valid_idx] = (class_name, confidence)
                logger.debug(f"Player color: {class_name}, confidence: {confidence:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch color detection failed: {e}")
            return [("unknown", 0.0)] * len(bboxes)


    def get_player_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Extract the dominant color of a single player's jersey.

        Args:
            frame: The image frame containing the player.
            bbox: The bounding box coordinates (x1, y1, x2, y2) of the player.

        Returns:
            Tuple of (color_class, confidence_score).
        """
        results = self.get_player_color_batch(frame, [bbox])
        return results[0]

    
    def get_player_team(
        self, 
        frame: np.ndarray, 
        player_id: int, 
        bbox: Tuple[int, int, int, int]
    ) -> int:
        """
        Assign a team to a player based on their jersey color with confidence filtering.

        Args:
            frame: The image frame containing the player.
            player_id: The unique identifier of the player.
            bbox: The bounding box coordinates (x1, y1, x2, y2) of the player.

        Returns:
            The team assignment for the player (1, 2, or 0 for unknown).
        """
        # Return cached assignment if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color, confidence = self.get_player_color(frame, bbox)
        
        # Filter by confidence threshold
        if confidence < self.confidence_threshold:
            logger.info(f"Player {player_id} confidence too low: {confidence:.2f}")
            return 0  # Unknown/uncertain
        
        # Assign team based on color
        team = 2 if color == self.team_2_class_name else 1
        
        self.player_team_dict[player_id] = team
        self.player_confidence_scores[player_id] = confidence
        logger.info(f"Player {player_id} assigned to Team {team} (confidence: {confidence:.2f})")
        
        return team


    def get_player_team_across_frames(
        self,
        video_frames: List[np.ndarray],
        player_tracks: List[Dict],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None
    ) -> List[Dict[int, int]]:
        """
        Assign teams to players across multiple video frames with optimized batch processing.

        Args:
            video_frames: List of video frames (numpy arrays).
            player_tracks: List of dictionaries mapping player IDs to their track info per frame.
            read_from_stub: Flag indicating whether to read from a stub file.
            stub_path: Path to the stub file for caching results.

        Returns:
            List of dictionaries mapping player IDs to their team assignments per frame.
        """
        # Try loading from cache
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            logger.info("Loaded player assignments from stub.")
            return player_assignment

        # Load model once
        self.load_model()
        
        player_assignment = []
        reassignment_interval = 100  # Reset team assignments every 100 frames
        
        try:
            for frame_num, player_track in enumerate(player_tracks):
                player_assignment.append({})
                
                # Periodic reassignment to handle jersey changes/lighting
                if frame_num % reassignment_interval == 0:
                    logger.info(f"Resetting team assignments at frame {frame_num}")
                    self.player_team_dict = {}
                
                # Batch process players in current frame
                player_ids = list(player_track.keys())
                bboxes = [player_track[pid]['bbox'] for pid in player_ids]
                
                # Process in batches for efficiency
                for i in range(0, len(player_ids), self.batch_size):
                    batch_ids = player_ids[i:i + self.batch_size]
                    batch_bboxes = bboxes[i:i + self.batch_size]
                    
                    # Get colors for new players only
                    new_players = [pid for pid in batch_ids if pid not in self.player_team_dict]
                    
                    if new_players:
                        new_bboxes = [player_track[pid]['bbox'] for pid in new_players]
                        colors_confidences = self.get_player_color_batch(
                            video_frames[frame_num],
                            new_bboxes
                        )
                        
                        # Assign teams based on results
                        for pid, (color, confidence) in zip(new_players, colors_confidences):
                            if confidence >= self.confidence_threshold:
                                team = 2 if color == self.team_2_class_name else 1
                                self.player_team_dict[pid] = team
                                self.player_confidence_scores[pid] = confidence
                                logger.info(f"Frame {frame_num}: Player {pid} -> Team {team} ({confidence:.2f})")
                    
                    # Assign teams to all players in batch
                    for pid in batch_ids:
                        team = self.player_team_dict.get(pid, 0)
                        player_assignment[frame_num][pid] = team
                
                # Progress logging
                if frame_num % 50 == 0:
                    logger.info(f"Processed {frame_num}/{len(video_frames)} frames")
            
            # Save to cache
            if stub_path:
                save_stub(stub_path, player_assignment)
                logger.info("Player assignments saved to stub.")
            
            return player_assignment
            
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            raise
        
        finally:
            # Clean up model from memory
            self.unload_model()




# --- DELETED CODE BELOW ---

# import cv2
# import logging
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from utils import read_stub, save_stub

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TeamAssigner:
#     """
#     A class that assigns players to teams based on their jersey colors using visual analysis.

#     The class uses a pre-trained vision model to classify players into teams based on their
#     appearance. It maintains a consistent team assignment for each player across frames.

#     Attributes:
#         team_colors (dict): Dictionary storing team color information.
#         player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
#         team_1_class_name (str): Description of Team 1's jersey appearance.
#         team_2_class_name (str): Description of Team 2's jersey appearance.
#     """
#     def __init__(self, team_1_class_name="white shirt", team_2_class_name="blue shirt"):
#         """
#         Initialize the TeamAssigner with specified team jersey descriptions.

#         Args:
#             team_1_class_name (str): Description of Team 1's jersey appearance.
#             team_2_class_name (str): Description of Team 2's jersey appearance.
#         """
#         self.team_colors = {}
#         self.player_team_dict = {}

#         self.team_1_class_name = team_1_class_name
#         self.team_2_class_name = team_2_class_name

#     def load_model(self):
#         """
#         Load the pre-trained CLIP model and processor for visual classification.
#         """
#         self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
#         logger.info("Model loaded successfully.")
#         self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
#         logger.info("Processor loaded successfully.")

#     def get_player_color(self, frame, bbox):
#         """
#         Extract the dominant color of a player's jersey from the given frame and bounding box.

#         Args:
#             frame (numpy.ndarray): The image frame containing the player.
#             bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) of the player.  

#         Returns:
#             str: The dominant color of the player's jersey.
#         """
#         image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(image)

#         classes = [self.team_1_class_name, self.team_2_class_name]

#         inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
#         outputs = self.model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1)

#         class_name = classes[probs.argmax(dim=1)[0]]
#         logger.info(f"Player color detected: {class_name}")
#         return class_name
    
#     def get_player_team(self, frame, player_id, bbox):
#         """
#         Assign a team to a player based on their jersey color.

#         Args:
#             frame (numpy.ndarray): The image frame containing the player.
#             player_id (int): The unique identifier of the player.
#             bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) of the player.

#         Returns:
#             str: The team assignment for the player ("team_1" or "team_2").
#         """
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]

#         color = self.get_player_color(frame, bbox)

#         team = 2
#         if color == self.team_1_class_name:
#             team = 1

#         self.player_team_dict[player_id] = team
#         logger.info(f"Player {player_id} assigned to {team}.")
#         return team

#     def get_player_team_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
#         """
#         Assign teams to players across multiple video frames.

#         Args:
#             video_frames (list): List of video frames (numpy arrays).
#             player_tracks (dict): Dictionary mapping player IDs to their bounding boxes in each frame.
#             read_from_stub (bool): Flag indicating whether to read from a stub file.
#             stub_path (str): Path to the stub file.

#         Returns:
#             dict: Dictionary mapping player IDs to their team assignments.
#         """
#         player_assignment = read_stub(read_from_stub,stub_path)
#         if player_assignment is not None:
#             if len(player_assignment) == len(video_frames):
#                 logger.info("Loaded player assignments from stub.")
#                 return player_assignment

#         self.load_model()

#         player_assignment=[]
#         for frame_num, player_track in enumerate(player_tracks):        
#             player_assignment.append({})
            
#             if frame_num %50 ==0:
#                 self.player_team_dict = {}

#             for player_id, track in player_track.items():
#                 team = self.get_player_team(video_frames[frame_num],   
#                                                     track['bbox'],
#                                                     player_id)
#                 player_assignment[frame_num][player_id] = team
        
#         save_stub(stub_path,player_assignment)
#         logger.info("Player assignments saved to stub.")

#         return player_assignment